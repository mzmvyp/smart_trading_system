# indicators/trend_analyzer.py
"""
📊 TREND ANALYZER - SMART TRADING SYSTEM v2.0
Análise de tendência multi-timeframe com hierarquia inteligente
1D (contexto) → 4H (setup) → 1H (entrada)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger
from utils.decorators import timing, cache
from utils.helpers import calculate_correlation
from core.market_data import MarketData, get_multiple_data
from core.market_structure import TrendDirection, MarketStructureAnalysis, analyze_market_structure
from config.settings import settings

logger = get_logger(__name__)

class TrendStrength(Enum):
    """💪 Força da tendência"""
    VERY_WEAK = "very_weak"      # 0-0.2
    WEAK = "weak"                # 0.2-0.4
    MODERATE = "moderate"        # 0.4-0.6
    STRONG = "strong"            # 0.6-0.8
    VERY_STRONG = "very_strong"  # 0.8-1.0

class TrendPhase(Enum):
    """🌊 Fase da tendência"""
    EARLY = "early"           # Início da tendência
    MIDDLE = "middle"         # Meio da tendência
    LATE = "late"            # Final da tendência (possível reversão)
    EXHAUSTION = "exhaustion" # Exaustão (reversão provável)

@dataclass
class TimeframeTrend:
    """📈 Análise de tendência para um timeframe específico"""
    timeframe: str
    direction: TrendDirection
    strength: float  # 0-1
    confidence: float  # 0-1
    phase: TrendPhase
    
    # Indicadores técnicos
    ma_20_direction: str  # 'up', 'down', 'flat'
    ma_50_direction: str
    ma_alignment: bool    # MAs alinhadas com a tendência
    momentum_score: float  # 0-1
    
    # Níveis importantes
    trend_line_slope: float
    key_support: float
    key_resistance: float
    
    # Estatísticas
    duration_candles: int
    price_change_pct: float
    avg_volume_ratio: float  # Volume médio vs média histórica
    
    # Sinais de reversão
    reversal_signals: List[str] = field(default_factory=list)
    reversal_probability: float = 0.0  # 0-1
    
    def get_strength_category(self) -> TrendStrength:
        """Retorna categoria de força"""
        if self.strength < 0.2:
            return TrendStrength.VERY_WEAK
        elif self.strength < 0.4:
            return TrendStrength.WEAK
        elif self.strength < 0.6:
            return TrendStrength.MODERATE
        elif self.strength < 0.8:
            return TrendStrength.STRONG
        else:
            return TrendStrength.VERY_STRONG
    
    def is_reliable(self) -> bool:
        """Verifica se tendência é confiável"""
        return (
            self.strength >= 0.5 and
            self.confidence >= 0.6 and
            self.ma_alignment and
            self.reversal_probability < 0.3
        )

@dataclass
class MultiTimeframeTrendAnalysis:
    """🎯 Análise completa multi-timeframe"""
    symbol: str
    timestamp: datetime
    
    # Análises por timeframe
    trend_1d: TimeframeTrend
    trend_4h: TimeframeTrend  
    trend_1h: TimeframeTrend
    
    # Alinhamento entre timeframes
    alignment_score: float  # 0-1 (1 = todos alinhados)
    dominant_direction: TrendDirection
    
    # Contexto hierárquico
    context_bias: TrendDirection    # 1D trend (contexto geral)
    setup_bias: TrendDirection      # 4H trend (setup)
    entry_bias: TrendDirection      # 1H trend (entrada)
    
    # Scores de confluência
    bullish_confluence: float  # 0-100
    bearish_confluence: float  # 0-100
    
    # Sinais de entrada
    trend_continuation_signal: bool = False
    trend_reversal_signal: bool = False
    breakout_signal: bool = False
    
    # Condições de mercado
    market_volatility: str = "normal"  # low, normal, high, extreme
    trend_maturity: str = "middle"     # early, middle, late, exhausted
    
    def get_trade_bias(self) -> TrendDirection:
        """Retorna bias para trading baseado na hierarquia"""
        # Peso dos timeframes (1D > 4H > 1H)
        weights = {
            'context': 0.5,  # 1D
            'setup': 0.3,    # 4H  
            'entry': 0.2     # 1H
        }
        
        bullish_weight = 0.0
        bearish_weight = 0.0
        
        # Context (1D)
        if self.context_bias == TrendDirection.BULLISH:
            bullish_weight += weights['context']
        elif self.context_bias == TrendDirection.BEARISH:
            bearish_weight += weights['context']
        
        # Setup (4H)
        if self.setup_bias == TrendDirection.BULLISH:
            bullish_weight += weights['setup']
        elif self.setup_bias == TrendDirection.BEARISH:
            bearish_weight += weights['setup']
        
        # Entry (1H)
        if self.entry_bias == TrendDirection.BULLISH:
            bullish_weight += weights['entry']
        elif self.entry_bias == TrendDirection.BEARISH:
            bearish_weight += weights['entry']
        
        # Determina bias final
        if bullish_weight > bearish_weight + 0.2:
            return TrendDirection.BULLISH
        elif bearish_weight > bullish_weight + 0.2:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def is_aligned_trend(self, min_alignment: float = 0.7) -> bool:
        """Verifica se há alinhamento suficiente entre timeframes"""
        return self.alignment_score >= min_alignment
    
    def get_entry_conditions(self) -> Dict[str, Any]:
        """Retorna condições para entrada"""
        return {
            'trade_bias': self.get_trade_bias().value,
            'alignment_score': self.alignment_score,
            'bullish_confluence': self.bullish_confluence,
            'bearish_confluence': self.bearish_confluence,
            'trend_continuation': self.trend_continuation_signal,
            'trend_reversal': self.trend_reversal_signal,
            'breakout_signal': self.breakout_signal,
            'market_volatility': self.market_volatility,
            'trend_maturity': self.trend_maturity
        }

class MultiTimeframeTrendAnalyzer:
    """📊 Analisador de tendência multi-timeframe"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Configurações hierárquicas
        self.timeframes = ["1d", "4h", "1h"]
        self.timeframe_weights = {
            "1d": 3.0,    # Contexto - maior peso
            "4h": 2.0,    # Setup - peso médio
            "1h": 1.0     # Entrada - menor peso
        }
        
        # Configurações de análise
        self.config = {
            'ma_periods': [20, 50, 200],
            'momentum_period': 14,
            'trend_min_duration': {
                '1h': 20,   # 20 candles = 20 horas
                '4h': 10,   # 10 candles = 40 horas  
                '1d': 5     # 5 candles = 5 dias
            },
            'reversal_lookback': 10,
            'volume_ma_period': 20
        }
        
        self.logger.info("📊 MultiTimeframeTrendAnalyzer inicializado")
    
    @timing(threshold_seconds=1.0)
    async def analyze_multi_timeframe_trend(self, symbol: str) -> MultiTimeframeTrendAnalysis:
        """🎯 Análise completa multi-timeframe"""
        
        self.logger.debug(f"🔍 Analisando tendência multi-TF: {symbol}")
        
        # Busca dados para todos os timeframes
        market_data_dict = await get_multiple_data([symbol], self.timeframes)
        
        if symbol not in market_data_dict:
            raise ValueError(f"Dados não disponíveis para {symbol}")
        
        symbol_data = market_data_dict[symbol]
        
        # Analisa cada timeframe
        trend_analyses = {}
        
        for tf in self.timeframes:
            if tf in symbol_data:
                market_data = symbol_data[tf]
                trend_analysis = await self._analyze_single_timeframe_trend(market_data)
                trend_analyses[tf] = trend_analysis
        
        # Verifica se temos dados suficientes
        if len(trend_analyses) < 2:
            raise ValueError(f"Dados insuficientes para análise multi-timeframe de {symbol}")
        
        # Análise de estrutura para complementar
        structure_analyses = {}
        for tf, market_data in symbol_data.items():
            if tf in self.timeframes:
                try:
                    structure_analysis = await analyze_market_structure(market_data)
                    structure_analyses[tf] = structure_analysis
                except Exception as e:
                    self.logger.warning(f"Estrutura não disponível para {tf}: {e}")
        
        # Calcula alinhamento entre timeframes
        alignment_score = self._calculate_alignment_score(trend_analyses)
        
        # Determina direção dominante
        dominant_direction = self._calculate_dominant_direction(trend_analyses)
        
        # Calcula confluência bullish/bearish
        bullish_confluence, bearish_confluence = self._calculate_confluence_scores(
            trend_analyses, structure_analyses
        )
        
        # Detecta sinais de entrada
        signals = self._detect_entry_signals(trend_analyses, structure_analyses)
        
        # Avalia condições de mercado
        market_conditions = self._assess_market_conditions(trend_analyses, structure_analyses)
        
        # Monta análise final
        analysis = MultiTimeframeTrendAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            trend_1d=trend_analyses.get('1d'),
            trend_4h=trend_analyses.get('4h'),
            trend_1h=trend_analyses.get('1h'),
            alignment_score=alignment_score,
            dominant_direction=dominant_direction,
            context_bias=trend_analyses.get('1d').direction if '1d' in trend_analyses else TrendDirection.UNKNOWN,
            setup_bias=trend_analyses.get('4h').direction if '4h' in trend_analyses else TrendDirection.UNKNOWN,
            entry_bias=trend_analyses.get('1h').direction if '1h' in trend_analyses else TrendDirection.UNKNOWN,
            bullish_confluence=bullish_confluence,
            bearish_confluence=bearish_confluence,
            trend_continuation_signal=signals.get('continuation', False),
            trend_reversal_signal=signals.get('reversal', False),
            breakout_signal=signals.get('breakout', False),
            market_volatility=market_conditions.get('volatility', 'normal'),
            trend_maturity=market_conditions.get('maturity', 'middle')
        )
        
        self.logger.info(
            f"✅ Análise multi-TF: {symbol} | "
            f"Bias: {analysis.get_trade_bias().value} | "
            f"Alignment: {alignment_score:.2f} | "
            f"Confluence: B{bullish_confluence:.0f}/B{bearish_confluence:.0f}"
        )
        
        return analysis
    
    async def _analyze_single_timeframe_trend(self, market_data: MarketData) -> TimeframeTrend:
        """📈 Analisa tendência de um timeframe específico"""
        
        df = market_data.ohlcv
        timeframe = market_data.timeframe
        
        if len(df) < 50:
            # Dados insuficientes - retorna análise básica
            return TimeframeTrend(
                timeframe=timeframe,
                direction=TrendDirection.UNKNOWN,
                strength=0.0,
                confidence=0.0,
                phase=TrendPhase.MIDDLE,
                ma_20_direction='flat',
                ma_50_direction='flat',
                ma_alignment=False,
                momentum_score=0.0,
                trend_line_slope=0.0,
                key_support=market_data.latest_price,
                key_resistance=market_data.latest_price,
                duration_candles=0,
                price_change_pct=0.0,
                avg_volume_ratio=1.0
            )
        
        # Calcula médias móveis
        ma_20 = df['close'].rolling(20).mean()
        ma_50 = df['close'].rolling(50).mean()
        ma_200 = df['close'].rolling(200).mean() if len(df) >= 200 else ma_50
        
        # Direção das médias móveis
        ma_20_direction = self._get_ma_direction(ma_20)
        ma_50_direction = self._get_ma_direction(ma_50)
        
        # Alinhamento das MAs
        current_price = df.iloc[-1]['close']
        ma_20_current = ma_20.iloc[-1] if not ma_20.empty else current_price
        ma_50_current = ma_50.iloc[-1] if not ma_50.empty else current_price
        
        # Determina direção da tendência baseada nas MAs
        if current_price > ma_20_current > ma_50_current:
            trend_direction = TrendDirection.BULLISH
            ma_alignment = True
        elif current_price < ma_20_current < ma_50_current:
            trend_direction = TrendDirection.BEARISH
            ma_alignment = True
        else:
            trend_direction = TrendDirection.SIDEWAYS
            ma_alignment = False
        
        # Calcula força da tendência
        strength = self._calculate_trend_strength(df, ma_20, ma_50)
        
        # Calcula confiança
        confidence = self._calculate_trend_confidence(df, trend_direction, ma_alignment)
        
        # Calcula momentum
        momentum_score = self._calculate_momentum_score(df)
        
        # Linha de tendência
        trend_line_slope = self._calculate_trend_line_slope(df['close'].tail(30))
        
        # Identifica suporte e resistência próximos
        key_support, key_resistance = self._identify_key_levels(df)
        
        # Estatísticas da tendência
        duration_candles = self._calculate_trend_duration(df, ma_20)
        price_change_pct = ((df.iloc[-1]['close'] - df.iloc[-30]['close']) / df.iloc[-30]['close']) * 100
        avg_volume_ratio = df['volume'].tail(20).mean() / df['volume'].mean() if 'volume' in df.columns else 1.0
        
        # Detecta sinais de reversão
        reversal_signals, reversal_probability = self._detect_reversal_signals(df, trend_direction)
        
        # Determina fase da tendência
        phase = self._determine_trend_phase(strength, duration_candles, reversal_probability, timeframe)
        
        return TimeframeTrend(
            timeframe=timeframe,
            direction=trend_direction,
            strength=strength,
            confidence=confidence,
            phase=phase,
            ma_20_direction=ma_20_direction,
            ma_50_direction=ma_50_direction,
            ma_alignment=ma_alignment,
            momentum_score=momentum_score,
            trend_line_slope=trend_line_slope,
            key_support=key_support,
            key_resistance=key_resistance,
            duration_candles=duration_candles,
            price_change_pct=price_change_pct,
            avg_volume_ratio=avg_volume_ratio,
            reversal_signals=reversal_signals,
            reversal_probability=reversal_probability
        )
    
    def _get_ma_direction(self, ma_series: pd.Series) -> str:
        """📊 Determina direção da média móvel"""
        if len(ma_series) < 5:
            return 'flat'
        
        recent_values = ma_series.tail(5).dropna()
        if len(recent_values) < 2:
            return 'flat'
        
        slope = (recent_values.iloc[-1] - recent_values.iloc[0]) / len(recent_values)
        relative_slope = slope / recent_values.iloc[-1] * 100  # Como % do preço
        
        if relative_slope > 0.1:
            return 'up'
        elif relative_slope < -0.1:
            return 'down'
        else:
            return 'flat'
    
    def _calculate_trend_strength(self, df: pd.DataFrame, ma_20: pd.Series, ma_50: pd.Series) -> float:
        """💪 Calcula força da tendência"""
        
        factors = []
        
        # 1. Consistência da direção (últimos 10 candles)
        recent_closes = df['close'].tail(10)
        up_moves = (recent_closes.diff() > 0).sum()
        consistency = up_moves / len(recent_closes) if len(recent_closes) > 0 else 0.5
        
        # Para bearish, inverte a lógica
        current_trend = self._get_basic_trend_direction(df)
        if current_trend == TrendDirection.BEARISH:
            consistency = 1 - consistency
        
        factors.append(consistency)
        
        # 2. Inclinação da MA20
        if len(ma_20) >= 10:
            ma_slope = (ma_20.iloc[-1] - ma_20.iloc[-10]) / ma_20.iloc[-10] * 100
            slope_strength = min(1.0, abs(ma_slope) / 5.0)  # 5% slope = strength 1.0
            factors.append(slope_strength)
        
        # 3. Separação entre preço e MA50
        if not ma_50.empty:
            current_price = df.iloc[-1]['close']
            ma_50_current = ma_50.iloc[-1]
            separation = abs(current_price - ma_50_current) / ma_50_current * 100
            separation_strength = min(1.0, separation / 10.0)  # 10% sep = strength 1.0
            factors.append(separation_strength)
        
        # 4. Volume confirmation
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(10).mean()
            avg_volume = df['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_strength = min(1.0, volume_ratio)
            factors.append(volume_strength)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _calculate_trend_confidence(self, df: pd.DataFrame, direction: TrendDirection, ma_alignment: bool) -> float:
        """🎯 Calcula confiança na tendência"""
        
        confidence_factors = []
        
        # 1. Alinhamento das médias móveis
        if ma_alignment:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)
        
        # 2. Duração da tendência
        trend_duration = self._calculate_trend_duration(df, df['close'].rolling(20).mean())
        min_duration = self.config['trend_min_duration'].get(df.attrs.get('timeframe', '1h'), 20)
        
        if trend_duration >= min_duration:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(trend_duration / min_duration)
        
        # 3. Consistência de highs e lows
        highs = df['high'].tail(20)
        lows = df['low'].tail(20)
        
        if direction == TrendDirection.BULLISH:
            higher_highs = (highs.rolling(5).max().diff() > 0).sum()
            higher_lows = (lows.rolling(5).min().diff() > 0).sum()
            structure_consistency = (higher_highs + higher_lows) / 30  # 15 checks cada
        elif direction == TrendDirection.BEARISH:
            lower_highs = (highs.rolling(5).max().diff() < 0).sum()
            lower_lows = (lows.rolling(5).min().diff() < 0).sum()
            structure_consistency = (lower_highs + lower_lows) / 30
        else:
            structure_consistency = 0.5
        
        confidence_factors.append(structure_consistency)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """🚀 Calcula score de momentum"""
        
        if len(df) < 20:
            return 0.5
        
        # RSI básico
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Rate of Change
        roc = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) >= 10 else 0
        
        # Normaliza momentum (0-1)
        rsi_momentum = abs(current_rsi - 50) / 50  # Distância do neutral
        roc_momentum = min(1.0, abs(roc) / 10.0)   # 10% ROC = momentum 1.0
        
        return (rsi_momentum + roc_momentum) / 2
    
    def _calculate_trend_line_slope(self, price_series: pd.Series) -> float:
        """📏 Calcula inclinação da linha de tendência"""
        
        if len(price_series) < 10:
            return 0.0
        
        x = np.arange(len(price_series))
        y = price_series.values
        
        try:
            slope, _ = np.polyfit(x, y, 1)
            # Normaliza slope como % por período
            return (slope / price_series.iloc[-1]) * 100
        except:
            return 0.0
    
    def _identify_key_levels(self, df: pd.DataFrame) -> Tuple[float, float]:
        """🎯 Identifica suporte e resistência próximos"""
        
        current_price = df.iloc[-1]['close']
        
        # Procura por níveis nos últimos 50 candles
        recent_data = df.tail(50)
        
        # Suporte = menor low acima do preço atual que foi testado
        potential_supports = []
        for i in range(len(recent_data)):
            low = recent_data.iloc[i]['low']
            if low < current_price:
                # Conta quantas vezes foi testado
                touches = ((recent_data['low'] <= low * 1.005) & 
                          (recent_data['low'] >= low * 0.995)).sum()
                if touches >= 2:
                    potential_supports.append(low)
        
        # Resistência = maior high acima do preço atual que foi testado
        potential_resistances = []
        for i in range(len(recent_data)):
            high = recent_data.iloc[i]['high']
            if high > current_price:
                touches = ((recent_data['high'] <= high * 1.005) & 
                          (recent_data['high'] >= high * 0.995)).sum()
                if touches >= 2:
                    potential_resistances.append(high)
        
        # Pega o suporte mais próximo (mais alto) e resistência mais próxima (mais baixa)
        key_support = max(potential_supports) if potential_supports else current_price * 0.95
        key_resistance = min(potential_resistances) if potential_resistances else current_price * 1.05
        
        return key_support, key_resistance
    
    def _calculate_trend_duration(self, df: pd.DataFrame, ma_series: pd.Series) -> int:
        """⏱️ Calcula duração da tendência atual"""
        
        if len(df) < 10 or ma_series.empty:
            return 0
        
        current_price = df.iloc[-1]['close']
        current_ma = ma_series.iloc[-1]
        
        # Determina se está acima ou abaixo da MA
        above_ma = current_price > current_ma
        
        # Conta candles consecutivos na mesma condição
        duration = 0
        for i in range(len(df) - 1, -1, -1):
            if i >= len(ma_series) or pd.isna(ma_series.iloc[i]):
                break
                
            price = df.iloc[i]['close']
            ma_value = ma_series.iloc[i]
            
            if (above_ma and price > ma_value) or (not above_ma and price < ma_value):
                duration += 1
            else:
                break
        
        return duration
    
    def _detect_reversal_signals(self, df: pd.DataFrame, trend_direction: TrendDirection) -> Tuple[List[str], float]:
        """🔄 Detecta sinais de reversão"""
        
        signals = []
        probability = 0.0
        
        if len(df) < 20:
            return signals, probability
        
        # 1. Divergência de momentum (simplificada)
        recent_highs = df['high'].tail(10)
        recent_lows = df['low'].tail(10)
        
        if trend_direction == TrendDirection.BULLISH:
            # Bearish divergence: preço faz higher high mas momentum não confirma
            if recent_highs.iloc[-1] > recent_highs.iloc[-5]:
                signals.append("potential_bearish_divergence")
                probability += 0.2
        
        elif trend_direction == TrendDirection.BEARISH:
            # Bullish divergence: preço faz lower low mas momentum não confirma
            if recent_lows.iloc[-1] < recent_lows.iloc[-5]:
                signals.append("potential_bullish_divergence")
                probability += 0.2
        
        # 2. Padrões de candlestick de reversão (simplificado)
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else last_candle
        
        # Doji em extremos
        body_size = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        
        if body_size < total_range * 0.1:  # Doji
            signals.append("doji_reversal")
            probability += 0.15
        
        # Hammer/Shooting star
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        if lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5:
            signals.append("hammer_pattern")
            probability += 0.2
        elif upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5:
            signals.append("shooting_star_pattern")
            probability += 0.2
        
        # 3. Volume de exaustão
        if 'volume' in df.columns:
            recent_volume = df['volume'].tail(5).mean()
            avg_volume = df['volume'].mean()
            
            if recent_volume > avg_volume * 2:  # Volume spike
                signals.append("volume_exhaustion")
                probability += 0.1
        
        return signals, min(1.0, probability)
    
    def _determine_trend_phase(self, strength: float, duration: int, reversal_prob: float, timeframe: str) -> TrendPhase:
        """🌊 Determina fase da tendência"""
        
        min_duration = self.config['trend_min_duration'].get(timeframe, 20)
        
        if duration < min_duration * 0.5:
            return TrendPhase.EARLY
        elif reversal_prob > 0.6:
            return TrendPhase.EXHAUSTION
        elif reversal_prob > 0.3 or duration > min_duration * 3:
            return TrendPhase.LATE
        else:
            return TrendPhase.MIDDLE
    
    def _get_basic_trend_direction(self, df: pd.DataFrame) -> TrendDirection:
        """📊 Determina direção básica da tendência"""
        if len(df) < 20:
            return TrendDirection.UNKNOWN
        
        ma_20 = df['close'].rolling(20).mean()
        current_price = df.iloc[-1]['close']
        current_ma = ma_20.iloc[-1]
        
        if current_price > current_ma:
            return TrendDirection.BULLISH
        elif current_price < current_ma:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_alignment_score(self, trend_analyses: Dict[str, TimeframeTrend]) -> float:
        """🎯 Calcula score de alinhamento entre timeframes"""
        
        if len(trend_analyses) < 2:
            return 0.0
        
        # Coleta direções de todos os timeframes
        directions = [analysis.direction for analysis in trend_analyses.values()]
        
        # Conta direções iguais
        bullish_count = directions.count(TrendDirection.BULLISH)
        bearish_count = directions.count(TrendDirection.BEARISH)
        sideways_count = directions.count(TrendDirection.SIDEWAYS)
        
        total_count = len(directions)
        max_aligned = max(bullish_count, bearish_count, sideways_count)
        
        base_alignment = max_aligned / total_count
        
        # Aplica pesos por timeframe (1D tem mais peso)
        weighted_score = 0.0
        total_weight = 0.0
        
        for tf, analysis in trend_analyses.items():
            weight = self.timeframe_weights.get(tf, 1.0)
            
            # Contribuição para alinhamento baseada na força da tendência
            contribution = analysis.strength * analysis.confidence
            weighted_score += contribution * weight
            total_weight += weight
        
        weighted_alignment = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Combina alignment básico com weighted
        return (base_alignment * 0.6 + weighted_alignment * 0.4)
    
    def _calculate_dominant_direction(self, trend_analyses: Dict[str, TimeframeTrend]) -> TrendDirection:
        """👑 Determina direção dominante com pesos hierárquicos"""
        
        weighted_bullish = 0.0
        weighted_bearish = 0.0
        total_weight = 0.0
        
        for tf, analysis in trend_analyses.items():
            weight = self.timeframe_weights.get(tf, 1.0)
            strength_confidence = analysis.strength * analysis.confidence
            
            if analysis.direction == TrendDirection.BULLISH:
                weighted_bullish += strength_confidence * weight
            elif analysis.direction == TrendDirection.BEARISH:
                weighted_bearish += strength_confidence * weight
            
            total_weight += weight
        
        if total_weight == 0:
            return TrendDirection.UNKNOWN
        
        bullish_score = weighted_bullish / total_weight
        bearish_score = weighted_bearish / total_weight
        
        if bullish_score > bearish_score + 0.1:
            return TrendDirection.BULLISH
        elif bearish_score > bullish_score + 0.1:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def _calculate_confluence_scores(self, trend_analyses: Dict[str, TimeframeTrend], 
                                   structure_analyses: Dict[str, MarketStructureAnalysis]) -> Tuple[float, float]:
        """🎯 Calcula scores de confluência bullish/bearish"""
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Pontuação baseada nas tendências
        for tf, analysis in trend_analyses.items():
            weight = self.timeframe_weights.get(tf, 1.0)
            strength_confidence = analysis.strength * analysis.confidence
            
            if analysis.direction == TrendDirection.BULLISH:
                bullish_score += strength_confidence * weight * 20
            elif analysis.direction == TrendDirection.BEARISH:
                bearish_score += strength_confidence * weight * 20
        
        # Pontuação baseada na estrutura
        for tf, structure in structure_analyses.items():
            weight = self.timeframe_weights.get(tf, 1.0) * 0.5  # Peso menor para estrutura
            
            bullish_score += structure.bullish_structure_score * weight * 0.3
            bearish_score += structure.bearish_structure_score * weight * 0.3
        
        # Normaliza para 0-100
        return min(100.0, bullish_score), min(100.0, bearish_score)
    
    def _detect_entry_signals(self, trend_analyses: Dict[str, TimeframeTrend], 
                            structure_analyses: Dict[str, MarketStructureAnalysis]) -> Dict[str, bool]:
        """🎯 Detecta sinais de entrada"""
        
        signals = {
            'continuation': False,
            'reversal': False,
            'breakout': False
        }
        
        # Sinal de continuação: alinhamento entre timeframes
        aligned_count = 0
        total_trends = len(trend_analyses)
        
        if total_trends >= 2:
            main_direction = None
            for analysis in trend_analyses.values():
                if analysis.direction != TrendDirection.SIDEWAYS:
                    if main_direction is None:
                        main_direction = analysis.direction
                    elif main_direction == analysis.direction:
                        aligned_count += 1
            
            if aligned_count >= total_trends - 1:  # Quase todos alinhados
                signals['continuation'] = True
        
        # Sinal de reversão: alta probabilidade de reversão em timeframe maior
        for tf, analysis in trend_analyses.items():
            if tf in ['1d', '4h'] and analysis.reversal_probability > 0.6:
                signals['reversal'] = True
                break
        
        # Sinal de breakout: baseado na estrutura
        for structure in structure_analyses.values():
            if structure.breakout_imminent_score > 75:
                signals['breakout'] = True
                break
        
        return signals
    
    def _assess_market_conditions(self, trend_analyses: Dict[str, TimeframeTrend], 
                                structure_analyses: Dict[str, MarketStructureAnalysis]) -> Dict[str, str]:
        """🌡️ Avalia condições gerais de mercado"""
        
        # Volatilidade baseada na força das tendências
        volatility_scores = []
        for analysis in trend_analyses.values():
            # Alta força + alta mudança de preço = alta volatilidade
            vol_score = analysis.strength * (abs(analysis.price_change_pct) / 10.0)
            volatility_scores.append(vol_score)
        
        avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 0.5
        
        if avg_volatility > 0.8:
            volatility = "extreme"
        elif avg_volatility > 0.6:
            volatility = "high"
        elif avg_volatility > 0.3:
            volatility = "normal"
        else:
            volatility = "low"
        
        # Maturidade baseada na fase das tendências
        mature_trends = 0
        for analysis in trend_analyses.values():
            if analysis.phase in [TrendPhase.LATE, TrendPhase.EXHAUSTION]:
                mature_trends += 1
        
        if mature_trends >= len(trend_analyses) * 0.6:
            maturity = "exhausted"
        elif mature_trends > 0:
            maturity = "late"
        else:
            # Verifica se há tendências early
            early_trends = sum(1 for a in trend_analyses.values() if a.phase == TrendPhase.EARLY)
            if early_trends > 0:
                maturity = "early"
            else:
                maturity = "middle"
        
        return {
            'volatility': volatility,
            'maturity': maturity
        }

# === FUNÇÕES DE CONVENIÊNCIA ===

async def analyze_multi_timeframe_trend(symbol: str) -> MultiTimeframeTrendAnalysis:
    """🎯 Função de conveniência para análise multi-timeframe"""
    analyzer = MultiTimeframeTrendAnalyzer()
    return await analyzer.analyze_multi_timeframe_trend(symbol)

def get_trend_bias(analysis: MultiTimeframeTrendAnalysis) -> str:
    """📊 Retorna bias simplificado"""
    return analysis.get_trade_bias().value

def is_strong_trend_alignment(analysis: MultiTimeframeTrendAnalysis) -> bool:
    """💪 Verifica se há alinhamento forte"""
    return analysis.is_aligned_trend(min_alignment=0.8)

def get_entry_timeframe_trend(analysis: MultiTimeframeTrendAnalysis) -> TimeframeTrend:
    """⏰ Retorna tendência do timeframe de entrada (1H)"""
    return analysis.trend_1h