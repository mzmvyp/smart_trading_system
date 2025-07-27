"""
Strategies: Trend Following Strategy
Estratégia de seguimento de tendência com pullbacks inteligentes
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.signal_manager import SignalType, SignalPriority
from indicators.confluence_analyzer import ConfluenceAnalyzer
from indicators.trend_analyzer import TrendAnalyzer
from filters.volatility_filter import VolatilityFilter
from filters.market_condition import MarketConditionFilter
from utils.logger import get_logger
from utils.helpers import (
    calculate_percentage_change,
    safe_divide,
    normalize_value,
    find_local_extremes
)


logger = get_logger(__name__)


class TrendSetup(Enum):
    """Tipos de setup de trend following"""
    TREND_CONTINUATION = "trend_continuation"
    PULLBACK_ENTRY = "pullback_entry"
    BREAKOUT_FOLLOW = "breakout_follow"
    MOMENTUM_ENTRY = "momentum_entry"
    MOVING_AVERAGE_CROSS = "ma_cross"


class TrendPhase(Enum):
    """Fases da tendência"""
    EARLY = "early"          # Início da tendência
    MATURE = "mature"        # Tendência madura
    EXHAUSTION = "exhaustion" # Possível exaustão


@dataclass
class TrendFollowingConfig:
    """Configurações da estratégia Trend Following"""
    # Moving Averages
    fast_ma_period: int = 10
    medium_ma_period: int = 21
    slow_ma_period: int = 50
    trend_ma_period: int = 200
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # ADX (Average Directional Index)
    adx_period: int = 14
    adx_strong_trend: float = 25
    adx_very_strong: float = 40
    
    # RSI para pullbacks
    rsi_period: int = 14
    rsi_pullback_bullish: float = 45  # RSI para pullback em uptrend
    rsi_pullback_bearish: float = 55  # RSI para pullback em downtrend
    
    # Momentum
    momentum_period: int = 14
    momentum_threshold: float = 5     # % mínimo de momentum
    
    # Volume
    volume_ma_period: int = 20
    min_volume_ratio: float = 1.1    # Volume 10% acima da média
    
    # Filtros
    min_trend_strength: float = 60   # Força mínima da tendência
    max_pullback_depth: float = 38.2 # Máxima retração em % (Fibonacci)
    min_confluence_score: float = 70
    
    # Risk Management
    atr_period: int = 14
    stop_atr_multiplier: float = 2.0
    profit_target_ratio: float = 3.0  # Risk:Reward mínimo
    max_holding_period: int = 100      # Máximo de barras


class TrendFollowingStrategy:
    """Estratégia de Trend Following inteligente"""
    
    def __init__(self, config: TrendFollowingConfig = None):
        self.config = config or TrendFollowingConfig()
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.volatility_filter = VolatilityFilter()
        self.market_filter = MarketConditionFilter()
        
        self.name = "trend_following"
        self.timeframes = ["4h", "1d"]  # Timeframes preferidos para trend following
        
        # Cache para otimização
        self._indicator_cache = {}
        self._trend_cache = {}
    
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Análise principal da estratégia"""
        
        if len(df) < 100:
            return {'signals': [], 'analysis': {'error': 'Dados insuficientes'}}
        
        try:
            # Calcula indicadores
            indicators = self._calculate_indicators(df)
            
            # Analisa tendência
            trend_analysis = self._analyze_trend(df, indicators, timeframe)
            
            # Aplica filtros
            filters_result = self._apply_filters(df, symbol, timeframe, indicators, trend_analysis)
            if not filters_result['passed']:
                return {
                    'signals': [],
                    'analysis': {
                        'filters_passed': False,
                        'filter_reasons': filters_result['reasons'],
                        'trend_analysis': trend_analysis
                    }
                }
            
            # Identifica setups
            setups = self._identify_setups(df, indicators, trend_analysis, timeframe)
            
            # Gera sinais
            signals = []
            for setup in setups:
                signal = self._generate_signal(df, setup, indicators, trend_analysis, symbol, timeframe)
                if signal:
                    signals.append(signal)
            
            # Análise de confluência
            confluence_analysis = self._analyze_confluence(df, indicators, trend_analysis, timeframe)
            
            return {
                'signals': signals,
                'analysis': {
                    'filters_passed': True,
                    'setups_found': len(setups),
                    'trend_analysis': trend_analysis,
                    'confluence': confluence_analysis,
                    'market_condition': filters_result['market_condition']
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise trend following: {e}")
            return {'signals': [], 'analysis': {'error': str(e)}}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcula todos os indicadores necessários"""
        
        indicators = {}
        
        try:
            # Moving Averages
            indicators['ema_fast'] = talib.EMA(df['close'], timeperiod=self.config.fast_ma_period)
            indicators['ema_medium'] = talib.EMA(df['close'], timeperiod=self.config.medium_ma_period)
            indicators['ema_slow'] = talib.EMA(df['close'], timeperiod=self.config.slow_ma_period)
            indicators['sma_trend'] = talib.SMA(df['close'], timeperiod=self.config.trend_ma_period)
            
            # MACD
            macd, signal, histogram = talib.MACD(
                df['close'],
                fastperiod=self.config.macd_fast,
                slowperiod=self.config.macd_slow,
                signalperiod=self.config.macd_signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = histogram
            
            # ADX (Trend Strength)
            indicators['adx'] = talib.ADX(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            indicators['plus_di'] = talib.PLUS_DI(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            indicators['minus_di'] = talib.MINUS_DI(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.adx_period
            )
            
            # RSI
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.config.rsi_period)
            
            # Momentum
            indicators['momentum'] = talib.MOM(df['close'], timeperiod=self.config.momentum_period)
            indicators['momentum_pct'] = (indicators['momentum'] / df['close']) * 100
            
            # ATR para stops
            indicators['atr'] = talib.ATR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.atr_period
            )
            
            # Volume Analysis
            indicators['volume_sma'] = talib.SMA(df['volume'], timeperiod=self.config.volume_ma_period)
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            # Slope das médias móveis (força da tendência)
            indicators['ema_fast_slope'] = self._calculate_slope(indicators['ema_fast'], 5)
            indicators['ema_slow_slope'] = self._calculate_slope(indicators['ema_slow'], 10)
            
            # Distância das médias
            indicators['price_vs_ema_fast'] = ((df['close'] - indicators['ema_fast']) / indicators['ema_fast']) * 100
            indicators['price_vs_sma_trend'] = ((df['close'] - indicators['sma_trend']) / indicators['sma_trend']) * 100
            
            # Higher Highs / Lower Lows
            indicators['hh_ll_pattern'] = self._detect_hh_ll_pattern(df)
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
        
        return indicators
    
    def _calculate_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula slope (inclinação) de uma série"""
        try:
            slopes = []
            for i in range(len(series)):
                if i < period:
                    slopes.append(0)
                else:
                    y2 = series.iloc[i]
                    y1 = series.iloc[i - period]
                    slope = ((y2 - y1) / y1) * 100
                    slopes.append(slope)
            return pd.Series(slopes, index=series.index)
        except:
            return pd.Series([0] * len(series), index=series.index)
    
    def _detect_hh_ll_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detecta padrões de Higher Highs / Lower Lows"""
        
        try:
            # Encontra extremos locais
            extremes = find_local_extremes(df['close'].tolist(), window=5)
            
            pattern = [0] * len(df)  # 0 = neutro, 1 = uptrend (HH/HL), -1 = downtrend (LH/LL)
            
            highs = extremes['highs']
            lows = extremes['lows']
            
            # Analisa sequência de highs
            if len(highs) >= 2:
                for i in range(1, len(highs)):
                    current_high = df['high'].iloc[highs[i]]
                    prev_high = df['high'].iloc[highs[i-1]]
                    
                    if current_high > prev_high:  # Higher High
                        for j in range(highs[i-1], min(highs[i] + 1, len(pattern))):
                            pattern[j] = max(pattern[j], 1)
            
            # Analisa sequência de lows
            if len(lows) >= 2:
                for i in range(1, len(lows)):
                    current_low = df['low'].iloc[lows[i]]
                    prev_low = df['low'].iloc[lows[i-1]]
                    
                    if current_low < prev_low:  # Lower Low
                        for j in range(lows[i-1], min(lows[i] + 1, len(pattern))):
                            pattern[j] = min(pattern[j], -1)
            
            return pd.Series(pattern, index=df.index)
        
        except Exception as e:
            logger.error(f"Erro ao detectar padrão HH/LL: {e}")
            return pd.Series([0] * len(df), index=df.index)
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict, timeframe: str) -> Dict:
        """Análise completa da tendência"""
        
        try:
            current_idx = len(df) - 1
            
            # Direção da tendência
            trend_direction = self._determine_trend_direction(indicators, current_idx)
            
            # Força da tendência
            trend_strength = self._calculate_trend_strength(indicators, current_idx)
            
            # Fase da tendência
            trend_phase = self._determine_trend_phase(df, indicators, current_idx)
            
            # Qualidade da tendência
            trend_quality = self._assess_trend_quality(df, indicators, current_idx)
            
            return {
                'direction': trend_direction,  # 'bullish', 'bearish', 'sideways'
                'strength': trend_strength,    # 0-100
                'phase': trend_phase,          # TrendPhase enum
                'quality': trend_quality,      # 0-100
                'adx_value': indicators['adx'].iloc[current_idx] if len(indicators['adx']) > current_idx else 0,
                'macd_bullish': indicators['macd'].iloc[current_idx] > indicators['macd_signal'].iloc[current_idx] if len(indicators['macd']) > current_idx else False,
                'ma_alignment': self._check_ma_alignment(indicators, current_idx)
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de tendência: {e}")
            return {
                'direction': 'sideways',
                'strength': 0,
                'phase': TrendPhase.MATURE,
                'quality': 0,
                'adx_value': 0,
                'macd_bullish': False,
                'ma_alignment': False
            }
    
    def _determine_trend_direction(self, indicators: Dict, current_idx: int) -> str:
        """Determina direção da tendência"""
        
        try:
            # Verifica alinhamento das médias móveis
            ema_fast = indicators['ema_fast'].iloc[current_idx]
            ema_slow = indicators['ema_slow'].iloc[current_idx]
            sma_trend = indicators['sma_trend'].iloc[current_idx]
            current_price = indicators['ema_fast'].index[current_idx]  # Simplificado
            
            # Score bullish
            bullish_score = 0
            
            if ema_fast > ema_slow:
                bullish_score += 1
            if ema_slow > sma_trend:
                bullish_score += 1
            if indicators['plus_di'].iloc[current_idx] > indicators['minus_di'].iloc[current_idx]:
                bullish_score += 1
            if indicators['macd'].iloc[current_idx] > indicators['macd_signal'].iloc[current_idx]:
                bullish_score += 1
            if indicators['ema_fast_slope'].iloc[current_idx] > 0:
                bullish_score += 1
            
            if bullish_score >= 4:
                return 'bullish'
            elif bullish_score <= 1:
                return 'bearish'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Erro ao determinar direção: {e}")
            return 'sideways'
    
    def _calculate_trend_strength(self, indicators: Dict, current_idx: int) -> float:
        """Calcula força da tendência (0-100)"""
        
        try:
            base_strength = 50
            
            # ADX
            adx = indicators['adx'].iloc[current_idx]
            if adx >= self.config.adx_very_strong:
                base_strength += 30
            elif adx >= self.config.adx_strong_trend:
                base_strength += 20
            else:
                base_strength -= 10
            
            # Slope das médias
            fast_slope = abs(indicators['ema_fast_slope'].iloc[current_idx])
            slow_slope = abs(indicators['ema_slow_slope'].iloc[current_idx])
            
            if fast_slope > 2:
                base_strength += 15
            if slow_slope > 1:
                base_strength += 10
            
            # Momentum
            momentum_pct = abs(indicators['momentum_pct'].iloc[current_idx])
            if momentum_pct > self.config.momentum_threshold:
                base_strength += 15
            
            # Volume
            volume_ratio = indicators['volume_ratio'].iloc[current_idx]
            if volume_ratio > 1.5:
                base_strength += 10
            elif volume_ratio < 0.8:
                base_strength -= 10
            
            return max(0, min(100, base_strength))
            
        except Exception as e:
            logger.error(f"Erro ao calcular força da tendência: {e}")
            return 0
    
    def _determine_trend_phase(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> TrendPhase:
        """Determina fase da tendência"""
        
        try:
            # Analisa duração da tendência
            trend_duration = self._calculate_trend_duration(indicators, current_idx)
            
            # Analisa momentum
            current_momentum = abs(indicators['momentum_pct'].iloc[current_idx])
            avg_momentum = np.mean(indicators['momentum_pct'].tail(10))
            
            # Analisa volume
            volume_trend = np.mean(indicators['volume_ratio'].tail(5))
            
            if trend_duration < 20 and current_momentum > avg_momentum * 1.2:
                return TrendPhase.EARLY
            elif volume_trend < 0.9 or current_momentum < avg_momentum * 0.7:
                return TrendPhase.EXHAUSTION
            else:
                return TrendPhase.MATURE
                
        except Exception as e:
            logger.error(f"Erro ao determinar fase: {e}")
            return TrendPhase.MATURE
    
    def _calculate_trend_duration(self, indicators: Dict, current_idx: int) -> int:
        """Calcula duração da tendência atual em barras"""
        
        try:
            # Simplificado: conta barras desde última mudança de direção
            direction_changes = 0
            lookback = min(50, current_idx)
            
            for i in range(current_idx - lookback, current_idx):
                if i > 0:
                    prev_macd = indicators['macd'].iloc[i-1] > indicators['macd_signal'].iloc[i-1]
                    curr_macd = indicators['macd'].iloc[i] > indicators['macd_signal'].iloc[i]
                    
                    if prev_macd != curr_macd:
                        direction_changes = current_idx - i
                        break
            
            return direction_changes if direction_changes > 0 else lookback
            
        except Exception as e:
            logger.error(f"Erro ao calcular duração: {e}")
            return 20
    
    def _assess_trend_quality(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> float:
        """Avalia qualidade da tendência (0-100)"""
        
        try:
            quality_score = 50
            
            # Consistência das médias móveis
            ma_alignment = self._check_ma_alignment(indicators, current_idx)
            if ma_alignment:
                quality_score += 20
            
            # Padrão HH/HL ou LH/LL
            hh_ll_pattern = indicators['hh_ll_pattern'].iloc[current_idx]
            if abs(hh_ll_pattern) == 1:
                quality_score += 15
            
            # Volume confirmation
            volume_ratio = indicators['volume_ratio'].iloc[current_idx]
            if volume_ratio > 1.2:
                quality_score += 15
            
            # MACD confirmation
            macd_histogram = indicators['macd_histogram'].iloc[current_idx]
            prev_histogram = indicators['macd_histogram'].iloc[current_idx - 1] if current_idx > 0 else 0
            
            if abs(macd_histogram) > abs(prev_histogram):  # Momentum increasing
                quality_score += 10
            
            return max(0, min(100, quality_score))
            
        except Exception as e:
            logger.error(f"Erro ao avaliar qualidade: {e}")
            return 50
    
    def _check_ma_alignment(self, indicators: Dict, current_idx: int) -> bool:
        """Verifica alinhamento das médias móveis"""
        
        try:
            ema_fast = indicators['ema_fast'].iloc[current_idx]
            ema_medium = indicators['ema_medium'].iloc[current_idx]
            ema_slow = indicators['ema_slow'].iloc[current_idx]
            
            # Bullish alignment
            bullish_aligned = ema_fast > ema_medium > ema_slow
            
            # Bearish alignment
            bearish_aligned = ema_fast < ema_medium < ema_slow
            
            return bullish_aligned or bearish_aligned
            
        except Exception as e:
            logger.error(f"Erro ao verificar alinhamento: {e}")
            return False
    
    def _apply_filters(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict, trend_analysis: Dict) -> Dict:
        """Aplica filtros da estratégia"""
        
        reasons = []
        
        # Filtro de força de tendência
        if trend_analysis['strength'] < self.config.min_trend_strength:
            reasons.append(f"Tendência fraca: {trend_analysis['strength']:.1f}")
        
        # Filtro de direção de tendência
        if trend_analysis['direction'] == 'sideways':
            reasons.append("Mercado lateral - não adequado para trend following")
        
        # Filtro de ADX
        if trend_analysis['adx_value'] < self.config.adx_strong_trend:
            reasons.append(f"ADX muito baixo: {trend_analysis['adx_value']:.1f}")
        
        # Filtro de volume
        current_volume_ratio = indicators['volume_ratio'].iloc[-1] if len(indicators['volume_ratio']) > 0 else 0
        if current_volume_ratio < self.config.min_volume_ratio:
            reasons.append(f"Volume insuficiente: {current_volume_ratio:.2f}")
        
        # Filtro de volatilidade
        volatility_result = self.volatility_filter.analyze(df, timeframe)
        if volatility_result['state'] == 'very_low':
            reasons.append("Volatilidade muito baixa para trend following")
        
        # Filtro de condição de mercado
        market_result = self.market_filter.analyze(df, symbol, timeframe)
        
        return {
            'passed': len(reasons) == 0,
            'reasons': reasons,
            'market_condition': market_result.get('condition', 'unknown')
        }
    
    def _identify_setups(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, timeframe: str) -> List[Dict]:
        """Identifica setups de trend following"""
        
        setups = []
        current_idx = len(df) - 1
        
        # Setup 1: Pullback Entry
        setups.extend(self._find_pullback_setups(df, indicators, trend_analysis, current_idx))
        
        # Setup 2: Breakout Follow
        setups.extend(self._find_breakout_setups(df, indicators, trend_analysis, current_idx))
        
        # Setup 3: Moving Average Cross
        setups.extend(self._find_ma_cross_setups(df, indicators, trend_analysis, current_idx))
        
        # Setup 4: Momentum Entry
        setups.extend(self._find_momentum_setups(df, indicators, trend_analysis, current_idx))
        
        return setups
    
    def _find_pullback_setups(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de pullback"""
        
        setups = []
        
        if trend_analysis['direction'] == 'sideways':
            return setups
        
        try:
            current_price = df['close'].iloc[current_idx]
            ema_fast = indicators['ema_fast'].iloc[current_idx]
            ema_slow = indicators['ema_slow'].iloc[current_idx]
            rsi = indicators['rsi'].iloc[current_idx]
            
            # Pullback em uptrend
            if trend_analysis['direction'] == 'bullish':
                
                # Preço próximo da EMA rápida mas acima da EMA lenta
                price_near_ema = abs((current_price - ema_fast) / ema_fast) < 0.02
                price_above_slow = current_price > ema_slow
                
                # RSI em nível de pullback
                rsi_pullback = self.config.rsi_pullback_bullish <= rsi <= 60
                
                # MACD ainda bullish
                macd_bullish = indicators['macd'].iloc[current_idx] > indicators['macd_signal'].iloc[current_idx]
                
                if price_near_ema and price_above_slow and rsi_pullback and macd_bullish:
                    
                    # Calcula profundidade do pullback
                    recent_high = df['high'].tail(20).max()
                    pullback_depth = ((recent_high - current_price) / recent_high) * 100
                    
                    if pullback_depth <= self.config.max_pullback_depth:
                        setup = {
                            'type': TrendSetup.PULLBACK_ENTRY,
                            'signal_type': SignalType.BUY,
                            'entry_price': current_price,
                            'confidence': self._calculate_pullback_confidence(pullback_depth, rsi, 'bullish'),
                            'metadata': {
                                'pullback_depth': pullback_depth,
                                'support_level': ema_fast,
                                'rsi_value': rsi,
                                'trend_direction': 'bullish'
                            }
                        }
                        setups.append(setup)
            
            # Pullback em downtrend
            elif trend_analysis['direction'] == 'bearish':
                
                price_near_ema = abs((current_price - ema_fast) / ema_fast) < 0.02
                price_below_slow = current_price < ema_slow
                rsi_pullback = 40 <= rsi <= self.config.rsi_pullback_bearish
                macd_bearish = indicators['macd'].iloc[current_idx] < indicators['macd_signal'].iloc[current_idx]
                
                if price_near_ema and price_below_slow and rsi_pullback and macd_bearish:
                    
                    recent_low = df['low'].tail(20).min()
                    pullback_depth = ((current_price - recent_low) / recent_low) * 100
                    
                    if pullback_depth <= self.config.max_pullback_depth:
                        setup = {
                            'type': TrendSetup.PULLBACK_ENTRY,
                            'signal_type': SignalType.SELL,
                            'entry_price': current_price,
                            'confidence': self._calculate_pullback_confidence(pullback_depth, rsi, 'bearish'),
                            'metadata': {
                                'pullback_depth': pullback_depth,
                                'resistance_level': ema_fast,
                                'rsi_value': rsi,
                                'trend_direction': 'bearish'
                            }
                        }
                        setups.append(setup)
                        
        except Exception as e:
            logger.error(f"Erro ao encontrar pullback setups: {e}")
        
        return setups
    
    def _find_breakout_setups(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de breakout"""
        
        setups = []
        
        try:
            # Identifica breakouts de consolidação
            # Simplificado: verifica se saiu de faixa de preços
            lookback_period = 20
            recent_data = df.tail(lookback_period)
            
            high_range = recent_data['high'].max()
            low_range = recent_data['low'].min()
            range_size = ((high_range - low_range) / low_range) * 100
            
            current_price = df['close'].iloc[current_idx]
            volume_ratio = indicators['volume_ratio'].iloc[current_idx]
            
            # Breakout bullish
            if (current_price > high_range and 
                trend_analysis['direction'] == 'bullish' and
                volume_ratio > 1.5 and
                range_size > 5):  # Consolidação significativa
                
                setup = {
                    'type': TrendSetup.BREAKOUT_FOLLOW,
                    'signal_type': SignalType.BUY,
                    'entry_price': current_price,
                    'confidence': self._calculate_breakout_confidence(volume_ratio, range_size, trend_analysis),
                    'metadata': {
                        'breakout_level': high_range,
                        'range_size': range_size,
                        'volume_ratio': volume_ratio,
                        'support_level': low_range
                    }
                }
                setups.append(setup)
            
            # Breakout bearish
            elif (current_price < low_range and 
                  trend_analysis['direction'] == 'bearish' and
                  volume_ratio > 1.5 and
                  range_size > 5):
                
                setup = {
                    'type': TrendSetup.BREAKOUT_FOLLOW,
                    'signal_type': SignalType.SELL,
                    'entry_price': current_price,
                    'confidence': self._calculate_breakout_confidence(volume_ratio, range_size, trend_analysis),
                    'metadata': {
                        'breakout_level': low_range,
                        'range_size': range_size,
                        'volume_ratio': volume_ratio,
                        'resistance_level': high_range
                    }
                }
                setups.append(setup)
                
        except Exception as e:
            logger.error(f"Erro ao encontrar breakout setups: {e}")
        
        return setups
    
    def _find_ma_cross_setups(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de cruzamento de médias"""
        
        setups = []
        
        try:
            if current_idx < 1:
                return setups
            
            # Verifica cruzamento EMA rápida x EMA média
            current_fast = indicators['ema_fast'].iloc[current_idx]
            current_medium = indicators['ema_medium'].iloc[current_idx]
            prev_fast = indicators['ema_fast'].iloc[current_idx - 1]
            prev_medium = indicators['ema_medium'].iloc[current_idx - 1]
            
            current_price = df['close'].iloc[current_idx]
            volume_ratio = indicators['volume_ratio'].iloc[current_idx]
            
            # Golden Cross (bullish)
            if (prev_fast <= prev_medium and current_fast > current_medium and
                trend_analysis['direction'] == 'bullish' and
                volume_ratio > self.config.min_volume_ratio):
                
                setup = {
                    'type': TrendSetup.MOVING_AVERAGE_CROSS,
                    'signal_type': SignalType.BUY,
                    'entry_price': current_price,
                    'confidence': self._calculate_ma_cross_confidence(trend_analysis, volume_ratio, 'bullish'),
                    'metadata': {
                        'cross_type': 'golden_cross',
                        'fast_ma': current_fast,
                        'medium_ma': current_medium,
                        'volume_confirmation': volume_ratio > 1.2
                    }
                }
                setups.append(setup)
            
            # Death Cross (bearish)
            elif (prev_fast >= prev_medium and current_fast < current_medium and
                  trend_analysis['direction'] == 'bearish' and
                  volume_ratio > self.config.min_volume_ratio):
                
                setup = {
                    'type': TrendSetup.MOVING_AVERAGE_CROSS,
                    'signal_type': SignalType.SELL,
                    'entry_price': current_price,
                    'confidence': self._calculate_ma_cross_confidence(trend_analysis, volume_ratio, 'bearish'),
                    'metadata': {
                        'cross_type': 'death_cross',
                        'fast_ma': current_fast,
                        'medium_ma': current_medium,
                        'volume_confirmation': volume_ratio > 1.2
                    }
                }
                setups.append(setup)
                
        except Exception as e:
            logger.error(f"Erro ao encontrar MA cross setups: {e}")
        
        return setups
    
    def _find_momentum_setups(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de momentum"""
        
        setups = []
        
        try:
            momentum_pct = indicators['momentum_pct'].iloc[current_idx]
            current_price = df['close'].iloc[current_idx]
            volume_ratio = indicators['volume_ratio'].iloc[current_idx]
            
            # Strong bullish momentum
            if (momentum_pct > self.config.momentum_threshold and
                trend_analysis['direction'] == 'bullish' and
                trend_analysis['phase'] == TrendPhase.EARLY and
                volume_ratio > 1.3):
                
                setup = {
                    'type': TrendSetup.MOMENTUM_ENTRY,
                    'signal_type': SignalType.BUY,
                    'entry_price': current_price,
                    'confidence': self._calculate_momentum_confidence(momentum_pct, volume_ratio, trend_analysis),
                    'metadata': {
                        'momentum_pct': momentum_pct,
                        'trend_phase': trend_analysis['phase'].value,
                        'volume_surge': volume_ratio > 2.0
                    }
                }
                setups.append(setup)
            
            # Strong bearish momentum
            elif (momentum_pct < -self.config.momentum_threshold and
                  trend_analysis['direction'] == 'bearish' and
                  trend_analysis['phase'] == TrendPhase.EARLY and
                  volume_ratio > 1.3):
                
                setup = {
                    'type': TrendSetup.MOMENTUM_ENTRY,
                    'signal_type': SignalType.SELL,
                    'entry_price': current_price,
                    'confidence': self._calculate_momentum_confidence(abs(momentum_pct), volume_ratio, trend_analysis),
                    'metadata': {
                        'momentum_pct': momentum_pct,
                        'trend_phase': trend_analysis['phase'].value,
                        'volume_surge': volume_ratio > 2.0
                    }
                }
                setups.append(setup)
                
        except Exception as e:
            logger.error(f"Erro ao encontrar momentum setups: {e}")
        
        return setups
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        setup: Dict,
        indicators: Dict,
        trend_analysis: Dict,
        symbol: str,
        timeframe: str
    ) -> Optional[Dict]:
        """Gera sinal final baseado no setup"""
        
        try:
            entry_price = setup['entry_price']
            signal_type = setup['signal_type']
            
            # Calcula stop loss e take profit
            stop_loss, take_profit = self._calculate_stop_and_target(
                df, indicators, entry_price, signal_type, setup, trend_analysis
            )
            
            # Calcula scores
            confluence_score = self._calculate_confluence_score(setup, trend_analysis)
            risk_score = self._calculate_risk_score(entry_price, stop_loss, take_profit)
            timing_score = setup['confidence']
            
            # Verifica score mínimo
            if confluence_score < self.config.min_confluence_score:
                logger.debug(f"Setup rejeitado por confluence score baixo: {confluence_score}")
                return None
            
            # Determina prioridade
            priority = self._determine_priority(confluence_score, setup['confidence'], trend_analysis)
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': self.name,
                'timeframe': timeframe,
                'priority': priority,
                'scores': {
                    'confluence': confluence_score,
                    'risk': risk_score,
                    'timing': timing_score
                },
                'setup_type': setup['type'].value,
                'metadata': {
                    **setup.get('metadata', {}),
                    'trend_strength': trend_analysis['strength'],
                    'trend_phase': trend_analysis['phase'].value
                },
                'expires_at': None  # Trend following sem expiração específica
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return None
    
    def _calculate_stop_and_target(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        entry_price: float,
        signal_type: SignalType,
        setup: Dict,
        trend_analysis: Dict
    ) -> Tuple[float, float]:
        """Calcula stop loss e take profit"""
        
        try:
            atr = indicators['atr'].iloc[-1]
            
            # Stop loss baseado no ATR
            if signal_type == SignalType.BUY:
                stop_loss = entry_price - (atr * self.config.stop_atr_multiplier)
                
                # Ajusta baseado no setup
                if setup['type'] == TrendSetup.PULLBACK_ENTRY:
                    support_level = setup['metadata'].get('support_level', entry_price)
                    stop_loss = max(stop_loss, support_level * 0.995)
                    
            else:  # SELL
                stop_loss = entry_price + (atr * self.config.stop_atr_multiplier)
                
                if setup['type'] == TrendSetup.PULLBACK_ENTRY:
                    resistance_level = setup['metadata'].get('resistance_level', entry_price)
                    stop_loss = min(stop_loss, resistance_level * 1.005)
            
            # Take profit baseado na relação risk:reward
            risk = abs(entry_price - stop_loss)
            reward = risk * self.config.profit_target_ratio
            
            if signal_type == SignalType.BUY:
                take_profit = entry_price + reward
            else:
                take_profit = entry_price - reward
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Erro ao calcular stop e target: {e}")
            # Fallback
            risk_pct = 0.02  # 2%
            if signal_type == SignalType.BUY:
                stop_loss = entry_price * (1 - risk_pct)
                take_profit = entry_price * (1 + risk_pct * 3)
            else:
                stop_loss = entry_price * (1 + risk_pct)
                take_profit = entry_price * (1 - risk_pct * 3)
            
            return stop_loss, take_profit
    
    def _calculate_confluence_score(self, setup: Dict, trend_analysis: Dict) -> float:
        """Calcula score de confluência"""
        
        base_score = setup['confidence']
        
        # Bonus por força da tendência
        base_score += trend_analysis['strength'] * 0.3
        
        # Bonus por qualidade da tendência
        base_score += trend_analysis['quality'] * 0.2
        
        # Bonus por tipo de setup
        setup_bonuses = {
            TrendSetup.PULLBACK_ENTRY: 15,
            TrendSetup.BREAKOUT_FOLLOW: 12,
            TrendSetup.MOMENTUM_ENTRY: 10,
            TrendSetup.MOVING_AVERAGE_CROSS: 8
        }
        
        base_score += setup_bonuses.get(setup['type'], 0)
        
        # Bonus por fase da tendência
        if trend_analysis['phase'] == TrendPhase.EARLY:
            base_score += 10
        elif trend_analysis['phase'] == TrendPhase.EXHAUSTION:
            base_score -= 15
        
        return max(0, min(100, base_score))
    
    def _calculate_risk_score(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calcula score de risco"""
        
        risk = abs(entry_price - stop_loss) / entry_price
        reward = abs(take_profit - entry_price) / entry_price
        
        rr_ratio = safe_divide(reward, risk, 0)
        
        # Score baseado na relação risco/recompensa
        if rr_ratio >= 4:
            return 95
        elif rr_ratio >= 3:
            return 85
        elif rr_ratio >= 2:
            return 75
        elif rr_ratio >= 1.5:
            return 65
        else:
            return 45
    
    def _calculate_pullback_confidence(self, pullback_depth: float, rsi: float, direction: str) -> float:
        """Calcula confiança do setup de pullback"""
        
        base_confidence = 60
        
        # Profundidade ideal do pullback (23.6% - 38.2% Fibonacci)
        if 20 <= pullback_depth <= 40:
            base_confidence += 20
        elif pullback_depth < 15:
            base_confidence -= 10
        elif pullback_depth > 50:
            base_confidence -= 20
        
        # RSI em zona ideal
        if direction == 'bullish' and 40 <= rsi <= 55:
            base_confidence += 15
        elif direction == 'bearish' and 45 <= rsi <= 60:
            base_confidence += 15
        
        return max(40, min(95, base_confidence))
    
    def _calculate_breakout_confidence(self, volume_ratio: float, range_size: float, trend_analysis: Dict) -> float:
        """Calcula confiança do setup de breakout"""
        
        base_confidence = 65
        
        # Volume confirmation
        if volume_ratio > 2.0:
            base_confidence += 20
        elif volume_ratio > 1.5:
            base_confidence += 10
        else:
            base_confidence -= 10
        
        # Tamanho da consolidação
        if range_size > 15:
            base_confidence += 15
        elif range_size > 8:
            base_confidence += 10
        
        # Força da tendência
        base_confidence += trend_analysis['strength'] * 0.2
        
        return max(40, min(95, base_confidence))
    
    def _calculate_ma_cross_confidence(self, trend_analysis: Dict, volume_ratio: float, direction: str) -> float:
        """Calcula confiança do setup de cruzamento de médias"""
        
        base_confidence = 55
        
        # Força da tendência
        base_confidence += trend_analysis['strength'] * 0.3
        
        # Volume confirmation
        if volume_ratio > 1.5:
            base_confidence += 15
        elif volume_ratio > 1.2:
            base_confidence += 10
        
        # Alinhamento das médias
        if trend_analysis['ma_alignment']:
            base_confidence += 15
        
        return max(40, min(90, base_confidence))
    
    def _calculate_momentum_confidence(self, momentum_pct: float, volume_ratio: float, trend_analysis: Dict) -> float:
        """Calcula confiança do setup de momentum"""
        
        base_confidence = 70
        
        # Força do momentum
        if momentum_pct > 10:
            base_confidence += 20
        elif momentum_pct > 7:
            base_confidence += 15
        
        # Volume surge
        if volume_ratio > 2.5:
            base_confidence += 15
        elif volume_ratio > 1.8:
            base_confidence += 10
        
        # Fase da tendência
        if trend_analysis['phase'] == TrendPhase.EARLY:
            base_confidence += 15
        
        return max(50, min(95, base_confidence))
    
    def _analyze_confluence(self, df: pd.DataFrame, indicators: Dict, trend_analysis: Dict, timeframe: str) -> Dict:
        """Análise de confluência para trend following"""
        
        try:
            confluence_result = self.confluence_analyzer.analyze(df, timeframe)
            
            # Adiciona fatores específicos de trend following
            trend_factors = []
            
            if trend_analysis['strength'] > 70:
                trend_factors.append({
                    'factor': 'strong_trend',
                    'value': trend_analysis['strength'],
                    'weight': 25
                })
            
            if trend_analysis['ma_alignment']:
                trend_factors.append({
                    'factor': 'ma_alignment',
                    'value': 1,
                    'weight': 20
                })
            
            if trend_analysis['adx_value'] > self.config.adx_very_strong:
                trend_factors.append({
                    'factor': 'very_strong_adx',
                    'value': trend_analysis['adx_value'],
                    'weight': 15
                })
            
            confluence_result['trend_factors'] = trend_factors
            
            return confluence_result
        
        except Exception as e:
            logger.error(f"Erro na análise de confluência: {e}")
            return {}
    
    def _determine_priority(self, confluence_score: float, setup_confidence: float, trend_analysis: Dict) -> SignalPriority:
        """Determina prioridade do sinal"""
        
        base_score = (confluence_score + setup_confidence) / 2
        
        # Bonus por fase da tendência
        if trend_analysis['phase'] == TrendPhase.EARLY:
            base_score += 10
        elif trend_analysis['phase'] == TrendPhase.EXHAUSTION:
            base_score -= 15
        
        # Bonus por força da tendência
        if trend_analysis['strength'] > 80:
            base_score += 5
        
        if base_score >= 85:
            return SignalPriority.CRITICAL
        elif base_score >= 75:
            return SignalPriority.HIGH
        elif base_score >= 65:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW


# Função de conveniência
def create_trend_following_strategy(custom_config: Dict = None) -> TrendFollowingStrategy:
    """Cria estratégia Trend Following com configuração customizada"""
    
    if custom_config:
        config = TrendFollowingConfig()
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return TrendFollowingStrategy(config)
    
    return TrendFollowingStrategy()