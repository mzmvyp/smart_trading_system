# core/market_structure.py
"""
🏗️ MARKET STRUCTURE ANALYZER - SMART TRADING SYSTEM v2.0
Análise de estrutura de mercado: HH/HL/LH/LL, trends, breakouts, consolidações
Base fundamental para todas as estratégias
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import get_logger
from utils.helpers import find_peaks_valleys, calculate_support_resistance_levels
from utils.decorators import timing, cache
from core.market_data import MarketData

logger = get_logger(__name__)

class TrendDirection(Enum):
    """📈 Direções de tendência"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class MarketPhase(Enum):
    """🌊 Fases do mercado"""
    ACCUMULATION = "accumulation"     # Lateralização após baixa
    MARKUP = "markup"                 # Tendência de alta
    DISTRIBUTION = "distribution"     # Lateralização após alta  
    MARKDOWN = "markdown"             # Tendência de baixa
    UNKNOWN = "unknown"

class StructureType(Enum):
    """🔧 Tipos de estrutura"""
    HIGHER_HIGH = "HH"      # Higher High
    HIGHER_LOW = "HL"       # Higher Low
    LOWER_HIGH = "LH"       # Lower High
    LOWER_LOW = "LL"        # Lower Low
    EQUAL_HIGH = "EH"       # Equal High
    EQUAL_LOW = "EL"        # Equal Low

@dataclass
class StructurePoint:
    """📍 Ponto de estrutura identificado"""
    index: int
    timestamp: datetime
    price: float
    structure_type: StructureType
    significance: float  # 0-1, quão significativo é o ponto
    volume: float = 0.0
    
    def __str__(self):
        return f"{self.structure_type.value}@{self.price:.4f}"

@dataclass
class TrendAnalysis:
    """📊 Análise completa de tendência"""
    direction: TrendDirection
    strength: float  # 0-1
    confidence: float  # 0-1
    duration_candles: int
    price_change_pct: float
    structure_points: List[StructurePoint] = field(default_factory=list)
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    trend_line_slope: float = 0.0
    trend_line_r2: float = 0.0  # R² da regressão linear
    
    def is_strong_trend(self) -> bool:
        """Verifica se é uma tendência forte"""
        return self.strength >= 0.7 and self.confidence >= 0.6
    
    def is_mature_trend(self) -> bool:
        """Verifica se tendência está madura (possível reversão)"""
        return (
            self.duration_candles > 50 and
            abs(self.price_change_pct) > 20 and
            self.strength > 0.8
        )

@dataclass
class BreakoutAnalysis:
    """💥 Análise de breakout"""
    level_broken: float
    breakout_type: str  # 'resistance', 'support', 'trendline'
    breakout_strength: float  # 0-1
    volume_confirmation: bool
    price_distance_pct: float  # % distância do nível
    candles_since_break: int
    retest_completed: bool = False
    false_breakout_risk: float = 0.0  # 0-1
    
    def is_valid_breakout(self) -> bool:
        """Verifica se é um breakout válido"""
        return (
            self.breakout_strength >= 0.6 and
            self.volume_confirmation and
            self.price_distance_pct >= 0.5 and
            self.false_breakout_risk <= 0.3
        )

@dataclass
class MarketStructureAnalysis:
    """🎯 Análise completa de estrutura de mercado"""
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Análises principais
    trend_analysis: TrendAnalysis
    market_phase: MarketPhase
    structure_points: List[StructurePoint]
    
    # Níveis importantes
    key_support_levels: List[float]
    key_resistance_levels: List[float]
    current_price: float
    
    # Breakouts
    recent_breakouts: List[BreakoutAnalysis]
    potential_breakout_levels: List[Dict[str, Any]]
    
    # Consolidações
    is_consolidating: bool
    consolidation_range: Optional[Tuple[float, float]] = None
    consolidation_duration: int = 0
    
    # Scores
    bullish_structure_score: float = 0.0  # 0-100
    bearish_structure_score: float = 0.0  # 0-100
    breakout_imminent_score: float = 0.0  # 0-100
    
    def get_bias(self) -> TrendDirection:
        """Retorna bias geral baseado na estrutura"""
        if self.bullish_structure_score > self.bearish_structure_score + 20:
            return TrendDirection.BULLISH
        elif self.bearish_structure_score > self.bullish_structure_score + 20:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def get_trade_context(self) -> Dict[str, Any]:
        """Retorna contexto para trading"""
        return {
            'bias': self.get_bias().value,
            'trend_strength': self.trend_analysis.strength,
            'market_phase': self.market_phase.value,
            'is_consolidating': self.is_consolidating,
            'breakout_imminent': self.breakout_imminent_score > 70,
            'key_levels': {
                'support': self.key_support_levels[:3],  # Top 3
                'resistance': self.key_resistance_levels[:3]
            }
        }

class MarketStructureAnalyzer:
    """🏗️ Analisador principal de estrutura de mercado"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Configurações para detecção de estrutura
        self.config = {
            'min_swing_size_pct': 1.0,       # Mínimo 1% para ser considerado swing
            'lookback_periods': 20,           # Períodos para análise de estrutura
            'trend_min_duration': 10,         # Mínima duração para tendência válida
            'consolidation_range_pct': 3.0,   # Máximo 3% range para consolidação
            'volume_spike_threshold': 1.5,    # 1.5x volume médio = spike
            'breakout_min_distance_pct': 0.5, # Mínimo 0.5% para breakout válido
            'structure_significance_period': 50, # Período para calcular significância
        }
        
        self.logger.info("🏗️ MarketStructureAnalyzer inicializado")
    
    @timing(threshold_seconds=0.5)
    async def analyze_structure(self, market_data: MarketData) -> MarketStructureAnalysis:
        """🎯 Análise completa de estrutura de mercado"""
        
        if not market_data.is_sufficient_data():
            raise ValueError(f"Dados insuficientes para análise de {market_data.symbol}")
        
        df = market_data.ohlcv
        symbol = market_data.symbol
        timeframe = market_data.timeframe
        
        self.logger.debug(f"🔍 Analisando estrutura: {symbol} {timeframe} ({len(df)} candles)")
        
        # 1. Identifica pontos de estrutura (swings)
        structure_points = self._identify_structure_points(df)
        
        # 2. Analisa tendência atual
        trend_analysis = self._analyze_trend(df, structure_points)
        
        # 3. Identifica fase do mercado
        market_phase = self._identify_market_phase(df, trend_analysis, structure_points)
        
        # 4. Calcula níveis de suporte e resistência
        support_levels, resistance_levels = self._calculate_key_levels(df, structure_points)
        
        # 5. Analisa breakouts recentes
        recent_breakouts = self._analyze_recent_breakouts(df, support_levels + resistance_levels)
        
        # 6. Identifica possíveis breakouts futuros
        potential_breakouts = self._identify_potential_breakouts(df, support_levels, resistance_levels)
        
        # 7. Analisa consolidação
        is_consolidating, consolidation_range, consolidation_duration = self._analyze_consolidation(df)
        
        # 8. Calcula scores de estrutura
        bullish_score, bearish_score, breakout_score = self._calculate_structure_scores(
            trend_analysis, structure_points, recent_breakouts, is_consolidating
        )
        
        # Monta análise final
        analysis = MarketStructureAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            trend_analysis=trend_analysis,
            market_phase=market_phase,
            structure_points=structure_points[-10:],  # Últimos 10 pontos
            key_support_levels=support_levels[:5],    # Top 5 supports
            key_resistance_levels=resistance_levels[:5], # Top 5 resistances
            current_price=market_data.latest_price,
            recent_breakouts=recent_breakouts,
            potential_breakout_levels=potential_breakouts,
            is_consolidating=is_consolidating,
            consolidation_range=consolidation_range,
            consolidation_duration=consolidation_duration,
            bullish_structure_score=bullish_score,
            bearish_structure_score=bearish_score,
            breakout_imminent_score=breakout_score
        )
        
        self.logger.info(
            f"✅ Estrutura analisada: {symbol} {timeframe} | "
            f"Trend: {trend_analysis.direction.value} ({trend_analysis.strength:.2f}) | "
            f"Phase: {market_phase.value} | "
            f"Scores: B{bullish_score:.0f}/B{bearish_score:.0f}/BO{breakout_score:.0f}"
        )
        
        return analysis
    
    def _identify_structure_points(self, df: pd.DataFrame) -> List[StructurePoint]:
        """📍 Identifica pontos de estrutura (HH, HL, LH, LL)"""
        
        # Encontra picos e vales
        peaks_valleys = find_peaks_valleys(df['close'].tolist(), prominence=0.01)
        peaks = peaks_valleys['peaks']
        valleys = peaks_valleys['valleys']
        
        # Combina e ordena todos os pontos
        all_points = []
        
        for peak_idx in peaks:
            if peak_idx < len(df):
                all_points.append({
                    'index': peak_idx,
                    'price': df.iloc[peak_idx]['high'],
                    'type': 'peak',
                    'timestamp': df.iloc[peak_idx]['timestamp'] if 'timestamp' in df.columns else datetime.now()
                })
        
        for valley_idx in valleys:
            if valley_idx < len(df):
                all_points.append({
                    'index': valley_idx,
                    'price': df.iloc[valley_idx]['low'],
                    'type': 'valley',
                    'timestamp': df.iloc[valley_idx]['timestamp'] if 'timestamp' in df.columns else datetime.now()
                })
        
        # Ordena por índice
        all_points.sort(key=lambda x: x['index'])
        
        # Classifica estrutura (HH, HL, LH, LL)
        structure_points = []
        
        for i in range(1, len(all_points)):
            current = all_points[i]
            previous = all_points[i-1]
            
            # Calcula significância baseada na distância dos preços
            price_change_pct = abs(current['price'] - previous['price']) / previous['price'] * 100
            significance = min(1.0, price_change_pct / self.config['min_swing_size_pct'])
            
            # Só considera swings significativos
            if significance < 0.3:
                continue
            
            # Determina tipo de estrutura
            structure_type = self._classify_structure_point(all_points, i)
            
            structure_point = StructurePoint(
                index=current['index'],
                timestamp=current['timestamp'],
                price=current['price'],
                structure_type=structure_type,
                significance=significance,
                volume=df.iloc[current['index']]['volume'] if 'volume' in df.columns else 0
            )
            
            structure_points.append(structure_point)
        
        return structure_points
    
    def _classify_structure_point(self, points: List[Dict], current_idx: int) -> StructureType:
        """🔧 Classifica tipo de estrutura de um ponto"""
        
        if current_idx == 0:
            return StructureType.UNKNOWN
        
        current = points[current_idx]
        previous = points[current_idx - 1]
        
        # Tolerância para níveis "iguais"
        equal_tolerance = 0.002  # 0.2%
        
        price_diff_pct = (current['price'] - previous['price']) / previous['price']
        
        if current['type'] == 'peak':
            if price_diff_pct > equal_tolerance:
                return StructureType.HIGHER_HIGH
            elif abs(price_diff_pct) <= equal_tolerance:
                return StructureType.EQUAL_HIGH
            else:
                return StructureType.LOWER_HIGH
        
        else:  # valley
            if price_diff_pct > equal_tolerance:
                return StructureType.HIGHER_LOW
            elif abs(price_diff_pct) <= equal_tolerance:
                return StructureType.EQUAL_LOW
            else:
                return StructureType.LOWER_LOW
    
    def _analyze_trend(self, df: pd.DataFrame, structure_points: List[StructurePoint]) -> TrendAnalysis:
        """📈 Analisa tendência atual baseada na estrutura"""
        
        if len(df) < 20 or len(structure_points) < 3:
            return TrendAnalysis(
                direction=TrendDirection.UNKNOWN,
                strength=0.0,
                confidence=0.0,
                duration_candles=0,
                price_change_pct=0.0
            )
        
        # Analisa últimos pontos de estrutura
        recent_points = structure_points[-6:]  # Últimos 6 pontos
        
        # Conta tipos de estrutura
        hh_count = sum(1 for p in recent_points if p.structure_type == StructureType.HIGHER_HIGH)
        hl_count = sum(1 for p in recent_points if p.structure_type == StructureType.HIGHER_LOW)
        lh_count = sum(1 for p in recent_points if p.structure_type == StructureType.LOWER_HIGH)
        ll_count = sum(1 for p in recent_points if p.structure_type == StructureType.LOWER_LOW)
        
        # Determina direção da tendência
        bullish_signals = hh_count + hl_count
        bearish_signals = lh_count + ll_count
        
        if bullish_signals > bearish_signals + 1:
            direction = TrendDirection.BULLISH
            strength = bullish_signals / len(recent_points)
        elif bearish_signals > bullish_signals + 1:
            direction = TrendDirection.BEARISH
            strength = bearish_signals / len(recent_points)
        else:
            direction = TrendDirection.SIDEWAYS
            strength = 0.5
        
        # Calcula mudança de preço
        start_price = df.iloc[-50]['close'] if len(df) >= 50 else df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        price_change_pct = ((end_price - start_price) / start_price) * 100
        
        # Calcula linha de tendência (regressão linear)
        y_values = df['close'].tail(30).values
        x_values = np.arange(len(y_values))
        
        try:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            r_squared = correlation ** 2
        except:
            slope, r_squared = 0.0, 0.0
        
        # Calcula confiança baseada em consistência
        confidence = min(1.0, (
            strength * 0.4 +           # Consistência da estrutura
            abs(r_squared) * 0.3 +     # Qualidade da linha de tendência
            min(1.0, abs(price_change_pct) / 10) * 0.3  # Magnitude da mudança
        ))
        
        # Identifica níveis de suporte e resistência da tendência
        support_levels = []
        resistance_levels = []
        
        if direction == TrendDirection.BULLISH:
            # Em uptrend, Higher Lows são suportes
            support_levels = [p.price for p in recent_points 
                            if p.structure_type in [StructureType.HIGHER_LOW, StructureType.EQUAL_LOW]]
        elif direction == TrendDirection.BEARISH:
            # Em downtrend, Lower Highs são resistências
            resistance_levels = [p.price for p in recent_points 
                               if p.structure_type in [StructureType.LOWER_HIGH, StructureType.EQUAL_HIGH]]
        
        return TrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=confidence,
            duration_candles=min(50, len(df)),
            price_change_pct=price_change_pct,
            structure_points=recent_points,
            support_levels=sorted(support_levels, reverse=True)[:3],
            resistance_levels=sorted(resistance_levels)[:3],
            trend_line_slope=slope,
            trend_line_r2=r_squared
        )
    
    def _identify_market_phase(self, df: pd.DataFrame, trend_analysis: TrendAnalysis, 
                             structure_points: List[StructurePoint]) -> MarketPhase:
        """🌊 Identifica fase atual do mercado"""
        
        direction = trend_analysis.direction
        strength = trend_analysis.strength
        price_change = abs(trend_analysis.price_change_pct)
        
        # Lógica simplificada de fases de mercado
        if direction == TrendDirection.BULLISH and strength > 0.6:
            return MarketPhase.MARKUP
        elif direction == TrendDirection.BEARISH and strength > 0.6:
            return MarketPhase.MARKDOWN
        elif direction == TrendDirection.SIDEWAYS:
            # Verifica se vem de alta ou baixa
            if len(df) >= 100:
                long_term_change = ((df.iloc[-1]['close'] - df.iloc[-100]['close']) / df.iloc[-100]['close']) * 100
                
                if long_term_change > 10:  # Vem de alta
                    return MarketPhase.DISTRIBUTION
                elif long_term_change < -10:  # Vem de baixa
                    return MarketPhase.ACCUMULATION
        
        return MarketPhase.UNKNOWN
    
    def _calculate_key_levels(self, df: pd.DataFrame, structure_points: List[StructurePoint]) -> Tuple[List[float], List[float]]:
        """🎯 Calcula níveis-chave de suporte e resistência"""
        
        # Usa função helper para calcular níveis básicos
        levels = calculate_support_resistance_levels(df['close'].tolist(), window=20, min_touches=2)
        support_levels = levels['support']
        resistance_levels = levels['resistance']
        
        # Adiciona pontos de estrutura significativos
        structure_supports = []
        structure_resistances = []
        
        for point in structure_points:
            if point.significance > 0.5:  # Apenas pontos significativos
                if point.structure_type in [StructureType.HIGHER_LOW, StructureType.LOWER_LOW]:
                    structure_supports.append(point.price)
                elif point.structure_type in [StructureType.HIGHER_HIGH, StructureType.LOWER_HIGH]:
                    structure_resistances.append(point.price)
        
        # Combina e remove duplicatas próximas
        all_supports = support_levels + structure_supports
        all_resistances = resistance_levels + structure_resistances
        
        # Remove níveis muito próximos (< 1% de diferença)
        def consolidate_levels(levels: List[float], tolerance: float = 0.01) -> List[float]:
            if not levels:
                return []
            
            levels_sorted = sorted(levels)
            consolidated = [levels_sorted[0]]
            
            for level in levels_sorted[1:]:
                last_level = consolidated[-1]
                if abs(level - last_level) / last_level > tolerance:
                    consolidated.append(level)
            
            return consolidated
        
        final_supports = consolidate_levels(all_supports)
        final_resistances = consolidate_levels(all_resistances)
        
        # Ordena por proximidade ao preço atual
        current_price = df.iloc[-1]['close']
        
        final_supports = sorted([s for s in final_supports if s < current_price], reverse=True)
        final_resistances = sorted([r for r in final_resistances if r > current_price])
        
        return final_supports[:5], final_resistances[:5]
    
    def _analyze_recent_breakouts(self, df: pd.DataFrame, key_levels: List[float]) -> List[BreakoutAnalysis]:
        """💥 Analisa breakouts recentes"""
        
        breakouts = []
        lookback = min(20, len(df))
        
        for i in range(len(df) - lookback, len(df)):
            if i < 1:
                continue
                
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            current_high = current_candle['high']
            current_low = current_candle['low']
            current_close = current_candle['close']
            previous_close = previous_candle['close']
            
            # Verifica breakout de cada nível
            for level in key_levels:
                # Breakout de resistência (para cima)
                if (previous_close <= level and current_high > level and 
                    current_close > level):
                    
                    distance_pct = ((current_close - level) / level) * 100
                    volume_ratio = (current_candle['volume'] / df['volume'].tail(20).mean()) if 'volume' in df.columns else 1.0
                    
                    breakout = BreakoutAnalysis(
                        level_broken=level,
                        breakout_type='resistance',
                        breakout_strength=min(1.0, distance_pct / 2.0),  # 2% = strength 1.0
                        volume_confirmation=volume_ratio > 1.3,
                        price_distance_pct=distance_pct,
                        candles_since_break=len(df) - i - 1,
                        false_breakout_risk=self._calculate_false_breakout_risk(df, i, level, 'resistance')
                    )
                    
                    breakouts.append(breakout)
                
                # Breakout de suporte (para baixo)
                elif (previous_close >= level and current_low < level and 
                      current_close < level):
                    
                    distance_pct = ((level - current_close) / level) * 100
                    volume_ratio = (current_candle['volume'] / df['volume'].tail(20).mean()) if 'volume' in df.columns else 1.0
                    
                    breakout = BreakoutAnalysis(
                        level_broken=level,
                        breakout_type='support',
                        breakout_strength=min(1.0, distance_pct / 2.0),
                        volume_confirmation=volume_ratio > 1.3,
                        price_distance_pct=distance_pct,
                        candles_since_break=len(df) - i - 1,
                        false_breakout_risk=self._calculate_false_breakout_risk(df, i, level, 'support')
                    )
                    
                    breakouts.append(breakout)
        
        # Retorna apenas breakouts recentes e válidos
        recent_breakouts = [b for b in breakouts if b.candles_since_break <= 10]
        return sorted(recent_breakouts, key=lambda x: x.candles_since_break)[:3]
    
    def _calculate_false_breakout_risk(self, df: pd.DataFrame, breakout_idx: int, 
                                     level: float, breakout_type: str) -> float:
        """🚨 Calcula risco de falso breakout"""
        
        if breakout_idx >= len(df) - 2:
            return 0.5  # Muito recente para avaliar
        
        # Verifica candles após o breakout
        candles_after = df.iloc[breakout_idx+1:breakout_idx+6]  # Próximos 5 candles
        
        if len(candles_after) == 0:
            return 0.5
        
        risk_score = 0.0
        
        for _, candle in candles_after.iterrows():
            if breakout_type == 'resistance':
                # Para breakout de resistência, se preço volta abaixo = risco
                if candle['close'] < level:
                    risk_score += 0.3
                if candle['low'] < level:
                    risk_score += 0.2
            else:  # support
                # Para breakout de suporte, se preço volta acima = risco
                if candle['close'] > level:
                    risk_score += 0.3
                if candle['high'] > level:
                    risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _identify_potential_breakouts(self, df: pd.DataFrame, supports: List[float], 
                                    resistances: List[float]) -> List[Dict[str, Any]]:
        """🎯 Identifica potenciais breakouts futuros"""
        
        current_price = df.iloc[-1]['close']
        potential_breakouts = []
        
        # Verifica proximidade a níveis importantes
        for resistance in resistances[:3]:  # Top 3 resistências
            distance_pct = ((resistance - current_price) / current_price) * 100
            
            if 0 < distance_pct <= 3:  # Até 3% de distância
                potential_breakouts.append({
                    'level': resistance,
                    'type': 'resistance',
                    'distance_pct': distance_pct,
                    'probability': max(0.1, 1.0 - distance_pct / 3.0),  # Mais próximo = maior probabilidade
                    'expected_move_pct': distance_pct * 1.5  # Estimativa de movimento
                })
        
        for support in supports[:3]:  # Top 3 suportes
            distance_pct = ((current_price - support) / current_price) * 100
            
            if 0 < distance_pct <= 3:  # Até 3% de distância
                potential_breakouts.append({
                    'level': support,
                    'type': 'support',
                    'distance_pct': distance_pct,
                    'probability': max(0.1, 1.0 - distance_pct / 3.0),
                    'expected_move_pct': distance_pct * 1.5
                })
        
        return sorted(potential_breakouts, key=lambda x: x['probability'], reverse=True)
    
    def _analyze_consolidation(self, df: pd.DataFrame) -> Tuple[bool, Optional[Tuple[float, float]], int]:
        """📦 Analisa se está em consolidação"""
        
        # Analisa últimos 20 candles
        recent_data = df.tail(20)
        
        if len(recent_data) < 10:
            return False, None, 0
        
        high = recent_data['high'].max()
        low = recent_data['low'].min()
        
        # Calcula range da consolidação
        range_pct = ((high - low) / low) * 100
        
        # Considera consolidação se range < 5%
        is_consolidating = range_pct < self.config['consolidation_range_pct']
        
        if is_consolidating:
            # Conta quantos candles estão no range
            consolidation_candles = 0
            for i in range(len(df) - 1, -1, -1):
                candle = df.iloc[i]
                if low <= candle['low'] <= high and low <= candle['high'] <= high:
                    consolidation_candles += 1
                else:
                    break
            
            return True, (low, high), consolidation_candles
        
        return False, None, 0
    
    def _calculate_structure_scores(self, trend_analysis: TrendAnalysis, 
                                  structure_points: List[StructurePoint],
                                  recent_breakouts: List[BreakoutAnalysis],
                                  is_consolidating: bool) -> Tuple[float, float, float]:
        """📊 Calcula scores de estrutura"""
        
        bullish_score = 0.0
        bearish_score = 0.0
        breakout_score = 0.0
        
        # Score baseado na tendência
        if trend_analysis.direction == TrendDirection.BULLISH:
            bullish_score += trend_analysis.strength * trend_analysis.confidence * 40
        elif trend_analysis.direction == TrendDirection.BEARISH:
            bearish_score += trend_analysis.strength * trend_analysis.confidence * 40
        
        # Score baseado na estrutura recente
        recent_structure = structure_points[-4:] if len(structure_points) >= 4 else structure_points
        
        for point in recent_structure:
            if point.structure_type in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW]:
                bullish_score += point.significance * 10
            elif point.structure_type in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW]:
                bearish_score += point.significance * 10
        
        # Score baseado em breakouts recentes
        for breakout in recent_breakouts:
            if breakout.is_valid_breakout():
                if breakout.breakout_type == 'resistance':
                    bullish_score += breakout.breakout_strength * 20
                else:  # support
                    bearish_score += breakout.breakout_strength * 20
        
        # Score de breakout iminente
        if is_consolidating:
            breakout_score += 30  # Consolidação aumenta chance de breakout
        
        # Adiciona volatilidade e momentum
        # TODO: Integrar com indicadores de momentum quando disponíveis
        
        # Normaliza scores (0-100)
        bullish_score = min(100, max(0, bullish_score))
        bearish_score = min(100, max(0, bearish_score))
        breakout_score = min(100, max(0, breakout_score))
        
        return bullish_score, bearish_score, breakout_score

# === FUNÇÕES DE CONVENIÊNCIA ===

async def analyze_market_structure(market_data: MarketData) -> MarketStructureAnalysis:
    """🎯 Função de conveniência para análise de estrutura"""
    analyzer = MarketStructureAnalyzer()
    return await analyzer.analyze_structure(market_data)

def get_trend_bias(structure_analysis: MarketStructureAnalysis) -> str:
    """📈 Retorna bias de tendência simplificado"""
    return structure_analysis.get_bias().value

def is_trending_market(structure_analysis: MarketStructureAnalysis) -> bool:
    """📊 Verifica se mercado está em tendência"""
    return (
        structure_analysis.trend_analysis.strength > 0.6 and
        structure_analysis.trend_analysis.confidence > 0.5 and
        not structure_analysis.is_consolidating
    )

def get_key_levels(structure_analysis: MarketStructureAnalysis) -> Dict[str, List[float]]:
    """🎯 Retorna níveis-chave simplificados"""
    return {
        'support': structure_analysis.key_support_levels[:3],
        'resistance': structure_analysis.key_resistance_levels[:3]
    }