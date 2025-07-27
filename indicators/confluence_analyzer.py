"""
🎯 CONFLUENCE ANALYZER - Smart Trading System v2.0

Sistema central de confluência que combina:
- Market Structure Analysis
- Trend Analysis (Multi-timeframe)
- Leading Indicators (Volume, Order Flow, Liquidity)
- Strategy Signals (Swing, Breakout)
- Support/Resistance Levels

Filosofia: Multiple Confirmations = Higher Probability = Better Risk/Reward
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

# Importar módulos do sistema
from ..core.market_structure import MarketStructureAnalyzer, StructureSignal
from ..indicators.trend_analyzer import TrendAnalyzer, TrendData
from ..indicators.leading_indicators import LeadingIndicatorsSystem, LeadingSignal
from ..strategies.swing_strategy import SwingStrategy, SwingSignal
from ..strategies.breakout_strategy import BreakoutStrategy, BreakoutSignal

logger = logging.getLogger(__name__)


class ConfluenceLevel(Enum):
    """Níveis de confluência"""
    VERY_LOW = "very_low"      # 0-30
    LOW = "low"                # 30-50
    MEDIUM = "medium"          # 50-70
    HIGH = "high"              # 70-85
    VERY_HIGH = "very_high"    # 85-100


class SignalDirection(Enum):
    """Direções dos sinais"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class ConfluenceFactor:
    """Fator individual de confluência"""
    factor_type: str              # 'structure', 'trend', 'leading', 'strategy'
    factor_name: str              # Nome específico do fator
    direction: SignalDirection
    strength: float               # 0-100 força do fator
    weight: float                 # 0-1 peso na confluência total
    confidence: float             # 0-100 confiança no fator
    timeframe: str
    details: Dict                 # Detalhes específicos do fator


@dataclass
class ConfluenceZone:
    """Zona de confluência identificada"""
    price_level: float
    zone_type: str                # 'support', 'resistance', 'pivot'
    confluence_factors: List[ConfluenceFactor]
    total_score: float            # 0-100 score total da zona
    direction_bias: SignalDirection
    strength: float               # 0-100 força da zona
    width_pct: float              # % largura da zona
    touches_count: int            # Número de toques históricos
    last_test_date: pd.Timestamp  # Último teste da zona


@dataclass
class ConfluenceSignal:
    """Sinal final de confluência"""
    signal_id: str
    direction: SignalDirection
    confluence_score: float       # 0-100 score total
    confidence_level: ConfluenceLevel
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    position_size_pct: float
    priority: int                 # 1-5 (1 = highest)
    factors: List[ConfluenceFactor]
    zones: List[ConfluenceZone]
    timeframe_analysis: Dict      # Análise por timeframe
    expiry_time: pd.Timestamp
    notes: str


class ConfluenceAnalyzer:
    """
    🎯 Sistema Principal de Análise de Confluência
    
    Combina todos os módulos para criar sinais de alta probabilidade:
    1. Market Structure (HH/HL/LH/LL, Breaks, Retests)
    2. Trend Analysis (Multi-timeframe alignment)
    3. Leading Indicators (Volume, Flow, Liquidity)
    4. Strategy Signals (Swing, Breakout setups)
    5. Support/Resistance Zones
    """
    
    def __init__(self,
                 min_confluence_score: float = 70.0,
                 min_factors_required: int = 3,
                 timeframes: List[str] = ["1H", "4H", "1D"]):
        
        self.min_confluence_score = min_confluence_score
        self.min_factors_required = min_factors_required
        self.timeframes = timeframes
        
        # Inicializar todos os analisadores
        self.structure_analyzer = MarketStructureAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.leading_system = LeadingIndicatorsSystem()
        self.swing_strategy = SwingStrategy()
        self.breakout_strategy = BreakoutStrategy()
        
        self.logger = logging.getLogger(f"{__name__}.ConfluenceAnalyzer")
        
        # Configuração de pesos para confluência
        self.weights = {
            'market_structure': 0.25,     # Estrutura de mercado
            'trend_alignment': 0.25,      # Alinhamento de trend
            'leading_indicators': 0.20,   # Indicadores leading
            'strategy_signals': 0.20,     # Sinais de estratégia
            'support_resistance': 0.10    # Suporte/Resistência
        }
        
        # Configuração de zonas
        self.zone_config = {
            'zone_width_pct': 0.015,      # 1.5% largura padrão
            'min_touches': 3,             # Mínimo toques para zona válida
            'lookback_periods': 100,      # Períodos para análise histórica
            'proximity_threshold': 0.02   # 2% proximidade para ativação
        }
    
    def analyze_confluence(self, 
                          market_data: Dict[str, pd.DataFrame],
                          symbol: str = "BTCUSDT") -> List[ConfluenceSignal]:
        """
        Análise principal de confluência
        
        Args:
            market_data: Dict com DataFrames por timeframe {"1H": df, "4H": df, "1D": df}
            symbol: Símbolo sendo analisado
            
        Returns:
            Lista de ConfluenceSignals de alta probabilidade
        """
        signals = []
        
        try:
            self.logger.info(f"Iniciando análise de confluência para {symbol}")
            
            # 1. Coletar fatores de confluência de todas as fontes
            all_factors = self._collect_confluence_factors(market_data, symbol)
            
            # 2. Identificar zonas de confluência
            confluence_zones = self._identify_confluence_zones(market_data, all_factors)
            
            # 3. Criar sinais baseados em confluência
            current_price = market_data["4H"]['close'].iloc[-1]
            signals = self._create_confluence_signals(
                all_factors, confluence_zones, current_price, symbol)
            
            # 4. Filtrar e ranquear sinais
            signals = self._filter_and_rank_signals(signals)
            
            self.logger.info(f"Análise concluída - {len(signals)} sinais de confluência encontrados")
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na análise de confluência: {e}")
            return signals
    
    def _collect_confluence_factors(self, 
                                  market_data: Dict[str, pd.DataFrame],
                                  symbol: str) -> List[ConfluenceFactor]:
        """Coleta fatores de confluência de todas as fontes"""
        factors = []
        
        try:
            # 1. Market Structure Factors
            structure_factors = self._analyze_structure_factors(market_data)
            factors.extend(structure_factors)
            
            # 2. Trend Analysis Factors
            trend_factors = self._analyze_trend_factors(market_data)
            factors.extend(trend_factors)
            
            # 3. Leading Indicators Factors
            leading_factors = self._analyze_leading_factors(market_data)
            factors.extend(leading_factors)
            
            # 4. Strategy Signal Factors
            strategy_factors = self._analyze_strategy_factors(market_data, symbol)
            factors.extend(strategy_factors)
            
            # 5. Support/Resistance Factors
            sr_factors = self._analyze_sr_factors(market_data)
            factors.extend(sr_factors)
            
            self.logger.info(f"Coletados {len(factors)} fatores de confluência")
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na coleta de fatores: {e}")
            return factors
    
    def _analyze_structure_factors(self, market_data: Dict[str, pd.DataFrame]) -> List[ConfluenceFactor]:
        """Analisa fatores de estrutura de mercado"""
        factors = []
        
        try:
            for tf, data in market_data.items():
                if len(data) < 50:
                    continue
                
                # Análise de estrutura para cada timeframe
                structure_analysis = self.structure_analyzer.analyze_market_structure(data)
                
                if 'signals' in structure_analysis:
                    for signal in structure_analysis['signals']:
                        if signal.strength > 60:  # Apenas sinais fortes
                            factor = ConfluenceFactor(
                                factor_type="structure",
                                factor_name=f"{signal.signal_type}_{tf}",
                                direction=SignalDirection(signal.direction),
                                strength=signal.strength,
                                weight=self.weights['market_structure'] / len(market_data),
                                confidence=signal.confidence,
                                timeframe=tf,
                                details=signal.details
                            )
                            factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na análise de estrutura: {e}")
            return factors
    
    def _analyze_trend_factors(self, market_data: Dict[str, pd.DataFrame]) -> List[ConfluenceFactor]:
        """Analisa fatores de tendência"""
        factors = []
        
        try:
            for tf, data in market_data.items():
                if len(data) < 50:
                    continue
                
                # Análise de trend para cada timeframe
                trend_data = self.trend_analyzer.analyze_trend(data, tf)
                
                if trend_data.strength > 50:  # Trend definido
                    factor = ConfluenceFactor(
                        factor_type="trend",
                        factor_name=f"trend_{tf}",
                        direction=SignalDirection(trend_data.direction),
                        strength=trend_data.strength,
                        weight=self.weights['trend_alignment'] / len(market_data),
                        confidence=trend_data.confidence,
                        timeframe=tf,
                        details={
                            'trend_strength': trend_data.strength,
                            'momentum': trend_data.momentum,
                            'volatility': trend_data.volatility
                        }
                    )
                    factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na análise de trend: {e}")
            return factors
    
    def _analyze_leading_factors(self, market_data: Dict[str, pd.DataFrame]) -> List[ConfluenceFactor]:
        """Analisa fatores de indicadores leading"""
        factors = []
        
        try:
            # Usar principalmente 4H para leading indicators
            if "4H" in market_data:
                data_4h = market_data["4H"]
                current_price = data_4h['close'].iloc[-1]
                
                # Análise de leading indicators
                leading_signals = self.leading_system.analyze_all_leading(data_4h, current_price, "4H")
                
                for signal in leading_signals:
                    if signal.strength > 55:  # Sinais significativos
                        factor = ConfluenceFactor(
                            factor_type="leading",
                            factor_name=signal.signal_type,
                            direction=SignalDirection(signal.direction),
                            strength=signal.strength,
                            weight=self.weights['leading_indicators'] / max(1, len(leading_signals)),
                            confidence=signal.confidence,
                            timeframe=signal.timeframe,
                            details=signal.details
                        )
                        factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na análise de leading: {e}")
            return factors
    
    def _analyze_strategy_factors(self, market_data: Dict[str, pd.DataFrame], symbol: str) -> List[ConfluenceFactor]:
        """Analisa fatores de sinais de estratégia"""
        factors = []
        
        try:
            # Swing Strategy Signals
            if all(tf in market_data for tf in ["1H", "4H", "1D"]):
                swing_signals = self.swing_strategy.analyze_swing_opportunity(
                    market_data["4H"], market_data["1D"], market_data["1H"], symbol)
                
                for signal in swing_signals:
                    if signal.signal_strength > 60:
                        factor = ConfluenceFactor(
                            factor_type="strategy",
                            factor_name=f"swing_{signal.setup.setup_type.value}",
                            direction=SignalDirection(signal.setup.direction.value),
                            strength=signal.signal_strength,
                            weight=self.weights['strategy_signals'] / 2,  # Dividir entre swing e breakout
                            confidence=signal.confidence,
                            timeframe=signal.setup.timeframe,
                            details={
                                'setup_type': signal.setup.setup_type.value,
                                'confluence_score': signal.setup.confluence_score,
                                'risk_reward': signal.setup.risk_reward_ratio
                            }
                        )
                        factors.append(factor)
            
            # Breakout Strategy Signals
            if "4H" in market_data:
                breakout_signals = self.breakout_strategy.analyze_breakout_opportunity(
                    market_data["4H"], symbol)
                
                for signal in breakout_signals:
                    if signal.signal_strength > 60:
                        factor = ConfluenceFactor(
                            factor_type="strategy",
                            factor_name=f"breakout_{signal.setup.breakout_type.value}",
                            direction=SignalDirection(signal.setup.direction),
                            strength=signal.signal_strength,
                            weight=self.weights['strategy_signals'] / 2,
                            confidence=signal.confidence,
                            timeframe=signal.setup.timeframe,
                            details={
                                'breakout_type': signal.setup.breakout_type.value,
                                'pattern_type': signal.setup.pattern.pattern_type,
                                'volume_surge': signal.setup.volume_surge
                            }
                        )
                        factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na análise de estratégias: {e}")
            return factors
    
    def _analyze_sr_factors(self, market_data: Dict[str, pd.DataFrame]) -> List[ConfluenceFactor]:
        """Analisa fatores de suporte e resistência"""
        factors = []
        
        try:
            # Usar 4H para S/R analysis
            if "4H" in market_data:
                data = market_data["4H"].tail(self.zone_config['lookback_periods'])
                current_price = data['close'].iloc[-1]
                
                # Identificar levels de S/R
                sr_levels = self._find_sr_levels(data)
                
                for level_data in sr_levels:
                    level = level_data['level']
                    level_type = level_data['type']
                    touches = level_data['touches']
                    strength = level_data['strength']
                    
                    # Verificar proximidade
                    distance_pct = abs(current_price - level) / level
                    
                    if distance_pct < self.zone_config['proximity_threshold']:
                        # Determinar direção baseada no tipo de level
                        if level_type == 'support':
                            direction = SignalDirection.BULLISH
                        elif level_type == 'resistance':
                            direction = SignalDirection.BEARISH
                        else:
                            direction = SignalDirection.NEUTRAL
                        
                        factor = ConfluenceFactor(
                            factor_type="support_resistance",
                            factor_name=f"{level_type}_level",
                            direction=direction,
                            strength=strength,
                            weight=self.weights['support_resistance'],
                            confidence=min(90, touches * 20),
                            timeframe="4H",
                            details={
                                'level': level,
                                'level_type': level_type,
                                'touches': touches,
                                'distance_pct': distance_pct
                            }
                        )
                        factors.append(factor)
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Erro na análise de S/R: {e}")
            return factors
    
    def _find_sr_levels(self, data: pd.DataFrame) -> List[Dict]:
        """Encontra níveis de suporte e resistência"""
        levels = []
        
        try:
            # Procurar por pivots significativos
            for i in range(10, len(data) - 10):
                current_high = data.iloc[i]['high']
                current_low = data.iloc[i]['low']
                
                # Verificar se é um pivot high
                is_pivot_high = True
                for j in range(i-5, i+6):
                    if j != i and data.iloc[j]['high'] >= current_high:
                        is_pivot_high = False
                        break
                
                # Verificar se é um pivot low
                is_pivot_low = True
                for j in range(i-5, i+6):
                    if j != i and data.iloc[j]['low'] <= current_low:
                        is_pivot_low = False
                        break
                
                # Contar toques nos levels
                if is_pivot_high:
                    touches = self._count_level_touches(data, current_high, 'resistance')
                    if touches >= self.zone_config['min_touches']:
                        levels.append({
                            'level': current_high,
                            'type': 'resistance',
                            'touches': touches,
                            'strength': min(100, touches * 25),
                            'index': i
                        })
                
                if is_pivot_low:
                    touches = self._count_level_touches(data, current_low, 'support')
                    if touches >= self.zone_config['min_touches']:
                        levels.append({
                            'level': current_low,
                            'type': 'support',
                            'touches': touches,
                            'strength': min(100, touches * 25),
                            'index': i
                        })
            
            # Filtrar levels muito próximos
            levels = self._filter_close_levels(levels)
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Erro na busca de S/R: {e}")
            return levels
    
    def _count_level_touches(self, data: pd.DataFrame, level: float, level_type: str) -> int:
        """Conta quantas vezes um level foi tocado"""
        touches = 0
        tolerance = level * 0.01  # 1% tolerance
        
        for _, row in data.iterrows():
            if level_type == 'resistance':
                if abs(row['high'] - level) <= tolerance:
                    touches += 1
            else:  # support
                if abs(row['low'] - level) <= tolerance:
                    touches += 1
        
        return touches
    
    def _filter_close_levels(self, levels: List[Dict]) -> List[Dict]:
        """Filtra levels muito próximos, mantendo os mais fortes"""
        if len(levels) <= 1:
            return levels
        
        # Ordenar por força
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        filtered = []
        for level in levels:
            # Verificar se está muito próximo de algum level já aceito
            too_close = False
            for accepted in filtered:
                distance = abs(level['level'] - accepted['level']) / level['level']
                if distance < 0.02:  # Dentro de 2%
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(level)
        
        return filtered
    
    def _identify_confluence_zones(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 factors: List[ConfluenceFactor]) -> List[ConfluenceZone]:
        """Identifica zonas de confluência baseadas nos fatores"""
        zones = []
        
        try:
            current_price = market_data["4H"]['close'].iloc[-1]
            
            # Agrupar fatores por proximidade de preço
            price_groups = defaultdict(list)
            
            for factor in factors:
                # Tentar extrair um nível de preço do fator
                price_level = self._extract_price_level(factor, current_price)
                if price_level:
                    # Agrupar por bins de preço (1% width)
                    price_bin = round(price_level / (current_price * 0.01)) * (current_price * 0.01)
                    price_groups[price_bin].append(factor)
            
            # Criar zonas onde há múltiplos fatores
            for price_level, group_factors in price_groups.items():
                if len(group_factors) >= 2:  # Mínimo 2 fatores para zona
                    
                    # Calcular score da zona
                    total_score = sum(f.strength * f.weight for f in group_factors)
                    
                    # Determinar bias direcional
                    bullish_score = sum(f.strength * f.weight for f in group_factors 
                                      if f.direction == SignalDirection.BULLISH)
                    bearish_score = sum(f.strength * f.weight for f in group_factors 
                                      if f.direction == SignalDirection.BEARISH)
                    
                    if bullish_score > bearish_score * 1.2:
                        direction_bias = SignalDirection.BULLISH
                    elif bearish_score > bullish_score * 1.2:
                        direction_bias = SignalDirection.BEARISH
                    else:
                        direction_bias = SignalDirection.NEUTRAL
                    
                    # Determinar tipo de zona
                    if price_level < current_price * 0.99:
                        zone_type = "support"
                    elif price_level > current_price * 1.01:
                        zone_type = "resistance"
                    else:
                        zone_type = "pivot"
                    
                    zone = ConfluenceZone(
                        price_level=price_level,
                        zone_type=zone_type,
                        confluence_factors=group_factors,
                        total_score=total_score,
                        direction_bias=direction_bias,
                        strength=min(100, total_score),
                        width_pct=self.zone_config['zone_width_pct'],
                        touches_count=len(group_factors),
                        last_test_date=pd.Timestamp.now()
                    )
                    zones.append(zone)
            
            # Ordenar zonas por score
            zones.sort(key=lambda x: x.total_score, reverse=True)
            
            self.logger.info(f"Identificadas {len(zones)} zonas de confluência")
            return zones[:5]  # Top 5 zonas
            
        except Exception as e:
            self.logger.error(f"Erro na identificação de zonas: {e}")
            return zones
    
    def _extract_price_level(self, factor: ConfluenceFactor, current_price: float) -> Optional[float]:
        """Extrai nível de preço relevante do fator"""
        try:
            # Para fatores de S/R, usar o level direto
            if factor.factor_type == "support_resistance":
                return factor.details.get('level')
            
            # Para outros fatores, usar preço atual como referência
            # (podem indicar direção sem level específico)
            return current_price
            
        except Exception:
            return None
    
    def _create_confluence_signals(self, 
                                 factors: List[ConfluenceFactor],
                                 zones: List[ConfluenceZone],
                                 current_price: float,
                                 symbol: str) -> List[ConfluenceSignal]:
        """Cria sinais finais baseados em confluência"""
        signals = []
        
        try:
            # Calcular confluência geral
            overall_confluence = self._calculate_overall_confluence(factors)
            
            if overall_confluence['score'] < self.min_confluence_score:
                return signals
            
            # Criar sinal baseado na confluência dominante
            direction = overall_confluence['direction']
            score = overall_confluence['score']
            
            # Definir levels baseados em zonas próximas
            entry_price, stop_loss, take_profit_1, take_profit_2 = self._calculate_signal_levels(
                direction, current_price, zones)
            
            if entry_price and stop_loss and take_profit_1:
                
                risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
                
                if risk_reward >= 1.5:  # Mínimo R:R
                    
                    # Determinar confidence level
                    confidence_level = self._get_confidence_level(score)
                    
                    # Position sizing baseado na confluência
                    position_size = self._calculate_position_size(score, risk_reward)
                    
                    # Priority baseado na qualidade
                    priority = 1 if score >= 85 else (2 if score >= 75 else 3)
                    
                    signal_id = f"{symbol}_confluence_{direction.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
                    
                    signal = ConfluenceSignal(
                        signal_id=signal_id,
                        direction=direction,
                        confluence_score=score,
                        confidence_level=confidence_level,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit_1=take_profit_1,
                        take_profit_2=take_profit_2,
                        risk_reward_ratio=risk_reward,
                        position_size_pct=position_size,
                        priority=priority,
                        factors=factors,
                        zones=zones,
                        timeframe_analysis=overall_confluence['timeframe_breakdown'],
                        expiry_time=pd.Timestamp.now() + pd.Timedelta(hours=48),
                        notes=f"Confluência {direction.value} - {len(factors)} fatores"
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na criação de sinais: {e}")
            return signals
    
    def _calculate_overall_confluence(self, factors: List[ConfluenceFactor]) -> Dict:
        """Calcula confluência geral dos fatores"""
        try:
            if not factors:
                return {'score': 0, 'direction': SignalDirection.NEUTRAL, 'timeframe_breakdown': {}}
            
            # Calcular scores por direção
            bullish_score = sum(f.strength * f.weight for f in factors 
                              if f.direction == SignalDirection.BULLISH)
            bearish_score = sum(f.strength * f.weight for f in factors 
                              if f.direction == SignalDirection.BEARISH)
            neutral_score = sum(f.strength * f.weight for f in factors 
                              if f.direction == SignalDirection.NEUTRAL)
            
            # Determinar direção dominante
            if bullish_score > bearish_score and bullish_score > neutral_score:
                direction = SignalDirection.BULLISH
                score = bullish_score
            elif bearish_score > bullish_score and bearish_score > neutral_score:
                direction = SignalDirection.BEARISH
                score = bearish_score
            else:
                direction = SignalDirection.NEUTRAL
                score = max(bullish_score, bearish_score, neutral_score)
            
            # Breakdown por timeframe
            timeframe_breakdown = defaultdict(lambda: {'count': 0, 'score': 0})
            for factor in factors:
                tf = factor.timeframe
                timeframe_breakdown[tf]['count'] += 1
                timeframe_breakdown[tf]['score'] += factor.strength * factor.weight
            
            return {
                'score': min(100, score),
                'direction': direction,
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'neutral_score': neutral_score,
                'timeframe_breakdown': dict(timeframe_breakdown)
            }
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de confluência: {e}")
            return {'score': 0, 'direction': SignalDirection.NEUTRAL, 'timeframe_breakdown': {}}
    
    def _calculate_signal_levels(self, 
                               direction: SignalDirection,
                               current_price: float,
                               zones: List[ConfluenceZone]) -> Tuple[float, float, float, float]:
        """Calcula levels do sinal baseado em zonas"""
        try:
            # Encontrar zonas relevantes
            support_zones = [z for z in zones if z.zone_type == "support"]
            resistance_zones = [z for z in zones if z.zone_type == "resistance"]
            
            if direction == SignalDirection.BULLISH:
                # Entry: próximo ao preço atual ou support
                entry_price = current_price * 1.002  # 0.2% acima
                
                # Stop: abaixo do support mais próximo
                if support_zones:
                    nearest_support = min(support_zones, key=lambda z: abs(z.price_level - current_price))
                    stop_loss = nearest_support.price_level * 0.98
                else:
                    stop_loss = current_price * 0.95  # 5% stop
                
                # Targets: próximas resistências ou % targets
                if resistance_zones:
                    resistance_above = [z for z in resistance_zones if z.price_level > current_price]
                    if resistance_above:
                        take_profit_1 = min(resistance_above, key=lambda z: z.price_level).price_level * 0.99
                        take_profit_2 = take_profit_1 * 1.08  # 8% adicional
                    else:
                        take_profit_1 = current_price * 1.06
                        take_profit_2 = current_price * 1.12
                else:
                    take_profit_1 = current_price * 1.06
                    take_profit_2 = current_price * 1.12
                    
            else:  # BEARISH
                # Entry: próximo ao preço atual ou resistance
                entry_price = current_price * 0.998  # 0.2% abaixo
                
                # Stop: acima da resistance mais próxima
                if resistance_zones:
                    nearest_resistance = min(resistance_zones, key=lambda z: abs(z.price_level - current_price))
                    stop_loss = nearest_resistance.price_level * 1.02
                else:
                    stop_loss = current_price * 1.05  # 5% stop
                
                # Targets: próximos supports ou % targets
                if support_zones:
                    support_below = [z for z in support_zones if z.price_level < current_price]
                    if support_below:
                        take_profit_1 = max(support_below, key=lambda z: z.price_level).price_level * 1.01
                        take_profit_2 = take_profit_1 * 0.92  # 8% adicional
                    else:
                        take_profit_1 = current_price * 0.94
                        take_profit_2 = current_price * 0.88
                else:
                    take_profit_1 = current_price * 0.94
                    take_profit_2 = current_price * 0.88
            
            return entry_price, stop_loss, take_profit_1, take_profit_2
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de levels: {e}")
            return None, None, None, None
    
    def _get_confidence_level(self, score: float) -> ConfluenceLevel:
        """Converte score em confidence level"""
        if score >= 85:
            return ConfluenceLevel.VERY_HIGH
        elif score >= 70:
            return ConfluenceLevel.HIGH
        elif score >= 50:
            return ConfluenceLevel.MEDIUM
        elif score >= 30:
            return ConfluenceLevel.LOW
        else:
            return ConfluenceLevel.VERY_LOW
    
    def _calculate_position_size(self, score: float, risk_reward: float) -> float:
        """Calcula position size baseado na confluência"""
        base_size = 0.02  # 2% base
        
        # Ajustar baseado no score
        score_multiplier = (score / 100) * 2  # 0-2x
        
        # Ajustar baseado no R:R
        rr_multiplier = min(2.0, risk_reward / 2)  # Max 2x
        
        position_size = base_size * score_multiplier * rr_multiplier
        
        return min(0.05, max(0.01, position_size))  # Entre 1-5%
    
    def _filter_and_rank_signals(self, signals: List[ConfluenceSignal]) -> List[ConfluenceSignal]:
        """Filtra e ranqueia sinais por qualidade"""
        # Filtrar sinais fracos
        filtered = [s for s in signals if s.confluence_score >= self.min_confluence_score]
        
        # Ordenar por score e priority
        filtered.sort(key=lambda x: (x.priority, -x.confluence_score))
        
        return filtered[:3]  # Top 3 sinais


def main():
    """Teste básico do sistema de confluência"""
    # Dados de exemplo
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    def create_sample_data(dates, base_price=50000):
        price_changes = np.random.randn(len(dates)).cumsum() * 50
        opens = base_price + price_changes
        closes = opens + np.random.randn(len(dates)) * 25
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(len(dates)) * 30)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(len(dates)) * 30)
        volumes = np.random.exponential(1000000, len(dates))
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
    
    # Criar dados para múltiplos timeframes
    market_data = {
        "1H": create_sample_data(dates),
        "4H": create_sample_data(dates[::4], 50000),
        "1D": create_sample_data(dates[::24], 50000)
    }
    
    # Testar sistema de confluência
    analyzer = ConfluenceAnalyzer()
    signals = analyzer.analyze_confluence(market_data, "BTCUSDT")
    
    print(f"\n🎯 CONFLUENCE ANALYSIS")
    print(f"Signals Found: {len(signals)}")
    
    for i, signal in enumerate(signals, 1):
        print(f"\n📊 CONFLUENCE SIGNAL {i}")
        print(f"   Direction: {signal.direction.value}")
        print(f"   Confluence Score: {signal.confluence_score:.1f}")
        print(f"   Confidence: {signal.confidence_level.value}")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Stop: ${signal.stop_loss:,.2f}")
        print(f"   Target 1: ${signal.take_profit_1:,.2f}")
        print(f"   R:R: {signal.risk_reward_ratio:.2f}")
        print(f"   Position Size: {signal.position_size_pct:.1%}")
        print(f"   Priority: {signal.priority}")
        print(f"   Factors: {len(signal.factors)}")
        print(f"   Zones: {len(signal.zones)}")


if __name__ == "__main__":
    main()