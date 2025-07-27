"""
🎯 SWING STRATEGY - Smart Trading System v2.0

Estratégia principal de Swing Trading focada em:
- Timeframes: 4H/1D para setup, 1H para entrada
- Market Structure: HH/HL para bull, LH/LL para bear
- Confluence: Multiple confirmations required
- Risk Management: Structure-based stops

Filosofia: Patience + Confluence + Structure = Profit
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

# Importar módulos do sistema
from ..core.market_structure import MarketStructureAnalyzer, StructureSignal
from ..indicators.trend_analyzer import TrendAnalyzer, TrendData
from ..indicators.leading_indicators import LeadingIndicatorsSystem, LeadingSignal

logger = logging.getLogger(__name__)


class SwingDirection(Enum):
    """Direções do swing"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SwingSetupType(Enum):
    """Tipos de setup de swing"""
    STRUCTURE_BREAK = "structure_break"          # Rompimento de estrutura
    PULLBACK_ENTRY = "pullback_entry"           # Entrada em pullback
    RETEST_SUPPORT = "retest_support"           # Reteste de suporte
    RETEST_RESISTANCE = "retest_resistance"     # Reteste de resistência


@dataclass
class SwingSetup:
    """Setup de swing trading identificado"""
    setup_type: SwingSetupType
    direction: SwingDirection
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    confluence_score: float               # 0-100
    setup_strength: float                 # 0-100
    timeframe: str
    timestamp: pd.Timestamp
    reasoning: str
    confluence_factors: List[str]
    market_structure: Dict
    invalidation_price: float


@dataclass
class SwingSignal:
    """Sinal de swing trading"""
    signal_id: str
    setup: SwingSetup
    signal_strength: float               # 0-100
    confidence: float                    # 0-100
    risk_level: str                      # 'low', 'medium', 'high'
    position_size_pct: float            # % do portfólio
    priority: int                        # 1-5 (1 = highest)
    expiry_time: pd.Timestamp
    notes: str


class SwingStrategy:
    """
    🎯 Swing Trading Strategy
    
    Estratégia focada em capturar movimentos de 3-10 dias baseada em:
    1. Market Structure (HH/HL/LH/LL)
    2. Trend Confirmation (multi-timeframe)
    3. Leading Indicators (Volume, Order Flow)
    4. Confluence Scoring
    """
    
    def __init__(self, 
                 primary_timeframe: str = "4H",
                 confirmation_timeframe: str = "1D",
                 entry_timeframe: str = "1H",
                 min_confluence_score: float = 65.0,
                 min_risk_reward: float = 2.0):
        
        self.primary_tf = primary_timeframe
        self.confirmation_tf = confirmation_timeframe  
        self.entry_tf = entry_timeframe
        self.min_confluence_score = min_confluence_score
        self.min_risk_reward = min_risk_reward
        
        # Inicializar analisadores
        self.structure_analyzer = MarketStructureAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.leading_system = LeadingIndicatorsSystem()
        
        self.logger = logging.getLogger(f"{__name__}.SwingStrategy")
        
        # Parâmetros de configuração
        self.config = {
            'structure_break_threshold': 0.5,    # % para confirmar rompimento
            'pullback_depth_min': 0.3,          # Mín 30% retração
            'pullback_depth_max': 0.7,          # Máx 70% retração
            'retest_tolerance': 0.02,           # 2% tolerância para reteste
            'confluence_weights': {             # Pesos para confluence
                'market_structure': 0.35,
                'trend_alignment': 0.25, 
                'leading_indicators': 0.25,
                'support_resistance': 0.15
            }
        }
    
    def analyze_swing_opportunity(self, 
                                data_4h: pd.DataFrame,
                                data_1d: pd.DataFrame,
                                data_1h: pd.DataFrame,
                                symbol: str = "BTCUSDT") -> List[SwingSignal]:
        """
        Análise principal para identificar oportunidades de swing
        
        Args:
            data_4h: Dados 4H (primary timeframe)
            data_1d: Dados 1D (confirmation)
            data_1h: Dados 1H (entry timing)
            symbol: Símbolo sendo analisado
            
        Returns:
            Lista de SwingSignals identificados
        """
        signals = []
        
        try:
            self.logger.info(f"Analisando swing para {symbol}")
            
            # 1. Análise de Market Structure (4H)
            structure_4h = self.structure_analyzer.analyze_market_structure(data_4h)
            
            # 2. Confirmação de Trend (1D)  
            trend_1d = self.trend_analyzer.analyze_trend(data_1d, "1D")
            
            # 3. Leading Indicators (4H)
            current_price = data_4h['close'].iloc[-1]
            leading_signals = self.leading_system.analyze_all_leading(
                data_4h, current_price, "4H")
            
            # 4. Identificar setups específicos
            setups = self._identify_swing_setups(
                structure_4h, trend_1d, leading_signals, data_4h, data_1h)
            
            # 5. Converter setups em sinais
            for setup in setups:
                signal = self._create_swing_signal(setup, symbol)
                if signal:
                    signals.append(signal)
            
            self.logger.info(f"Encontrados {len(signals)} sinais de swing para {symbol}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na análise de swing: {e}")
            return signals
    
    def _identify_swing_setups(self, 
                              structure_data: Dict,
                              trend_data: TrendData,
                              leading_signals: List[LeadingSignal],
                              data_4h: pd.DataFrame,
                              data_1h: pd.DataFrame) -> List[SwingSetup]:
        """Identifica setups específicos de swing trading"""
        setups = []
        
        try:
            current_price = data_4h['close'].iloc[-1]
            
            # Setup 1: Structure Break + Trend Alignment
            structure_break_setup = self._analyze_structure_break(
                structure_data, trend_data, current_price, data_4h)
            if structure_break_setup:
                setups.append(structure_break_setup)
            
            # Setup 2: Pullback Entry in Trend
            pullback_setup = self._analyze_pullback_entry(
                structure_data, trend_data, leading_signals, current_price, data_4h)
            if pullback_setup:
                setups.append(pullback_setup)
            
            # Setup 3: Support/Resistance Retest
            retest_setup = self._analyze_retest_setup(
                structure_data, trend_data, current_price, data_4h)
            if retest_setup:
                setups.append(retest_setup)
            
            return setups
            
        except Exception as e:
            self.logger.error(f"Erro na identificação de setups: {e}")
            return setups
    
    def _analyze_structure_break(self, 
                               structure_data: Dict,
                               trend_data: TrendData,
                               current_price: float,
                               data_4h: pd.DataFrame) -> Optional[SwingSetup]:
        """Analisa setup de rompimento de estrutura"""
        try:
            if not structure_data or 'signals' not in structure_data:
                return None
            
            # Procurar sinais de rompimento recentes
            structure_signals = structure_data['signals']
            recent_breaks = [s for s in structure_signals 
                           if s.signal_type in ['structure_break', 'trend_change'] 
                           and s.strength > 70]
            
            if not recent_breaks:
                return None
            
            latest_break = recent_breaks[-1]  # Mais recente
            
            # Verificar alinhamento com trend 1D
            trend_direction = trend_data.direction
            break_direction = latest_break.direction
            
            if trend_direction != break_direction:
                return None  # Sem alinhamento
            
            # Definir levels baseados na estrutura
            if break_direction == "bullish":
                # Break acima de resistência
                resistance_level = latest_break.details.get('level', current_price)
                entry_price = resistance_level * 1.005  # 0.5% acima
                stop_loss = latest_break.details.get('previous_low', entry_price * 0.95)
                take_profit_1 = entry_price * 1.06   # 6% target
                take_profit_2 = entry_price * 1.12   # 12% target
                direction = SwingDirection.BULLISH
                
            else:  # bearish
                # Break abaixo de suporte
                support_level = latest_break.details.get('level', current_price)
                entry_price = support_level * 0.995  # 0.5% abaixo
                stop_loss = latest_break.details.get('previous_high', entry_price * 1.05)
                take_profit_1 = entry_price * 0.94   # 6% target
                take_profit_2 = entry_price * 0.88   # 12% target
                direction = SwingDirection.BEARISH
            
            # Calcular confluence
            confluence_factors = [
                f"Structure break {break_direction}",
                f"Trend alignment 1D {trend_direction}",
                f"Break strength {latest_break.strength:.1f}"
            ]
            
            confluence_score = self._calculate_confluence_score({
                'structure_signal': latest_break,
                'trend_data': trend_data,
                'leading_signals': []
            })
            
            risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
            
            if confluence_score >= self.min_confluence_score and risk_reward >= self.min_risk_reward:
                return SwingSetup(
                    setup_type=SwingSetupType.STRUCTURE_BREAK,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    risk_reward_ratio=risk_reward,
                    confluence_score=confluence_score,
                    setup_strength=latest_break.strength,
                    timeframe=self.primary_tf,
                    timestamp=pd.Timestamp.now(),
                    reasoning=f"Structure break {break_direction} with trend alignment",
                    confluence_factors=confluence_factors,
                    market_structure=structure_data,
                    invalidation_price=stop_loss * (1.02 if direction == SwingDirection.BEARISH else 0.98)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de structure break: {e}")
            return None
    
    def _analyze_pullback_entry(self, 
                              structure_data: Dict,
                              trend_data: TrendData,
                              leading_signals: List[LeadingSignal],
                              current_price: float,
                              data_4h: pd.DataFrame) -> Optional[SwingSetup]:
        """Analisa setup de entrada em pullback"""
        try:
            # Verificar se estamos em trend definido
            if trend_data.strength < 60 or trend_data.direction == "sideways":
                return None
            
            # Calcular retração desde o último extremo
            if trend_data.direction == "bullish":
                # Procurar último high e calcular pullback
                recent_highs = data_4h['high'].rolling(10).max()
                last_high = recent_highs.iloc[-10:].max()
                pullback_depth = (last_high - current_price) / last_high
                
                if pullback_depth < self.config['pullback_depth_min'] or \
                   pullback_depth > self.config['pullback_depth_max']:
                    return None
                
                # Setup bullish pullback
                entry_price = current_price * 1.002
                stop_loss = current_price * (1 - pullback_depth * 1.2)  # Abaixo do pullback
                take_profit_1 = last_high * 1.02  # 2% acima do high anterior
                take_profit_2 = last_high * 1.08  # 8% acima
                direction = SwingDirection.BULLISH
                
            else:  # bearish trend
                # Procurar último low e calcular rally
                recent_lows = data_4h['low'].rolling(10).min()
                last_low = recent_lows.iloc[-10:].min()
                rally_depth = (current_price - last_low) / last_low
                
                if rally_depth < self.config['pullback_depth_min'] or \
                   rally_depth > self.config['pullback_depth_max']:
                    return None
                
                # Setup bearish rally
                entry_price = current_price * 0.998
                stop_loss = current_price * (1 + rally_depth * 1.2)  # Acima do rally
                take_profit_1 = last_low * 0.98   # 2% abaixo do low anterior
                take_profit_2 = last_low * 0.92   # 8% abaixo
                direction = SwingDirection.BEARISH
            
            # Verificar confluência com leading indicators
            leading_score = 50  # Neutro por padrão
            leading_factors = []
            
            for signal in leading_signals:
                if signal.direction == direction.value:
                    leading_score = max(leading_score, signal.strength)
                    leading_factors.append(f"{signal.signal_type} {signal.direction}")
            
            confluence_factors = [
                f"Trend {trend_data.direction} (strength {trend_data.strength:.1f})",
                f"Pullback depth {pullback_depth:.1%}",
                f"Leading score {leading_score:.1f}"
            ] + leading_factors
            
            confluence_score = self._calculate_confluence_score({
                'trend_data': trend_data,
                'leading_signals': leading_signals,
                'pullback_quality': pullback_depth
            })
            
            risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
            
            if confluence_score >= self.min_confluence_score and risk_reward >= self.min_risk_reward:
                return SwingSetup(
                    setup_type=SwingSetupType.PULLBACK_ENTRY,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    risk_reward_ratio=risk_reward,
                    confluence_score=confluence_score,
                    setup_strength=trend_data.strength,
                    timeframe=self.primary_tf,
                    timestamp=pd.Timestamp.now(),
                    reasoning=f"Pullback entry in {trend_data.direction} trend",
                    confluence_factors=confluence_factors,
                    market_structure=structure_data,
                    invalidation_price=stop_loss * (1.03 if direction == SwingDirection.BEARISH else 0.97)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de pullback: {e}")
            return None
    
    def _analyze_retest_setup(self, 
                            structure_data: Dict,
                            trend_data: TrendData,
                            current_price: float,
                            data_4h: pd.DataFrame) -> Optional[SwingSetup]:
        """Analisa setup de reteste de support/resistance"""
        try:
            # Identificar levels importantes dos últimos 50 períodos
            lookback_df = data_4h.tail(50)
            
            # Support levels (lows que foram respeitados)
            support_levels = []
            resistance_levels = []
            
            # Simplified S/R detection
            for i in range(5, len(lookback_df)-5):
                current_low = lookback_df.iloc[i]['low']
                current_high = lookback_df.iloc[i]['high']
                
                # Check if it's a significant low (support)
                if (current_low <= lookback_df.iloc[i-5:i]['low'].min() and 
                    current_low <= lookback_df.iloc[i+1:i+6]['low'].min()):
                    support_levels.append(current_low)
                
                # Check if it's a significant high (resistance)
                if (current_high >= lookback_df.iloc[i-5:i]['high'].max() and 
                    current_high >= lookback_df.iloc[i+1:i+6]['high'].max()):
                    resistance_levels.append(current_high)
            
            # Encontrar level mais próximo sendo retestado
            all_levels = [(level, 'support') for level in support_levels] + \
                        [(level, 'resistance') for level in resistance_levels]
            
            if not all_levels:
                return None
            
            # Level mais próximo
            closest_level, level_type = min(all_levels, 
                key=lambda x: abs(x[0] - current_price))
            
            distance_pct = abs(closest_level - current_price) / current_price
            
            # Verificar se está próximo o suficiente (< 2%)
            if distance_pct > self.config['retest_tolerance']:
                return None
            
            # Setup baseado no tipo de level
            if level_type == 'support' and trend_data.direction == "bullish":
                # Retest de suporte em uptrend
                direction = SwingDirection.BULLISH
                entry_price = closest_level * 1.005
                stop_loss = closest_level * 0.97
                take_profit_1 = closest_level * 1.06
                take_profit_2 = closest_level * 1.12
                setup_type = SwingSetupType.RETEST_SUPPORT
                
            elif level_type == 'resistance' and trend_data.direction == "bearish":
                # Retest de resistência em downtrend
                direction = SwingDirection.BEARISH
                entry_price = closest_level * 0.995
                stop_loss = closest_level * 1.03
                take_profit_1 = closest_level * 0.94
                take_profit_2 = closest_level * 0.88
                setup_type = SwingSetupType.RETEST_RESISTANCE
                
            else:
                return None  # Não há alinhamento
            
            confluence_factors = [
                f"Retest {level_type} at {closest_level:.2f}",
                f"Trend alignment {trend_data.direction}",
                f"Distance {distance_pct:.1%}"
            ]
            
            confluence_score = self._calculate_confluence_score({
                'trend_data': trend_data,
                'level_strength': 70,  # Assumir level forte
                'distance_quality': (1 - distance_pct) * 100
            })
            
            risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
            
            if confluence_score >= self.min_confluence_score and risk_reward >= self.min_risk_reward:
                return SwingSetup(
                    setup_type=setup_type,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    risk_reward_ratio=risk_reward,
                    confluence_score=confluence_score,
                    setup_strength=70,
                    timeframe=self.primary_tf,
                    timestamp=pd.Timestamp.now(),
                    reasoning=f"Retest {level_type} in {trend_data.direction} trend",
                    confluence_factors=confluence_factors,
                    market_structure={'closest_level': closest_level, 'level_type': level_type},
                    invalidation_price=stop_loss * (1.02 if direction == SwingDirection.BEARISH else 0.98)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de retest: {e}")
            return None
    
    def _calculate_confluence_score(self, factors: Dict) -> float:
        """Calcula score de confluência baseado nos fatores"""
        try:
            weights = self.config['confluence_weights']
            total_score = 0
            
            # Market Structure Score
            if 'structure_signal' in factors:
                structure_score = factors['structure_signal'].strength
                total_score += structure_score * weights['market_structure']
            
            # Trend Alignment Score
            if 'trend_data' in factors:
                trend_score = factors['trend_data'].strength
                total_score += trend_score * weights['trend_alignment']
            
            # Leading Indicators Score
            if 'leading_signals' in factors:
                leading_signals = factors['leading_signals']
                if leading_signals:
                    leading_score = np.mean([s.strength for s in leading_signals])
                else:
                    leading_score = 50  # Neutro
                total_score += leading_score * weights['leading_indicators']
            
            # Support/Resistance Score
            sr_score = 50  # Default
            if 'level_strength' in factors:
                sr_score = factors['level_strength']
            elif 'pullback_quality' in factors:
                sr_score = (1 - factors['pullback_quality']) * 100
            
            total_score += sr_score * weights['support_resistance']
            
            return min(100, max(0, total_score))
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de confluência: {e}")
            return 50
    
    def _create_swing_signal(self, setup: SwingSetup, symbol: str) -> Optional[SwingSignal]:
        """Converte setup em sinal de trading"""
        try:
            # Determinar prioridade baseada na qualidade
            if setup.confluence_score >= 85:
                priority = 1
                risk_level = "low"
                position_size = 0.05  # 5%
            elif setup.confluence_score >= 75:
                priority = 2
                risk_level = "medium"
                position_size = 0.03  # 3%
            elif setup.confluence_score >= 65:
                priority = 3
                risk_level = "medium"
                position_size = 0.02  # 2%
            else:
                return None  # Não passa no filtro
            
            # Signal strength combinada
            signal_strength = (setup.confluence_score + setup.setup_strength) / 2
            
            # Confiança baseada em risk/reward
            confidence = min(85, setup.confluence_score * (setup.risk_reward_ratio / 3))
            
            # Expiração do sinal (72h para swing)
            expiry_time = pd.Timestamp.now() + pd.Timedelta(hours=72)
            
            signal_id = f"{symbol}_{setup.setup_type.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
            
            return SwingSignal(
                signal_id=signal_id,
                setup=setup,
                signal_strength=signal_strength,
                confidence=confidence,
                risk_level=risk_level,
                position_size_pct=position_size,
                priority=priority,
                expiry_time=expiry_time,
                notes=f"Swing {setup.direction.value} - {setup.reasoning}"
            )
            
        except Exception as e:
            self.logger.error(f"Erro na criação do sinal: {e}")
            return None
    
    def validate_signal(self, signal: SwingSignal, current_data: pd.DataFrame) -> bool:
        """Valida se o sinal ainda é válido"""
        try:
            current_price = current_data['close'].iloc[-1]
            setup = signal.setup
            
            # Verificar se não foi invalidado
            if setup.direction == SwingDirection.BULLISH:
                if current_price <= setup.invalidation_price:
                    return False
            else:
                if current_price >= setup.invalidation_price:
                    return False
            
            # Verificar expiração
            if pd.Timestamp.now() > signal.expiry_time:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro na validação do sinal: {e}")
            return False


def main():
    """Teste básico da estratégia"""
    # Dados de exemplo
    dates_4h = pd.date_range(start='2024-01-01', periods=200, freq='4H')
    dates_1d = pd.date_range(start='2024-01-01', periods=50, freq='1D')
    dates_1h = pd.date_range(start='2024-01-01', periods=800, freq='1H')
    
    np.random.seed(42)
    
    # Simular dados OHLCV
    def create_sample_data(dates, base_price=50000):
        price_changes = np.random.randn(len(dates)).cumsum() * 100
        opens = base_price + price_changes
        closes = opens + np.random.randn(len(dates)) * 50
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
    
    data_4h = create_sample_data(dates_4h)
    data_1d = create_sample_data(dates_1d)
    data_1h = create_sample_data(dates_1h)
    
    # Testar estratégia
    strategy = SwingStrategy()
    signals = strategy.analyze_swing_opportunity(data_4h, data_1d, data_1h, "BTCUSDT")
    
    print(f"\n🎯 SWING STRATEGY ANALYSIS")
    print(f"Signals Found: {len(signals)}")
    
    for i, signal in enumerate(signals, 1):
        setup = signal.setup
        print(f"\n📊 SIGNAL {i}")
        print(f"   Type: {setup.setup_type.value}")
        print(f"   Direction: {setup.direction.value}")
        print(f"   Entry: ${setup.entry_price:,.2f}")
        print(f"   Stop: ${setup.stop_loss:,.2f}")
        print(f"   Target 1: ${setup.take_profit_1:,.2f}")
        print(f"   R:R: {setup.risk_reward_ratio:.2f}")
        print(f"   Confluence: {setup.confluence_score:.1f}")
        print(f"   Priority: {signal.priority}")
        print(f"   Position Size: {signal.position_size_pct:.1%}")


if __name__ == "__main__":
    main()