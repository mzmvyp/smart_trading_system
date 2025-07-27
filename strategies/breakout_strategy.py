"""
💥 BREAKOUT STRATEGY - Smart Trading System v2.0

Estratégia focada em breakouts de alta probabilidade:
- Consolidation Pattern Breaks
- Volume Confirmation Required
- Retest Entry Logic
- False Breakout Filtering

Filosofia: Volume Confirms Direction + Patience for Retest = High Win Rate
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Importar módulos do sistema
from ..core.market_structure import MarketStructureAnalyzer, StructureSignal
from ..indicators.trend_analyzer import TrendAnalyzer, TrendData
from ..indicators.leading_indicators import LeadingIndicatorsSystem, LeadingSignal

logger = logging.getLogger(__name__)


class BreakoutType(Enum):
    """Tipos de breakout"""
    RESISTANCE_BREAK = "resistance_break"
    SUPPORT_BREAK = "support_break"
    RANGE_BREAK_UP = "range_break_up"
    RANGE_BREAK_DOWN = "range_break_down"
    TRIANGLE_BREAK = "triangle_break"
    FLAG_BREAK = "flag_break"


class BreakoutPhase(Enum):
    """Fases do breakout"""
    INITIAL_BREAK = "initial_break"       # Rompimento inicial
    RETEST_PHASE = "retest_phase"         # Aguardando reteste
    CONTINUATION = "continuation"         # Continuação confirmada


@dataclass
class ConsolidationPattern:
    """Padrão de consolidação identificado"""
    pattern_type: str                     # 'range', 'triangle', 'flag', 'pennant'
    support_level: float
    resistance_level: float
    duration_periods: int                 # Duração em períodos
    volume_profile: Dict                  # Volume durante consolidação
    breakout_level: float                 # Level para breakout
    pattern_strength: float               # 0-100 qualidade do padrão
    min_target: float                     # Target mínimo
    max_target: float                     # Target máximo


@dataclass
class BreakoutSetup:
    """Setup de breakout identificado"""
    breakout_type: BreakoutType
    direction: str                        # 'bullish', 'bearish'
    breakout_level: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    volume_confirmation: bool
    pattern: ConsolidationPattern
    risk_reward_ratio: float
    breakout_strength: float             # 0-100
    volume_surge: float                  # Volume vs média
    retest_expected: bool
    invalidation_level: float
    timeframe: str
    timestamp: pd.Timestamp


@dataclass
class BreakoutSignal:
    """Sinal de breakout trading"""
    signal_id: str
    setup: BreakoutSetup
    phase: BreakoutPhase
    signal_strength: float               # 0-100
    confidence: float                    # 0-100
    position_size_pct: float            # % do portfólio
    priority: int                        # 1-5
    expiry_time: pd.Timestamp
    notes: str
    entry_conditions: List[str]          # Condições para entrada


class BreakoutStrategy:
    """
    💥 Breakout Trading Strategy
    
    Focada em capturar movimentos explosivos após consolidações:
    1. Pattern Recognition (Range, Triangle, Flag)
    2. Volume Confirmation (2x+ average)
    3. Retest Logic (patient entry)
    4. False Breakout Filtering
    """
    
    def __init__(self, 
                 primary_timeframe: str = "4H",
                 volume_surge_threshold: float = 2.0,
                 min_consolidation_periods: int = 15,
                 max_consolidation_periods: int = 50,
                 retest_patience_hours: int = 48):
        
        self.primary_tf = primary_timeframe
        self.volume_threshold = volume_surge_threshold
        self.min_consolidation = min_consolidation_periods
        self.max_consolidation = max_consolidation_periods
        self.retest_patience = retest_patience_hours
        
        # Inicializar analisadores
        self.structure_analyzer = MarketStructureAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.leading_system = LeadingIndicatorsSystem()
        
        self.logger = logging.getLogger(f"{__name__}.BreakoutStrategy")
        
        # Configurações de padrões
        self.pattern_config = {
            'range_threshold': 0.02,          # 2% min range size
            'triangle_convergence': 0.7,      # 70% convergência
            'flag_angle_max': 30,             # Máx 30° slope
            'volume_decline_threshold': 0.6,   # 60% volume decline
            'breakout_confirmation': 0.005,   # 0.5% above/below level
        }
    
    def analyze_breakout_opportunity(self, 
                                   data: pd.DataFrame,
                                   symbol: str = "BTCUSDT") -> List[BreakoutSignal]:
        """
        Análise principal para identificar oportunidades de breakout
        
        Args:
            data: DataFrame com dados OHLCV
            symbol: Símbolo sendo analisado
            
        Returns:
            Lista de BreakoutSignals identificados
        """
        signals = []
        
        try:
            self.logger.info(f"Analisando breakouts para {symbol}")
            
            # 1. Identificar padrões de consolidação
            patterns = self._identify_consolidation_patterns(data)
            
            # 2. Analisar breakouts em progresso
            current_price = data['close'].iloc[-1]
            
            for pattern in patterns:
                breakout_setup = self._analyze_pattern_breakout(pattern, data, current_price)
                if breakout_setup:
                    signal = self._create_breakout_signal(breakout_setup, symbol)
                    if signal:
                        signals.append(signal)
            
            # 3. Verificar retestes de breakouts anteriores
            retest_signals = self._analyze_retest_opportunities(data, symbol)
            signals.extend(retest_signals)
            
            self.logger.info(f"Encontrados {len(signals)} sinais de breakout para {symbol}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na análise de breakout: {e}")
            return signals
    
    def _identify_consolidation_patterns(self, data: pd.DataFrame) -> List[ConsolidationPattern]:
        """Identifica padrões de consolidação"""
        patterns = []
        
        try:
            # Analisar diferentes janelas de consolidação
            for window in range(self.min_consolidation, min(len(data), self.max_consolidation + 1)):
                if window > len(data):
                    continue
                
                recent_data = data.tail(window)
                
                # Range Pattern
                range_pattern = self._detect_range_pattern(recent_data, window)
                if range_pattern:
                    patterns.append(range_pattern)
                
                # Triangle Pattern
                triangle_pattern = self._detect_triangle_pattern(recent_data, window)
                if triangle_pattern:
                    patterns.append(triangle_pattern)
                
                # Flag Pattern
                flag_pattern = self._detect_flag_pattern(recent_data, window)
                if flag_pattern:
                    patterns.append(flag_pattern)
            
            # Filtrar padrões sobrepostos (manter o melhor)
            patterns = self._filter_overlapping_patterns(patterns)
            
            self.logger.info(f"Identificados {len(patterns)} padrões de consolidação")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Erro na identificação de padrões: {e}")
            return patterns
    
    def _detect_range_pattern(self, data: pd.DataFrame, window: int) -> Optional[ConsolidationPattern]:
        """Detecta padrão de range/rectangle"""
        try:
            # Identificar suporte e resistência
            highs = data['high']
            lows = data['low']
            
            # Resistance = média dos 3 maiores highs
            top_highs = highs.nlargest(3).mean()
            # Support = média dos 3 menores lows  
            bottom_lows = lows.nsmallest(3).mean()
            
            range_size = (top_highs - bottom_lows) / bottom_lows
            
            # Verificar se é um range válido (2-15%)
            if range_size < self.pattern_config['range_threshold'] or range_size > 0.15:
                return None
            
            # Verificar se preço respeitou os levels
            respect_count = 0
            for _, row in data.iterrows():
                # Toques na resistência
                if abs(row['high'] - top_highs) / top_highs < 0.01:
                    respect_count += 1
                # Toques no suporte
                if abs(row['low'] - bottom_lows) / bottom_lows < 0.01:
                    respect_count += 1
            
            if respect_count < 4:  # Mínimo 4 toques
                return None
            
            # Volume profile durante range
            avg_volume = data['volume'].mean()
            volume_decline = data['volume'].tail(window//3).mean() / data['volume'].head(window//3).mean()
            
            # Qualidade do padrão
            pattern_strength = min(100, (respect_count * 15) + (volume_decline * 30) + 40)
            
            # Targets baseados no range size
            breakout_up = top_highs * (1 + self.pattern_config['breakout_confirmation'])
            breakout_down = bottom_lows * (1 - self.pattern_config['breakout_confirmation'])
            
            target_distance = top_highs - bottom_lows
            
            return ConsolidationPattern(
                pattern_type="range",
                support_level=bottom_lows,
                resistance_level=top_highs,
                duration_periods=window,
                volume_profile={'avg_volume': avg_volume, 'decline_ratio': volume_decline},
                breakout_level=top_highs,  # Para breakout bullish
                pattern_strength=pattern_strength,
                min_target=top_highs + target_distance * 0.5,
                max_target=top_highs + target_distance * 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de range: {e}")
            return None
    
    def _detect_triangle_pattern(self, data: pd.DataFrame, window: int) -> Optional[ConsolidationPattern]:
        """Detecta padrão de triângulo"""
        try:
            if window < 20:  # Mínimo para triângulo
                return None
            
            # Dividir em duas metades para analisar convergência
            first_half = data.head(window//2)
            second_half = data.tail(window//2)
            
            # Highs e lows de cada metade
            first_high = first_half['high'].max()
            first_low = first_half['low'].min()
            second_high = second_half['high'].max()
            second_low = second_half['low'].min()
            
            # Calcular convergência
            high_convergence = (first_high - second_high) / first_high
            low_convergence = (second_low - first_low) / first_low
            
            # Verificar se está convergindo
            total_convergence = high_convergence + low_convergence
            
            if total_convergence < self.pattern_config['triangle_convergence']:
                return None
            
            # Apex (ponto de convergência)
            current_high = data['high'].tail(5).max()
            current_low = data['low'].tail(5).min()
            apex_level = (current_high + current_low) / 2
            
            # Volume deve declinar
            volume_decline = data['volume'].tail(window//3).mean() / data['volume'].head(window//3).mean()
            
            if volume_decline > 0.8:  # Pouco decline
                return None
            
            pattern_strength = min(100, total_convergence * 100 + (1 - volume_decline) * 50)
            
            # Target = altura do triângulo
            triangle_height = first_high - first_low
            
            return ConsolidationPattern(
                pattern_type="triangle",
                support_level=current_low,
                resistance_level=current_high,
                duration_periods=window,
                volume_profile={'decline_ratio': volume_decline},
                breakout_level=apex_level,
                pattern_strength=pattern_strength,
                min_target=apex_level + triangle_height * 0.618,
                max_target=apex_level + triangle_height * 1.0
            )
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de triângulo: {e}")
            return None
    
    def _detect_flag_pattern(self, data: pd.DataFrame, window: int) -> Optional[ConsolidationPattern]:
        """Detecta padrão de flag/pennant"""
        try:
            if window < 10 or window > 25:  # Flags são curtas
                return None
            
            # Flag deve seguir movimento forte (flagpole)
            pre_flag = data.iloc[:-window]
            if len(pre_flag) < 10:
                return None
            
            # Verificar se houve movimento forte antes
            flagpole_start = pre_flag['close'].iloc[-10]
            flagpole_end = pre_flag['close'].iloc[-1]
            flagpole_move = abs(flagpole_end - flagpole_start) / flagpole_start
            
            if flagpole_move < 0.05:  # Mínimo 5% move
                return None
            
            # Flag data
            flag_data = data.tail(window)
            
            # Flag deve ser lateral/counter-trend
            flag_start = flag_data['close'].iloc[0]
            flag_end = flag_data['close'].iloc[-1]
            flag_move = abs(flag_end - flag_start) / flag_start
            
            if flag_move > 0.03:  # Máximo 3% move na flag
                return None
            
            # Direção da flag vs flagpole
            flagpole_direction = "up" if flagpole_end > flagpole_start else "down"
            
            # Levels da flag
            flag_high = flag_data['high'].max()
            flag_low = flag_data['low'].min()
            
            # Volume deve declinar na flag
            flag_volume = flag_data['volume'].mean()
            pre_volume = pre_flag['volume'].tail(10).mean()
            volume_ratio = flag_volume / pre_volume
            
            if volume_ratio > 0.7:  # Muito volume na flag
                return None
            
            pattern_strength = min(100, flagpole_move * 500 + (1 - volume_ratio) * 100)
            
            # Target = altura do flagpole
            if flagpole_direction == "up":
                breakout_level = flag_high
                target = flag_high + (flagpole_end - flagpole_start)
            else:
                breakout_level = flag_low
                target = flag_low - (flagpole_start - flagpole_end)
            
            return ConsolidationPattern(
                pattern_type="flag",
                support_level=flag_low,
                resistance_level=flag_high,
                duration_periods=window,
                volume_profile={'volume_ratio': volume_ratio, 'flagpole_move': flagpole_move},
                breakout_level=breakout_level,
                pattern_strength=pattern_strength,
                min_target=target,
                max_target=target
            )
            
        except Exception as e:
            self.logger.error(f"Erro na detecção de flag: {e}")
            return None
    
    def _filter_overlapping_patterns(self, patterns: List[ConsolidationPattern]) -> List[ConsolidationPattern]:
        """Filtra padrões sobrepostos, mantendo os melhores"""
        if len(patterns) <= 1:
            return patterns
        
        # Ordenar por força
        patterns.sort(key=lambda x: x.pattern_strength, reverse=True)
        
        filtered = []
        for pattern in patterns:
            # Verificar sobreposição com padrões já aceitos
            overlaps = False
            for accepted in filtered:
                # Se períodos se sobrepõem significativamente
                overlap_threshold = min(pattern.duration_periods, accepted.duration_periods) * 0.5
                if abs(pattern.duration_periods - accepted.duration_periods) < overlap_threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(pattern)
        
        return filtered[:3]  # Máximo 3 padrões
    
    def _analyze_pattern_breakout(self, 
                                pattern: ConsolidationPattern,
                                data: pd.DataFrame,
                                current_price: float) -> Optional[BreakoutSetup]:
        """Analisa se padrão está em breakout"""
        try:
            # Verificar se breakout já aconteceu
            recent_data = data.tail(5)  # Últimos 5 períodos
            
            breakout_occurred = False
            breakout_direction = None
            breakout_level = 0
            
            # Verificar breakout bullish (acima da resistência)
            if current_price > pattern.resistance_level * (1 + self.pattern_config['breakout_confirmation']):
                breakout_occurred = True
                breakout_direction = "bullish"
                breakout_level = pattern.resistance_level
            
            # Verificar breakout bearish (abaixo do suporte)
            elif current_price < pattern.support_level * (1 - self.pattern_config['breakout_confirmation']):
                breakout_occurred = True
                breakout_direction = "bearish"
                breakout_level = pattern.support_level
            
            if not breakout_occurred:
                return None
            
            # Verificar confirmação de volume
            recent_volume = recent_data['volume'].mean()
            historical_volume = data.tail(30)['volume'].mean()  # Média de 30 períodos
            volume_surge = recent_volume / historical_volume
            
            volume_confirmation = volume_surge >= self.volume_threshold
            
            # Definir entry, stop e targets
            if breakout_direction == "bullish":
                entry_price = pattern.resistance_level * 1.005  # 0.5% acima
                stop_loss = pattern.support_level * 0.98       # 2% abaixo do suporte
                take_profit_1 = pattern.min_target
                take_profit_2 = pattern.max_target
                invalidation = pattern.support_level
                breakout_type = BreakoutType.RESISTANCE_BREAK
                
            else:  # bearish
                entry_price = pattern.support_level * 0.995    # 0.5% abaixo
                stop_loss = pattern.resistance_level * 1.02    # 2% acima da resistência
                take_profit_1 = current_price - (pattern.resistance_level - pattern.support_level) * 0.5
                take_profit_2 = current_price - (pattern.resistance_level - pattern.support_level) * 1.0
                invalidation = pattern.resistance_level
                breakout_type = BreakoutType.SUPPORT_BREAK
            
            risk_reward = abs(take_profit_1 - entry_price) / abs(entry_price - stop_loss)
            
            # Força do breakout
            price_distance = abs(current_price - breakout_level) / breakout_level
            breakout_strength = min(100, pattern.pattern_strength * 0.7 + price_distance * 1000 + volume_surge * 10)
            
            # Expectativa de retest (padrões maiores tendem a retest)
            retest_expected = pattern.duration_periods > 20 or volume_surge < 3
            
            return BreakoutSetup(
                breakout_type=breakout_type,
                direction=breakout_direction,
                breakout_level=breakout_level,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                volume_confirmation=volume_confirmation,
                pattern=pattern,
                risk_reward_ratio=risk_reward,
                breakout_strength=breakout_strength,
                volume_surge=volume_surge,
                retest_expected=retest_expected,
                invalidation_level=invalidation,
                timeframe=self.primary_tf,
                timestamp=pd.Timestamp.now()
            )
            
        except Exception as e:
            self.logger.error(f"Erro na análise de breakout: {e}")
            return None
    
    def _analyze_retest_opportunities(self, data: pd.DataFrame, symbol: str) -> List[BreakoutSignal]:
        """Analisa oportunidades de reteste de breakouts anteriores"""
        signals = []
        
        try:
            # Simplificado: procurar por retestes em support/resistance recentes
            current_price = data['close'].iloc[-1]
            recent_data = data.tail(50)
            
            # Identificar levels importantes que foram quebrados
            significant_levels = []
            
            for i in range(10, len(recent_data) - 5):
                high = recent_data.iloc[i]['high']
                low = recent_data.iloc[i]['low']
                
                # Verificar se foi um level significativo (tocado múltiplas vezes)
                touches_before = 0
                touches_after = 0
                
                for j in range(max(0, i-10), i):
                    if abs(recent_data.iloc[j]['high'] - high) / high < 0.01:
                        touches_before += 1
                    if abs(recent_data.iloc[j]['low'] - low) / low < 0.01:
                        touches_before += 1
                
                for j in range(i+1, min(len(recent_data), i+10)):
                    if abs(recent_data.iloc[j]['high'] - high) / high < 0.01:
                        touches_after += 1
                    if abs(recent_data.iloc[j]['low'] - low) / low < 0.01:
                        touches_after += 1
                
                if touches_before >= 2 and touches_after == 0:  # Level que foi quebrado
                    significant_levels.append({'level': high, 'type': 'resistance', 'index': i})
                    significant_levels.append({'level': low, 'type': 'support', 'index': i})
            
            # Verificar retestes próximos
            for level_data in significant_levels:
                level = level_data['level']
                level_type = level_data['type']
                distance = abs(current_price - level) / level
                
                if distance < 0.02:  # Dentro de 2%
                    # Possível reteste
                    # ... implementar lógica de reteste específica
                    pass
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na análise de reteste: {e}")
            return signals
    
    def _create_breakout_signal(self, setup: BreakoutSetup, symbol: str) -> Optional[BreakoutSignal]:
        """Converte setup em sinal de trading"""
        try:
            # Determinar fase do breakout
            if setup.retest_expected and not setup.volume_confirmation:
                phase = BreakoutPhase.INITIAL_BREAK
                entry_conditions = ["Aguardar reteste", "Volume confirmation"]
                position_size = 0.02  # Posição menor até confirmação
                priority = 3
            elif setup.volume_confirmation:
                phase = BreakoutPhase.CONTINUATION
                entry_conditions = ["Enter immediately", "Volume confirmed"]
                position_size = 0.04  # Posição normal
                priority = 2
            else:
                phase = BreakoutPhase.RETEST_PHASE
                entry_conditions = ["Wait for retest", "Volume surge"]
                position_size = 0.015
                priority = 4
            
            # Signal strength
            signal_strength = setup.breakout_strength
            
            # Confiança baseada em volume e padrão
            pattern_quality = setup.pattern.pattern_strength
            volume_quality = min(100, setup.volume_surge * 30)
            confidence = (pattern_quality + volume_quality) / 2
            
            # Filtrar sinais fracos
            if signal_strength < 60 or confidence < 50:
                return None
            
            # Expiração
            expiry_hours = 24 if phase == BreakoutPhase.CONTINUATION else self.retest_patience
            expiry_time = pd.Timestamp.now() + pd.Timedelta(hours=expiry_hours)
            
            signal_id = f"{symbol}_{setup.breakout_type.value}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
            
            return BreakoutSignal(
                signal_id=signal_id,
                setup=setup,
                phase=phase,
                signal_strength=signal_strength,
                confidence=confidence,
                position_size_pct=position_size,
                priority=priority,
                expiry_time=expiry_time,
                notes=f"{setup.pattern.pattern_type.title()} breakout {setup.direction}",
                entry_conditions=entry_conditions
            )
            
        except Exception as e:
            self.logger.error(f"Erro na criação do sinal: {e}")
            return None


def main():
    """Teste básico da estratégia"""
    # Criar dados de exemplo com padrão de range
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    # Simular consolidação seguida de breakout
    base_price = 50000
    
    # Primeira metade: range
    range_data = []
    for i in range(50):
        # Range entre 49500-50500
        noise = np.random.randn() * 50
        price = base_price + noise
        range_data.append({
            'open': price,
            'high': price + abs(np.random.randn() * 100),
            'low': price - abs(np.random.randn() * 100),
            'close': price + np.random.randn() * 30,
            'volume': np.random.exponential(1000000)
        })
    
    # Segunda metade: breakout
    breakout_data = []
    breakout_price = 50500
    for i in range(50):
        breakout_price += np.random.randn() * 100 + 50  # Trend up
        breakout_data.append({
            'open': breakout_price,
            'high': breakout_price + abs(np.random.randn() * 150),
            'low': breakout_price - abs(np.random.randn() * 50),
            'close': breakout_price + np.random.randn() * 75,
            'volume': np.random.exponential(2000000)  # Volume maior
        })
    
    all_data = range_data + breakout_data
    df = pd.DataFrame(all_data, index=dates)
    
    # Testar estratégia
    strategy = BreakoutStrategy()
    signals = strategy.analyze_breakout_opportunity(df, "BTCUSDT")
    
    print(f"\n💥 BREAKOUT STRATEGY ANALYSIS")
    print(f"Signals Found: {len(signals)}")
    
    for i, signal in enumerate(signals, 1):
        setup = signal.setup
        print(f"\n📊 SIGNAL {i}")
        print(f"   Type: {setup.breakout_type.value}")
        print(f"   Direction: {setup.direction}")
        print(f"   Phase: {signal.phase.value}")
        print(f"   Pattern: {setup.pattern.pattern_type}")
        print(f"   Entry: ${setup.entry_price:,.2f}")
        print(f"   Stop: ${setup.stop_loss:,.2f}")
        print(f"   Target: ${setup.take_profit_1:,.2f}")
        print(f"   Volume Surge: {setup.volume_surge:.1f}x")
        print(f"   Strength: {signal.signal_strength:.1f}")
        print(f"   Priority: {signal.priority}")


if __name__ == "__main__":
    main()