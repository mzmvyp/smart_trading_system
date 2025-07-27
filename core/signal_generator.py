"""
🎯 SIGNAL GENERATOR - Smart Trading System v2.0

Motor principal que orquestra TODOS os componentes do sistema:
- Market Structure Analysis
- Trend Analysis (Multi-timeframe)
- Leading Indicators (Volume, Order Flow, Liquidity)
- Trading Strategies (Swing, Breakout)
- Confluence Analysis
- Risk Management
- All Filters (Market Condition, Volatility, Time, Fundamental)

Filosofia: All Components Working in Harmony = Ultimate Trading Edge
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone
import uuid

# Importar todos os componentes do sistema
from .market_data import MarketDataProvider
from .market_structure import MarketStructureAnalyzer
from ..indicators.trend_analyzer import TrendAnalyzer
from ..indicators.leading_indicators import LeadingIndicatorsSystem
from ..indicators.confluence_analyzer import ConfluenceAnalyzer
from ..strategies.swing_strategy import SwingStrategy
from ..strategies.breakout_strategy import BreakoutStrategy
from .risk_manager import RiskManager, PositionSizingInput
from ..filters.market_condition import MarketConditionFilter
from ..filters.volatility_filter import VolatilityFilter
from ..filters.time_filter import TimeFilter
from ..filters.fundamental_filter import FundamentalFilter

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Qualidade do sinal"""
    EXCEPTIONAL = "exceptional"    # 90-100 score
    EXCELLENT = "excellent"        # 80-90 score
    GOOD = "good"                 # 70-80 score
    MODERATE = "moderate"         # 60-70 score
    WEAK = "weak"                 # 50-60 score
    POOR = "poor"                 # <50 score


class SignalStatus(Enum):
    """Status do sinal"""
    ACTIVE = "active"             # Sinal ativo
    PENDING = "pending"           # Aguardando condições
    EXECUTED = "executed"         # Sinal executado
    CANCELLED = "cancelled"       # Sinal cancelado
    EXPIRED = "expired"           # Sinal expirado


class TradeType(Enum):
    """Tipo de trade"""
    LONG = "long"
    SHORT = "short"


@dataclass
class ComponentAnalysis:
    """Análise de um componente individual"""
    component_name: str
    enabled: bool
    analysis_result: Any
    score: float                  # 0-100 score do componente
    weight: float                 # Peso na decisão final
    confidence: float             # Confiança na análise
    execution_time_ms: float      # Tempo de execução
    error: Optional[str] = None


@dataclass
class FilterResult:
    """Resultado de um filtro"""
    filter_name: str
    passed: bool
    score: float                  # 0-100 score do filtro
    adjustments: Dict[str, float] # Ajustes recomendados
    reasons: List[str]            # Razões para pass/fail
    risk_level: str               # Nível de risco detectado


@dataclass
class MasterSignal:
    """Sinal mestre do sistema"""
    signal_id: str
    symbol: str
    signal_type: str              # 'swing_long', 'breakout_short', etc.
    trade_type: TradeType
    
    # Scores e qualidade
    final_score: float            # 0-100 score final
    confidence: float             # 0-100 confiança
    quality: SignalQuality
    status: SignalStatus
    
    # Trading parameters
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float]
    position_size_pct: float      # % do portfólio
    risk_reward_ratio: float
    
    # Timing
    generated_at: datetime
    valid_until: datetime
    best_entry_time: Optional[datetime]
    
    # Component analysis
    market_structure: ComponentAnalysis
    trend_analysis: ComponentAnalysis
    leading_indicators: ComponentAnalysis
    strategy_signals: ComponentAnalysis
    confluence_analysis: ComponentAnalysis
    
    # Filter results
    market_condition_filter: FilterResult
    volatility_filter: FilterResult
    time_filter: FilterResult
    fundamental_filter: FilterResult
    
    # Risk management
    risk_analysis: Dict
    position_sizing: Dict
    
    # Metadata
    timeframes_analyzed: List[str]
    total_analysis_time_ms: float
    confluence_factors_count: int
    reasoning: str                # Razão principal do sinal
    warnings: List[str]           # Avisos importantes
    
    def to_dict(self) -> Dict:
        """Converte para dicionário para persistência"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'trade_type': self.trade_type.value,
            'final_score': self.final_score,
            'confidence': self.confidence,
            'quality': self.quality.value,
            'status': self.status.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'position_size_pct': self.position_size_pct,
            'risk_reward_ratio': self.risk_reward_ratio,
            'generated_at': self.generated_at.isoformat(),
            'valid_until': self.valid_until.isoformat(),
            'reasoning': self.reasoning,
            'warnings': self.warnings
        }


class SignalGenerator:
    """
    🎯 Signal Generator Principal
    
    Orquestra todo o sistema de trading:
    1. Coleta dados de múltiplos timeframes
    2. Executa análises de todos os componentes
    3. Aplica todos os filtros
    4. Calcula confluência final
    5. Gera sinais de alta qualidade
    6. Aplica risk management
    """
    
    def __init__(self,
                 market_data_provider: MarketDataProvider,
                 risk_manager: RiskManager,
                 min_signal_score: float = 70.0,
                 min_confluence_factors: int = 4,
                 enabled_timeframes: List[str] = None):
        
        self.market_data_provider = market_data_provider
        self.risk_manager = risk_manager
        self.min_signal_score = min_signal_score
        self.min_confluence_factors = min_confluence_factors
        self.enabled_timeframes = enabled_timeframes or ["1H", "4H", "1D"]
        
        self.logger = logging.getLogger(f"{__name__}.SignalGenerator")
        
        # Inicializar todos os componentes
        self._initialize_components()
        
        # Configurações de pesos dos componentes
        self.component_weights = {
            'market_structure': 0.20,
            'trend_analysis': 0.20,
            'leading_indicators': 0.15,
            'strategy_signals': 0.25,
            'confluence_analysis': 0.20
        }
        
        # Configurações de filtros
        self.filter_weights = {
            'market_condition': 0.30,
            'volatility': 0.25,
            'time': 0.25,
            'fundamental': 0.20
        }
        
        # Cache de sinais gerados
        self.signals_cache: Dict[str, MasterSignal] = {}
        self.generation_stats = {
            'total_generated': 0,
            'total_filtered_out': 0,
            'avg_generation_time': 0,
            'success_rate': 0.0
        }
    
    def _initialize_components(self):
        """Inicializa todos os componentes do sistema"""
        try:
            # Core analyzers
            self.market_structure_analyzer = MarketStructureAnalyzer()
            self.trend_analyzer = TrendAnalyzer()
            self.leading_indicators_system = LeadingIndicatorsSystem()
            self.confluence_analyzer = ConfluenceAnalyzer()
            
            # Strategies
            self.swing_strategy = SwingStrategy()
            self.breakout_strategy = BreakoutStrategy()
            
            # Filters
            self.market_condition_filter = MarketConditionFilter()
            self.volatility_filter = VolatilityFilter()
            self.time_filter = TimeFilter()
            self.fundamental_filter = FundamentalFilter()
            
            self.logger.info("Todos os componentes inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro na inicialização de componentes: {e}")
            raise
    
    def generate_signals(self, 
                        symbols: List[str],
                        current_time: Optional[datetime] = None) -> List[MasterSignal]:
        """
        Gera sinais para múltiplos símbolos
        
        Args:
            symbols: Lista de símbolos para análise
            current_time: Timestamp atual
            
        Returns:
            Lista de MasterSignals de alta qualidade
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        all_signals = []
        
        for symbol in symbols:
            try:
                self.logger.info(f"Gerando sinais para {symbol}")
                
                # Gerar sinais para o símbolo
                symbol_signals = self.generate_symbol_signals(symbol, current_time)
                all_signals.extend(symbol_signals)
                
            except Exception as e:
                self.logger.error(f"Erro na geração de sinais para {symbol}: {e}")
                continue
        
        # Ranquear e filtrar sinais finais
        final_signals = self._rank_and_filter_signals(all_signals)
        
        # Atualizar estatísticas
        self._update_generation_stats(len(symbols), len(final_signals))
        
        self.logger.info(f"Gerados {len(final_signals)} sinais finais de {len(symbols)} símbolos")
        return final_signals
    
    def generate_symbol_signals(self, 
                               symbol: str,
                               current_time: datetime) -> List[MasterSignal]:
        """
        Gera sinais para um símbolo específico
        
        Args:
            symbol: Símbolo para análise
            current_time: Timestamp atual
            
        Returns:
            Lista de MasterSignals para o símbolo
        """
        start_time = pd.Timestamp.now()
        signals = []
        
        try:
            # 1. Coletar dados de mercado
            market_data = self._collect_market_data(symbol)
            if not self._validate_market_data(market_data):
                self.logger.warning(f"Dados insuficientes para {symbol}")
                return signals
            
            # 2. Executar análises de componentes
            component_analyses = self._execute_component_analyses(symbol, market_data, current_time)
            
            # 3. Executar filtros
            filter_results = self._execute_filters(symbol, market_data, current_time)
            
            # 4. Verificar se passou nos filtros críticos
            if not self._check_critical_filters(filter_results):
                self.logger.info(f"Símbolo {symbol} não passou nos filtros críticos")
                return signals
            
            # 5. Identificar oportunidades de trading
            trading_opportunities = self._identify_trading_opportunities(
                component_analyses, filter_results)
            
            # 6. Gerar sinais para cada oportunidade
            for opportunity in trading_opportunities:
                signal = self._create_master_signal(
                    symbol, opportunity, component_analyses, 
                    filter_results, current_time)
                
                if signal and signal.final_score >= self.min_signal_score:
                    signals.append(signal)
                    
                    # Cache do sinal
                    self.signals_cache[signal.signal_id] = signal
            
            analysis_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
            self.logger.info(f"Análise de {symbol} concluída em {analysis_time:.1f}ms - "
                           f"{len(signals)} sinais gerados")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na geração de sinais para {symbol}: {e}")
            return signals
    
    def _collect_market_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Coleta dados de mercado para todos os timeframes"""
        try:
            market_data = {}
            
            for timeframe in self.enabled_timeframes:
                df = self.market_data_provider.get_ohlcv(symbol, timeframe, limit=200)
                if df is not None and len(df) >= 50:
                    market_data[timeframe] = df
                else:
                    self.logger.warning(f"Dados insuficientes para {symbol} {timeframe}")
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Erro na coleta de dados para {symbol}: {e}")
            return {}
    
    def _validate_market_data(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """Valida qualidade dos dados de mercado"""
        try:
            # Verificar se temos dados para pelo menos 2 timeframes
            if len(market_data) < 2:
                return False
            
            # Verificar se temos dados suficientes
            for timeframe, df in market_data.items():
                if len(df) < 50:  # Mínimo 50 períodos
                    return False
                
                # Verificar se não há gaps grandes
                if df.isnull().sum().sum() > len(df) * 0.1:  # Mais de 10% missing
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _execute_component_analyses(self, 
                                  symbol: str,
                                  market_data: Dict[str, pd.DataFrame],
                                  current_time: datetime) -> Dict[str, ComponentAnalysis]:
        """Executa análises de todos os componentes"""
        analyses = {}
        
        # 1. Market Structure Analysis
        analyses['market_structure'] = self._analyze_component(
            'market_structure',
            lambda: self.market_structure_analyzer.analyze_market_structure(market_data.get("4H")),
            weight=self.component_weights['market_structure']
        )
        
        # 2. Trend Analysis (Multi-timeframe)
        analyses['trend_analysis'] = self._analyze_component(
            'trend_analysis',
            lambda: self._multi_timeframe_trend_analysis(market_data),
            weight=self.component_weights['trend_analysis']
        )
        
        # 3. Leading Indicators
        analyses['leading_indicators'] = self._analyze_component(
            'leading_indicators',
            lambda: self.leading_indicators_system.analyze_all_leading(
                market_data.get("4H"), market_data.get("4H")['close'].iloc[-1], "4H"),
            weight=self.component_weights['leading_indicators']
        )
        
        # 4. Strategy Signals
        analyses['strategy_signals'] = self._analyze_component(
            'strategy_signals',
            lambda: self._analyze_strategy_signals(symbol, market_data),
            weight=self.component_weights['strategy_signals']
        )
        
        # 5. Confluence Analysis
        analyses['confluence_analysis'] = self._analyze_component(
            'confluence_analysis',
            lambda: self.confluence_analyzer.analyze_confluence(market_data, symbol),
            weight=self.component_weights['confluence_analysis']
        )
        
        return analyses
    
    def _analyze_component(self, component_name: str, analysis_func, weight: float) -> ComponentAnalysis:
        """Executa análise de um componente com error handling"""
        start_time = pd.Timestamp.now()
        
        try:
            result = analysis_func()
            
            # Calcular score baseado no resultado
            score = self._calculate_component_score(component_name, result)
            confidence = self._calculate_component_confidence(component_name, result)
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
            
            return ComponentAnalysis(
                component_name=component_name,
                enabled=True,
                analysis_result=result,
                score=score,
                weight=weight,
                confidence=confidence,
                execution_time_ms=execution_time,
                error=None
            )
            
        except Exception as e:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds() * 1000
            self.logger.error(f"Erro na análise de {component_name}: {e}")
            
            return ComponentAnalysis(
                component_name=component_name,
                enabled=False,
                analysis_result=None,
                score=0,
                weight=weight,
                confidence=0,
                execution_time_ms=execution_time,
                error=str(e)
            )
    
    def _multi_timeframe_trend_analysis(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Análise de tendência multi-timeframe"""
        trend_results = {}
        
        for timeframe, df in market_data.items():
            if len(df) >= 50:
                trend_data = self.trend_analyzer.analyze_trend(df, timeframe)
                trend_results[timeframe] = trend_data
        
        # Calcular alinhamento de tendências
        if len(trend_results) >= 2:
            directions = [td.direction for td in trend_results.values()]
            bullish_count = directions.count('bullish')
            bearish_count = directions.count('bearish')
            total = len(directions)
            
            if bullish_count / total >= 0.7:
                alignment = 'strong_bullish'
                alignment_score = bullish_count / total * 100
            elif bearish_count / total >= 0.7:
                alignment = 'strong_bearish'
                alignment_score = bearish_count / total * 100
            elif bullish_count > bearish_count:
                alignment = 'weak_bullish'
                alignment_score = 60
            elif bearish_count > bullish_count:
                alignment = 'weak_bearish'
                alignment_score = 60
            else:
                alignment = 'neutral'
                alignment_score = 40
        else:
            alignment = 'insufficient_data'
            alignment_score = 30
        
        return {
            'timeframe_trends': trend_results,
            'alignment': alignment,
            'alignment_score': alignment_score,
            'strongest_trend': max(trend_results.values(), key=lambda x: x.strength) if trend_results else None
        }
    
    def _analyze_strategy_signals(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analisa sinais de todas as estratégias"""
        strategy_results = {
            'swing_signals': [],
            'breakout_signals': [],
            'total_signals': 0,
            'best_signal': None,
            'strategy_scores': {}
        }
        
        try:
            # Swing Strategy
            if all(tf in market_data for tf in ["1H", "4H", "1D"]):
                swing_signals = self.swing_strategy.analyze_swing_opportunity(
                    market_data["4H"], market_data["1D"], market_data["1H"], symbol)
                strategy_results['swing_signals'] = swing_signals
                
                if swing_signals:
                    avg_score = np.mean([s.signal_strength for s in swing_signals])
                    strategy_results['strategy_scores']['swing'] = avg_score
            
            # Breakout Strategy
            if "4H" in market_data:
                breakout_signals = self.breakout_strategy.analyze_breakout_opportunity(
                    market_data["4H"], symbol)
                strategy_results['breakout_signals'] = breakout_signals
                
                if breakout_signals:
                    avg_score = np.mean([s.signal_strength for s in breakout_signals])
                    strategy_results['strategy_scores']['breakout'] = avg_score
            
            # Combinar todos os sinais
            all_signals = strategy_results['swing_signals'] + strategy_results['breakout_signals']
            strategy_results['total_signals'] = len(all_signals)
            
            if all_signals:
                # Encontrar melhor sinal
                best_signal = max(all_signals, key=lambda x: x.signal_strength)
                strategy_results['best_signal'] = best_signal
            
            return strategy_results
            
        except Exception as e:
            self.logger.error(f"Erro na análise de estratégias: {e}")
            return strategy_results
    
    def _execute_filters(self, 
                        symbol: str,
                        market_data: Dict[str, pd.DataFrame],
                        current_time: datetime) -> Dict[str, FilterResult]:
        """Executa todos os filtros"""
        filter_results = {}
        
        # 1. Market Condition Filter
        filter_results['market_condition'] = self._execute_filter(
            'market_condition',
            lambda: self.market_condition_filter.analyze_market_condition(market_data, symbol),
            lambda analysis: analysis.confidence > 50,  # Criteria
            self.filter_weights['market_condition']
        )
        
        # 2. Volatility Filter
        filter_results['volatility'] = self._execute_filter(
            'volatility',
            lambda: self.volatility_filter.analyze_volatility(market_data.get("4H"), symbol),
            lambda signal: signal.entry_timing != 'avoid',
            self.filter_weights['volatility']
        )
        
        # 3. Time Filter
        filter_results['time'] = self._execute_filter(
            'time',
            lambda: self.time_filter.analyze_timing(market_data.get("4H"), current_time, "swing"),
            lambda signal: signal.entry_timing != 'avoid',
            self.filter_weights['time']
        )
        
        # 4. Fundamental Filter
        filter_results['fundamental'] = self._execute_filter(
            'fundamental',
            lambda: self.fundamental_filter.analyze_fundamentals(symbol, current_time),
            lambda signal: signal.recommended_action != 'avoid',
            self.filter_weights['fundamental']
        )
        
        return filter_results
    
    def _execute_filter(self, filter_name: str, analysis_func, pass_criteria, weight: float) -> FilterResult:
        """Executa um filtro individual"""
        try:
            analysis_result = analysis_func()
            passed = pass_criteria(analysis_result)
            
            # Calcular score do filtro
            score = self._calculate_filter_score(filter_name, analysis_result)
            
            # Extrair ajustes recomendados
            adjustments = self._extract_filter_adjustments(filter_name, analysis_result)
            
            # Gerar razões
            reasons = self._generate_filter_reasons(filter_name, analysis_result, passed)
            
            # Determinar nível de risco
            risk_level = self._determine_filter_risk_level(filter_name, analysis_result)
            
            return FilterResult(
                filter_name=filter_name,
                passed=passed,
                score=score,
                adjustments=adjustments,
                reasons=reasons,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Erro no filtro {filter_name}: {e}")
            return FilterResult(
                filter_name=filter_name,
                passed=False,
                score=0,
                adjustments={},
                reasons=[f"Erro no filtro: {str(e)}"],
                risk_level='high'
            )
    
    def _calculate_component_score(self, component_name: str, result: Any) -> float:
        """Calcula score de um componente"""
        try:
            if component_name == 'market_structure':
                if result and 'signals' in result:
                    signals = result['signals']
                    if signals:
                        return np.mean([s.strength for s in signals])
                return 30
                
            elif component_name == 'trend_analysis':
                if result and 'alignment_score' in result:
                    return result['alignment_score']
                return 40
                
            elif component_name == 'leading_indicators':
                if result:
                    leading_score = self.leading_indicators_system.get_leading_score(result)
                    return leading_score.get('overall_score', 30)
                return 30
                
            elif component_name == 'strategy_signals':
                if result and result['best_signal']:
                    return result['best_signal'].signal_strength
                return 30
                
            elif component_name == 'confluence_analysis':
                if result:
                    scores = [s.confluence_score for s in result]
                    return np.mean(scores) if scores else 30
                return 30
                
            else:
                return 50  # Default
                
        except Exception:
            return 30
    
    def _calculate_component_confidence(self, component_name: str, result: Any) -> float:
        """Calcula confiança de um componente"""
        try:
            if component_name == 'market_structure':
                return 75 if result and 'signals' in result and result['signals'] else 40
            elif component_name == 'trend_analysis':
                if result and 'strongest_trend' in result and result['strongest_trend']:
                    return result['strongest_trend'].confidence
                return 50
            elif component_name == 'leading_indicators':
                return 70 if result else 30
            elif component_name == 'strategy_signals':
                if result and result['best_signal']:
                    return result['best_signal'].confidence
                return 40
            elif component_name == 'confluence_analysis':
                if result:
                    confidences = [s.confidence for s in result]
                    return np.mean(confidences) if confidences else 50
                return 50
            else:
                return 60
        except Exception:
            return 50
    
    def _calculate_filter_score(self, filter_name: str, result: Any) -> float:
        """Calcula score de um filtro"""
        try:
            if filter_name == 'market_condition':
                return result.confidence if hasattr(result, 'confidence') else 50
            elif filter_name == 'volatility':
                return result.confidence if hasattr(result, 'confidence') else 50
            elif filter_name == 'time':
                return result.overall_score if hasattr(result, 'overall_score') else 50
            elif filter_name == 'fundamental':
                return result.confidence if hasattr(result, 'confidence') else 50
            else:
                return 50
        except Exception:
            return 30
    
    def _extract_filter_adjustments(self, filter_name: str, result: Any) -> Dict[str, float]:
        """Extrai ajustes recomendados de um filtro"""
        try:
            adjustments = {}
            
            if filter_name == 'market_condition':
                if hasattr(result, 'position_sizing_modifier'):
                    adjustments['position_size_multiplier'] = result.position_sizing_modifier
                    
            elif filter_name == 'volatility':
                if hasattr(result, 'position_size_adjustment'):
                    adjustments['position_size_multiplier'] = result.position_size_adjustment
                if hasattr(result, 'stop_loss_adjustment'):
                    adjustments['stop_loss_multiplier'] = result.stop_loss_adjustment
                    
            elif filter_name == 'time':
                if hasattr(result, 'position_size_adjustment'):
                    adjustments['position_size_multiplier'] = result.position_size_adjustment
                    
            elif filter_name == 'fundamental':
                if hasattr(result, 'position_size_adjustment'):
                    adjustments['position_size_multiplier'] = result.position_size_adjustment
                if hasattr(result, 'risk_adjustment'):
                    adjustments['risk_multiplier'] = result.risk_adjustment
            
            return adjustments
            
        except Exception:
            return {}
    
    def _generate_filter_reasons(self, filter_name: str, result: Any, passed: bool) -> List[str]:
        """Gera razões para o resultado do filtro"""
        reasons = []
        
        try:
            if passed:
                reasons.append(f"{filter_name} filter passed")
                
                if filter_name == 'market_condition' and hasattr(result, 'condition'):
                    reasons.append(f"Market condition: {result.condition.value}")
                elif filter_name == 'volatility' and hasattr(result, 'regime'):
                    reasons.append(f"Volatility regime: {result.regime.value}")
                elif filter_name == 'time' and hasattr(result, 'current_session'):
                    reasons.append(f"Trading session: {result.current_session.value}")
                elif filter_name == 'fundamental' and hasattr(result, 'direction'):
                    reasons.append(f"Fundamental direction: {result.direction}")
            else:
                reasons.append(f"{filter_name} filter failed")
                
                # Adicionar razões específicas do porque falhou
                if filter_name == 'volatility' and hasattr(result, 'entry_timing'):
                    if result.entry_timing == 'avoid':
                        reasons.append("High volatility - avoid trading")
                elif filter_name == 'time' and hasattr(result, 'entry_timing'):
                    if result.entry_timing == 'avoid':
                        reasons.append("Poor timing conditions")
                elif filter_name == 'fundamental' and hasattr(result, 'recommended_action'):
                    if result.recommended_action == 'avoid':
                        reasons.append("Negative fundamental outlook")
        
        except Exception:
            reasons.append(f"Error analyzing {filter_name} filter")
        
        return reasons
    
    def _determine_filter_risk_level(self, filter_name: str, result: Any) -> str:
        """Determina nível de risco do filtro"""
        try:
            if filter_name == 'volatility':
                if hasattr(result, 'regime'):
                    if result.regime.value in ['extremely_high', 'high']:
                        return 'high'
                    elif result.regime.value in ['low', 'extremely_low']:
                        return 'low'
                    else:
                        return 'medium'
                        
            elif filter_name == 'fundamental':
                if hasattr(result, 'black_swan_potential') and result.black_swan_potential:
                    return 'extreme'
                elif hasattr(result, 'high_volatility_warning') and result.high_volatility_warning:
                    return 'high'
                else:
                    return 'medium'
                    
            elif filter_name == 'time':
                if hasattr(result, 'low_liquidity_warning') and result.low_liquidity_warning:
                    return 'high'
                else:
                    return 'medium'
                    
            else:
                return 'medium'
                
        except Exception:
            return 'high'
    
    def _check_critical_filters(self, filter_results: Dict[str, FilterResult]) -> bool:
        """Verifica se passou nos filtros críticos"""
        try:
            # Fundamental filter é crítico
            if 'fundamental' in filter_results:
                if filter_results['fundamental'].risk_level == 'extreme':
                    return False
            
            # Volatility filter é crítico
            if 'volatility' in filter_results:
                if not filter_results['volatility'].passed:
                    return False
            
            # Pelo menos 2 filtros devem passar
            passed_filters = sum(1 for fr in filter_results.values() if fr.passed)
            if passed_filters < 2:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _identify_trading_opportunities(self, 
                                      component_analyses: Dict[str, ComponentAnalysis],
                                      filter_results: Dict[str, FilterResult]) -> List[Dict]:
        """Identifica oportunidades de trading baseadas nas análises"""
        opportunities = []
        
        try:
            # Verificar se há sinais de estratégias
            if 'strategy_signals' in component_analyses:
                strategy_analysis = component_analyses['strategy_signals']
                
                if strategy_analysis.enabled and strategy_analysis.analysis_result:
                    result = strategy_analysis.analysis_result
                    
                    # Swing opportunities
                    for swing_signal in result.get('swing_signals', []):
                        if swing_signal.signal_strength > 60:
                            opportunity = {
                                'type': 'swing',
                                'signal': swing_signal,
                                'score': swing_signal.signal_strength,
                                'confidence': swing_signal.confidence
                            }
                            opportunities.append(opportunity)
                    
                    # Breakout opportunities
                    for breakout_signal in result.get('breakout_signals', []):
                        if breakout_signal.signal_strength > 60:
                            opportunity = {
                                'type': 'breakout',
                                'signal': breakout_signal,
                                'score': breakout_signal.signal_strength,
                                'confidence': breakout_signal.confidence
                            }
                            opportunities.append(opportunity)
            
            # Filtrar oportunidades baseadas nos filtros
            filtered_opportunities = []
            for opp in opportunities:
                if self._validate_opportunity_with_filters(opp, filter_results):
                    filtered_opportunities.append(opp)
            
            return filtered_opportunities
            
        except Exception as e:
            self.logger.error(f"Erro na identificação de oportunidades: {e}")
            return opportunities
    
    def _validate_opportunity_with_filters(self, opportunity: Dict, filter_results: Dict[str, FilterResult]) -> bool:
        """Valida oportunidade contra os filtros"""
        try:
            # Check volatility filter
            if 'volatility' in filter_results:
                vol_filter = filter_results['volatility']
                if vol_filter.risk_level == 'extreme':
                    return False
            
            # Check time filter
            if 'time' in filter_results:
                time_filter = filter_results['time']
                if not time_filter.passed and time_filter.risk_level == 'high':
                    return False
            
            return True
            
        except Exception:
            return True  # Conservative - allow if can't validate
    
    def _create_master_signal(self, 
                            symbol: str,
                            opportunity: Dict,
                            component_analyses: Dict[str, ComponentAnalysis],
                            filter_results: Dict[str, FilterResult],
                            current_time: datetime) -> Optional[MasterSignal]:
        """Cria um Master Signal"""
        try:
            signal_id = f"{symbol}_{opportunity['type']}_{current_time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Extrair informações do sinal da estratégia
            strategy_signal = opportunity['signal']
            
            # Determinar tipo de trade
            if hasattr(strategy_signal, 'setup'):
                setup = strategy_signal.setup
                trade_type = TradeType.LONG if setup.direction.value == 'bullish' else TradeType.SHORT
                entry_price = setup.entry_price
                stop_loss = setup.stop_loss
                take_profit_1 = setup.take_profit_1
                take_profit_2 = setup.take_profit_2
                risk_reward_ratio = setup.risk_reward_ratio
            elif hasattr(strategy_signal, 'direction'):
                # Breakout signal
                trade_type = TradeType.LONG if strategy_signal.setup.direction == 'bullish' else TradeType.SHORT
                entry_price = strategy_signal.setup.entry_price
                stop_loss = strategy_signal.setup.stop_loss
                take_profit_1 = strategy_signal.setup.take_profit_1
                take_profit_2 = strategy_signal.setup.take_profit_2
                risk_reward_ratio = strategy_signal.setup.risk_reward_ratio
            else:
                self.logger.warning(f"Formato de sinal não reconhecido para {symbol}")
                return None
            
            # Calcular score final
            final_score = self._calculate_final_score(opportunity, component_analyses, filter_results)
            
            # Calcular confiança final
            confidence = self._calculate_final_confidence(component_analyses, filter_results)
            
            # Determinar qualidade
            quality = self._determine_signal_quality(final_score)
            
            # Aplicar risk management
            risk_analysis, position_sizing = self._apply_risk_management(
                symbol, entry_price, stop_loss, strategy_signal, filter_results)
            
            # Gerar reasoning
            reasoning = self._generate_signal_reasoning(opportunity, component_analyses, filter_results)
            
            # Gerar warnings
            warnings = self._generate_signal_warnings(filter_results, risk_analysis)
            
            # Calcular timing
            valid_until = current_time + pd.Timedelta(hours=48)  # 48h validity
            best_entry_time = self._calculate_best_entry_time(filter_results, current_time)
            
            # Calcular tempo total de análise
            total_analysis_time = sum(ca.execution_time_ms for ca in component_analyses.values())
            
            master_signal = MasterSignal(
                signal_id=signal_id,
                symbol=symbol,
                signal_type=f"{opportunity['type']}_{trade_type.value}",
                trade_type=trade_type,
                
                final_score=final_score,
                confidence=confidence,
                quality=quality,
                status=SignalStatus.ACTIVE,
                
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                position_size_pct=position_sizing.get('position_size_pct', 0.02),
                risk_reward_ratio=risk_reward_ratio,
                
                generated_at=current_time,
                valid_until=valid_until,
                best_entry_time=best_entry_time,
                
                market_structure=component_analyses['market_structure'],
                trend_analysis=component_analyses['trend_analysis'],
                leading_indicators=component_analyses['leading_indicators'],
                strategy_signals=component_analyses['strategy_signals'],
                confluence_analysis=component_analyses['confluence_analysis'],
                
                market_condition_filter=filter_results['market_condition'],
                volatility_filter=filter_results['volatility'],
                time_filter=filter_results['time'],
                fundamental_filter=filter_results['fundamental'],
                
                risk_analysis=risk_analysis,
                position_sizing=position_sizing,
                
                timeframes_analyzed=self.enabled_timeframes,
                total_analysis_time_ms=total_analysis_time,
                confluence_factors_count=self._count_confluence_factors(component_analyses),
                reasoning=reasoning,
                warnings=warnings
            )
            
            return master_signal
            
        except Exception as e:
            self.logger.error(f"Erro na criação do master signal: {e}")
            return None
    
    def _calculate_final_score(self, 
                             opportunity: Dict,
                             component_analyses: Dict[str, ComponentAnalysis],
                             filter_results: Dict[str, FilterResult]) -> float:
        """Calcula score final do sinal"""
        try:
            # Base score da oportunidade
            base_score = opportunity['score']
            
            # Weighted score dos componentes
            component_score = 0
            total_weight = 0
            
            for name, analysis in component_analyses.items():
                if analysis.enabled:
                    component_score += analysis.score * analysis.weight
                    total_weight += analysis.weight
            
            if total_weight > 0:
                component_score /= total_weight
            else:
                component_score = 50
            
            # Filter score
            filter_score = 0
            filter_weight = 0
            
            for name, filter_result in filter_results.items():
                weight = self.filter_weights.get(name, 0.25)
                filter_score += filter_result.score * weight
                filter_weight += weight
            
            if filter_weight > 0:
                filter_score /= filter_weight
            else:
                filter_score = 50
            
            # Combinar scores
            final_score = (base_score * 0.4 + component_score * 0.4 + filter_score * 0.2)
            
            return min(100, max(0, final_score))
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo do score final: {e}")
            return 50
    
    def _calculate_final_confidence(self, 
                                  component_analyses: Dict[str, ComponentAnalysis],
                                  filter_results: Dict[str, FilterResult]) -> float:
        """Calcula confiança final"""
        try:
            confidences = []
            
            # Component confidences
            for analysis in component_analyses.values():
                if analysis.enabled:
                    confidences.append(analysis.confidence)
            
            # Filter confidences (simplified)
            for filter_result in filter_results.values():
                if filter_result.passed:
                    confidences.append(filter_result.score * 0.8)  # Reduce confidence if filter not optimal
                else:
                    confidences.append(30)  # Low confidence if filter failed
            
            return np.mean(confidences) if confidences else 50
            
        except Exception:
            return 50
    
    def _determine_signal_quality(self, final_score: float) -> SignalQuality:
        """Determina qualidade do sinal baseada no score"""
        if final_score >= 90:
            return SignalQuality.EXCEPTIONAL
        elif final_score >= 80:
            return SignalQuality.EXCELLENT
        elif final_score >= 70:
            return SignalQuality.GOOD
        elif final_score >= 60:
            return SignalQuality.MODERATE
        elif final_score >= 50:
            return SignalQuality.WEAK
        else:
            return SignalQuality.POOR
    
    def _apply_risk_management(self, 
                             symbol: str,
                             entry_price: float,
                             stop_loss: float,
                             strategy_signal: Any,
                             filter_results: Dict[str, FilterResult]) -> Tuple[Dict, Dict]:
        """Aplica risk management ao sinal"""
        try:
            # Calcular volatilidade de mercado
            market_volatility = 0.025  # Default 2.5%
            
            # Extrair ajustes dos filtros
            position_adjustments = []
            for filter_result in filter_results.values():
                if 'position_size_multiplier' in filter_result.adjustments:
                    position_adjustments.append(filter_result.adjustments['position_size_multiplier'])
            
            # Calcular adjustment médio
            avg_adjustment = np.mean(position_adjustments) if position_adjustments else 1.0
            
            # Criar input para position sizing
            sizing_input = PositionSizingInput(
                signal_strength=strategy_signal.signal_strength,
                confidence=strategy_signal.confidence,
                risk_reward_ratio=abs(entry_price - strategy_signal.setup.take_profit_1) / abs(entry_price - stop_loss),
                entry_price=entry_price,
                stop_loss=stop_loss,
                market_volatility=market_volatility,
                portfolio_balance=self.risk_manager.current_balance,
                existing_exposure=sum(pos.position_size_pct for pos in self.risk_manager.active_positions.values()),
                correlation_factor=0.3  # Default
            )
            
            # Calcular position size
            position_sizing = self.risk_manager.calculate_position_size(sizing_input, symbol)
            
            # Aplicar ajustes dos filtros
            if position_sizing.get('approved', False):
                position_sizing['position_size_pct'] *= avg_adjustment
                position_sizing['position_size_usd'] *= avg_adjustment
            
            # Risk analysis
            risk_analysis = {
                'base_risk_pct': abs(entry_price - stop_loss) / entry_price,
                'portfolio_risk_pct': position_sizing.get('risk_pct', 0.02),
                'risk_level': 'low' if position_sizing.get('risk_pct', 0.02) < 0.02 else 'medium',
                'filter_adjustments': position_adjustments,
                'volatility_risk': market_volatility,
                'correlation_risk': 0.3
            }
            
            return risk_analysis, position_sizing
            
        except Exception as e:
            self.logger.error(f"Erro no risk management: {e}")
            return {}, {'approved': False, 'position_size_pct': 0.01}
    
    def _generate_signal_reasoning(self, 
                                 opportunity: Dict,
                                 component_analyses: Dict[str, ComponentAnalysis],
                                 filter_results: Dict[str, FilterResult]) -> str:
        """Gera reasoning do sinal"""
        try:
            reasoning_parts = []
            
            # Strategy reasoning
            reasoning_parts.append(f"{opportunity['type'].title()} setup detected")
            
            # Component insights
            strong_components = [name for name, analysis in component_analyses.items() 
                               if analysis.enabled and analysis.score > 70]
            if strong_components:
                reasoning_parts.append(f"Strong {', '.join(strong_components)} signals")
            
            # Filter insights
            passed_filters = [name for name, filter_result in filter_results.items() 
                            if filter_result.passed]
            if passed_filters:
                reasoning_parts.append(f"Passed {len(passed_filters)}/4 filters")
            
            return ". ".join(reasoning_parts)
            
        except Exception:
            return f"{opportunity.get('type', 'Unknown')} trading opportunity"
    
    def _generate_signal_warnings(self, 
                                filter_results: Dict[str, FilterResult],
                                risk_analysis: Dict) -> List[str]:
        """Gera warnings do sinal"""
        warnings = []
        
        try:
            # Filter warnings
            for name, filter_result in filter_results.items():
                if filter_result.risk_level in ['high', 'extreme']:
                    warnings.append(f"High {name} risk detected")
                
                if not filter_result.passed:
                    warnings.append(f"{name} filter failed")
            
            # Risk warnings
            if risk_analysis.get('portfolio_risk_pct', 0) > 0.03:
                warnings.append("High portfolio risk")
            
            if risk_analysis.get('volatility_risk', 0) > 0.04:
                warnings.append("High market volatility")
            
        except Exception:
            warnings.append("Error generating warnings")
        
        return warnings
    
    def _calculate_best_entry_time(self, 
                                 filter_results: Dict[str, FilterResult],
                                 current_time: datetime) -> Optional[datetime]:
        """Calcula melhor horário de entrada"""
        try:
            # Check time filter
            if 'time' in filter_results:
                time_filter = filter_results['time']
                if hasattr(time_filter, 'next_optimal_time'):
                    return time_filter.next_optimal_time
            
            # Default: próxima hora
            return current_time + pd.Timedelta(hours=1)
            
        except Exception:
            return None
    
    def _count_confluence_factors(self, component_analyses: Dict[str, ComponentAnalysis]) -> int:
        """Conta fatores de confluência"""
        try:
            count = 0
            
            for analysis in component_analyses.values():
                if analysis.enabled and analysis.score > 60:
                    count += 1
            
            return count
            
        except Exception:
            return 0
    
    def _rank_and_filter_signals(self, signals: List[MasterSignal]) -> List[MasterSignal]:
        """Ranqueia e filtra sinais finais"""
        try:
            # Filtrar por score mínimo
            qualified_signals = [s for s in signals if s.final_score >= self.min_signal_score]
            
            # Ranquear por score e qualidade
            ranked_signals = sorted(qualified_signals, 
                                  key=lambda x: (x.final_score, x.confidence), 
                                  reverse=True)
            
            # Limitar a top 10 sinais
            final_signals = ranked_signals[:10]
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Erro no ranking de sinais: {e}")
            return signals
    
    def _update_generation_stats(self, symbols_analyzed: int, signals_generated: int):
        """Atualiza estatísticas de geração"""
        try:
            self.generation_stats['total_generated'] += signals_generated
            success_rate = signals_generated / symbols_analyzed if symbols_analyzed > 0 else 0
            
            # Update running average
            prev_rate = self.generation_stats['success_rate']
            self.generation_stats['success_rate'] = (prev_rate + success_rate) / 2
            
        except Exception:
            pass
    
    def get_signal_by_id(self, signal_id: str) -> Optional[MasterSignal]:
        """Retorna sinal por ID"""
        return self.signals_cache.get(signal_id)
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[MasterSignal]:
        """Retorna sinais ativos"""
        try:
            active_signals = [s for s in self.signals_cache.values() 
                            if s.status == SignalStatus.ACTIVE]
            
            if symbol:
                active_signals = [s for s in active_signals if s.symbol == symbol]
            
            return sorted(active_signals, key=lambda x: x.final_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar sinais ativos: {e}")
            return []
    
    def update_signal_status(self, signal_id: str, new_status: SignalStatus):
        """Atualiza status de um sinal"""
        try:
            if signal_id in self.signals_cache:
                self.signals_cache[signal_id].status = new_status
                self.logger.info(f"Signal {signal_id} status updated to {new_status.value}")
            else:
                self.logger.warning(f"Signal {signal_id} not found in cache")
        except Exception as e:
            self.logger.error(f"Erro ao atualizar status do sinal: {e}")


def main():
    """Teste básico do signal generator"""
    # Mock dos componentes necessários
    from .market_data import MarketDataProvider
    
    # Criar dados de exemplo
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
    
    # Mock market data provider
    class MockMarketDataProvider:
        def __init__(self):
            self.data = {
                "1H": create_sample_data(dates),
                "4H": create_sample_data(dates[::4]),
                "1D": create_sample_data(dates[::24])
            }
        
        def get_ohlcv(self, symbol, timeframe, limit=200):
            return self.data.get(timeframe)
    
    # Mock risk manager
    class MockRiskManager:
        def __init__(self):
            self.current_balance = 100000
            self.active_positions = {}
        
        def calculate_position_size(self, sizing_input, symbol):
            return {
                'approved': True,
                'position_size_pct': 0.03,
                'position_size_usd': 3000,
                'risk_pct': 0.015
            }
    
    # Testar signal generator
    market_data_provider = MockMarketDataProvider()
    risk_manager = MockRiskManager()
    
    signal_generator = SignalGenerator(
        market_data_provider=market_data_provider,
        risk_manager=risk_manager,
        min_signal_score=60.0
    )
    
    # Gerar sinais
    symbols = ["BTCUSDT", "ETHUSDT"]
    signals = signal_generator.generate_signals(symbols)
    
    print(f"\n🎯 SIGNAL GENERATOR TEST")
    print(f"Symbols Analyzed: {len(symbols)}")
    print(f"Signals Generated: {len(signals)}")
    
    for i, signal in enumerate(signals, 1):
        print(f"\n📊 SIGNAL {i}")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Type: {signal.signal_type}")
        print(f"   Score: {signal.final_score:.1f}")
        print(f"   Quality: {signal.quality.value}")
        print(f"   Confidence: {signal.confidence:.1f}%")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Stop: ${signal.stop_loss:,.2f}")
        print(f"   Target: ${signal.take_profit_1:,.2f}")
        print(f"   R:R: {signal.risk_reward_ratio:.2f}")
        print(f"   Position Size: {signal.position_size_pct:.2%}")
        print(f"   Reasoning: {signal.reasoning}")
        if signal.warnings:
            print(f"   Warnings: {', '.join(signal.warnings)}")
    
    print(f"\n📈 GENERATION STATS")
    print(f"Success Rate: {signal_generator.generation_stats['success_rate']:.1%}")
    print(f"Total Generated: {signal_generator.generation_stats['total_generated']}")


if __name__ == "__main__":
    main()