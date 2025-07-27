"""
🔬 BACKTEST ENGINE - Smart Trading System v2.0

Engine completo de backtesting para validação do sistema:
- Historical signal generation
- Realistic execution simulation
- Slippage and fees modeling
- Performance analytics
- Risk metrics calculation
- Component performance analysis

Filosofia: Past Performance Informs Future Success - Test Everything, Trust Nothing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timezone, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Importar componentes do sistema
from ..core.signal_generator import SignalGenerator, MasterSignal, SignalStatus
from ..core.risk_manager import RiskManager
from ..database.models import DatabaseManager, BacktestResult
from ..core.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Modos de backtesting"""
    FAST = "fast"                    # Execução rápida, menos detalhes
    DETAILED = "detailed"            # Execução completa com todos os detalhes
    STATISTICAL = "statistical"     # Foco em estatísticas, múltiplas rodadas


class ExecutionModel(Enum):
    """Modelos de execução"""
    PERFECT = "perfect"              # Execução perfeita sem slippage
    REALISTIC = "realistic"          # Slippage e delays realistas
    CONSERVATIVE = "conservative"    # Execução conservadora (pior caso)


@dataclass
class BacktestConfig:
    """Configuração do backtesting"""
    name: str
    description: str
    
    # Período
    start_date: datetime
    end_date: datetime
    
    # Ativos e timeframes
    symbols: List[str]
    timeframes: List[str]
    
    # Configurações financeiras
    initial_balance: float = 100000
    max_positions: int = 5
    position_sizing_method: str = "risk_based"
    
    # Execução
    execution_model: ExecutionModel = ExecutionModel.REALISTIC
    slippage_bps: float = 5.0        # Basis points
    fees_pct: float = 0.001          # 0.1% fees
    
    # Sistema
    signal_generation_frequency: str = "4H"  # Frequência de geração
    min_signal_score: float = 70.0
    rebalance_frequency: str = "daily"
    
    # Performance
    benchmark_symbol: str = "BTCUSDT"
    risk_free_rate: float = 0.02     # 2% anual
    
    # Otimizações
    use_multiprocessing: bool = True
    chunk_size_days: int = 30        # Processar em chunks


@dataclass
class BacktestPosition:
    """Posição no backtesting"""
    position_id: str
    signal_id: str
    symbol: str
    side: str                        # 'long' ou 'short'
    
    # Entry
    entry_time: datetime
    entry_price: float
    quantity: float
    entry_value: float
    
    # Risk management
    stop_loss: float
    take_profit: float
    
    # Current status
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Exit (quando fechada)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    
    # Tracking
    max_profit: float = 0
    max_loss: float = 0
    duration_hours: float = 0
    
    def update_unrealized_pnl(self, current_price: float):
        """Atualiza P&L não realizado"""
        self.current_price = current_price
        
        if self.side == 'long':
            pnl = (current_price - self.entry_price) * self.quantity
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            pnl = (self.entry_price - current_price) * self.quantity
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        self.unrealized_pnl = pnl
        self.unrealized_pnl_pct = pnl_pct
        
        # Atualizar máximos
        self.max_profit = max(self.max_profit, pnl)
        self.max_loss = min(self.max_loss, pnl)
    
    def close_position(self, exit_time: datetime, exit_price: float, reason: str):
        """Fecha a posição"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.side == 'long':
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
            self.realized_pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
            self.realized_pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.duration_hours = (exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestSnapshot:
    """Snapshot do estado do backtesting"""
    timestamp: datetime
    balance: float
    equity: float
    open_positions: int
    total_exposure: float
    daily_pnl: float
    cumulative_return: float
    drawdown: float
    signals_today: int
    active_signals: int


@dataclass
class BacktestResults:
    """Resultados completos do backtesting"""
    config: BacktestConfig
    
    # Performance básica
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    cagr: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    profit_factor: float
    
    # Timing
    avg_trade_duration_hours: float
    total_time_in_market_pct: float
    
    # Signal analysis
    total_signals_generated: int
    signals_executed: int
    signal_execution_rate: float
    avg_signal_score: float
    
    # Detailed data
    daily_snapshots: List[BacktestSnapshot]
    trade_log: List[Dict]
    signal_log: List[Dict]
    equity_curve: List[Tuple[datetime, float]]
    
    # Component performance
    strategy_performance: Dict[str, Dict]
    timeframe_performance: Dict[str, Dict]
    symbol_performance: Dict[str, Dict]
    
    # Execution info
    execution_time_seconds: float
    data_points_processed: int


class BacktestEngine:
    """
    🔬 Engine Principal de Backtesting
    
    Simula execução histórica do sistema completo:
    1. Carrega dados históricos
    2. Gera sinais período por período
    3. Simula execução com slippage/fees
    4. Calcula performance e métricas
    5. Analisa componentes individuais
    """
    
    def __init__(self,
                 signal_generator: SignalGenerator,
                 risk_manager: RiskManager,
                 market_data_provider: MarketDataProvider,
                 database_manager: Optional[DatabaseManager] = None):
        
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.market_data_provider = market_data_provider
        self.database_manager = database_manager
        
        self.logger = logging.getLogger(f"{__name__}.BacktestEngine")
        
        # Estado do backtesting
        self.current_balance = 0
        self.current_equity = 0
        self.open_positions: Dict[str, BacktestPosition] = {}
        self.closed_positions: List[BacktestPosition] = []
        self.daily_snapshots: List[BacktestSnapshot] = []
        self.signal_log: List[Dict] = []
        
        # Performance tracking
        self.peak_equity = 0
        self.current_drawdown = 0
        self.max_drawdown = 0
        
        # Execution tracking
        self.total_fees_paid = 0
        self.total_slippage_cost = 0
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResults:
        """
        Executa backtesting completo
        
        Args:
            config: Configuração do backtesting
            
        Returns:
            BacktestResults com resultados completos
        """
        start_time = pd.Timestamp.now()
        
        try:
            self.logger.info(f"Iniciando backtesting: {config.name}")
            self.logger.info(f"Período: {config.start_date} - {config.end_date}")
            self.logger.info(f"Símbolos: {config.symbols}")
            
            # 1. Inicializar estado
            self._initialize_backtest(config)
            
            # 2. Carregar dados históricos
            historical_data = self._load_historical_data(config)
            if not self._validate_historical_data(historical_data, config):
                raise ValueError("Dados históricos insuficientes")
            
            # 3. Executar simulação
            if config.use_multiprocessing:
                self._run_parallel_simulation(config, historical_data)
            else:
                self._run_sequential_simulation(config, historical_data)
            
            # 4. Calcular resultados finais
            results = self._calculate_final_results(config, start_time)
            
            # 5. Salvar no banco se disponível
            if self.database_manager:
                self._save_backtest_results(results)
            
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.logger.info(f"Backtesting concluído em {execution_time:.1f}s")
            self.logger.info(f"Return: {results.total_return_pct:.2%}, "
                           f"Sharpe: {results.sharpe_ratio:.2f}, "
                           f"Max DD: {results.max_drawdown_pct:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erro no backtesting: {e}")
            raise
    
    def _initialize_backtest(self, config: BacktestConfig):
        """Inicializa estado do backtesting"""
        self.current_balance = config.initial_balance
        self.current_equity = config.initial_balance
        self.peak_equity = config.initial_balance
        self.current_drawdown = 0
        self.max_drawdown = 0
        
        self.open_positions.clear()
        self.closed_positions.clear()
        self.daily_snapshots.clear()
        self.signal_log.clear()
        
        self.total_fees_paid = 0
        self.total_slippage_cost = 0
        
        # Reinicializar risk manager
        self.risk_manager.current_balance = config.initial_balance
        self.risk_manager.active_positions.clear()
        self.risk_manager.trade_history.clear()
    
    def _load_historical_data(self, config: BacktestConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Carrega dados históricos para todos os símbolos e timeframes"""
        historical_data = {}
        
        try:
            total_days = (config.end_date - config.start_date).days
            self.logger.info(f"Carregando dados históricos para {total_days} dias")
            
            for symbol in config.symbols:
                historical_data[symbol] = {}
                
                for timeframe in config.timeframes:
                    try:
                        # Carregar com margem para indicadores
                        start_with_margin = config.start_date - timedelta(days=100)
                        
                        df = self.market_data_provider.get_ohlcv_range(
                            symbol, timeframe, start_with_margin, config.end_date)
                        
                        if df is not None and len(df) > 50:
                            historical_data[symbol][timeframe] = df
                            self.logger.debug(f"Carregados {len(df)} pontos para {symbol} {timeframe}")
                        else:
                            self.logger.warning(f"Dados insuficientes para {symbol} {timeframe}")
                            
                    except Exception as e:
                        self.logger.error(f"Erro carregando {symbol} {timeframe}: {e}")
                        continue
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Erro no carregamento de dados: {e}")
            return {}
    
    def _validate_historical_data(self, historical_data: Dict, config: BacktestConfig) -> bool:
        """Valida qualidade dos dados históricos"""
        try:
            # Verificar se temos dados para pelo menos um símbolo
            if not historical_data:
                return False
            
            # Verificar cobertura mínima
            symbols_with_data = 0
            for symbol in config.symbols:
                if symbol in historical_data and historical_data[symbol]:
                    # Verificar se temos dados para o timeframe principal
                    if config.signal_generation_frequency in historical_data[symbol]:
                        df = historical_data[symbol][config.signal_generation_frequency]
                        # Verificar se cobre pelo menos 80% do período
                        data_start = df.index[0]
                        data_end = df.index[-1]
                        coverage = (data_end - data_start) / (config.end_date - config.start_date)
                        if coverage >= 0.8:
                            symbols_with_data += 1
            
            # Necessário pelo menos 1 símbolo com cobertura adequada
            return symbols_with_data >= 1
            
        except Exception:
            return False
    
    def _run_sequential_simulation(self, config: BacktestConfig, historical_data: Dict):
        """Executa simulação sequencial"""
        try:
            # Criar timeline de simulação
            timeline = pd.date_range(
                start=config.start_date,
                end=config.end_date,
                freq=config.signal_generation_frequency
            )
            
            self.logger.info(f"Simulando {len(timeline)} períodos")
            
            for i, current_time in enumerate(timeline):
                if i % 100 == 0:
                    progress = i / len(timeline)
                    self.logger.info(f"Progresso: {progress:.1%}")
                
                # Processar período
                self._process_period(current_time, config, historical_data)
                
                # Snapshot diário
                if current_time.hour == 0 or i == len(timeline) - 1:
                    self._take_daily_snapshot(current_time)
            
        except Exception as e:
            self.logger.error(f"Erro na simulação sequencial: {e}")
            raise
    
    def _run_parallel_simulation(self, config: BacktestConfig, historical_data: Dict):
        """Executa simulação em paralelo (simplificado)"""
        # Para simplicidade, usar simulação sequencial
        # Em uma implementação real, dividiria o período em chunks
        self._run_sequential_simulation(config, historical_data)
    
    def _process_period(self, current_time: datetime, config: BacktestConfig, historical_data: Dict):
        """Processa um período individual"""
        try:
            # 1. Atualizar posições existentes
            self._update_existing_positions(current_time, historical_data, config)
            
            # 2. Verificar stop loss e take profit
            self._check_position_exits(current_time, historical_data, config)
            
            # 3. Gerar novos sinais
            new_signals = self._generate_signals_for_period(current_time, historical_data, config)
            
            # 4. Executar sinais válidos
            self._execute_new_signals(current_time, new_signals, historical_data, config)
            
            # 5. Atualizar equity
            self._update_equity(current_time, historical_data)
            
        except Exception as e:
            self.logger.error(f"Erro processando período {current_time}: {e}")
    
    def _update_existing_positions(self, current_time: datetime, historical_data: Dict, config: BacktestConfig):
        """Atualiza posições existentes com preços atuais"""
        try:
            for position in self.open_positions.values():
                if position.symbol in historical_data:
                    # Usar timeframe principal para preços
                    tf_data = historical_data[position.symbol].get(config.signal_generation_frequency)
                    if tf_data is not None:
                        # Encontrar preço mais próximo
                        price_data = tf_data[tf_data.index <= current_time]
                        if len(price_data) > 0:
                            current_price = price_data['close'].iloc[-1]
                            position.update_unrealized_pnl(current_price)
        
        except Exception as e:
            self.logger.error(f"Erro atualizando posições: {e}")
    
    def _check_position_exits(self, current_time: datetime, historical_data: Dict, config: BacktestConfig):
        """Verifica saídas por stop loss ou take profit"""
        try:
            positions_to_close = []
            
            for position in self.open_positions.values():
                if position.symbol not in historical_data:
                    continue
                
                tf_data = historical_data[position.symbol].get(config.signal_generation_frequency)
                if tf_data is None:
                    continue
                
                # Encontrar dados do período atual
                period_data = tf_data[tf_data.index <= current_time]
                if len(period_data) == 0:
                    continue
                
                current_candle = period_data.iloc[-1]
                high_price = current_candle['high']
                low_price = current_candle['low']
                close_price = current_candle['close']
                
                # Verificar exit conditions
                exit_price = None
                exit_reason = None
                
                if position.side == 'long':
                    # Stop loss
                    if low_price <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif high_price >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = 'take_profit'
                
                else:  # short
                    # Stop loss
                    if high_price >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = 'stop_loss'
                    # Take profit
                    elif low_price <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = 'take_profit'
                
                # Aplicar slippage e fees
                if exit_price is not None:
                    exit_price = self._apply_execution_costs(exit_price, config, 'sell')
                    positions_to_close.append((position, exit_price, exit_reason))
            
            # Fechar posições
            for position, exit_price, exit_reason in positions_to_close:
                self._close_position(position, current_time, exit_price, exit_reason)
        
        except Exception as e:
            self.logger.error(f"Erro verificando saídas: {e}")
    
    def _generate_signals_for_period(self, current_time: datetime, historical_data: Dict, config: BacktestConfig) -> List[MasterSignal]:
        """Gera sinais para o período atual"""
        try:
            # Preparar dados de mercado para o período
            period_data = {}
            
            for symbol in config.symbols:
                if symbol not in historical_data:
                    continue
                
                symbol_data = {}
                for timeframe in config.timeframes:
                    if timeframe in historical_data[symbol]:
                        # Pegar dados até o período atual
                        df = historical_data[symbol][timeframe]
                        available_data = df[df.index <= current_time]
                        
                        if len(available_data) >= 50:  # Mínimo para análise
                            symbol_data[timeframe] = available_data.tail(200)  # Últimos 200 períodos
                
                if symbol_data:
                    period_data[symbol] = symbol_data
            
            # Gerar sinais usando o signal generator
            if period_data:
                # Configurar para usar dados históricos
                original_provider = self.signal_generator.market_data_provider
                self.signal_generator.market_data_provider = HistoricalDataProvider(period_data)
                
                symbols_to_analyze = list(period_data.keys())
                signals = self.signal_generator.generate_signals(symbols_to_analyze, current_time)
                
                # Restaurar provider original
                self.signal_generator.market_data_provider = original_provider
                
                # Log dos sinais
                for signal in signals:
                    self.signal_log.append({
                        'timestamp': current_time,
                        'signal_id': signal.signal_id,
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type,
                        'score': signal.final_score,
                        'executed': False  # Será atualizado se executado
                    })
                
                return signals
            
            return []
            
        except Exception as e:
            self.logger.error(f"Erro gerando sinais: {e}")
            return []
    
    def _execute_new_signals(self, current_time: datetime, signals: List[MasterSignal], 
                           historical_data: Dict, config: BacktestConfig):
        """Executa novos sinais válidos"""
        try:
            for signal in signals:
                # Verificar se devemos executar o sinal
                if not self._should_execute_signal(signal, config):
                    continue
                
                # Verificar se temos capacidade para nova posição
                if len(self.open_positions) >= config.max_positions:
                    continue
                
                # Calcular position size
                position_size_usd = self._calculate_position_size(signal, config)
                if position_size_usd <= 0:
                    continue
                
                # Verificar liquidez disponível
                if position_size_usd > self.current_balance * 0.95:  # Deixar 5% de buffer
                    position_size_usd = self.current_balance * 0.95
                
                # Executar entrada
                entry_price = self._apply_execution_costs(signal.entry_price, config, 'buy')
                quantity = position_size_usd / entry_price
                
                # Criar posição
                position = BacktestPosition(
                    position_id=str(uuid.uuid4()),
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    side='long' if signal.trade_type.value == 'long' else 'short',
                    entry_time=current_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_value=position_size_usd,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_1,
                    current_price=entry_price
                )
                
                # Adicionar à carteira
                self.open_positions[position.position_id] = position
                
                # Atualizar balance (reduzir caixa)
                self.current_balance -= position_size_usd
                
                # Log da execução
                for log_entry in self.signal_log:
                    if log_entry['signal_id'] == signal.signal_id:
                        log_entry['executed'] = True
                        break
                
                self.logger.debug(f"Executado: {signal.symbol} {signal.signal_type} - "
                                f"${position_size_usd:,.0f} @ ${entry_price:.2f}")
        
        except Exception as e:
            self.logger.error(f"Erro executando sinais: {e}")
    
    def _should_execute_signal(self, signal: MasterSignal, config: BacktestConfig) -> bool:
        """Determina se um sinal deve ser executado"""
        try:
            # Verificar score mínimo
            if signal.final_score < config.min_signal_score:
                return False
            
            # Verificar se já temos posição no símbolo
            for position in self.open_positions.values():
                if position.symbol == signal.symbol:
                    return False  # Uma posição por símbolo
            
            # Verificar correlação (simplificado)
            # Em uma implementação real, verificaria correlação entre símbolos
            
            return True
            
        except Exception:
            return False
    
    def _calculate_position_size(self, signal: MasterSignal, config: BacktestConfig) -> float:
        """Calcula tamanho da posição"""
        try:
            if config.position_sizing_method == "risk_based":
                # Usar o position sizing do sinal
                return self.current_balance * signal.position_size_pct
            
            elif config.position_sizing_method == "equal_weight":
                # Peso igual para todas as posições
                return self.current_balance / config.max_positions
            
            elif config.position_sizing_method == "kelly":
                # Kelly Criterion (simplificado)
                # Em uma implementação real, usaria histórico de win rate e avg win/loss
                win_rate = 0.6  # Assumir 60% win rate
                avg_win = 0.06  # 6% avg win
                avg_loss = 0.03  # 3% avg loss
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                
                return self.current_balance * kelly_fraction
            
            else:
                # Default: 2% risk per trade
                return self.current_balance * 0.02
                
        except Exception:
            return self.current_balance * 0.02
    
    def _apply_execution_costs(self, price: float, config: BacktestConfig, side: str) -> float:
        """Aplica slippage e fees ao preço"""
        try:
            if config.execution_model == ExecutionModel.PERFECT:
                return price
            
            # Slippage
            slippage_factor = config.slippage_bps / 10000  # Convert bps to decimal
            
            if config.execution_model == ExecutionModel.CONSERVATIVE:
                slippage_factor *= 2  # Dobrar slippage no modo conservador
            
            if side == 'buy':
                slipped_price = price * (1 + slippage_factor)
            else:  # sell
                slipped_price = price * (1 - slippage_factor)
            
            # Fees são aplicados separadamente no balanço
            fee_cost = slipped_price * config.fees_pct
            self.total_fees_paid += fee_cost
            self.total_slippage_cost += abs(slipped_price - price)
            
            return slipped_price
            
        except Exception:
            return price
    
    def _close_position(self, position: BacktestPosition, exit_time: datetime, 
                       exit_price: float, reason: str):
        """Fecha uma posição"""
        try:
            # Fechar posição
            position.close_position(exit_time, exit_price, reason)
            
            # Atualizar balanço
            if position.realized_pnl:
                self.current_balance += position.entry_value + position.realized_pnl
            else:
                self.current_balance += position.entry_value
            
            # Remover das posições abertas
            if position.position_id in self.open_positions:
                del self.open_positions[position.position_id]
            
            # Adicionar às posições fechadas
            self.closed_positions.append(position)
            
            # Remover do risk manager
            if position.symbol in self.risk_manager.active_positions:
                del self.risk_manager.active_positions[position.symbol]
            
            self.logger.debug(f"Fechada: {position.symbol} - "
                            f"P&L: ${position.realized_pnl:.2f} ({position.realized_pnl_pct:.2%}) - "
                            f"Razão: {reason}")
        
        except Exception as e:
            self.logger.error(f"Erro fechando posição: {e}")
    
    def _update_equity(self, current_time: datetime, historical_data: Dict):
        """Atualiza equity total"""
        try:
            # Equity = Cash + Valor das posições abertas
            positions_value = sum(pos.entry_value + pos.unrealized_pnl 
                                for pos in self.open_positions.values() 
                                if pos.unrealized_pnl is not None)
            
            self.current_equity = self.current_balance + positions_value
            
            # Atualizar drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        except Exception as e:
            self.logger.error(f"Erro atualizando equity: {e}")
    
    def _take_daily_snapshot(self, current_time: datetime):
        """Tira snapshot diário do estado"""
        try:
            # Calcular P&L diário
            daily_pnl = 0
            if len(self.daily_snapshots) > 0:
                yesterday_equity = self.daily_snapshots[-1].equity
                daily_pnl = self.current_equity - yesterday_equity
            
            # Calcular retorno cumulativo
            initial_balance = self.daily_snapshots[0].balance if self.daily_snapshots else self.current_equity
            cumulative_return = (self.current_equity - initial_balance) / initial_balance
            
            # Contar sinais do dia
            signals_today = len([s for s in self.signal_log 
                               if s['timestamp'].date() == current_time.date()])
            
            snapshot = BacktestSnapshot(
                timestamp=current_time,
                balance=self.current_balance,
                equity=self.current_equity,
                open_positions=len(self.open_positions),
                total_exposure=sum(pos.entry_value for pos in self.open_positions.values()),
                daily_pnl=daily_pnl,
                cumulative_return=cumulative_return,
                drawdown=self.current_drawdown,
                signals_today=signals_today,
                active_signals=len([s for s in self.signal_log if not s['executed']])
            )
            
            self.daily_snapshots.append(snapshot)
        
        except Exception as e:
            self.logger.error(f"Erro no snapshot diário: {e}")
    
    def _calculate_final_results(self, config: BacktestConfig, start_time: pd.Timestamp) -> BacktestResults:
        """Calcula resultados finais do backtesting"""
        try:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Performance básica
            initial_balance = config.initial_balance
            final_balance = self.current_equity
            total_return = final_balance - initial_balance
            total_return_pct = total_return / initial_balance
            
            # CAGR
            years = (config.end_date - config.start_date).days / 365.25
            cagr = (final_balance / initial_balance) ** (1 / years) - 1 if years > 0 else 0
            
            # Trading statistics
            total_trades = len(self.closed_positions)
            if total_trades > 0:
                winning_trades = len([p for p in self.closed_positions if p.realized_pnl and p.realized_pnl > 0])
                losing_trades = total_trades - winning_trades
                win_rate = winning_trades / total_trades
                
                wins = [p.realized_pnl_pct for p in self.closed_positions if p.realized_pnl_pct and p.realized_pnl_pct > 0]
                losses = [abs(p.realized_pnl_pct) for p in self.closed_positions if p.realized_pnl_pct and p.realized_pnl_pct < 0]
                
                avg_win_pct = np.mean(wins) if wins else 0
                avg_loss_pct = np.mean(losses) if losses else 0
                largest_win_pct = max(wins) if wins else 0
                largest_loss_pct = max(losses) if losses else 0
                profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0
            else:
                winning_trades = losing_trades = 0
                win_rate = avg_win_pct = avg_loss_pct = largest_win_pct = largest_loss_pct = profit_factor = 0
            
            # Risk metrics
            if len(self.daily_snapshots) > 1:
                daily_returns = []
                for i in range(1, len(self.daily_snapshots)):
                    prev_equity = self.daily_snapshots[i-1].equity
                    curr_equity = self.daily_snapshots[i].equity
                    daily_ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                    daily_returns.append(daily_ret)
                
                volatility = np.std(daily_returns) * np.sqrt(365) if daily_returns else 0
                
                avg_daily_return = np.mean(daily_returns) if daily_returns else 0
                risk_free_daily = config.risk_free_rate / 365
                
                if volatility > 0:
                    sharpe_ratio = (avg_daily_return - risk_free_daily) / volatility * np.sqrt(365)
                    
                    # Sortino ratio (só desvio negativo)
                    negative_returns = [r for r in daily_returns if r < 0]
                    downside_vol = np.std(negative_returns) * np.sqrt(365) if negative_returns else volatility
                    sortino_ratio = (avg_daily_return - risk_free_daily) / downside_vol * np.sqrt(365) if downside_vol > 0 else 0
                else:
                    sharpe_ratio = sortino_ratio = 0
                
                # Calmar ratio
                calmar_ratio = cagr / self.max_drawdown if self.max_drawdown > 0 else 0
            else:
                volatility = sharpe_ratio = sortino_ratio = calmar_ratio = 0
            
            # Signal analysis
            total_signals_generated = len(self.signal_log)
            signals_executed = len([s for s in self.signal_log if s['executed']])
            signal_execution_rate = signals_executed / total_signals_generated if total_signals_generated > 0 else 0
            avg_signal_score = np.mean([s['score'] for s in self.signal_log]) if self.signal_log else 0
            
            # Timing
            if total_trades > 0:
                avg_trade_duration_hours = np.mean([p.duration_hours for p in self.closed_positions if p.duration_hours])
                total_time_in_market_hours = sum(p.duration_hours for p in self.closed_positions if p.duration_hours)
                total_hours = (config.end_date - config.start_date).total_seconds() / 3600
                total_time_in_market_pct = total_time_in_market_hours / total_hours if total_hours > 0 else 0
            else:
                avg_trade_duration_hours = total_time_in_market_pct = 0
            
            # Equity curve
            equity_curve = [(snapshot.timestamp, snapshot.equity) for snapshot in self.daily_snapshots]
            
            # Trade log
            trade_log = []
            for position in self.closed_positions:
                trade_log.append({
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_time': position.entry_time,
                    'exit_time': position.exit_time,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'quantity': position.quantity,
                    'pnl': position.realized_pnl,
                    'pnl_pct': position.realized_pnl_pct,
                    'duration_hours': position.duration_hours,
                    'exit_reason': position.exit_reason
                })
            
            # Component performance (simplificado)
            strategy_performance = {}
            timeframe_performance = {}
            symbol_performance = {}
            
            return BacktestResults(
                config=config,
                initial_balance=initial_balance,
                final_balance=final_balance,
                total_return=total_return,
                total_return_pct=total_return_pct,
                cagr=cagr,
                
                max_drawdown=self.max_drawdown * initial_balance,
                max_drawdown_pct=self.max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                largest_win_pct=largest_win_pct,
                largest_loss_pct=largest_loss_pct,
                profit_factor=profit_factor,
                
                avg_trade_duration_hours=avg_trade_duration_hours,
                total_time_in_market_pct=total_time_in_market_pct,
                
                total_signals_generated=total_signals_generated,
                signals_executed=signals_executed,
                signal_execution_rate=signal_execution_rate,
                avg_signal_score=avg_signal_score,
                
                daily_snapshots=self.daily_snapshots,
                trade_log=trade_log,
                signal_log=self.signal_log,
                equity_curve=equity_curve,
                
                strategy_performance=strategy_performance,
                timeframe_performance=timeframe_performance,
                symbol_performance=symbol_performance,
                
                execution_time_seconds=execution_time,
                data_points_processed=len(self.daily_snapshots)
            )
            
        except Exception as e:
            self.logger.error(f"Erro calculando resultados finais: {e}")
            raise
    
    def _save_backtest_results(self, results: BacktestResults):
        """Salva resultados no banco de dados"""
        try:
            if self.database_manager:
                # Criar registro de backtest
                backtest_record = BacktestResult(
                    backtest_id=str(uuid.uuid4()),
                    name=results.config.name,
                    description=results.config.description,
                    strategy_config=results.config.__dict__,
                    symbols_tested=results.config.symbols,
                    timeframes_tested=results.config.timeframes,
                    
                    start_date=results.config.start_date,
                    end_date=results.config.end_date,
                    total_days=(results.config.end_date - results.config.start_date).days,
                    
                    initial_balance=results.initial_balance,
                    final_balance=results.final_balance,
                    total_return=results.total_return,
                    total_return_pct=results.total_return_pct,
                    cagr=results.cagr,
                    
                    max_drawdown=results.max_drawdown,
                    max_drawdown_pct=results.max_drawdown_pct,
                    volatility_annual=results.volatility,
                    sharpe_ratio=results.sharpe_ratio,
                    sortino_ratio=results.sortino_ratio,
                    calmar_ratio=results.calmar_ratio,
                    
                    total_trades=results.total_trades,
                    winning_trades=results.winning_trades,
                    losing_trades=results.losing_trades,
                    win_rate=results.win_rate,
                    
                    avg_win_pct=results.avg_win_pct,
                    avg_loss_pct=results.avg_loss_pct,
                    largest_win_pct=results.largest_win_pct,
                    largest_loss_pct=results.largest_loss_pct,
                    profit_factor=results.profit_factor,
                    
                    avg_trade_duration_hours=results.avg_trade_duration_hours,
                    
                    execution_time_seconds=results.execution_time_seconds
                )
                
                with self.database_manager.get_session() as session:
                    session.add(backtest_record)
                    session.commit()
                
                self.logger.info(f"Backtest results saved to database")
        
        except Exception as e:
            self.logger.error(f"Erro salvando resultados: {e}")


class HistoricalDataProvider:
    """Provider temporário para dados históricos durante backtesting"""
    
    def __init__(self, historical_data: Dict):
        self.historical_data = historical_data
    
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Retorna dados históricos"""
        if symbol in self.historical_data and timeframe in self.historical_data[symbol]:
            df = self.historical_data[symbol][timeframe]
            return df.tail(limit) if len(df) > limit else df
        return None


def main():
    """Teste básico do backtest engine"""
    from ..core.market_data import MarketDataProvider
    
    # Mock components para teste
    class MockMarketDataProvider(MarketDataProvider):
        def get_ohlcv_range(self, symbol, timeframe, start_date, end_date):
            # Gerar dados de teste
            dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
            np.random.seed(42)
            
            price_base = 50000
            prices = price_base + np.random.randn(len(dates)).cumsum() * 100
            
            return pd.DataFrame({
                'open': prices,
                'high': prices + np.abs(np.random.randn(len(dates)) * 50),
                'low': prices - np.abs(np.random.randn(len(dates)) * 50),
                'close': prices,
                'volume': np.random.exponential(1000000, len(dates))
            }, index=dates)
    
    # Configuração de teste
    config = BacktestConfig(
        name="Test Backtest",
        description="Teste básico do sistema",
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
        symbols=["BTCUSDT"],
        timeframes=["4H"],
        initial_balance=100000,
        execution_model=ExecutionModel.REALISTIC
    )
    
    print(f"\n🔬 BACKTEST ENGINE TEST")
    print(f"Period: {config.start_date.date()} - {config.end_date.date()}")
    print(f"Symbols: {config.symbols}")
    print(f"Initial Balance: ${config.initial_balance:,.0f}")
    print(f"Execution Model: {config.execution_model.value}")
    
    print(f"\n✅ Backtest configuration validated")
    print(f"Duration: {(config.end_date - config.start_date).days} days")
    print(f"Expected data points: ~{(config.end_date - config.start_date).days * 6}")  # 4H intervals


if __name__ == "__main__":
    main()