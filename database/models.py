"""
💾 DATABASE MODELS - Smart Trading System v2.0

Modelos SQLAlchemy para persistência completa do sistema:
- Signals e executions
- Market data cache
- Performance tracking
- Risk metrics
- Backtesting results

Filosofia: Data is the Foundation of Intelligence - Store Everything, Learn from Everything
"""

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime, timezone
import enum
import json
from typing import Dict, List, Optional, Any
import uuid

Base = declarative_base()


# Enums for database
class SignalQualityEnum(enum.Enum):
    EXCEPTIONAL = "exceptional"
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    WEAK = "weak"
    POOR = "poor"


class SignalStatusEnum(enum.Enum):
    ACTIVE = "active"
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TradeTypeEnum(enum.Enum):
    LONG = "long"
    SHORT = "short"


class TradeStatusEnum(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class MarketConditionEnum(enum.Enum):
    STRONG_BULL = "strong_bull"
    MODERATE_BULL = "moderate_bull"
    WEAK_BULL = "weak_bull"
    SIDEWAYS_BULL = "sideways_bull"
    SIDEWAYS = "sideways"
    SIDEWAYS_BEAR = "sideways_bear"
    WEAK_BEAR = "weak_bear"
    MODERATE_BEAR = "moderate_bear"
    STRONG_BEAR = "strong_bear"


class Signal(Base):
    """
    📊 Tabela principal de sinais gerados
    """
    __tablename__ = 'signals'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Basic Info
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(50), nullable=False)  # swing_long, breakout_short, etc.
    trade_type = Column(SQLEnum(TradeTypeEnum), nullable=False)
    
    # Scores and Quality
    final_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    quality = Column(SQLEnum(SignalQualityEnum), nullable=False)
    status = Column(SQLEnum(SignalStatusEnum), nullable=False, default=SignalStatusEnum.ACTIVE)
    
    # Trading Parameters
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float)
    position_size_pct = Column(Float, nullable=False)
    risk_reward_ratio = Column(Float, nullable=False)
    
    # Timing
    generated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    valid_until = Column(DateTime(timezone=True), nullable=False)
    best_entry_time = Column(DateTime(timezone=True))
    executed_at = Column(DateTime(timezone=True))
    cancelled_at = Column(DateTime(timezone=True))
    
    # Analysis Results (JSON fields)
    component_scores = Column(JSON)  # Scores de cada componente
    filter_results = Column(JSON)   # Resultados dos filtros
    risk_analysis = Column(JSON)    # Análise de risco
    confluence_factors = Column(JSON)  # Fatores de confluência
    
    # Metadata
    timeframes_analyzed = Column(JSON)
    total_analysis_time_ms = Column(Float)
    confluence_factors_count = Column(Integer)
    reasoning = Column(Text)
    warnings = Column(JSON)
    
    # Relationships
    trades = relationship("Trade", back_populates="signal")
    performance_metrics = relationship("SignalPerformance", back_populates="signal")
    
    # Indexes
    __table_args__ = (
        Index('idx_signals_symbol_status', 'symbol', 'status'),
        Index('idx_signals_generated_score', 'generated_at', 'final_score'),
        Index('idx_signals_quality_type', 'quality', 'signal_type'),
    )
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
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
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'reasoning': self.reasoning,
            'warnings': self.warnings
        }


class Trade(Base):
    """
    💰 Tabela de trades executados
    """
    __tablename__ = 'trades'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Foreign Key
    signal_id = Column(String(100), ForeignKey('signals.signal_id'), nullable=False)
    
    # Basic Info
    symbol = Column(String(20), nullable=False, index=True)
    trade_type = Column(SQLEnum(TradeTypeEnum), nullable=False)
    status = Column(SQLEnum(TradeStatusEnum), nullable=False, default=TradeStatusEnum.OPEN)
    
    # Entry Details
    entry_price = Column(Float, nullable=False)
    entry_quantity = Column(Float, nullable=False)
    entry_value = Column(Float, nullable=False)  # USD value
    entry_time = Column(DateTime(timezone=True), nullable=False, default=func.now())
    entry_fees = Column(Float, default=0)
    
    # Exit Details
    exit_price = Column(Float)
    exit_quantity = Column(Float)
    exit_value = Column(Float)
    exit_time = Column(DateTime(timezone=True))
    exit_fees = Column(Float, default=0)
    exit_reason = Column(String(100))  # 'take_profit', 'stop_loss', 'manual', 'timeout'
    
    # Risk Management
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    trailing_stop = Column(Float)
    max_loss_pct = Column(Float)
    
    # Performance
    pnl_gross = Column(Float)
    pnl_net = Column(Float)  # After fees
    pnl_pct = Column(Float)
    max_unrealized_profit = Column(Float)
    max_unrealized_loss = Column(Float)
    duration_hours = Column(Float)
    
    # Market Context
    market_condition_entry = Column(SQLEnum(MarketConditionEnum))
    market_condition_exit = Column(SQLEnum(MarketConditionEnum))
    volatility_entry = Column(Float)
    volatility_exit = Column(Float)
    
    # Execution Quality
    slippage_entry = Column(Float)  # Diferença entre expected e actual
    slippage_exit = Column(Float)
    execution_delay_ms = Column(Integer)
    
    # Metadata
    notes = Column(Text)
    tags = Column(JSON)  # Tags customizáveis
    
    # Relationships
    signal = relationship("Signal", back_populates="trades")
    
    # Indexes
    __table_args__ = (
        Index('idx_trades_symbol_status', 'symbol', 'status'),
        Index('idx_trades_entry_time', 'entry_time'),
        Index('idx_trades_pnl', 'pnl_pct'),
    )
    
    def calculate_performance(self):
        """Calcula métricas de performance do trade"""
        if self.exit_price and self.entry_price:
            if self.trade_type == TradeTypeEnum.LONG:
                self.pnl_gross = (self.exit_price - self.entry_price) * self.entry_quantity
                self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
            else:  # SHORT
                self.pnl_gross = (self.entry_price - self.exit_price) * self.entry_quantity
                self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
            
            self.pnl_net = self.pnl_gross - (self.entry_fees + self.exit_fees)
            
            if self.exit_time and self.entry_time:
                self.duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600


class MarketData(Base):
    """
    📈 Cache de dados de mercado
    """
    __tablename__ = 'market_data'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Basic Info
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # OHLCV Data
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional Metrics
    vwap = Column(Float)  # Volume Weighted Average Price
    number_of_trades = Column(Integer)
    taker_buy_volume = Column(Float)
    taker_buy_quote_volume = Column(Float)
    
    # Technical Indicators (computed)
    sma_20 = Column(Float)
    ema_20 = Column(Float)
    rsi_14 = Column(Float)
    atr_14 = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    
    # Metadata
    data_source = Column(String(50), default='binance')
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('symbol', 'timeframe', 'timestamp'),
        Index('idx_market_data_symbol_tf_time', 'symbol', 'timeframe', 'timestamp'),
    )


class PerformanceMetrics(Base):
    """
    📊 Métricas de performance do sistema
    """
    __tablename__ = 'performance_metrics'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Time Period
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly'
    
    # Portfolio Metrics
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    
    # Risk Metrics
    max_drawdown = Column(Float)
    max_drawdown_pct = Column(Float)
    volatility = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Trading Metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    profit_factor = Column(Float)
    
    # Signal Generation Metrics
    signals_generated = Column(Integer, default=0)
    signals_executed = Column(Integer, default=0)
    signal_success_rate = Column(Float)
    avg_signal_score = Column(Float)
    
    # Market Exposure
    avg_exposure = Column(Float)  # Average portfolio exposure
    max_exposure = Column(Float)  # Maximum portfolio exposure
    time_in_market_pct = Column(Float)
    
    # Component Performance
    best_performing_strategy = Column(String(50))
    worst_performing_strategy = Column(String(50))
    best_performing_timeframe = Column(String(10))
    component_scores = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_period', 'period_type', 'period_start'),
    )


class SignalPerformance(Base):
    """
    🎯 Performance individual de sinais
    """
    __tablename__ = 'signal_performance'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Foreign Key
    signal_id = Column(String(100), ForeignKey('signals.signal_id'), nullable=False)
    
    # Performance Metrics
    was_executed = Column(Boolean, default=False)
    execution_delay_hours = Column(Float)  # Delay from generation to execution
    
    # Price Movement Analysis
    price_at_generation = Column(Float)
    price_at_expiry = Column(Float)
    max_favorable_move = Column(Float)  # Best price during signal validity
    max_adverse_move = Column(Float)    # Worst price during signal validity
    
    # Directional Accuracy
    direction_correct = Column(Boolean)
    price_move_pct = Column(Float)  # Actual price movement
    predicted_move_pct = Column(Float)  # Expected from take_profit
    
    # Component Accuracy
    structure_accuracy = Column(Float)   # How accurate was structure analysis
    trend_accuracy = Column(Float)       # How accurate was trend analysis
    timing_accuracy = Column(Float)      # How good was the timing
    risk_accuracy = Column(Float)        # How accurate was risk assessment
    
    # Market Context at Expiry
    market_condition_at_expiry = Column(SQLEnum(MarketConditionEnum))
    volatility_realized = Column(Float)
    volatility_predicted = Column(Float)
    
    # Learning Data
    what_went_right = Column(JSON)  # Factors that worked
    what_went_wrong = Column(JSON)  # Factors that didn't work
    lessons_learned = Column(Text)
    
    # Metadata
    analyzed_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    signal = relationship("Signal", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_signal_perf_execution', 'was_executed', 'direction_correct'),
    )


class BacktestResult(Base):
    """
    🔬 Resultados de backtesting
    """
    __tablename__ = 'backtest_results'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    backtest_id = Column(String(100), unique=True, nullable=False)
    
    # Backtest Configuration
    name = Column(String(200), nullable=False)
    description = Column(Text)
    strategy_config = Column(JSON, nullable=False)
    symbols_tested = Column(JSON, nullable=False)
    timeframes_tested = Column(JSON, nullable=False)
    
    # Time Range
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    total_days = Column(Integer)
    
    # Initial Parameters
    initial_balance = Column(Float, nullable=False)
    position_sizing_method = Column(String(50))
    max_positions = Column(Integer)
    
    # Performance Results
    final_balance = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    total_return_pct = Column(Float, nullable=False)
    cagr = Column(Float)  # Compound Annual Growth Rate
    
    # Risk Metrics
    max_drawdown = Column(Float, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    volatility_annual = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Trading Statistics
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    win_rate = Column(Float, nullable=False)
    
    # Trade Analysis
    avg_win_pct = Column(Float)
    avg_loss_pct = Column(Float)
    largest_win_pct = Column(Float)
    largest_loss_pct = Column(Float)
    profit_factor = Column(Float)
    
    # Timing Analysis
    avg_trade_duration_hours = Column(Float)
    avg_time_to_profit = Column(Float)
    avg_time_to_loss = Column(Float)
    
    # Market Conditions Performance
    bull_market_return = Column(Float)
    bear_market_return = Column(Float)
    sideways_market_return = Column(Float)
    high_vol_return = Column(Float)
    low_vol_return = Column(Float)
    
    # Component Performance
    best_strategy = Column(String(50))
    worst_strategy = Column(String(50))
    best_timeframe = Column(String(10))
    worst_timeframe = Column(String(10))
    component_analysis = Column(JSON)
    
    # Detailed Results
    monthly_returns = Column(JSON)
    trade_log_summary = Column(JSON)
    equity_curve_data = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    execution_time_seconds = Column(Float)
    
    # Indexes
    __table_args__ = (
        Index('idx_backtest_performance', 'total_return_pct', 'sharpe_ratio'),
        Index('idx_backtest_dates', 'start_date', 'end_date'),
    )


class SystemConfig(Base):
    """
    ⚙️ Configurações do sistema
    """
    __tablename__ = 'system_config'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    
    # Config Identity
    config_name = Column(String(100), unique=True, nullable=False)
    config_version = Column(String(20), nullable=False)
    description = Column(Text)
    
    # Configuration Data
    config_data = Column(JSON, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    created_by = Column(String(100))
    last_used = Column(DateTime(timezone=True))
    
    # Performance tracking
    total_signals_generated = Column(Integer, default=0)
    avg_signal_score = Column(Float)
    success_rate = Column(Float)
    
    # Indexes
    __table_args__ = (
        Index('idx_config_active', 'is_active', 'is_default'),
    )


class TradingSession(Base):
    """
    🕐 Sessões de trading e seus resultados
    """
    __tablename__ = 'trading_sessions'
    
    # Primary Key
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    
    # Session Info
    session_name = Column(String(200))
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True))
    duration_hours = Column(Float)
    
    # Market Context
    market_condition = Column(SQLEnum(MarketConditionEnum))
    avg_volatility = Column(Float)
    major_news_events = Column(JSON)
    
    # Session Performance
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float)
    session_return = Column(Float)
    session_return_pct = Column(Float)
    
    # Trading Activity
    signals_generated = Column(Integer, default=0)
    trades_executed = Column(Integer, default=0)
    active_positions = Column(Integer, default=0)
    
    # Risk Metrics
    max_exposure = Column(Float)
    max_single_position = Column(Float)
    realized_vol = Column(Float)
    
    # System Performance
    avg_analysis_time_ms = Column(Float)
    system_uptime_pct = Column(Float)
    error_count = Column(Integer, default=0)
    
    # Notes and learnings
    session_notes = Column(Text)
    key_decisions = Column(JSON)
    lessons_learned = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_sessions_performance', 'session_return_pct', 'start_time'),
    )


# Database utility functions
class DatabaseManager:
    """
    💾 Gerenciador do banco de dados
    """
    
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Cria todas as tabelas"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Retorna uma sessão do banco"""
        return self.SessionLocal()
    
    def save_signal(self, master_signal) -> Signal:
        """Salva um MasterSignal no banco"""
        with self.get_session() as session:
            try:
                signal = Signal(
                    signal_id=master_signal.signal_id,
                    symbol=master_signal.symbol,
                    signal_type=master_signal.signal_type,
                    trade_type=TradeTypeEnum(master_signal.trade_type.value),
                    final_score=master_signal.final_score,
                    confidence=master_signal.confidence,
                    quality=SignalQualityEnum(master_signal.quality.value),
                    status=SignalStatusEnum(master_signal.status.value),
                    entry_price=master_signal.entry_price,
                    stop_loss=master_signal.stop_loss,
                    take_profit_1=master_signal.take_profit_1,
                    take_profit_2=master_signal.take_profit_2,
                    position_size_pct=master_signal.position_size_pct,
                    risk_reward_ratio=master_signal.risk_reward_ratio,
                    generated_at=master_signal.generated_at,
                    valid_until=master_signal.valid_until,
                    best_entry_time=master_signal.best_entry_time,
                    component_scores={
                        'market_structure': master_signal.market_structure.score,
                        'trend_analysis': master_signal.trend_analysis.score,
                        'leading_indicators': master_signal.leading_indicators.score,
                        'strategy_signals': master_signal.strategy_signals.score,
                        'confluence_analysis': master_signal.confluence_analysis.score
                    },
                    filter_results={
                        'market_condition': master_signal.market_condition_filter.score,
                        'volatility': master_signal.volatility_filter.score,
                        'time': master_signal.time_filter.score,
                        'fundamental': master_signal.fundamental_filter.score
                    },
                    risk_analysis=master_signal.risk_analysis,
                    timeframes_analyzed=master_signal.timeframes_analyzed,
                    total_analysis_time_ms=master_signal.total_analysis_time_ms,
                    confluence_factors_count=master_signal.confluence_factors_count,
                    reasoning=master_signal.reasoning,
                    warnings=master_signal.warnings
                )
                
                session.add(signal)
                session.commit()
                session.refresh(signal)
                return signal
                
            except Exception as e:
                session.rollback()
                raise e
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[Signal]:
        """Retorna sinais ativos"""
        with self.get_session() as session:
            query = session.query(Signal).filter(Signal.status == SignalStatusEnum.ACTIVE)
            
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            return query.order_by(Signal.final_score.desc()).all()
    
    def save_trade(self, trade_data: Dict) -> Trade:
        """Salva um trade no banco"""
        with self.get_session() as session:
            try:
                trade = Trade(**trade_data)
                session.add(trade)
                session.commit()
                session.refresh(trade)
                return trade
            except Exception as e:
                session.rollback()
                raise e
    
    def get_performance_metrics(self, period_days: int = 30) -> PerformanceMetrics:
        """Retorna métricas de performance"""
        with self.get_session() as session:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - pd.Timedelta(days=period_days)
            
            return session.query(PerformanceMetrics).filter(
                PerformanceMetrics.period_start >= start_date,
                PerformanceMetrics.period_end <= end_date
            ).order_by(PerformanceMetrics.period_end.desc()).first()
    
    def calculate_portfolio_performance(self) -> Dict:
        """Calcula performance atual do portfólio"""
        with self.get_session() as session:
            # Trades fechados nos últimos 30 dias
            thirty_days_ago = datetime.now(timezone.utc) - pd.Timedelta(days=30)
            
            trades = session.query(Trade).filter(
                Trade.status == TradeStatusEnum.CLOSED,
                Trade.exit_time >= thirty_days_ago
            ).all()
            
            if not trades:
                return {'total_trades': 0, 'win_rate': 0, 'total_return': 0}
            
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.pnl_net > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.pnl_net for t in trades if t.pnl_net)
            total_return_pct = sum(t.pnl_pct for t in trades if t.pnl_pct)
            
            avg_win = np.mean([t.pnl_pct for t in trades if t.pnl_pct and t.pnl_pct > 0])
            avg_loss = np.mean([abs(t.pnl_pct) for t in trades if t.pnl_pct and t.pnl_pct < 0])
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_return': total_pnl,
                'total_return_pct': total_return_pct,
                'avg_win_pct': avg_win if not pd.isna(avg_win) else 0,
                'avg_loss_pct': avg_loss if not pd.isna(avg_loss) else 0,
                'profit_factor': avg_win / avg_loss if avg_loss and avg_loss > 0 else 0
            }


def main():
    """Teste básico do sistema de banco de dados"""
    # Usar SQLite em memória para teste
    db_manager = DatabaseManager("sqlite:///:memory:")
    
    # Criar tabelas
    db_manager.create_tables()
    print("✅ Tabelas criadas com sucesso")
    
    # Testar salvamento de dados de mercado
    with db_manager.get_session() as session:
        market_data = MarketData(
            symbol="BTCUSDT",
            timeframe="4H",
            timestamp=datetime.now(timezone.utc),
            open_price=50000,
            high_price=51000,
            low_price=49500,
            close_price=50500,
            volume=1000000,
            vwap=50250
        )
        
        session.add(market_data)
        session.commit()
        print("✅ Dados de mercado salvos")
    
    # Testar recuperação
    with db_manager.get_session() as session:
        data = session.query(MarketData).filter(MarketData.symbol == "BTCUSDT").first()
        print(f"✅ Dados recuperados: {data.symbol} - {data.close_price}")
    
    # Testar performance
    performance = db_manager.calculate_portfolio_performance()
    print(f"✅ Performance calculada: {performance}")
    
    print("\n💾 DATABASE SYSTEM TEST COMPLETED")
    print(f"Tables created: {len(Base.metadata.tables)}")
    print(f"Available tables: {list(Base.metadata.tables.keys())}")


if __name__ == "__main__":
    main()