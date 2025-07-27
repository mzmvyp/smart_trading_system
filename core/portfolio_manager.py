"""
Core: Portfolio Manager
Gerenciamento global do portfólio, exposição e correlações
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

from database.database import db_manager
from utils.logger import get_logger
from utils.helpers import (
    calculate_percentage_change,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    weighted_average,
    sanitize_numeric_input
)


logger = get_logger(__name__)


@dataclass
class Position:
    """Representa uma posição no portfólio"""
    symbol: str
    side: str                    # 'long' ou 'short'
    size: float                  # Tamanho da posição
    entry_price: float           # Preço médio de entrada
    current_price: float         # Preço atual
    unrealized_pnl: float = 0.0  # PnL não realizado
    realized_pnl: float = 0.0    # PnL realizado
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        """Valor de mercado da posição"""
        return self.size * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """PnL em percentual"""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'long':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # short
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    @property
    def risk_amount(self) -> float:
        """Valor em risco (distância até stop loss)"""
        if not self.stop_loss:
            return 0.0
        
        if self.side == 'long':
            risk = max(0, self.entry_price - self.stop_loss)
        else:
            risk = max(0, self.stop_loss - self.entry_price)
        
        return self.size * risk


@dataclass
class PortfolioSnapshot:
    """Snapshot do portfólio em um momento"""
    timestamp: datetime
    total_balance: float
    available_balance: float
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    positions_count: int
    open_orders: int
    risk_percentage: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


class PortfolioAnalyzer:
    """Analisador de métricas do portfólio"""
    
    @staticmethod
    def calculate_correlation_matrix(symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Calcula matriz de correlação entre símbolos"""
        try:
            price_data = {}
            
            for symbol in symbols:
                # Obtém dados históricos
                market_data = db_manager.get_market_data(
                    symbol=symbol,
                    timeframe='1d',
                    start_date=datetime.now() - timedelta(days=days)
                )
                
                if not market_data.empty:
                    # Calcula retornos diários
                    returns = market_data['close'].pct_change().dropna()
                    price_data[symbol] = returns
            
            if price_data:
                df = pd.DataFrame(price_data)
                return df.corr()
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erro ao calcular correlações: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_portfolio_volatility(positions: List[Position]) -> float:
        """Calcula volatilidade do portfólio"""
        if not positions:
            return 0.0
        
        try:
            # Coleta retornos de cada posição
            returns_data = []
            
            for position in positions:
                # Obtém dados históricos
                market_data = db_manager.get_market_data(
                    symbol=position.symbol,
                    timeframe='1d',
                    limit=30
                )
                
                if not market_data.empty:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        returns_data.append(returns.tolist())
            
            if returns_data:
                # Calcula volatilidade média ponderada
                all_returns = np.concatenate(returns_data)
                return np.std(all_returns) * np.sqrt(252)  # Anualizada
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular volatilidade: {e}")
            return 0.0
    
    @staticmethod
    def calculate_value_at_risk(
        positions: List[Position], 
        confidence: float = 0.95,
        time_horizon: int = 1
    ) -> float:
        """Calcula Value at Risk (VaR)"""
        if not positions:
            return 0.0
        
        try:
            total_portfolio_value = sum(pos.market_value for pos in positions)
            
            if total_portfolio_value == 0:
                return 0.0
            
            # Calcula retornos históricos do portfólio
            portfolio_returns = []
            
            # Simplificado: usa volatilidade média das posições
            volatilities = []
            for position in positions:
                market_data = db_manager.get_market_data(
                    symbol=position.symbol,
                    timeframe='1d',
                    limit=30
                )
                
                if not market_data.empty:
                    returns = market_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        vol = calculate_volatility(market_data['close'].tolist())
                        volatilities.append(vol)
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                # VaR paramétrico
                z_score = 1.645 if confidence == 0.95 else 2.33  # 95% ou 99%
                var = total_portfolio_value * avg_volatility * z_score * np.sqrt(time_horizon)
                return var
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular VaR: {e}")
            return 0.0


class PortfolioManager:
    """Gerenciador principal do portfólio"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.snapshots: List[PortfolioSnapshot] = []
        
        # Configurações de risco
        self.max_portfolio_risk = 10.0  # % máximo em risco
        self.max_position_size = 20.0   # % máximo por posição
        self.max_correlation_exposure = 50.0  # % máximo em ativos correlacionados
        
        # Cache de análises
        self._correlation_cache = {}
        self._last_correlation_update = None
        
        self.analyzer = PortfolioAnalyzer()
        
        # Carrega estado do banco
        self._load_portfolio_state()
    
    def _load_portfolio_state(self):
        """Carrega estado do portfólio do banco"""
        try:
            # Carrega posições abertas
            open_trades = db_manager.get_open_trades()
            
            for trade_data in open_trades:
                symbol = trade_data['symbol']
                
                if symbol not in self.positions:
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=trade_data.get('side', 'long'),
                        size=sanitize_numeric_input(trade_data.get('size', 0)),
                        entry_price=sanitize_numeric_input(trade_data.get('entry_price', 0)),
                        current_price=sanitize_numeric_input(trade_data.get('current_price', 0)),
                        stop_loss=trade_data.get('stop_loss'),
                        take_profit=trade_data.get('take_profit'),
                        opened_at=trade_data.get('entry_time')
                    )
                else:
                    # Atualiza posição existente (média ponderada)
                    existing = self.positions[symbol]
                    new_size = sanitize_numeric_input(trade_data.get('size', 0))
                    new_price = sanitize_numeric_input(trade_data.get('entry_price', 0))
                    
                    # Calcula preço médio
                    total_cost = (existing.size * existing.entry_price) + (new_size * new_price)
                    total_size = existing.size + new_size
                    
                    if total_size > 0:
                        existing.entry_price = total_cost / total_size
                        existing.size = total_size
            
            logger.info(f"Carregadas {len(self.positions)} posições")
            
        except Exception as e:
            logger.error(f"Erro ao carregar estado do portfólio: {e}")
    
    def add_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None
    ) -> bool:
        """Adiciona nova posição ao portfólio"""
        try:
            # Validações
            if size <= 0 or entry_price <= 0:
                logger.error("Tamanho e preço devem ser positivos")
                return False
            
            position_value = size * entry_price
            
            # Verifica limite de exposição por posição
            max_position_value = self.get_total_equity() * (self.max_position_size / 100)
            if position_value > max_position_value:
                logger.warning(f"Posição muito grande: {position_value} > {max_position_value}")
                return False
            
            # Verifica risco total do portfólio
            if not self._check_portfolio_risk_limits(symbol, size, entry_price, stop_loss):
                return False
            
            # Cria ou atualiza posição
            if symbol in self.positions:
                # Atualiza posição existente
                existing = self.positions[symbol]
                
                if existing.side == side:
                    # Mesmo lado: calcula preço médio
                    total_cost = (existing.size * existing.entry_price) + (size * entry_price)
                    total_size = existing.size + size
                    
                    existing.entry_price = total_cost / total_size
                    existing.size = total_size
                    existing.updated_at = datetime.now()
                else:
                    # Lado oposto: reduz posição existente
                    if size >= existing.size:
                        # Fecha posição e abre nova
                        self._close_position(symbol, existing.size, entry_price)
                        remaining_size = size - existing.size
                        
                        if remaining_size > 0:
                            self.positions[symbol] = Position(
                                symbol=symbol,
                                side=side,
                                size=remaining_size,
                                entry_price=entry_price,
                                current_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                opened_at=datetime.now()
                            )
                    else:
                        # Reduz posição existente
                        existing.size -= size
                        # Registra PnL realizado parcial
                        partial_pnl = self._calculate_pnl(existing, entry_price, size)
                        existing.realized_pnl += partial_pnl
            else:
                # Nova posição
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    opened_at=datetime.now()
                )
            
            # Salva no banco
            trade_data = {
                'symbol': symbol,
                'side': side,
                'size': size,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'open'
            }
            
            db_manager.save_trade(trade_data)
            
            logger.info(f"Posição adicionada: {symbol} {side} {size}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar posição: {e}")
            return False
    
    def close_position(
        self,
        symbol: str,
        size: float = None,
        exit_price: float = None
    ) -> bool:
        """Fecha posição (total ou parcial)"""
        try:
            if symbol not in self.positions:
                logger.warning(f"Posição não encontrada: {symbol}")
                return False
            
            position = self.positions[symbol]
            close_size = size or position.size
            
            if close_size > position.size:
                logger.error("Tamanho de fechamento maior que posição")
                return False
            
            # Usa preço atual se não fornecido
            if exit_price is None:
                exit_price = position.current_price
            
            # Calcula PnL
            pnl = self._calculate_pnl(position, exit_price, close_size)
            
            # Atualiza saldo
            self.current_balance += pnl
            
            # Atualiza posição
            if close_size >= position.size:
                # Fecha posição completamente
                del self.positions[symbol]
            else:
                # Fechamento parcial
                position.size -= close_size
                position.realized_pnl += pnl
                position.updated_at = datetime.now()
            
            # Registra trade histórico
            self.trade_history.append({
                'symbol': symbol,
                'side': position.side,
                'size': close_size,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_time': datetime.now()
            })
            
            # Atualiza no banco
            db_manager.update_trade(
                symbol,  # Simplificado
                {
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'pnl': pnl,
                    'status': 'closed' if close_size >= position.size else 'partial'
                }
            )
            
            logger.info(f"Posição fechada: {symbol} PnL: {pnl}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")
            return False
    
    def update_prices(self, price_updates: Dict[str, float]):
        """Atualiza preços atuais das posições"""
        for symbol, price in price_updates.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.unrealized_pnl = self._calculate_pnl(position, price, position.size)
                position.updated_at = datetime.now()
    
    def _calculate_pnl(self, position: Position, current_price: float, size: float) -> float:
        """Calcula PnL de uma posição"""
        if position.side == 'long':
            return (current_price - position.entry_price) * size
        else:  # short
            return (position.entry_price - current_price) * size
    
    def _check_portfolio_risk_limits(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_loss: float
    ) -> bool:
        """Verifica se nova posição respeita limites de risco"""
        
        # Calcula risco da nova posição
        if stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            position_risk = size * risk_per_unit
        else:
            # Se não tem stop loss, considera 5% de risco
            position_risk = size * entry_price * 0.05
        
        # Risco atual do portfólio
        current_risk = sum(pos.risk_amount for pos in self.positions.values())
        total_risk = current_risk + position_risk
        
        # Verifica limite de risco total
        max_risk = self.get_total_equity() * (self.max_portfolio_risk / 100)
        if total_risk > max_risk:
            logger.warning(f"Risco total muito alto: {total_risk} > {max_risk}")
            return False
        
        # Verifica correlações (se houver posições similares)
        correlated_exposure = self._calculate_correlated_exposure(symbol)
        position_value = size * entry_price
        
        if correlated_exposure + position_value > self.get_total_equity() * (self.max_correlation_exposure / 100):
            logger.warning("Muito exposição em ativos correlacionados")
            return False
        
        return True
    
    def _calculate_correlated_exposure(self, symbol: str) -> float:
        """Calcula exposição em ativos correlacionados"""
        # Implementação simplificada
        # Em produção, usaria matriz de correlação real
        
        correlated_symbols = []
        
        # Agrupa símbolos similares (BTC, ETH, etc.)
        if 'BTC' in symbol:
            correlated_symbols = [s for s in self.positions.keys() if 'BTC' in s]
        elif 'ETH' in symbol:
            correlated_symbols = [s for s in self.positions.keys() if 'ETH' in s]
        
        total_exposure = 0.0
        for corr_symbol in correlated_symbols:
            if corr_symbol in self.positions:
                total_exposure += self.positions[corr_symbol].market_value
        
        return total_exposure
    
    def get_total_equity(self) -> float:
        """Calcula patrimônio total (saldo + posições)"""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.current_balance + unrealized_pnl
    
    def get_available_balance(self) -> float:
        """Calcula saldo disponível para novas posições"""
        used_margin = sum(abs(pos.market_value) for pos in self.positions.values())
        return max(0, self.current_balance - used_margin)
    
    def get_portfolio_summary(self) -> Dict:
        """Obtém resumo completo do portfólio"""
        try:
            total_equity = self.get_total_equity()
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized = sum(pos.realized_pnl for pos in self.positions.values())
            
            # Calcula métricas de performance
            performance_metrics = self._calculate_performance_metrics()
            
            return {
                'timestamp': datetime.now(),
                'balance': {
                    'initial': self.initial_balance,
                    'current': self.current_balance,
                    'total_equity': total_equity,
                    'available': self.get_available_balance(),
                    'unrealized_pnl': total_unrealized,
                    'realized_pnl': total_realized,
                    'total_return': ((total_equity - self.initial_balance) / self.initial_balance) * 100
                },
                'positions': {
                    'count': len(self.positions),
                    'total_value': sum(pos.market_value for pos in self.positions.values()),
                    'long_positions': len([p for p in self.positions.values() if p.side == 'long']),
                    'short_positions': len([p for p in self.positions.values() if p.side == 'short'])
                },
                'risk': {
                    'total_risk': sum(pos.risk_amount for pos in self.positions.values()),
                    'risk_percentage': (sum(pos.risk_amount for pos in self.positions.values()) / total_equity) * 100 if total_equity > 0 else 0,
                    'var_95': self.analyzer.calculate_value_at_risk(list(self.positions.values())),
                    'portfolio_volatility': self.analyzer.calculate_portfolio_volatility(list(self.positions.values()))
                },
                'performance': performance_metrics,
                'positions_detail': [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'market_value': pos.market_value,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'pnl_percentage': pos.pnl_percentage,
                        'opened_at': pos.opened_at
                    }
                    for pos in self.positions.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar resumo do portfólio: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calcula métricas de performance"""
        try:
            if not self.trade_history:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Coleta dados dos trades
            pnls = [trade['pnl'] for trade in self.trade_history]
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            # Calcula métricas
            total_trades = len(pnls)
            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Sharpe ratio dos retornos
            if len(pnls) > 1:
                returns = [pnl / self.initial_balance for pnl in pnls]
                sharpe = calculate_sharpe_ratio(returns)
            else:
                sharpe = 0
            
            # Max drawdown
            equity_curve = [self.initial_balance]
            running_equity = self.initial_balance
            
            for pnl in pnls:
                running_equity += pnl
                equity_curve.append(running_equity)
            
            dd_info = calculate_max_drawdown(equity_curve)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'max_drawdown': dd_info['max_drawdown']
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}
    
    def save_snapshot(self):
        """Salva snapshot atual do portfólio"""
        try:
            summary = self.get_portfolio_summary()
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_balance=summary['balance']['current'],
                available_balance=summary['balance']['available'],
                total_equity=summary['balance']['total_equity'],
                unrealized_pnl=summary['balance']['unrealized_pnl'],
                realized_pnl=summary['balance']['realized_pnl'],
                daily_pnl=0,  # Calcular baseado no snapshot anterior
                positions_count=summary['positions']['count'],
                open_orders=0,  # Implementar contagem de ordens
                risk_percentage=summary['risk']['risk_percentage'],
                sharpe_ratio=summary['performance']['sharpe_ratio'],
                max_drawdown=summary['performance']['max_drawdown'],
                win_rate=summary['performance']['win_rate']
            )
            
            self.snapshots.append(snapshot)
            
            # Salva no banco
            snapshot_data = {
                'timestamp': snapshot.timestamp,
                'total_equity': snapshot.total_equity,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'realized_pnl': snapshot.realized_pnl,
                'positions_count': snapshot.positions_count,
                'risk_percentage': snapshot.risk_percentage,
                'metadata': json.dumps({
                    'sharpe_ratio': snapshot.sharpe_ratio,
                    'max_drawdown': snapshot.max_drawdown,
                    'win_rate': snapshot.win_rate
                })
            }
            
            db_manager.save_performance_snapshot(snapshot_data)
            
            logger.debug("Snapshot do portfólio salvo")
            
        except Exception as e:
            logger.error(f"Erro ao salvar snapshot: {e}")


# Instância global do gerenciador
portfolio_manager = PortfolioManager()