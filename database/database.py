"""
Database: Database Management
Gerenciamento de conexões, operações e cache do banco de dados
"""
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from threading import Lock

from .models import (
    Base, Signal, Trade, PerformanceMetrics, MarketData, 
    SignalPerformance, BacktestResult
)
from utils.logger import get_logger
from utils.helpers import generate_cache_key


logger = get_logger(__name__)


class DatabaseManager:
    """Gerenciador principal do banco de dados"""
    
    def __init__(self, database_url: str = None, use_cache: bool = True):
        self.database_url = database_url or self._get_default_database_url()
        self.use_cache = use_cache
        self._engine = None
        self._session_factory = None
        self._metadata = None
        self._cache = {}
        self._cache_lock = Lock()
        self._max_cache_size = 1000
        self._cache_ttl = 300  # 5 minutos
        
        self._initialize_database()
    
    def _get_default_database_url(self) -> str:
        """Obtém URL padrão do banco de dados"""
        db_path = os.getenv('DATABASE_PATH', 'data/trading_system.db')
        
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        return f"sqlite:///{db_path}"
    
    def _initialize_database(self):
        """Inicializa conexão e tabelas do banco"""
        try:
            # Configurações específicas para SQLite
            if 'sqlite' in self.database_url:
                self._engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 30
                    },
                    echo=False
                )
            else:
                self._engine = create_engine(self.database_url, echo=False)
            
            # Cria session factory
            self._session_factory = sessionmaker(bind=self._engine)
            
            # Cria tabelas se não existirem
            Base.metadata.create_all(self._engine)
            
            # Inicializa metadata
            self._metadata = MetaData()
            self._metadata.reflect(bind=self._engine)
            
            logger.info(f"Database inicializado: {self.database_url}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager para sessões do banco"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Erro na sessão do banco: {e}")
            raise
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Executa query SQL e retorna DataFrame"""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Erro ao executar query: {e}")
            raise
    
    def execute_raw_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Executa query e retorna lista de dicionários"""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return [dict(row) for row in result.fetchall()]
        except Exception as e:
            logger.error(f"Erro ao executar raw query: {e}")
            raise
    
    # ==========================================================================
    # OPERAÇÕES DE CACHE
    # ==========================================================================
    
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """Gera chave de cache"""
        return generate_cache_key(operation, **kwargs)
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Obtém item do cache se válido"""
        if not self.use_cache:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                item = self._cache[cache_key]
                
                # Verifica TTL
                if datetime.now() - item['timestamp'] < timedelta(seconds=self._cache_ttl):
                    return item['data']
                else:
                    # Remove item expirado
                    del self._cache[cache_key]
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Salva item no cache"""
        if not self.use_cache:
            return
        
        with self._cache_lock:
            # Limpa cache se muito grande
            if len(self._cache) >= self._max_cache_size:
                # Remove 25% dos itens mais antigos
                oldest_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1]['timestamp']
                )[:self._max_cache_size // 4]
                
                for key, _ in oldest_items:
                    del self._cache[key]
            
            self._cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
    
    def clear_cache(self):
        """Limpa todo o cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cache limpo")
    
    # ==========================================================================
    # OPERAÇÕES COM SINAIS
    # ==========================================================================
    
    def save_signal(self, signal_data: Dict) -> int:
        """Salva novo sinal no banco"""
        try:
            with self.get_session() as session:
                signal = Signal(**signal_data)
                session.add(signal)
                session.flush()
                signal_id = signal.id
                logger.debug(f"Sinal salvo: ID {signal_id}")
                return signal_id
        except Exception as e:
            logger.error(f"Erro ao salvar sinal: {e}")
            raise
    
    def update_signal(self, signal_id: int, updates: Dict) -> bool:
        """Atualiza sinal existente"""
        try:
            with self.get_session() as session:
                signal = session.query(Signal).filter(Signal.id == signal_id).first()
                if signal:
                    for key, value in updates.items():
                        setattr(signal, key, value)
                    signal.updated_at = datetime.now()
                    logger.debug(f"Sinal atualizado: ID {signal_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Erro ao atualizar sinal: {e}")
            raise
    
    def get_active_signals(self, symbol: str = None) -> List[Dict]:
        """Obtém sinais ativos"""
        cache_key = self._get_cache_key('active_signals', symbol=symbol)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        try:
            query = """
                SELECT * FROM signals 
                WHERE status = 'active'
            """
            params = {}
            
            if symbol:
                query += " AND symbol = :symbol"
                params['symbol'] = symbol
            
            query += " ORDER BY created_at DESC"
            
            result = self.execute_raw_query(query, params)
            self._save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Erro ao obter sinais ativos: {e}")
            raise
    
    def get_signal_history(
        self, 
        symbol: str = None, 
        days: int = 30,
        status: str = None
    ) -> pd.DataFrame:
        """Obtém histórico de sinais"""
        try:
            query = """
                SELECT * FROM signals 
                WHERE created_at >= :start_date
            """
            params = {
                'start_date': datetime.now() - timedelta(days=days)
            }
            
            if symbol:
                query += " AND symbol = :symbol"
                params['symbol'] = symbol
            
            if status:
                query += " AND status = :status"
                params['status'] = status
            
            query += " ORDER BY created_at DESC"
            
            return self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Erro ao obter histórico de sinais: {e}")
            raise
    
    # ==========================================================================
    # OPERAÇÕES COM TRADES
    # ==========================================================================
    
    def save_trade(self, trade_data: Dict) -> int:
        """Salva novo trade"""
        try:
            with self.get_session() as session:
                trade = Trade(**trade_data)
                session.add(trade)
                session.flush()
                trade_id = trade.id
                logger.debug(f"Trade salvo: ID {trade_id}")
                return trade_id
        except Exception as e:
            logger.error(f"Erro ao salvar trade: {e}")
            raise
    
    def update_trade(self, trade_id: int, updates: Dict) -> bool:
        """Atualiza trade existente"""
        try:
            with self.get_session() as session:
                trade = session.query(Trade).filter(Trade.id == trade_id).first()
                if trade:
                    for key, value in updates.items():
                        setattr(trade, key, value)
                    trade.updated_at = datetime.now()
                    logger.debug(f"Trade atualizado: ID {trade_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Erro ao atualizar trade: {e}")
            raise
    
    def get_open_trades(self, symbol: str = None) -> List[Dict]:
        """Obtém trades abertos"""
        try:
            query = """
                SELECT * FROM trades 
                WHERE status IN ('open', 'partial')
            """
            params = {}
            
            if symbol:
                query += " AND symbol = :symbol"
                params['symbol'] = symbol
            
            query += " ORDER BY entry_time DESC"
            
            return self.execute_raw_query(query, params)
        except Exception as e:
            logger.error(f"Erro ao obter trades abertos: {e}")
            raise
    
    def get_trade_history(
        self, 
        symbol: str = None, 
        days: int = 30
    ) -> pd.DataFrame:
        """Obtém histórico de trades"""
        try:
            query = """
                SELECT * FROM trades 
                WHERE entry_time >= :start_date
            """
            params = {
                'start_date': datetime.now() - timedelta(days=days)
            }
            
            if symbol:
                query += " AND symbol = :symbol"
                params['symbol'] = symbol
            
            query += " ORDER BY entry_time DESC"
            
            return self.execute_query(query, params)
        except Exception as e:
            logger.error(f"Erro ao obter histórico de trades: {e}")
            raise
    
    # ==========================================================================
    # OPERAÇÕES COM DADOS DE MERCADO
    # ==========================================================================
    
    def save_market_data(self, data_list: List[Dict]) -> int:
        """Salva dados de mercado em lote"""
        try:
            with self.get_session() as session:
                market_data_objects = [MarketData(**data) for data in data_list]
                session.add_all(market_data_objects)
                session.flush()
                count = len(market_data_objects)
                logger.debug(f"{count} registros de market data salvos")
                return count
        except Exception as e:
            logger.error(f"Erro ao salvar market data: {e}")
            raise
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtém dados de mercado"""
        cache_key = self._get_cache_key(
            'market_data',
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            query = """
                SELECT * FROM market_data 
                WHERE symbol = :symbol AND timeframe = :timeframe
            """
            params = {
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = self.execute_query(query, params)
            self._save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Erro ao obter market data: {e}")
            raise
    
    # ==========================================================================
    # OPERAÇÕES COM PERFORMANCE
    # ==========================================================================
    
    def save_performance_snapshot(self, performance_data: Dict) -> int:
        """Salva snapshot de performance"""
        try:
            with self.get_session() as session:
                performance = PerformanceMetrics(**performance_data)
                session.add(performance)
                session.flush()
                performance_id = performance.id
                logger.debug(f"Performance snapshot salvo: ID {performance_id}")
                return performance_id
        except Exception as e:
            logger.error(f"Erro ao salvar performance: {e}")
            raise
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """Obtém métricas de performance"""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades 
                WHERE entry_time >= :start_date
                AND status = 'closed'
            """
            params = {
                'start_date': datetime.now() - timedelta(days=days)
            }
            
            result = self.execute_raw_query(query, params)
            if result:
                metrics = result[0]
                # Calcula win rate
                if metrics['total_trades'] > 0:
                    metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
                else:
                    metrics['win_rate'] = 0
                
                return metrics
            
            return {}
        except Exception as e:
            logger.error(f"Erro ao obter métricas de performance: {e}")
            raise
    
    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    
    def get_table_info(self, table_name: str) -> Dict:
        """Obtém informações sobre uma tabela"""
        try:
            inspector = inspect(self._engine)
            
            if table_name not in inspector.get_table_names():
                return {}
            
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            
            return {
                'columns': columns,
                'indexes': indexes,
                'exists': True
            }
        except Exception as e:
            logger.error(f"Erro ao obter info da tabela {table_name}: {e}")
            return {'exists': False}
    
    def backup_database(self, backup_path: str) -> bool:
        """Cria backup do banco de dados"""
        try:
            if 'sqlite' in self.database_url:
                # Para SQLite, copia o arquivo
                import shutil
                db_path = self.database_url.replace('sqlite:///', '')
                shutil.copy2(db_path, backup_path)
                logger.info(f"Backup criado: {backup_path}")
                return True
            else:
                logger.warning("Backup não implementado para este tipo de banco")
                return False
        except Exception as e:
            logger.error(f"Erro ao criar backup: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Obtém estatísticas do banco de dados"""
        try:
            stats = {}
            
            # Contagem de registros por tabela
            tables = ['signals', 'trades', 'market_data', 'performance']
            
            for table in tables:
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {table}"
                    result = self.execute_raw_query(count_query)
                    stats[f"{table}_count"] = result[0]['count'] if result else 0
                except:
                    stats[f"{table}_count"] = 0
            
            # Tamanho do cache
            stats['cache_size'] = len(self._cache)
            stats['cache_max_size'] = self._max_cache_size
            
            return stats
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}
    
    def close(self):
        """Fecha conexões do banco"""
        if self._engine:
            self._engine.dispose()
        self.clear_cache()
        logger.info("Database connection closed")


# Instância global do gerenciador
db_manager = DatabaseManager()