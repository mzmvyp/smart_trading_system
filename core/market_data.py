# core/market_data.py
"""
📊 MARKET DATA PROVIDER - SMART TRADING SYSTEM v2.0
Provider inteligente de dados OHLCV com cache, validação e múltiplas fontes
Otimizado para timeframes altos (1H/4H/1D) e qualidade de dados
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# Importações locais
from config.settings import settings
from config.symbols import get_symbol_config, SymbolConfig
from utils.logger import get_logger
from utils.validators import validate_ohlcv_data
from utils.helpers import retry_with_backoff

logger = get_logger(__name__)

@dataclass
class MarketData:
    """📊 Estrutura padronizada para dados de mercado"""
    symbol: str
    timeframe: str
    timestamp: datetime
    ohlcv: pd.DataFrame
    metadata: Dict = field(default_factory=dict)
    
    # Indicadores calculados automaticamente
    sma_20: Optional[pd.Series] = None
    sma_50: Optional[pd.Series] = None
    atr_14: Optional[pd.Series] = None
    volume_ma_20: Optional[pd.Series] = None
    
    # Estatísticas rápidas
    total_candles: int = 0
    data_quality_score: float = 0.0
    last_price: float = 0.0
    avg_volume: float = 0.0
    volatility_atr: float = 0.0
    
    def __post_init__(self):
        """Calcula indicadores e estatísticas após inicialização"""
        if not self.ohlcv.empty:
            self._calculate_indicators()
            self._calculate_statistics()
    
    def _calculate_indicators(self):
        """Calcula indicadores básicos para análise rápida"""
        try:
            df = self.ohlcv
            
            # Moving averages
            self.sma_20 = df['close'].rolling(20).mean()
            self.sma_50 = df['close'].rolling(50).mean()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.atr_14 = true_range.rolling(14).mean()
            
            # Volume MA
            self.volume_ma_20 = df['volume'].rolling(20).mean()
            
        except Exception as e:
            logger.warning(f"Erro ao calcular indicadores para {self.symbol}: {e}")
    
    def _calculate_statistics(self):
        """Calcula estatísticas básicas"""
        try:
            df = self.ohlcv
            
            self.total_candles = len(df)
            self.last_price = float(df['close'].iloc[-1]) if len(df) > 0 else 0.0
            self.avg_volume = float(df['volume'].mean()) if len(df) > 0 else 0.0
            
            # Volatilidade como % do preço
            if self.atr_14 is not None and len(self.atr_14) > 0:
                latest_atr = self.atr_14.iloc[-1]
                self.volatility_atr = (latest_atr / self.last_price * 100) if self.last_price > 0 else 0.0
            
            # Score de qualidade dos dados (0-100)
            self.data_quality_score = self._calculate_quality_score()
            
        except Exception as e:
            logger.warning(f"Erro ao calcular estatísticas para {self.symbol}: {e}")
    
    def _calculate_quality_score(self) -> float:
        """Calcula score de qualidade dos dados (0-100)"""
        try:
            df = self.ohlcv
            if df.empty:
                return 0.0
            
            score = 100.0
            
            # Penaliza dados faltantes
            missing_data_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            score -= missing_data_pct * 10  # -10 pontos por % de dados faltantes
            
            # Penaliza gaps temporais grandes
            if 'timestamp' in df.columns:
                time_diffs = df['timestamp'].diff().dt.total_seconds()
                expected_interval = self._get_expected_interval_seconds()
                large_gaps = (time_diffs > expected_interval * 2).sum()
                gap_penalty = min(large_gaps * 5, 20)  # Máximo -20 pontos
                score -= gap_penalty
            
            # Penaliza volumes zero
            zero_volume_pct = (df['volume'] == 0).sum() / len(df) * 100
            score -= zero_volume_pct * 5  # -5 pontos por % de volume zero
            
            # Penaliza preços inconsistentes (high < low, etc)
            inconsistent_prices = (df['high'] < df['low']).sum()
            if inconsistent_prices > 0:
                score -= inconsistent_prices * 10
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Erro ao calcular quality score: {e}")
            return 50.0  # Score neutro em caso de erro
    
    def _get_expected_interval_seconds(self) -> int:
        """Retorna intervalo esperado em segundos para o timeframe"""
        intervals = {
            '1m': 60,
            '5m': 300, 
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return intervals.get(self.timeframe, 3600)
    
    def is_sufficient_data(self, min_candles: int = None) -> bool:
        """Verifica se há dados suficientes para análise"""
        if min_candles is None:
            min_candles = settings.trading.TIMEFRAMES[self.timeframe].min_candles
        
        return (
            self.total_candles >= min_candles and
            self.data_quality_score >= 70.0 and
            self.last_price > 0
        )
    
    def get_latest_candle(self) -> Optional[Dict]:
        """Retorna o último candle como dicionário"""
        if self.ohlcv.empty:
            return None
        
        latest = self.ohlcv.iloc[-1]
        return {
            'timestamp': latest.get('timestamp'),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'close': float(latest['close']),
            'volume': float(latest['volume'])
        }

class DataProvider(ABC):
    """🌐 Interface abstrata para provedores de dados"""
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, 
                         limit: int = 500, since: Optional[datetime] = None) -> pd.DataFrame:
        """Busca dados OHLCV"""
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Busca preço mais recente"""
        pass
    
    @abstractmethod
    async def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """Busca estatísticas 24h"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica conexão"""
        pass

class BinanceDataProvider(DataProvider):
    """🔶 Provider de dados da Binance"""
    
    def __init__(self):
        self.client = None
        self.rate_limiter = {}
        self.last_request_time = {}
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa cliente Binance"""
        try:
            # Simularemos conexão - substituir por client real
            logger.info("🔶 Binance data provider inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Binance client: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_ohlcv(self, symbol: str, timeframe: str, 
                         limit: int = 500, since: Optional[datetime] = None) -> pd.DataFrame:
        """Busca dados OHLCV da Binance"""
        try:
            # Rate limiting
            await self._wait_for_rate_limit()
            
            logger.debug(f"📊 Buscando dados OHLCV: {symbol} {timeframe} (limit: {limit})")
            
            # TODO: Substituir por chamada real à API da Binance
            # Por agora, simula dados para desenvolvimento
            data = self._simulate_ohlcv_data(symbol, timeframe, limit, since)
            
            if data.empty:
                logger.warning(f"Nenhum dado retornado para {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Validação dos dados
            if not validate_ohlcv_data(data):
                logger.error(f"Dados inválidos recebidos para {symbol} {timeframe}")
                return pd.DataFrame()
            
            logger.debug(f"✅ Dados obtidos: {len(data)} candles para {symbol} {timeframe}")
            return data
            
        except Exception as e:
            logger.error(f"Erro ao buscar dados OHLCV {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Busca preço mais recente"""
        try:
            await self._wait_for_rate_limit()
            
            # TODO: Implementar chamada real
            # Por agora simula preço baseado no symbol
            base_prices = {
                'BTCUSDT': 50000.0,
                'ETHUSDT': 3000.0, 
                'BNBUSDT': 400.0,
                'ADAUSDT': 0.5,
                'SOLUSDT': 100.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            # Adiciona variação aleatória pequena
            import random
            variation = random.uniform(-0.02, 0.02)  # ±2%
            price = base_price * (1 + variation)
            
            return round(price, 4)
            
        except Exception as e:
            logger.error(f"Erro ao buscar preço de {symbol}: {e}")
            return None
    
    async def get_24h_stats(self, symbol: str) -> Optional[Dict]:
        """Busca estatísticas 24h"""
        try:
            await self._wait_for_rate_limit()
            
            # TODO: Implementar chamada real
            price = await self.get_latest_price(symbol)
            if not price:
                return None
            
            # Simula estatísticas 24h
            import random
            change_pct = random.uniform(-10.0, 10.0)
            
            return {
                'symbol': symbol,
                'price': price,
                'price_change_percent': change_pct,
                'high_24h': price * 1.05,
                'low_24h': price * 0.95,
                'volume_24h': random.uniform(1000000, 50000000),
                'count': random.randint(50000, 200000)
            }
            
        except Exception as e:
            logger.error(f"Erro ao buscar stats 24h de {symbol}: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Verifica conexão com Binance"""
        # TODO: Implementar verificação real
        return True
    
    async def _wait_for_rate_limit(self):
        """Rate limiting inteligente"""
        current_time = time.time()
        min_interval = 1.0 / settings.api.REQUESTS_PER_SECOND
        
        last_time = self.last_request_time.get('global', 0)
        time_since_last = current_time - last_time
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time['global'] = time.time()
    
    def _simulate_ohlcv_data(self, symbol: str, timeframe: str, 
                           limit: int, since: Optional[datetime]) -> pd.DataFrame:
        """Simula dados OHLCV para desenvolvimento"""
        try:
            # Calcula período baseado no timeframe
            interval_minutes = {
                '1m': 1, '5m': 5, '15m': 15, 
                '1h': 60, '4h': 240, '1d': 1440
            }
            
            minutes = interval_minutes.get(timeframe, 60)
            
            # Data inicial
            if since:
                start_time = since
            else:
                start_time = datetime.now() - timedelta(days=limit * minutes / 1440)
            
            # Gera timestamps
            timestamps = []
            current_time = start_time
            
            for i in range(limit):
                timestamps.append(current_time)
                current_time += timedelta(minutes=minutes)
            
            # Preço base por symbol
            base_prices = {
                'BTCUSDT': 50000.0,
                'ETHUSDT': 3000.0,
                'BNBUSDT': 400.0, 
                'ADAUSDT': 0.5,
                'SOLUSDT': 100.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Gera dados OHLCV realistas
            ohlcv_data = []
            current_price = base_price
            
            for timestamp in timestamps:
                # Variação aleatória
                daily_volatility = 0.03  # 3% volatilidade diária base
                interval_volatility = daily_volatility * np.sqrt(minutes / 1440)
                
                price_change = np.random.normal(0, interval_volatility)
                current_price *= (1 + price_change)
                
                # OHLC baseado no preço atual
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = current_price
                close_price = current_price
                
                # Volume baseado no symbol
                base_volume = {
                    'BTCUSDT': 1000,
                    'ETHUSDT': 800,
                    'BNBUSDT': 500,
                    'ADAUSDT': 10000,
                    'SOLUSDT': 2000
                }.get(symbol, 1000)
                
                volume = base_volume * np.random.uniform(0.5, 2.0)
                
                ohlcv_data.append({
                    'timestamp': timestamp,
                    'open': round(open_price, 4),
                    'high': round(high, 4),
                    'low': round(low, 4),
                    'close': round(close_price, 4),
                    'volume': round(volume, 2)
                })
            
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao simular dados para {symbol}: {e}")
            return pd.DataFrame()

class MarketDataManager:
    """🏗️ Gerenciador principal de dados de mercado"""
    
    def __init__(self):
        self.providers = {}
        self.cache = {}
        self.cache_timestamps = {}
        self.stats = {
            'requests_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors_count': 0
        }
        
        # Inicializa providers
        self._initialize_providers()
        
        logger.info("🏗️ MarketDataManager inicializado")
    
    def _initialize_providers(self):
        """Inicializa provedores de dados"""
        try:
            # Provider principal: Binance
            self.providers['binance'] = BinanceDataProvider()
            self.primary_provider = 'binance'
            
            logger.info(f"✅ Provider principal: {self.primary_provider}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar providers: {e}")
    
    async def get_market_data(self, symbol: str, timeframe: str, 
                            limit: int = None, use_cache: bool = True) -> Optional[MarketData]:
        """🎯 Método principal para obter dados de mercado"""
        try:
            self.stats['requests_count'] += 1
            
            # Configuração padrão baseada no timeframe
            if limit is None:
                tf_config = settings.get_timeframe_config(timeframe)
                limit = tf_config.min_candles * 2  # 2x para ter margem
            
            # Verifica cache primeiro
            if use_cache:
                cached_data = self._get_from_cache(symbol, timeframe)
                if cached_data:
                    self.stats['cache_hits'] += 1
                    logger.debug(f"📦 Cache hit: {symbol} {timeframe}")
                    return cached_data
            
            self.stats['cache_misses'] += 1
            
            # Busca dados do provider
            provider = self.providers.get(self.primary_provider)
            if not provider:
                logger.error("Nenhum provider disponível")
                return None
            
            logger.debug(f"🔍 Buscando dados: {symbol} {timeframe} (limit: {limit})")
            ohlcv_df = await provider.fetch_ohlcv(symbol, timeframe, limit)
            
            if ohlcv_df.empty:
                logger.warning(f"Dados vazios para {symbol} {timeframe}")
                return None
            
            # Cria objeto MarketData
            market_data = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                ohlcv=ohlcv_df,
                metadata={
                    'provider': self.primary_provider,
                    'limit': limit,
                    'fetched_at': datetime.now().isoformat()
                }
            )
            
            # Validação de qualidade
            if not market_data.is_sufficient_data():
                logger.warning(f"Dados insuficientes: {symbol} {timeframe} "
                             f"(candles: {market_data.total_candles}, "
                             f"quality: {market_data.data_quality_score:.1f})")
                return None
            
            # Salva no cache
            if use_cache:
                self._save_to_cache(symbol, timeframe, market_data)
            
            logger.debug(f"✅ Dados obtidos: {symbol} {timeframe} "
                        f"({market_data.total_candles} candles, "
                        f"quality: {market_data.data_quality_score:.1f})")
            
            return market_data
            
        except Exception as e:
            self.stats['errors_count'] += 1
            logger.error(f"Erro ao obter dados {symbol} {timeframe}: {e}")
            return None
    
    async def get_multiple_symbols_data(self, symbols: List[str], timeframe: str,
                                      limit: int = None) -> Dict[str, MarketData]:
        """📊 Busca dados para múltiplos símbolos em paralelo"""
        try:
            logger.info(f"📊 Buscando dados para {len(symbols)} símbolos ({timeframe})")
            
            tasks = []
            for symbol in symbols:
                task = self.get_market_data(symbol, timeframe, limit)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processa resultados
            market_data_dict = {}
            successful = 0
            
            for symbol, result in zip(symbols, results):
                if isinstance(result, MarketData):
                    market_data_dict[symbol] = result
                    successful += 1
                elif isinstance(result, Exception):
                    logger.error(f"Erro ao buscar {symbol}: {result}")
                else:
                    logger.warning(f"Dados inválidos para {symbol}")
            
            logger.info(f"✅ Dados obtidos: {successful}/{len(symbols)} símbolos")
            return market_data_dict
            
        except Exception as e:
            logger.error(f"Erro ao buscar múltiplos símbolos: {e}")
            return {}
    
    async def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """💰 Busca preços atuais para múltiplos símbolos"""
        try:
            provider = self.providers.get(self.primary_provider)
            if not provider:
                return {}
            
            tasks = [provider.get_latest_price(symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            prices = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, (int, float)) and result > 0:
                    prices[symbol] = float(result)
                elif isinstance(result, Exception):
                    logger.error(f"Erro ao buscar preço de {symbol}: {result}")
            
            return prices
            
        except Exception as e:
            logger.error(f"Erro ao buscar preços: {e}")
            return {}
    
    def _get_from_cache(self, symbol: str, timeframe: str) -> Optional[MarketData]:
        """Busca dados do cache"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            if cache_key not in self.cache:
                return None
            
            # Verifica se cache ainda é válido
            cache_time = self.cache_timestamps.get(cache_key)
            if not cache_time:
                return None
            
            age_minutes = (datetime.now() - cache_time).total_seconds() / 60
            if age_minutes > settings.database.CACHE_TTL_SECONDS / 60:
                # Cache expirado
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
                return None
            
            return self.cache[cache_key]
            
        except Exception as e:
            logger.warning(f"Erro ao acessar cache: {e}")
            return None
    
    def _save_to_cache(self, symbol: str, timeframe: str, market_data: MarketData):
        """Salva dados no cache"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            self.cache[cache_key] = market_data
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Limita tamanho do cache
            if len(self.cache) > 100:  # Máximo 100 entradas
                # Remove entrada mais antiga
                oldest_key = min(self.cache_timestamps, key=self.cache_timestamps.get)
                del self.cache[oldest_key]
                del self.cache_timestamps[oldest_key]
                
        except Exception as e:
            logger.warning(f"Erro ao salvar no cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Estatísticas do cache e manager"""
        total_requests = self.stats['requests_count']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'], 
            'cache_hit_rate_pct': round(hit_rate, 1),
            'errors_count': self.stats['errors_count'],
            'cache_size': len(self.cache),
            'providers_available': list(self.providers.keys()),
            'primary_provider': self.primary_provider
        }
    
    def clear_cache(self):
        """Limpa cache manualmente"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("🧹 Cache limpo manualmente")

# Instância global do manager
market_data_manager = MarketDataManager()

# Funções de conveniência
async def get_market_data(symbol: str, timeframe: str, limit: int = None) -> Optional[MarketData]:
    """Função de conveniência para buscar dados de mercado"""
    return await market_data_manager.get_market_data(symbol, timeframe, limit)

async def get_multiple_data(symbols: List[str], timeframe: str) -> Dict[str, MarketData]:
    """Função de conveniência para buscar múltiplos símbolos"""
    return await market_data_manager.get_multiple_symbols_data(symbols, timeframe)

async def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Função de conveniência para buscar preços atuais"""
    return await market_data_manager.get_latest_prices(symbols)