"""
API: Data Provider
Provedor abstrato de dados para múltiplas exchanges
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import time
import json

from .binance_client import BinanceClient
from utils.logger import get_logger
from utils.helpers import (
    validate_symbol,
    validate_timeframe,
    parse_timeframe_to_minutes,
    sanitize_numeric_input
)


logger = get_logger(__name__)


class Exchange(Enum):
    """Exchanges suportadas"""
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    # Adicionar outras exchanges conforme necessário


@dataclass
class DataProviderConfig:
    """Configurações do provedor de dados"""
    default_exchange: Exchange = Exchange.BINANCE
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30
    rate_limit: int = 1000  # Requests por minuto
    cache_enabled: bool = True
    cache_ttl: int = 300    # TTL em segundos
    
    # Configurações específicas por exchange
    exchange_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.exchange_configs is None:
            self.exchange_configs = {
                'binance': {
                    'base_url': 'https://api.binance.com',
                    'weight_limit': 1200,
                    'order_limit': 10
                },
                'binance_us': {
                    'base_url': 'https://api.binance.us',
                    'weight_limit': 1200,
                    'order_limit': 10
                }
            }


class BaseDataProvider(ABC):
    """Classe base abstrata para provedores de dados"""
    
    def __init__(self, config: DataProviderConfig = None):
        self.config = config or DataProviderConfig()
        self.cache = {}
        self.last_requests = []  # Para rate limiting
        
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtém dados históricos OHLCV"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Obtém preço atual de um símbolo"""
        pass
    
    @abstractmethod
    async def get_ticker_info(self, symbol: str) -> Dict:
        """Obtém informações do ticker (24h stats)"""
        pass
    
    @abstractmethod
    async def get_symbols_info(self) -> List[Dict]:
        """Obtém informações de todos os símbolos disponíveis"""
        pass
    
    @abstractmethod
    def validate_symbol(self, symbol: str) -> bool:
        """Valida se símbolo existe na exchange"""
        pass
    
    def _check_rate_limit(self):
        """Verifica rate limiting"""
        now = time.time()
        
        # Remove requests antigas (mais de 1 minuto)
        self.last_requests = [
            req_time for req_time in self.last_requests
            if now - req_time < 60
        ]
        
        # Verifica se excedeu o limite
        if len(self.last_requests) >= self.config.rate_limit:
            sleep_time = 60 - (now - self.last_requests[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit atingido, aguardando {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.last_requests.append(now)
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Gera chave de cache"""
        key_data = f"{method}_{json.dumps(kwargs, sort_keys=True, default=str)}"
        return key_data
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Obtém dados do cache se válidos"""
        if not self.config.cache_enabled:
            return None
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            
            # Verifica TTL
            if time.time() - timestamp < self.config.cache_ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Salva dados no cache"""
        if self.config.cache_enabled:
            self.cache[cache_key] = (data, time.time())


class BinanceDataProvider(BaseDataProvider):
    """Provedor de dados para Binance"""
    
    def __init__(self, config: DataProviderConfig = None, api_key: str = None, api_secret: str = None):
        super().__init__(config)
        self.client = BinanceClient(api_key, api_secret)
        self.exchange_info = None
        
    async def initialize(self):
        """Inicializa o provedor carregando informações da exchange"""
        try:
            self.exchange_info = await self.get_symbols_info()
            logger.info(f"Binance provider inicializado com {len(self.exchange_info)} símbolos")
        except Exception as e:
            logger.error(f"Erro ao inicializar Binance provider: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtém dados históricos da Binance"""
        
        try:
            # Valida parâmetros
            symbol = validate_symbol(symbol)
            timeframe = validate_timeframe(timeframe)
            
            # Verifica cache
            cache_key = self._get_cache_key(
                'historical_data',
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Rate limiting
            self._check_rate_limit()
            
            # Chama API da Binance
            klines = await self.client.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                start_str=start_date.isoformat() if start_date else None,
                end_str=end_date.isoformat() if end_date else None,
                limit=limit
            )
            
            if not klines:
                return pd.DataFrame()
            
            # Converte para DataFrame
            df = self._klines_to_dataframe(klines)
            
            # Salva no cache
            self._save_to_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_current_price(self, symbol: str) -> float:
        """Obtém preço atual da Binance"""
        
        try:
            symbol = validate_symbol(symbol)
            
            # Verifica cache
            cache_key = self._get_cache_key('current_price', symbol=symbol)
            cached_price = self._get_from_cache(cache_key)
            if cached_price is not None:
                return cached_price
            
            # Rate limiting
            self._check_rate_limit()
            
            # Chama API
            ticker = await self.client.get_symbol_ticker(symbol)
            price = sanitize_numeric_input(ticker.get('price', 0))
            
            # Cache com TTL menor para preços
            if self.config.cache_enabled:
                self.cache[cache_key] = (price, time.time())
            
            return price
            
        except Exception as e:
            logger.error(f"Erro ao obter preço atual {symbol}: {e}")
            return 0.0
    
    async def get_ticker_info(self, symbol: str) -> Dict:
        """Obtém informações do ticker 24h"""
        
        try:
            symbol = validate_symbol(symbol)
            
            cache_key = self._get_cache_key('ticker_info', symbol=symbol)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            self._check_rate_limit()
            
            ticker = await self.client.get_ticker_24hr(symbol)
            
            # Padroniza dados
            ticker_info = {
                'symbol': ticker.get('symbol'),
                'price': sanitize_numeric_input(ticker.get('lastPrice', 0)),
                'price_change': sanitize_numeric_input(ticker.get('priceChange', 0)),
                'price_change_percent': sanitize_numeric_input(ticker.get('priceChangePercent', 0)),
                'high_24h': sanitize_numeric_input(ticker.get('highPrice', 0)),
                'low_24h': sanitize_numeric_input(ticker.get('lowPrice', 0)),
                'volume_24h': sanitize_numeric_input(ticker.get('volume', 0)),
                'quote_volume_24h': sanitize_numeric_input(ticker.get('quoteVolume', 0)),
                'open_price': sanitize_numeric_input(ticker.get('openPrice', 0)),
                'prev_close': sanitize_numeric_input(ticker.get('prevClosePrice', 0)),
                'bid_price': sanitize_numeric_input(ticker.get('bidPrice', 0)),
                'ask_price': sanitize_numeric_input(ticker.get('askPrice', 0)),
                'timestamp': datetime.now()
            }
            
            self._save_to_cache(cache_key, ticker_info)
            
            return ticker_info
            
        except Exception as e:
            logger.error(f"Erro ao obter ticker info {symbol}: {e}")
            return {}
    
    async def get_symbols_info(self) -> List[Dict]:
        """Obtém informações de todos os símbolos"""
        
        try:
            cache_key = self._get_cache_key('symbols_info')
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            self._check_rate_limit()
            
            exchange_info = await self.client.get_exchange_info()
            symbols_data = []
            
            for symbol_info in exchange_info.get('symbols', []):
                if symbol_info.get('status') == 'TRADING':
                    symbols_data.append({
                        'symbol': symbol_info.get('symbol'),
                        'base_asset': symbol_info.get('baseAsset'),
                        'quote_asset': symbol_info.get('quoteAsset'),
                        'status': symbol_info.get('status'),
                        'is_spot_trading_allowed': symbol_info.get('isSpotTradingAllowed', False),
                        'is_margin_trading_allowed': symbol_info.get('isMarginTradingAllowed', False),
                        'permissions': symbol_info.get('permissions', [])
                    })
            
            # Cache com TTL maior (info de símbolos muda raramente)
            if self.config.cache_enabled:
                self.cache[cache_key] = (symbols_data, time.time())
            
            return symbols_data
            
        except Exception as e:
            logger.error(f"Erro ao obter informações dos símbolos: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Valida se símbolo existe na Binance"""
        
        if not self.exchange_info:
            return False
        
        symbol = symbol.upper()
        
        for symbol_info in self.exchange_info:
            if symbol_info['symbol'] == symbol:
                return symbol_info['status'] == 'TRADING'
        
        return False
    
    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """Converte dados de klines para DataFrame"""
        
        try:
            # Colunas da Binance klines API
            columns = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            
            # Converte tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'number_of_trades',
                             'taker_buy_base_volume', 'taker_buy_quote_volume']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Define timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            # Mantém apenas colunas essenciais
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Remove dados inválidos
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao converter klines: {e}")
            return pd.DataFrame()


class MultiExchangeDataProvider:
    """Provedor que agrega múltiplas exchanges"""
    
    def __init__(self, config: DataProviderConfig = None):
        self.config = config or DataProviderConfig()
        self.providers: Dict[Exchange, BaseDataProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa provedores das exchanges"""
        
        try:
            # Binance
            binance_config = DataProviderConfig(
                default_exchange=Exchange.BINANCE,
                **self.config.exchange_configs.get('binance', {})
            )
            self.providers[Exchange.BINANCE] = BinanceDataProvider(binance_config)
            
            logger.info(f"Inicializados {len(self.providers)} provedores")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar provedores: {e}")
    
    async def initialize_all(self):
        """Inicializa todos os provedores"""
        
        for exchange, provider in self.providers.items():
            try:
                if hasattr(provider, 'initialize'):
                    await provider.initialize()
                logger.info(f"Provider {exchange.value} inicializado")
            except Exception as e:
                logger.error(f"Erro ao inicializar {exchange.value}: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        exchange: Exchange = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Obtém dados históricos de uma exchange específica"""
        
        if exchange is None:
            exchange = self.config.default_exchange
        
        if exchange not in self.providers:
            logger.error(f"Exchange {exchange.value} não disponível")
            return pd.DataFrame()
        
        provider = self.providers[exchange]
        return await provider.get_historical_data(
            symbol, timeframe, start_date, end_date, limit
        )
    
    async def get_current_prices(
        self,
        symbols: List[str],
        exchange: Exchange = None
    ) -> Dict[str, float]:
        """Obtém preços atuais de múltiplos símbolos"""
        
        if exchange is None:
            exchange = self.config.default_exchange
        
        if exchange not in self.providers:
            return {}
        
        provider = self.providers[exchange]
        prices = {}
        
        # Obtém preços em paralelo
        tasks = [provider.get_current_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(symbols, results):
            if not isinstance(result, Exception):
                prices[symbol] = result
            else:
                logger.error(f"Erro ao obter preço {symbol}: {result}")
                prices[symbol] = 0.0
        
        return prices
    
    async def get_market_data_batch(
        self,
        symbols: List[str],
        timeframe: str,
        exchange: Exchange = None,
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Obtém dados históricos para múltiplos símbolos"""
        
        if exchange is None:
            exchange = self.config.default_exchange
        
        if exchange not in self.providers:
            return {}
        
        provider = self.providers[exchange]
        market_data = {}
        
        # Obtém dados em paralelo (com limite para não sobrecarregar API)
        batch_size = 5  # Processa 5 símbolos por vez
        
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            tasks = [
                provider.get_historical_data(symbol, timeframe, limit=limit)
                for symbol in batch_symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch_symbols, results):
                if not isinstance(result, Exception) and not result.empty:
                    market_data[symbol] = result
                else:
                    logger.error(f"Erro ao obter dados {symbol}: {result}")
            
            # Pausa entre batches para respeitar rate limits
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        return market_data
    
    async def get_top_symbols(
        self,
        quote_asset: str = 'USDT',
        limit: int = 50,
        sort_by: str = 'volume',
        exchange: Exchange = None
    ) -> List[str]:
        """Obtém top símbolos por volume ou outros critérios"""
        
        if exchange is None:
            exchange = self.config.default_exchange
        
        if exchange not in self.providers:
            return []
        
        try:
            provider = self.providers[exchange]
            symbols_info = await provider.get_symbols_info()
            
            # Filtra por quote asset
            filtered_symbols = [
                info['symbol'] for info in symbols_info
                if info.get('quote_asset') == quote_asset
            ]
            
            if sort_by == 'volume':
                # Obtém informações de ticker para ordenar por volume
                ticker_tasks = [
                    provider.get_ticker_info(symbol)
                    for symbol in filtered_symbols[:limit * 2]  # Obtém mais para filtrar
                ]
                
                ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
                
                # Ordena por volume
                valid_tickers = []
                for symbol, ticker in zip(filtered_symbols[:limit * 2], ticker_results):
                    if not isinstance(ticker, Exception) and ticker:
                        ticker['symbol'] = symbol
                        valid_tickers.append(ticker)
                
                valid_tickers.sort(key=lambda x: x.get('volume_24h', 0), reverse=True)
                
                return [ticker['symbol'] for ticker in valid_tickers[:limit]]
            
            else:
                # Retorna primeiros símbolos se não há critério de ordenação específico
                return filtered_symbols[:limit]
            
        except Exception as e:
            logger.error(f"Erro ao obter top símbolos: {e}")
            return []
    
    def get_available_exchanges(self) -> List[Exchange]:
        """Retorna exchanges disponíveis"""
        return list(self.providers.keys())
    
    def is_symbol_available(self, symbol: str, exchange: Exchange = None) -> bool:
        """Verifica se símbolo está disponível"""
        
        if exchange is None:
            exchange = self.config.default_exchange
        
        if exchange not in self.providers:
            return False
        
        provider = self.providers[exchange]
        return provider.validate_symbol(symbol)


# Instância global do data provider
data_provider = MultiExchangeDataProvider()


# Funções de conveniência
async def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    exchange: str = 'binance'
) -> pd.DataFrame:
    """Função de conveniência para obter dados de mercado"""
    
    exchange_enum = Exchange(exchange.lower())
    return await data_provider.get_historical_data(
        symbol, timeframe, exchange_enum, limit=limit
    )


async def get_current_price(symbol: str, exchange: str = 'binance') -> float:
    """Função de conveniência para obter preço atual"""
    
    exchange_enum = Exchange(exchange.lower())
    if exchange_enum not in data_provider.providers:
        return 0.0
    
    provider = data_provider.providers[exchange_enum]
    return await provider.get_current_price(symbol)


async def get_multiple_prices(symbols: List[str], exchange: str = 'binance') -> Dict[str, float]:
    """Função de conveniência para obter múltiplos preços"""
    
    exchange_enum = Exchange(exchange.lower())
    return await data_provider.get_current_prices(symbols, exchange_enum)