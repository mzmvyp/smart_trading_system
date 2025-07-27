"""
🔗 BINANCE CLIENT - Smart Trading System v2.0

Cliente robusto para Binance API com:
- Rate limiting inteligente
- Retry logic avançado
- Error handling completo
- WebSocket streams
- Order management

Filosofia: Robust Connection = Reliable Data = Better Decisions
"""

import time
import hmac
import hashlib
import requests
import json
import threading
import websocket
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging
from urllib.parse import urlencode
import queue

logger = logging.getLogger(__name__)


@dataclass
class BinanceConfig:
    """Configuração do cliente Binance"""
    api_key: str
    api_secret: str
    base_url: str = "https://api.binance.com"
    testnet: bool = False
    rate_limit_buffer: float = 0.1  # Buffer de 10% no rate limit
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


class RateLimiter:
    """Rate limiter inteligente para Binance API"""
    
    def __init__(self, requests_per_minute: int = 1200, weight_limit: int = 6000):
        self.requests_per_minute = requests_per_minute
        self.weight_limit = weight_limit
        self.request_timestamps = []
        self.weight_usage = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self, weight: int = 1):
        """Espera se necessário para respeitar rate limits"""
        with self.lock:
            now = time.time()
            
            # Limpar timestamps antigos (> 1 minuto)
            minute_ago = now - 60
            self.request_timestamps = [ts for ts in self.request_timestamps if ts > minute_ago]
            self.weight_usage = [w for w, ts in zip(self.weight_usage, self.request_timestamps) if ts > minute_ago]
            
            # Verificar limites
            current_requests = len(self.request_timestamps)
            current_weight = sum(self.weight_usage)
            
            # Calcular espera necessária
            wait_time = 0
            
            if current_requests >= self.requests_per_minute * 0.9:  # 90% do limite
                wait_time = max(wait_time, 60 - (now - min(self.request_timestamps)))
            
            if current_weight + weight >= self.weight_limit * 0.9:  # 90% do limite
                wait_time = max(wait_time, 60 - (now - min(self.request_timestamps)))
            
            if wait_time > 0:
                logger.warning(f"Rate limit approached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                now = time.time()
            
            # Registrar nova requisição
            self.request_timestamps.append(now)
            self.weight_usage.append(weight)


class BinanceClient:
    """
    🔗 Cliente Principal Binance
    
    Gerencia todas as interações com a API Binance:
    - Market data (OHLCV, orderbook, trades)
    - Account management
    - Order placement/management
    - WebSocket streams
    """
    
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.session = requests.Session()
        self.rate_limiter = RateLimiter()
        
        # Headers padrão
        self.session.headers.update({
            'X-MBX-APIKEY': config.api_key,
            'Content-Type': 'application/json'
        })
        
        # WebSocket streams
        self.ws_connections = {}
        self.ws_callbacks = {}
        
        self.logger = logging.getLogger(f"{__name__}.BinanceClient")
        
        # Testar conectividade
        self._test_connectivity()
    
    def _test_connectivity(self):
        """Testa conectividade com Binance"""
        try:
            response = self._make_request('GET', '/api/v3/ping')
            if response.status_code == 200:
                self.logger.info("✅ Conectado à Binance API")
            else:
                raise Exception(f"Ping failed: {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ Falha na conectividade: {e}")
            raise
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Assina requisição para endpoints privados"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                     signed: bool = False, weight: int = 1) -> requests.Response:
        """Faz requisição HTTP com retry logic"""
        if params is None:
            params = {}
        
        # Rate limiting
        self.rate_limiter.wait_if_needed(weight)
        
        # Adicionar timestamp para endpoints assinados
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign_request(params)
        
        url = f"{self.config.base_url}{endpoint}"
        
        # Retry logic
        for attempt in range(self.config.max_retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=self.config.timeout)
                elif method == 'POST':
                    response = self.session.post(url, json=params, timeout=self.config.timeout)
                elif method == 'DELETE':
                    response = self.session.delete(url, params=params, timeout=self.config.timeout)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                # Verificar rate limit headers
                if 'X-MBX-USED-WEIGHT-1M' in response.headers:
                    weight_used = int(response.headers['X-MBX-USED-WEIGHT-1M'])
                    if weight_used > 5000:  # 83% do limite
                        self.logger.warning(f"High API weight usage: {weight_used}/6000")
                
                # Verificar erro de rate limit
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue
                
                # Verificar outros erros
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    error_code = error_data.get('code', response.status_code)
                    error_msg = error_data.get('msg', response.text)
                    
                    if error_code in [-1021, -1022]:  # Timestamp errors
                        self.logger.warning("Timestamp error, adjusting...")
                        time.sleep(1)
                        continue
                    
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    
                    raise Exception(f"API Error {error_code}: {error_msg}")
                
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(f"Network error (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                raise
        
        raise Exception(f"Max retries exceeded for {method} {endpoint}")
    
    # =============================================================================
    # MARKET DATA METHODS
    # =============================================================================
    
    def get_server_time(self) -> int:
        """Retorna timestamp do servidor"""
        response = self._make_request('GET', '/api/v3/time')
        return response.json()['serverTime']
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """Retorna informações da exchange"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('GET', '/api/v3/exchangeInfo', params)
        return response.json()
    
    def get_ticker_24hr(self, symbol: Optional[str] = None) -> Dict:
        """Retorna ticker 24h"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('GET', '/api/v3/ticker/24hr', params, weight=1 if symbol else 40)
        return response.json()
    
    def get_ticker_price(self, symbol: Optional[str] = None) -> Dict:
        """Retorna preço atual"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        response = self._make_request('GET', '/api/v3/ticker/price', params, weight=1 if symbol else 2)
        return response.json()
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Retorna orderbook"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        weight = 1 if limit <= 100 else 5 if limit <= 500 else 10
        response = self._make_request('GET', '/api/v3/depth', params, weight=weight)
        return response.json()
    
    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Retorna trades recentes"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = self._make_request('GET', '/api/v3/trades', params, weight=1)
        return response.json()
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                   start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[List]:
        """
        Retorna dados OHLCV (candlesticks)
        
        Args:
            symbol: Par de trading (ex: BTCUSDT)
            interval: Intervalo (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Número de candles (max 1000)
            start_time: Timestamp de início
            end_time: Timestamp de fim
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        response = self._make_request('GET', '/api/v3/klines', params, weight=1)
        return response.json()
    
    def get_klines_dataframe(self, symbol: str, interval: str, limit: int = 500,
                           start_time: Optional[int] = None, end_time: Optional[int] = None) -> pd.DataFrame:
        """Retorna klines como DataFrame pandas"""
        klines = self.get_klines(symbol, interval, limit, start_time, end_time)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Converter tipos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df[numeric_columns]
    
    # =============================================================================
    # ACCOUNT METHODS (PRIVATE)
    # =============================================================================
    
    def get_account_info(self) -> Dict:
        """Retorna informações da conta"""
        response = self._make_request('GET', '/api/v3/account', signed=True, weight=10)
        return response.json()
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Retorna ordens abertas"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        weight = 3 if symbol else 40
        response = self._make_request('GET', '/api/v3/openOrders', params, signed=True, weight=weight)
        return response.json()
    
    def get_all_orders(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Retorna histórico de ordens"""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        response = self._make_request('GET', '/api/v3/allOrders', params, signed=True, weight=10)
        return response.json()
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                   price: Optional[float] = None, time_in_force: str = 'GTC',
                   stop_price: Optional[float] = None) -> Dict:
        """
        Coloca ordem
        
        Args:
            symbol: Par de trading
            side: BUY ou SELL
            order_type: LIMIT, MARKET, STOP_LOSS, etc.
            quantity: Quantidade
            price: Preço (para ordens LIMIT)
            time_in_force: GTC, IOC, FOK
            stop_price: Preço de stop (para stop orders)
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if price:
            params['price'] = price
        if order_type in ['LIMIT', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT']:
            params['timeInForce'] = time_in_force
        if stop_price:
            params['stopPrice'] = stop_price
        
        response = self._make_request('POST', '/api/v3/order', params, signed=True, weight=1)
        return response.json()
    
    def cancel_order(self, symbol: str, order_id: Optional[int] = None, 
                    orig_client_order_id: Optional[str] = None) -> Dict:
        """Cancela ordem"""
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise ValueError("Must provide either order_id or orig_client_order_id")
        
        response = self._make_request('DELETE', '/api/v3/order', params, signed=True, weight=1)
        return response.json()
    
    def cancel_all_orders(self, symbol: str) -> List[Dict]:
        """Cancela todas as ordens de um símbolo"""
        params = {'symbol': symbol}
        response = self._make_request('DELETE', '/api/v3/openOrders', params, signed=True, weight=1)
        return response.json()
    
    # =============================================================================
    # WEBSOCKET METHODS
    # =============================================================================
    
    def start_ticker_stream(self, symbol: str, callback: Callable[[Dict], None]):
        """Inicia stream de ticker"""
        stream_name = f"{symbol.lower()}@ticker"
        self._start_websocket_stream(stream_name, callback)
    
    def start_kline_stream(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """Inicia stream de klines"""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self._start_websocket_stream(stream_name, callback)
    
    def start_depth_stream(self, symbol: str, callback: Callable[[Dict], None]):
        """Inicia stream de orderbook"""
        stream_name = f"{symbol.lower()}@depth"
        self._start_websocket_stream(stream_name, callback)
    
    def _start_websocket_stream(self, stream_name: str, callback: Callable[[Dict], None]):
        """Inicia stream WebSocket genérico"""
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                callback(data)
            except Exception as e:
                self.logger.error(f"Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error for {stream_name}: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"WebSocket closed for {stream_name}")
        
        def on_open(ws):
            self.logger.info(f"WebSocket opened for {stream_name}")
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Executar em thread separada
        def run_websocket():
            ws.run_forever()
        
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        self.ws_connections[stream_name] = ws
        self.ws_callbacks[stream_name] = callback
        
        return ws
    
    def stop_stream(self, stream_name: str):
        """Para stream WebSocket"""
        if stream_name in self.ws_connections:
            self.ws_connections[stream_name].close()
            del self.ws_connections[stream_name]
            del self.ws_callbacks[stream_name]
    
    def stop_all_streams(self):
        """Para todos os streams WebSocket"""
        for stream_name in list(self.ws_connections.keys()):
            self.stop_stream(stream_name)
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Retorna informações específicas de um símbolo"""
        exchange_info = self.get_exchange_info()
        
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                return symbol_info
        
        raise ValueError(f"Symbol {symbol} not found")
    
    def format_quantity(self, symbol: str, quantity: float) -> float:
        """Formata quantidade respeitando step size"""
        symbol_info = self.get_symbol_info(symbol)
        
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                step_size = float(filter_info['stepSize'])
                # Arredondar para o step size mais próximo
                return round(quantity / step_size) * step_size
        
        return quantity
    
    def format_price(self, symbol: str, price: float) -> float:
        """Formata preço respeitando tick size"""
        symbol_info = self.get_symbol_info(symbol)
        
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'PRICE_FILTER':
                tick_size = float(filter_info['tickSize'])
                # Arredondar para o tick size mais próximo
                return round(price / tick_size) * tick_size
        
        return price
    
    def get_balance(self, asset: str) -> Dict:
        """Retorna saldo de um ativo específico"""
        account_info = self.get_account_info()
        
        for balance in account_info['balances']:
            if balance['asset'] == asset:
                return {
                    'asset': balance['asset'],
                    'free': float(balance['free']),
                    'locked': float(balance['locked']),
                    'total': float(balance['free']) + float(balance['locked'])
                }
        
        return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}
    
    def __del__(self):
        """Cleanup ao destruir objeto"""
        try:
            self.stop_all_streams()
        except:
            pass


def main():
    """Teste básico do cliente Binance"""
    # Configuração de teste (use suas próprias chaves)
    config = BinanceConfig(
        api_key="your_api_key_here",
        api_secret="your_secret_here",
        testnet=True  # Usar testnet para testes
    )
    
    try:
        # Inicializar cliente
        client = BinanceClient(config)
        
        # Testar market data
        print("🔗 Testing Binance Client")
        
        # Server time
        server_time = client.get_server_time()
        print(f"Server time: {datetime.fromtimestamp(server_time/1000)}")
        
        # Ticker
        ticker = client.get_ticker_24hr("BTCUSDT")
        print(f"BTC/USDT: ${float(ticker['lastPrice']):,.2f}")
        
        # Klines
        df = client.get_klines_dataframe("BTCUSDT", "1h", limit=10)
        print(f"OHLCV data shape: {df.shape}")
        print(f"Latest close: ${df['close'].iloc[-1]:,.2f}")
        
        # Test WebSocket (comentado para não manter conexão)
        """
        def on_ticker_update(data):
            print(f"Ticker update: {data['s']} = ${float(data['c']):,.2f}")
        
        client.start_ticker_stream("BTCUSDT", on_ticker_update)
        time.sleep(10)  # Listen for 10 seconds
        client.stop_all_streams()
        """
        
        print("✅ Binance client test completed successfully")
        
    except Exception as e:
        print(f"❌ Error testing Binance client: {e}")


if __name__ == "__main__":
    main()