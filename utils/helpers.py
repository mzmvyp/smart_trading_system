"""
Utils: Helper Functions
Funções auxiliares matemáticas, formatação e utilitários gerais
"""
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from datetime import datetime, timedelta
import json
import hashlib
from functools import wraps
import time


# =============================================================================
# FUNÇÕES MATEMÁTICAS E ESTATÍSTICAS
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura evitando divisão por zero"""
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        result = numerator / denominator
        return default if np.isnan(result) or np.isinf(result) else result
    except:
        return default


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normaliza valor entre 0 e 1"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def denormalize_value(normalized: float, min_val: float, max_val: float) -> float:
    """Desnormaliza valor de 0-1 para range original"""
    return min_val + (normalized * (max_val - min_val))


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calcula mudança percentual"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Média ponderada"""
    if len(values) != len(weights) or sum(weights) == 0:
        return np.mean(values) if values else 0.0
    
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def calculate_volatility(prices: List[float], periods: int = 20) -> float:
    """Calcula volatilidade como desvio padrão dos retornos"""
    if len(prices) < 2:
        return 0.0
    
    returns = [calculate_percentage_change(prices[i-1], prices[i]) 
               for i in range(1, len(prices))]
    
    return np.std(returns[-periods:]) if len(returns) >= periods else np.std(returns)


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calcula Sharpe Ratio"""
    if not returns or np.std(returns) == 0:
        return 0.0
    
    excess_return = np.mean(returns) - risk_free_rate
    return excess_return / np.std(returns)


def calculate_max_drawdown(values: List[float]) -> Dict[str, float]:
    """Calcula máximo drawdown"""
    if not values:
        return {'max_drawdown': 0.0, 'drawdown_duration': 0}
    
    peak = values[0]
    max_dd = 0.0
    dd_duration = 0
    current_dd_duration = 0
    
    for value in values:
        if value > peak:
            peak = value
            current_dd_duration = 0
        else:
            current_dd_duration += 1
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                dd_duration = current_dd_duration
    
    return {
        'max_drawdown': max_dd * 100,  # Em percentual
        'drawdown_duration': dd_duration
    }


# =============================================================================
# FUNÇÕES DE FORMATAÇÃO E CONVERSÃO
# =============================================================================

def format_currency(value: float, decimals: int = 2, symbol: str = "$") -> str:
    """Formata valor como moeda"""
    return f"{symbol}{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata valor como percentual"""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 4) -> str:
    """Formata número com decimais específicos"""
    return f"{value:.{decimals}f}"


def round_to_precision(value: float, precision: int, round_up: bool = False) -> float:
    """Arredonda para precisão específica"""
    multiplier = 10 ** precision
    if round_up:
        return np.ceil(value * multiplier) / multiplier
    return np.floor(value * multiplier) / multiplier


def truncate_decimals(value: float, decimals: int) -> float:
    """Trunca casas decimais (não arredonda)"""
    factor = 10 ** decimals
    return int(value * factor) / factor


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Converte timeframe string para minutos"""
    timeframe = timeframe.lower()
    
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 1440
    else:
        raise ValueError(f"Timeframe inválido: {timeframe}")


def minutes_to_timeframe(minutes: int) -> str:
    """Converte minutos para string de timeframe"""
    if minutes < 60:
        return f"{minutes}m"
    elif minutes < 1440:
        return f"{minutes // 60}h"
    else:
        return f"{minutes // 1440}d"


# =============================================================================
# FUNÇÕES DE VALIDAÇÃO E SANITIZAÇÃO
# =============================================================================

def validate_price_data(df: pd.DataFrame) -> bool:
    """Valida estrutura de dados OHLCV"""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Verifica se high >= low, close/open entre high/low
    valid_ohlc = (
        (df['high'] >= df['low']) &
        (df['high'] >= df['open']) &
        (df['high'] >= df['close']) &
        (df['low'] <= df['open']) &
        (df['low'] <= df['close']) &
        (df['volume'] >= 0)
    )
    
    return valid_ohlc.all()


def sanitize_numeric_input(value: Any, default: float = 0.0) -> float:
    """Sanitiza entrada numérica"""
    try:
        if pd.isna(value) or value is None:
            return default
        
        float_val = float(value)
        return default if np.isnan(float_val) or np.isinf(float_val) else float_val
    except:
        return default


def validate_symbol(symbol: str) -> str:
    """Valida e padroniza símbolo de trading"""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Símbolo deve ser uma string válida")
    
    # Remove espaços e converte para uppercase
    symbol = symbol.strip().upper()
    
    if len(symbol) < 3:
        raise ValueError("Símbolo deve ter pelo menos 3 caracteres")
    
    return symbol


def validate_timeframe(timeframe: str) -> str:
    """Valida formato de timeframe"""
    timeframe = timeframe.lower().strip()
    
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w']
    
    if timeframe not in valid_timeframes:
        raise ValueError(f"Timeframe '{timeframe}' não é válido. Use: {valid_timeframes}")
    
    return timeframe


# =============================================================================
# FUNÇÕES DE CACHE E PERFORMANCE
# =============================================================================

def generate_cache_key(*args, **kwargs) -> str:
    """Gera chave única para cache"""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def timing_decorator(func):
    """Decorator para medir tempo de execução"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        print(f"{func.__name__} executado em {execution_time:.4f}s")
        
        return result
    return wrapper


def memoize(maxsize: int = 128):
    """Decorator simples de memoização"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = generate_cache_key(*args, **kwargs)
            
            if key in cache:
                return cache[key]
            
            result = func(*args, **kwargs)
            
            if len(cache) >= maxsize:
                # Remove item mais antigo
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            cache[key] = result
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: {'size': len(cache), 'maxsize': maxsize}
        
        return wrapper
    return decorator


# =============================================================================
# FUNÇÕES DE DATA E TEMPO
# =============================================================================

def timestamp_to_datetime(timestamp: Union[int, float]) -> datetime:
    """Converte timestamp para datetime"""
    # Binance usa milliseconds
    if timestamp > 1e12:
        timestamp = timestamp / 1000
    
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: datetime) -> int:
    """Converte datetime para timestamp (milliseconds)"""
    return int(dt.timestamp() * 1000)


def get_market_session(dt: datetime) -> str:
    """Identifica sessão de mercado baseada na hora UTC"""
    hour = dt.hour
    
    if 21 <= hour or hour < 6:
        return "sydney"
    elif 6 <= hour < 14:
        return "london"  
    elif 14 <= hour < 21:
        return "new_york"
    else:
        return "unknown"


def is_market_hours(dt: datetime) -> bool:
    """Verifica se está em horário de alta liquidez"""
    hour = dt.hour
    # Londres + NY overlap (12-17 UTC) ou NY + Sydney overlap (21-24 UTC)
    return (12 <= hour < 17) or (21 <= hour < 24)


# =============================================================================
# FUNÇÕES DE TRADING ESPECÍFICAS
# =============================================================================

def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    leverage: float = 1.0
) -> float:
    """Calcula tamanho da posição baseado no risco"""
    
    if entry_price <= 0 or stop_loss <= 0 or account_balance <= 0:
        return 0.0
    
    # Risco máximo em valor absoluto
    max_risk_amount = account_balance * (risk_per_trade / 100)
    
    # Risco por unidade
    risk_per_unit = abs(entry_price - stop_loss)
    
    if risk_per_unit == 0:
        return 0.0
    
    # Tamanho base da posição
    position_size = max_risk_amount / risk_per_unit
    
    # Aplica alavancagem
    return position_size * leverage


def calculate_pip_value(symbol: str, lot_size: float, exchange_rate: float = 1.0) -> float:
    """Calcula valor do pip para um símbolo"""
    # Para crypto, geralmente 1 pip = 0.0001 do preço
    if 'BTC' in symbol.upper():
        return lot_size * 0.01  # Bitcoin: $0.01 por pip
    elif 'ETH' in symbol.upper():
        return lot_size * 0.001  # Ethereum: $0.001 por pip
    else:
        return lot_size * 0.0001  # Padrão: $0.0001 por pip


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calcula relação risco/recompensa"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    return safe_divide(reward, risk, 0.0)


def price_to_percentage_distance(price1: float, price2: float) -> float:
    """Calcula distância percentual entre dois preços"""
    return abs(calculate_percentage_change(price1, price2))


# =============================================================================
# FUNÇÕES DE ANÁLISE TÉCNICA AUXILIARES
# =============================================================================

def find_local_extremes(prices: List[float], window: int = 5) -> Dict[str, List[int]]:
    """Encontra máximos e mínimos locais"""
    if len(prices) < window * 2 + 1:
        return {'highs': [], 'lows': []}
    
    highs = []
    lows = []
    
    for i in range(window, len(prices) - window):
        # Verifica se é máximo local
        if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1)):
            if prices[i] > max(prices[i - window:i] + prices[i + 1:i + window + 1]):
                highs.append(i)
        
        # Verifica se é mínimo local
        if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1)):
            if prices[i] < min(prices[i - window:i] + prices[i + 1:i + window + 1]):
                lows.append(i)
    
    return {'highs': highs, 'lows': lows}


def calculate_support_resistance_levels(
    prices: List[float], 
    method: str = 'pivot',
    sensitivity: float = 0.02
) -> Dict[str, List[float]]:
    """Calcula níveis de suporte e resistência"""
    
    if method == 'pivot':
        extremes = find_local_extremes(prices)
        
        resistance_levels = [prices[i] for i in extremes['highs']]
        support_levels = [prices[i] for i in extremes['lows']]
        
        # Remove níveis muito próximos
        resistance_levels = remove_nearby_levels(resistance_levels, sensitivity)
        support_levels = remove_nearby_levels(support_levels, sensitivity)
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels)
        }
    
    return {'resistance': [], 'support': []}


def remove_nearby_levels(levels: List[float], threshold: float = 0.02) -> List[float]:
    """Remove níveis muito próximos entre si"""
    if not levels:
        return []
    
    cleaned_levels = [levels[0]]
    
    for level in levels[1:]:
        # Verifica se está suficientemente distante dos níveis existentes
        if all(price_to_percentage_distance(level, existing) > threshold * 100 
               for existing in cleaned_levels):
            cleaned_levels.append(level)
    
    return cleaned_levels


# =============================================================================
# CONSTANTES ÚTEIS
# =============================================================================

CRYPTO_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
    'BNBUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT'
]

TIMEFRAME_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '6h': 360,
    '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080
}

MARKET_SESSIONS = {
    'sydney': {'start': 21, 'end': 6},
    'tokyo': {'start': 23, 'end': 8},
    'london': {'start': 7, 'end': 16},
    'new_york': {'start': 12, 'end': 21}
}