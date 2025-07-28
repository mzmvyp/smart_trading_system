# config/symbols.py
"""
💰 CONFIGURAÇÕES AVANÇADAS DE SÍMBOLOS
Lista de criptomoedas com configurações específicas para trading inteligente
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class MarketCap(Enum):
    """Classificação por market cap"""
    LARGE_CAP = "large_cap"      # > $10B
    MID_CAP = "mid_cap"          # $1B - $10B  
    SMALL_CAP = "small_cap"      # $100M - $1B
    MICRO_CAP = "micro_cap"      # < $100M

class Volatility(Enum):
    """Classificação por volatilidade"""
    LOW = "low"          # < 3% daily ATR
    MEDIUM = "medium"    # 3-8% daily ATR
    HIGH = "high"        # 8-15% daily ATR
    EXTREME = "extreme"  # > 15% daily ATR

@dataclass
class SymbolConfig:
    """Configuração completa de um símbolo"""
    symbol: str
    base_asset: str
    quote_asset: str
    
    # Classificações
    market_cap: MarketCap
    volatility: Volatility
    sector: str
    
    # Trading parameters
    min_volume_usd: float
    min_price: float
    max_price: float
    price_precision: int
    quantity_precision: int
    
    # Risk parameters específicos
    max_position_risk: float    # Override do global se necessário
    volatility_multiplier: float  # Ajuste no position sizing
    correlation_group: str      # Grupo de correlação
    
    # Filtros específicos
    min_atr_percent: float
    max_atr_percent: float
    min_daily_volume: float
    
    # Strategy preferences (weight por estratégia)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    is_active: bool = True
    last_updated: str = ""
    notes: str = ""

# === CONFIGURAÇÕES PRINCIPAIS ===

# Top cryptocurrencies com configurações otimizadas
CRYPTO_SYMBOLS: Dict[str, SymbolConfig] = {
    
    # === LARGE CAP - TIER 1 ===
    "BTCUSDT": SymbolConfig(
        symbol="BTCUSDT",
        base_asset="BTC",
        quote_asset="USDT",
        market_cap=MarketCap.LARGE_CAP,
        volatility=Volatility.MEDIUM,
        sector="digital_gold",
        min_volume_usd=50_000_000,  # $50M min volume
        min_price=20_000,
        max_price=200_000,
        price_precision=2,
        quantity_precision=5,
        max_position_risk=0.025,    # 2.5% max (maior que default)
        volatility_multiplier=1.0,
        correlation_group="btc_dominance",
        min_atr_percent=2.0,
        max_atr_percent=12.0,
        min_daily_volume=100_000_000,
        strategy_weights={
            "swing_strategy": 0.35,      # BTC é ótimo para swing
            "trend_following": 0.30,     # Trends longos
            "breakout_strategy": 0.25,   # Breakouts claros
            "mean_reversion": 0.10       # Menos mean reversion
        },
        notes="Bitcoin - Líder de mercado, trends claros"
    ),
    
    "ETHUSDT": SymbolConfig(
        symbol="ETHUSDT",
        base_asset="ETH",
        quote_asset="USDT", 
        market_cap=MarketCap.LARGE_CAP,
        volatility=Volatility.MEDIUM,
        sector="smart_contracts",
        min_volume_usd=30_000_000,
        min_price=1_000,
        max_price=10_000,
        price_precision=2,
        quantity_precision=4,
        max_position_risk=0.025,
        volatility_multiplier=1.1,    # Slightly more volatile than BTC
        correlation_group="eth_ecosystem",
        min_atr_percent=2.5,
        max_atr_percent=15.0,
        min_daily_volume=50_000_000,
        strategy_weights={
            "swing_strategy": 0.30,
            "trend_following": 0.25,
            "breakout_strategy": 0.30,   # ETH tem bons breakouts
            "mean_reversion": 0.15
        },
        notes="Ethereum - Smart contracts leader"
    ),
    
    # === MID CAP - TIER 2 ===
    "BNBUSDT": SymbolConfig(
        symbol="BNBUSDT",
        base_asset="BNB",
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.MEDIUM,
        sector="exchange_token",
        min_volume_usd=5_000_000,
        min_price=200,
        max_price=1_000,
        price_precision=2,
        quantity_precision=3,
        max_position_risk=0.020,      # Slightly lower
        volatility_multiplier=1.2,
        correlation_group="exchange_tokens",
        min_atr_percent=3.0,
        max_atr_percent=18.0,
        min_daily_volume=10_000_000,
        strategy_weights={
            "swing_strategy": 0.25,
            "trend_following": 0.20,
            "breakout_strategy": 0.35,   # Exchange tokens quebram bem
            "mean_reversion": 0.20
        },
        notes="Binance Coin - Exchange utility token"
    ),
    
    "ADAUSDT": SymbolConfig(
        symbol="ADAUSDT",
        base_asset="ADA",
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="smart_contracts",
        min_volume_usd=3_000_000,
        min_price=0.2,
        max_price=3.0,
        price_precision=4,
        quantity_precision=1,
        max_position_risk=0.018,
        volatility_multiplier=1.4,    # More volatile
        correlation_group="alt_smart_contracts",
        min_atr_percent=4.0,
        max_atr_percent=25.0,
        min_daily_volume=8_000_000,
        strategy_weights={
            "swing_strategy": 0.20,
            "trend_following": 0.15,
            "breakout_strategy": 0.30,
            "mean_reversion": 0.35       # Bom para mean reversion
        },
        notes="Cardano - Academic approach to blockchain"
    ),
    
    "SOLUSDT": SymbolConfig(
        symbol="SOLUSDT", 
        base_asset="SOL",
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="smart_contracts",
        min_volume_usd=8_000_000,
        min_price=20,
        max_price=300,
        price_precision=3,
        quantity_precision=2,
        max_position_risk=0.020,
        volatility_multiplier=1.5,    # High volatility
        correlation_group="sol_ecosystem",
        min_atr_percent=5.0,
        max_atr_percent=30.0,
        min_daily_volume=15_000_000,
        strategy_weights={
            "swing_strategy": 0.20,
            "trend_following": 0.20,
            "breakout_strategy": 0.40,   # SOL tem breakouts explosivos
            "mean_reversion": 0.20
        },
        notes="Solana - High performance blockchain"
    ),
    
    # === ADDITIONAL TOKENS ===
    "LINKUSDT": SymbolConfig(
        symbol="LINKUSDT",
        base_asset="LINK", 
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="oracle",
        min_volume_usd=2_000_000,
        min_price=5,
        max_price=50,
        price_precision=3,
        quantity_precision=2,
        max_position_risk=0.018,
        volatility_multiplier=1.3,
        correlation_group="defi_infrastructure",
        min_atr_percent=4.0,
        max_atr_percent=25.0,
        min_daily_volume=5_000_000,
        strategy_weights={
            "swing_strategy": 0.25,
            "trend_following": 0.20,
            "breakout_strategy": 0.30,
            "mean_reversion": 0.25
        },
        notes="Chainlink - Decentralized oracle network"
    ),
    
    "DOTUSDT": SymbolConfig(
        symbol="DOTUSDT",
        base_asset="DOT",
        quote_asset="USDT", 
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="interoperability",
        min_volume_usd=1_500_000,
        min_price=4,
        max_price=40,
        price_precision=3,
        quantity_precision=2,
        max_position_risk=0.017,
        volatility_multiplier=1.4,
        correlation_group="alt_smart_contracts",
        min_atr_percent=4.5,
        max_atr_percent=28.0,
        min_daily_volume=3_000_000,
        strategy_weights={
            "swing_strategy": 0.20,
            "trend_following": 0.15,
            "breakout_strategy": 0.30,
            "mean_reversion": 0.35
        },
        notes="Polkadot - Multi-chain protocol"
    ),
    
    "AVAXUSDT": SymbolConfig(
        symbol="AVAXUSDT",
        base_asset="AVAX",
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="smart_contracts",
        min_volume_usd=3_000_000,
        min_price=10,
        max_price=100,
        price_precision=3,
        quantity_precision=2,
        max_position_risk=0.018,
        volatility_multiplier=1.4,
        correlation_group="avalanche_ecosystem",
        min_atr_percent=5.0,
        max_atr_percent=30.0,
        min_daily_volume=6_000_000,
        strategy_weights={
            "swing_strategy": 0.20,
            "trend_following": 0.20,
            "breakout_strategy": 0.35,
            "mean_reversion": 0.25
        },
        notes="Avalanche - Fast consensus protocol"
    ),
    
    "MATICUSDT": SymbolConfig(
        symbol="MATICUSDT",
        base_asset="MATIC",
        quote_asset="USDT",
        market_cap=MarketCap.MID_CAP,
        volatility=Volatility.HIGH,
        sector="scaling",
        min_volume_usd=2_000_000,
        min_price=0.5,
        max_price=3.0,
        price_precision=4,
        quantity_precision=1,
        max_position_risk=0.017,
        volatility_multiplier=1.3,
        correlation_group="eth_scaling",
        min_atr_percent=4.0,
        max_atr_percent=25.0,
        min_daily_volume=4_000_000,
        strategy_weights={
            "swing_strategy": 0.25,
            "trend_following": 0.20,
            "breakout_strategy": 0.30,
            "mean_reversion": 0.25
        },
        notes="Polygon - Ethereum scaling solution"
    )
}

# === GRUPOS DE CORRELAÇÃO ===
CORRELATION_GROUPS = {
    "btc_dominance": ["BTCUSDT"],
    "eth_ecosystem": ["ETHUSDT", "MATICUSDT"],
    "exchange_tokens": ["BNBUSDT"],
    "alt_smart_contracts": ["ADAUSDT", "DOTUSDT"],
    "sol_ecosystem": ["SOLUSDT"], 
    "defi_infrastructure": ["LINKUSDT"],
    "avalanche_ecosystem": ["AVAXUSDT"],
    "eth_scaling": ["MATICUSDT"]
}

# === TIER CLASSIFICATION ===
TIER_CLASSIFICATION = {
    "tier_1": ["BTCUSDT", "ETHUSDT"],                                    # Top tier
    "tier_2": ["BNBUSDT", "ADAUSDT", "SOLUSDT"],                       # Major alts  
    "tier_3": ["LINKUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT"]         # Smaller alts
}

# === FUNÇÕES UTILITÁRIAS ===

def get_active_symbols() -> List[str]:
    """Retorna lista de símbolos ativos para análise"""
    return [symbol for symbol, config in CRYPTO_SYMBOLS.items() if config.is_active]

# Add alias for compatibility with main.py
def get_enabled_symbols() -> List[str]:
    """Alias for get_active_symbols for compatibility"""
    return get_active_symbols()

def get_symbols_by_tier(tier: int) -> List[str]:
    """Retorna símbolos por tier (1, 2, 3)"""
    tier_key = f"tier_{tier}"
    return TIER_CLASSIFICATION.get(tier_key, [])

def get_symbols_by_market_cap(market_cap: MarketCap) -> List[str]:
    """Retorna símbolos por market cap"""
    return [symbol for symbol, config in CRYPTO_SYMBOLS.items() 
            if config.market_cap == market_cap and config.is_active]

def get_symbols_by_volatility(volatility: Volatility) -> List[str]:
    """Retorna símbolos por volatilidade"""
    return [symbol for symbol, config in CRYPTO_SYMBOLS.items()
            if config.volatility == volatility and config.is_active]

def get_symbols_by_sector(sector: str) -> List[str]:
    """Retorna símbolos por setor"""
    return [symbol for symbol, config in CRYPTO_SYMBOLS.items()
            if config.sector == sector and config.is_active]

def get_correlation_group(symbol: str) -> Optional[str]:
    """Retorna grupo de correlação de um símbolo"""
    config = CRYPTO_SYMBOLS.get(symbol)
    return config.correlation_group if config else None

def get_correlated_symbols(symbol: str) -> List[str]:
    """Retorna símbolos correlacionados"""
    correlation_group = get_correlation_group(symbol)
    if not correlation_group:
        return []
    
    return CORRELATION_GROUPS.get(correlation_group, [])

def get_symbol_config(symbol: str) -> Optional[SymbolConfig]:
    """Retorna configuração de um símbolo"""
    return CRYPTO_SYMBOLS.get(symbol)

def is_high_volume_symbol(symbol: str) -> bool:
    """Verifica se símbolo tem alto volume"""
    config = get_symbol_config(symbol)
    if not config:
        return False
    
    return config.min_daily_volume >= 10_000_000  # $10M+

def get_recommended_position_size(symbol: str, portfolio_value: float) -> float:
    """Calcula position size recomendado baseado na volatilidade"""
    config = get_symbol_config(symbol)
    if not config:
        return 0.0
    
    # Ajusta position size pela volatilidade
    base_risk = config.max_position_risk
    adjusted_risk = base_risk / config.volatility_multiplier
    
    return portfolio_value * adjusted_risk

def validate_symbol_config() -> Dict[str, List[str]]:
    """Valida configurações dos símbolos"""
    errors = []
    warnings = []
    
    for symbol, config in CRYPTO_SYMBOLS.items():
        # Validações críticas
        if config.max_position_risk > 0.05:  # 5% max
            errors.append(f"{symbol}: Position risk muito alto ({config.max_position_risk})")
        
        if config.min_volume_usd < 100_000:  # $100K min
            errors.append(f"{symbol}: Volume mínimo muito baixo ({config.min_volume_usd})")
        
        # Warnings
        if config.volatility_multiplier > 2.0:
            warnings.append(f"{symbol}: Volatility multiplier alto ({config.volatility_multiplier})")
        
        # Validar strategy weights
        weights_sum = sum(config.strategy_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            errors.append(f"{symbol}: Strategy weights não somam 1.0 ({weights_sum})")
    
    return {"errors": errors, "warnings": warnings}

# === CONFIGURAÇÕES DE MERCADO ===

# Horários de alta atividade por região
MARKET_ACTIVITY_SCHEDULE = {
    "asian_session": {
        "start_utc": "22:00",
        "end_utc": "08:00", 
        "activity_level": "medium",
        "preferred_symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    },
    "london_session": {
        "start_utc": "08:00",
        "end_utc": "16:00",
        "activity_level": "high", 
        "preferred_symbols": get_active_symbols()  # Todos ativos
    },
    "ny_session": {
        "start_utc": "13:00",
        "end_utc": "22:00",
        "activity_level": "highest",
        "preferred_symbols": get_active_symbols()
    },
    "overlap_session": {
        "start_utc": "13:00", 
        "end_utc": "16:00",
        "activity_level": "highest",
        "preferred_symbols": get_symbols_by_tier(1) + get_symbols_by_tier(2)
    }
}

# Blacklist de pares (se necessário)
SYMBOL_BLACKLIST = [
    # Adicionar symbols problemáticos aqui
    # "LUNAUSDT",  # Exemplo de token colapsado
]

# Configurações especiais para eventos
EVENT_CONFIGS = {
    "bitcoin_halving": {
        "affected_symbols": ["BTCUSDT"],
        "volatility_adjustment": 1.5,
        "volume_threshold_multiplier": 2.0
    },
    "ethereum_upgrade": {
        "affected_symbols": ["ETHUSDT", "MATICUSDT"],
        "volatility_adjustment": 1.3,
        "volume_threshold_multiplier": 1.5
    }
}