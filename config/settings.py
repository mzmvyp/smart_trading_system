# config/settings.py
"""
🎯 CONFIGURAÇÕES PRINCIPAIS - SMART TRADING SYSTEM v2.0
Configurações otimizadas para timeframes altos e qualidade de sinais
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

# Caminhos do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_DIR = PROJECT_ROOT / "database"

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, LOGS_DIR, DB_DIR]:
    dir_path.mkdir(exist_ok=True)

@dataclass
class TimeframeConfig:
    """Configuração específica por timeframe"""
    name: str
    weight: float  # Peso na análise (1D=3, 4H=2, 1H=1)
    role: str      # context, setup, entry
    min_candles: int
    lookback_days: int
    confidence_multiplier: float

@dataclass
class TradingConfig:
    """🎯 CONFIGURAÇÕES PRINCIPAIS DE TRADING"""
    
    # === TIMEFRAMES HIERÁRQUICOS ===
    TIMEFRAMES: Dict[str, TimeframeConfig] = field(default_factory=lambda: {
        "1d": TimeframeConfig(
            name="1d",
            weight=3.0,
            role="context",
            min_candles=30,
            lookback_days=90,
            confidence_multiplier=1.5
        ),
        "4h": TimeframeConfig(
            name="4h", 
            weight=2.0,
            role="setup",
            min_candles=50,
            lookback_days=30,
            confidence_multiplier=1.2
        ),
        "1h": TimeframeConfig(
            name="1h",
            weight=1.0, 
            role="entry",
            min_candles=100,
            lookback_days=10,
            confidence_multiplier=1.0
        )
    })
    
    # Timeframe principal para entrada
    PRIMARY_TIMEFRAME: str = "1h"
    CONTEXT_TIMEFRAME: str = "1d"
    SETUP_TIMEFRAME: str = "4h"
    
    # === RISK MANAGEMENT AVANÇADO ===
    MAX_PORTFOLIO_RISK: float = 0.10        # 10% máximo do portfolio
    MAX_POSITION_RISK: float = 0.02         # 2% máximo por posição
    MAX_CORRELATION_EXPOSURE: float = 0.15  # 15% máx em ativos correlacionados
    MAX_DRAWDOWN_STOP: float = 0.20         # Para sistema automático se DD > 20%
    
    # Ratios mínimos
    MIN_REWARD_RISK_RATIO: float = 2.0      # Mínimo 2:1 R/R
    MIN_WIN_RATE_THRESHOLD: float = 0.35    # 35% win rate mínimo
    
    # === SCORING SYSTEM ===
    CONFLUENCE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "trend_alignment": 0.25,    # 25% - Alinhamento multi-TF
        "market_structure": 0.20,   # 20% - HH/HL/LH/LL
        "volume_confirmation": 0.15, # 15% - Volume profile
        "support_resistance": 0.15, # 15% - Níveis importantes  
        "momentum_divergence": 0.10, # 10% - Divergências
        "market_condition": 0.10,   # 10% - Bull/Bear/Lateral
        "volatility_environment": 0.05 # 5% - Volatilidade adequada
    })
    
    # Thresholds de scoring
    MIN_CONFLUENCE_SCORE: float = 70.0      # Score mínimo para sinal
    MIN_RISK_SCORE: float = 60.0            # Risk assessment mínimo
    MIN_TIMING_SCORE: float = 65.0          # Timing quality mínimo
    
    # === FILTROS DE QUALIDADE ===
    VOLUME_FILTERS: Dict[str, float] = field(default_factory=lambda: {
        "min_volume_ratio": 1.5,      # 1.5x volume médio
        "min_daily_volume_usd": 1000000, # $1M volume diário mínimo
        "volume_spike_threshold": 3.0   # 3x volume = spike
    })
    
    VOLATILITY_FILTERS: Dict[str, float] = field(default_factory=lambda: {
        "min_atr_percent": 2.0,        # Mínimo 2% ATR
        "max_atr_percent": 15.0,       # Máximo 15% ATR
        "volatility_percentile_min": 20, # Percentil 20 mínimo
        "volatility_percentile_max": 80  # Percentil 80 máximo
    })
    
    # === ESTRATÉGIAS ATIVAS ===
    ACTIVE_STRATEGIES: List[str] = field(default_factory=lambda: [
        "swing_strategy",
        "breakout_strategy", 
        "mean_reversion",
        "trend_following"
    ])
    
    STRATEGY_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "swing_strategy": 0.30,      # 30% - Mais conservadora
        "breakout_strategy": 0.25,   # 25% - Boa R/R
        "trend_following": 0.25,     # 25% - Trends longos
        "mean_reversion": 0.20       # 20% - Mais arriscada
    })
    
    # === CONFIGURAÇÕES DE MERCADO ===
    MARKET_SESSIONS: Dict[str, Dict] = field(default_factory=lambda: {
        "asian": {"start": "22:00", "end": "08:00", "activity": "low"},
        "london": {"start": "08:00", "end": "16:00", "activity": "high"},
        "new_york": {"start": "13:00", "end": "22:00", "activity": "high"},
        "overlap": {"start": "13:00", "end": "16:00", "activity": "highest"}
    })
    
    # Filtros de tempo
    AVOID_WEEKEND_GAPS: bool = True
    MIN_MARKET_ACTIVITY: str = "medium"  # low, medium, high, highest
    
    # === CONFIGURAÇÕES TÉCNICAS ===
    MAX_CONCURRENT_SIGNALS: int = 5       # Máximo 5 sinais simultâneos
    SIGNAL_COOLDOWN_HOURS: int = 24       # 24h entre sinais do mesmo ativo
    MAX_ANALYSIS_TIME_SECONDS: int = 30   # Timeout para análise
    
    # Cache e performance
    CACHE_DURATION_MINUTES: int = 30      # Cache de dados por 30min
    MAX_HISTORICAL_DAYS: int = 365        # Máximo 1 ano de dados
    
    # === NOTIFICAÇÕES ===
    NOTIFICATION_CHANNELS: List[str] = field(default_factory=lambda: [
        "console", "file", "webhook"
    ])
    
    MIN_NOTIFICATION_SCORE: float = 80.0  # Score mínimo para notificar

@dataclass 
class DatabaseConfig:
    """💾 CONFIGURAÇÕES DE BANCO DE DADOS"""
    
    # URLs de conexão
    MAIN_DB_URL: str = f"sqlite:///{DB_DIR}/smart_trading.db"
    CACHE_DB_URL: str = f"sqlite:///{DB_DIR}/cache.db"
    BACKTEST_DB_URL: str = f"sqlite:///{DB_DIR}/backtest.db"
    
    # Configurações de pool
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_TIMEOUT: int = 30
    
    # Configurações de cache
    CACHE_TTL_SECONDS: int = 1800  # 30 minutos
    MAX_CACHE_SIZE_MB: int = 500   # 500MB máximo
    
    # Backup
    AUTO_BACKUP: bool = True
    BACKUP_INTERVAL_HOURS: int = 24
    MAX_BACKUP_FILES: int = 7

@dataclass
class APIConfig:
    """🌐 CONFIGURAÇÕES DE APIs"""
    
    # Binance
    BINANCE_API_KEY: str = os.getenv("NcXoZQpJX1howG8esjMdFonmxOg7CdJNTGmRHCKrms6PwizmyoWQVkfbuYSKKf6h", "")
    BINANCE_SECRET_KEY: str = os.getenv("AZUtoYUYgXIEQYwx6YnCIyp5tx35c6spTdwNScBS0kC3XQ3Ckp0yNrVLHNevyE2D", "")
    BINANCE_TESTNET: bool = os.getenv("TRADING_MODE", "paper") != "live"
    
    # Rate limiting
    REQUESTS_PER_MINUTE: int = 1200
    REQUESTS_PER_SECOND: int = 10
    
    # Timeouts e retries
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_RECONNECT_ATTEMPTS: int = 5
    
    # Data providers
    PRIMARY_PROVIDER: str = "binance"
    BACKUP_PROVIDERS: List[str] = field(default_factory=lambda: ["coinbase", "kraken"])

@dataclass
class LoggingConfig:
    """📝 CONFIGURAÇÕES DE LOGGING"""
    
    # Níveis de log
    CONSOLE_LEVEL: str = "INFO"
    FILE_LEVEL: str = "DEBUG" 
    
    # Arquivos de log
    MAIN_LOG_FILE: str = str(LOGS_DIR / "smart_trading.log")
    ERROR_LOG_FILE: str = str(LOGS_DIR / "errors.log")
    TRADES_LOG_FILE: str = str(LOGS_DIR / "trades.log")
    PERFORMANCE_LOG_FILE: str = str(LOGS_DIR / "performance.log")
    
    # Rotação de logs
    MAX_FILE_SIZE_MB: int = 50
    BACKUP_COUNT: int = 5
    
    # Formatação
    LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Structured logging
    USE_JSON_FORMAT: bool = False
    INCLUDE_EXTRA_FIELDS: bool = True

@dataclass
class BacktestConfig:
    """📊 CONFIGURAÇÕES DE BACKTESTING"""
    
    # Período padrão
    DEFAULT_START_DATE: str = "2023-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"
    
    # Capital inicial
    INITIAL_CAPITAL: float = 10000.0
    
    # Custos de trading
    MAKER_FEE: float = 0.001      # 0.1%
    TAKER_FEE: float = 0.001      # 0.1%
    SLIPPAGE_BPS: float = 2.0     # 2 basis points
    
    # Configurações avançadas
    COMPOUND_RETURNS: bool = True
    REINVEST_PROFITS: bool = True
    REBALANCE_FREQUENCY: str = "monthly"
    
    # Métricas obrigatórias
    REQUIRED_METRICS: List[str] = field(default_factory=lambda: [
        "total_return", "sharpe_ratio", "max_drawdown", 
        "win_rate", "profit_factor", "calmar_ratio"
    ])

# === CONFIGURAÇÕES GLOBAIS ===
class Settings:
    """⚙️ CONFIGURAÇÕES GLOBAIS DO SISTEMA"""
    
    def __init__(self):
        self.trading = TradingConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.backtest = BacktestConfig()
        
        # Modo de operação
        self.MODE = os.getenv("TRADING_MODE", "paper")  # paper, live, backtest
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # Validação inicial
        self._validate_config()
    
    def _validate_config(self):
        """Validação das configurações"""
        # Validar timeframes
        required_tfs = {"1h", "4h", "1d"}
        available_tfs = set(self.trading.TIMEFRAMES.keys())
        if not required_tfs.issubset(available_tfs):
            raise ValueError(f"Timeframes obrigatórios: {required_tfs}")
        
        # Validar risk management
        if self.trading.MAX_POSITION_RISK >= self.trading.MAX_PORTFOLIO_RISK:
            raise ValueError("Position risk deve ser menor que portfolio risk")
        
        # Validar scoring weights
        confluence_sum = sum(self.trading.CONFLUENCE_WEIGHTS.values())
        if abs(confluence_sum - 1.0) > 0.01:
            raise ValueError(f"Confluence weights devem somar 1.0, soma atual: {confluence_sum}")
        
        strategy_sum = sum(self.trading.STRATEGY_WEIGHTS.values())
        if abs(strategy_sum - 1.0) > 0.01:
            raise ValueError(f"Strategy weights devem somar 1.0, soma atual: {strategy_sum}")
    
    def get_timeframe_config(self, timeframe: str) -> TimeframeConfig:
        """Retorna configuração de um timeframe específico"""
        if timeframe not in self.trading.TIMEFRAMES:
            raise ValueError(f"Timeframe '{timeframe}' não configurado")
        return self.trading.TIMEFRAMES[timeframe]
    
    def is_valid_trading_time(self) -> bool:
        """Verifica se é horário válido para trading"""
        # TODO: Implementar lógica de horário baseada em market_sessions
        return True
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Retorna limites de risco atuais"""
        return {
            "max_portfolio_risk": self.trading.MAX_PORTFOLIO_RISK,
            "max_position_risk": self.trading.MAX_POSITION_RISK,
            "max_correlation_exposure": self.trading.MAX_CORRELATION_EXPOSURE,
            "min_reward_risk": self.trading.MIN_REWARD_RISK_RATIO
        }

# Instância global das configurações
settings = Settings()

# Funções de conveniência
def get_timeframes() -> List[str]:
    """Retorna lista de timeframes configurados"""
    return list(settings.trading.TIMEFRAMES.keys())

def get_primary_timeframe() -> str:
    """Retorna timeframe principal"""
    return settings.trading.PRIMARY_TIMEFRAME

def is_live_mode() -> bool:
    """Verifica se está em modo live"""
    return settings.MODE == "live"

def is_paper_mode() -> bool:
    """Verifica se está em modo paper"""
    return settings.MODE == "paper"

def is_backtest_mode() -> bool:
    """Verifica se está em modo backtest"""
    return settings.MODE == "backtest"

# === FUNÇÕES UTILITÁRIAS ===

def load_config(config_path: str = "config/config.json") -> TradingConfig:
    """
    Carrega configuração de arquivo JSON ou retorna configuração padrão
    
    Args:
        config_path: Caminho para arquivo de configuração
        
    Returns:
        TradingConfig: Instância da configuração
    """
    import json
    import os
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Cria instância com configurações customizadas
            config = TradingConfig()
            
            # Atualiza com dados do arquivo (se necessário)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            return config
            
        except Exception as e:
            print(f"Erro ao carregar configuração de {config_path}: {e}")
            print("Usando configuração padrão...")
    
    # Retorna configuração padrão
    return TradingConfig()


def save_config(config: TradingConfig, config_path: str = "config/config.json") -> bool:
    """
    Salva configuração em arquivo JSON
    
    Args:
        config: Instância da configuração
        config_path: Caminho para salvar o arquivo
        
    Returns:
        bool: True se salvou com sucesso
    """
    import json
    import os
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Converte para dicionário
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_'):
                value = getattr(config, attr)
                if not callable(value):
                    config_dict[attr] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Erro ao salvar configuração: {e}")
        return False