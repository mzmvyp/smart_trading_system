"""
Config: Timeframes Configuration
Configurações hierárquicas de timeframes para análise multi-temporal
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class TimeframeType(Enum):
    """Tipos de timeframe para classificação"""
    MACRO = "macro"        # 1D, 4H - Contexto e bias
    ENTRY = "entry"        # 1H - Setup e entrada
    MANAGEMENT = "mgmt"    # 15m - Gestão de posição


@dataclass
class TimeframeConfig:
    """Configuração individual de timeframe"""
    name: str
    minutes: int
    type: TimeframeType
    priority: int          # 1=maior prioridade
    min_candles: int       # Mínimo de candles para análise
    max_lookback: int      # Máximo de candles históricos
    confluence_weight: float   # Peso na confluência (0-1)
    
    @property
    def binance_interval(self) -> str:
        """Converte para formato Binance"""
        if self.minutes < 60:
            return f"{self.minutes}m"
        elif self.minutes < 1440:
            hours = self.minutes // 60
            return f"{hours}h"
        else:
            days = self.minutes // 1440
            return f"{days}d"


# Configurações principais de timeframes
TIMEFRAMES = {
    '1d': TimeframeConfig(
        name='1d',
        minutes=1440,
        type=TimeframeType.MACRO,
        priority=1,
        min_candles=50,
        max_lookback=200,
        confluence_weight=0.4
    ),
    
    '4h': TimeframeConfig(
        name='4h', 
        minutes=240,
        type=TimeframeType.MACRO,
        priority=2,
        min_candles=100,
        max_lookback=500,
        confluence_weight=0.35
    ),
    
    '1h': TimeframeConfig(
        name='1h',
        minutes=60,
        type=TimeframeType.ENTRY,
        priority=3,
        min_candles=200,
        max_lookback=1000,
        confluence_weight=0.25
    ),
    
    '15m': TimeframeConfig(
        name='15m',
        minutes=15,
        type=TimeframeType.MANAGEMENT,
        priority=4,
        min_candles=100,
        max_lookback=500,
        confluence_weight=0.0  # Não usado para confluência
    )
}


class TimeframeHierarchy:
    """Gerenciador da hierarquia de timeframes"""
    
    def __init__(self):
        self.timeframes = TIMEFRAMES
        self._hierarchy = self._build_hierarchy()
    
    def _build_hierarchy(self) -> Dict[str, Dict]:
        """Constrói hierarquia de timeframes"""
        hierarchy = {}
        
        for tf_name, tf_config in self.timeframes.items():
            hierarchy[tf_name] = {
                'config': tf_config,
                'higher': self._get_higher_timeframe(tf_name),
                'lower': self._get_lower_timeframe(tf_name),
                'confluence_group': self._get_confluence_group(tf_name)
            }
        
        return hierarchy
    
    def _get_higher_timeframe(self, timeframe: str) -> Optional[str]:
        """Retorna timeframe superior"""
        current_minutes = self.timeframes[timeframe].minutes
        
        candidates = [
            (tf, cfg) for tf, cfg in self.timeframes.items()
            if cfg.minutes > current_minutes
        ]
        
        if not candidates:
            return None
            
        # Retorna o menor timeframe superior
        return min(candidates, key=lambda x: x[1].minutes)[0]
    
    def _get_lower_timeframe(self, timeframe: str) -> Optional[str]:
        """Retorna timeframe inferior"""
        current_minutes = self.timeframes[timeframe].minutes
        
        candidates = [
            (tf, cfg) for tf, cfg in self.timeframes.items()
            if cfg.minutes < current_minutes
        ]
        
        if not candidates:
            return None
            
        # Retorna o maior timeframe inferior
        return max(candidates, key=lambda x: x[1].minutes)[0]
    
    def _get_confluence_group(self, timeframe: str) -> List[str]:
        """Retorna grupo de timeframes para confluência"""
        tf_config = self.timeframes[timeframe]
        
        if tf_config.type == TimeframeType.MANAGEMENT:
            return []
        
        # Para análise de confluência, usa timeframe atual + superior
        group = [timeframe]
        higher = self._get_higher_timeframe(timeframe)
        if higher:
            group.append(higher)
            
        return group
    
    def get_analysis_timeframes(self) -> List[str]:
        """Retorna timeframes para análise (exceto management)"""
        return [
            tf for tf, cfg in self.timeframes.items()
            if cfg.type != TimeframeType.MANAGEMENT
        ]
    
    def get_confluence_timeframes(self) -> List[str]:
        """Retorna timeframes usados para confluência"""
        return [
            tf for tf, cfg in self.timeframes.items()
            if cfg.confluence_weight > 0
        ]
    
    def get_timeframe_config(self, timeframe: str) -> TimeframeConfig:
        """Retorna configuração de um timeframe"""
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe '{timeframe}' não configurado")
        return self.timeframes[timeframe]
    
    def get_hierarchy_info(self, timeframe: str) -> Dict:
        """Retorna informações hierárquicas de um timeframe"""
        if timeframe not in self._hierarchy:
            raise ValueError(f"Timeframe '{timeframe}' não encontrado")
        return self._hierarchy[timeframe]


# Configurações específicas para diferentes tipos de análise
ANALYSIS_CONFIG = {
    'trend_analysis': {
        'primary_timeframes': ['1d', '4h'],
        'confirmation_timeframes': ['1h'],
        'min_confluence_score': 60
    },
    
    'entry_analysis': {
        'primary_timeframes': ['4h', '1h'],
        'confirmation_timeframes': ['1h'],
        'min_confluence_score': 70
    },
    
    'risk_management': {
        'stop_timeframe': '15m',
        'target_timeframes': ['1h', '4h'],
        'trailing_timeframe': '15m'
    }
}


# Pesos para diferentes tipos de sinais por timeframe
SIGNAL_WEIGHTS = {
    '1d': {
        'trend': 0.5,
        'support_resistance': 0.3,
        'momentum': 0.2
    },
    '4h': {
        'trend': 0.4,
        'support_resistance': 0.35,
        'momentum': 0.25
    },
    '1h': {
        'trend': 0.3,
        'support_resistance': 0.4,
        'momentum': 0.3
    }
}


def get_optimal_timeframe_for_strategy(strategy_type: str) -> str:
    """Retorna timeframe ótimo para tipo de estratégia"""
    strategy_timeframes = {
        'swing': '4h',
        'breakout': '1h', 
        'mean_reversion': '1h',
        'trend_following': '4h'
    }
    
    return strategy_timeframes.get(strategy_type, '1h')


def validate_timeframe_data(timeframe: str, candles_count: int) -> bool:
    """Valida se há dados suficientes para análise"""
    if timeframe not in TIMEFRAMES:
        return False
    
    min_required = TIMEFRAMES[timeframe].min_candles
    return candles_count >= min_required


# Instância global do gerenciador
timeframe_manager = TimeframeHierarchy()