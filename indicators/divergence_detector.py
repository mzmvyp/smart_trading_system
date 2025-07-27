"""
Indicators: Divergence Detector
Detector inteligente de divergências entre preço e indicadores
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import talib

from utils.logger import get_logger
from utils.helpers import find_local_extremes, safe_divide


logger = get_logger(__name__)


class DivergenceType(Enum):
    """Tipos de divergência"""
    BULLISH_REGULAR = "bullish_regular"        # Preço faz LL, indicador faz HL
    BEARISH_REGULAR = "bearish_regular"        # Preço faz HH, indicador faz LH
    BULLISH_HIDDEN = "bullish_hidden"          # Preço faz HL, indicador faz LL
    BEARISH_HIDDEN = "bearish_hidden"          # Preço faz LH, indicador faz HH


class DivergenceStrength(Enum):
    """Força da divergência"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class DivergenceSignal:
    """Sinal de divergência detectado"""
    type: DivergenceType
    strength: DivergenceStrength
    indicator: str
    timeframe: str
    
    # Pontos de divergência
    price_points: List[Tuple[int, float]]      # [(index, price), ...]
    indicator_points: List[Tuple[int, float]]  # [(index, value), ...]
    
    # Scores
    confidence: float           # 0-100
    reliability: float          # 0-100
    timing_score: float         # 0-100
    
    # Metadata
    detected_at: int           # Índice de detecção
    bars_duration: int         # Duração em barras
    price_change: float        # Mudança de preço (%)
    indicator_change: float    # Mudança do indicador (%)


class DivergenceDetector:
    """Detector principal de divergências"""
    
    def __init__(self):
        self.min_bars_between_peaks = 5
        self.max_bars_between_peaks = 50
        self.min_price_change = 1.0      # % mínimo de mudança no preço
        self.min_indicator_change = 5.0   # % mínimo de mudança no indicador
        self.lookback_periods = 100
        
        # Configurações por indicador
        self.indicator_configs = {
            'rsi': {
                'period': 14,
                'overbought': 70,
                'oversold': 30,
                'weight': 0.8
            },
            'macd': {
                'fast': 12,
                'slow': 26,
                'signal': 9,
                'weight': 0.9
            },
            'stoch': {
                'k_period': 14,
                'd_period': 3,
                'weight': 0.7
            },
            'momentum': {
                'period': 14,
                'weight': 0.6
            },
            'williams_r': {
                'period': 14,
                'weight': 0.6
            },
            'cci': {
                'period': 20,
                'weight': 0.7
            }
        }
    
    def detect_all_divergences(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        indicators: List[str] = None
    ) -> List[DivergenceSignal]:
        """Detecta todas as divergências em múltiplos indicadores"""
        
        if indicators is None:
            indicators = ['rsi', 'macd', 'stoch']
        
        all_divergences = []
        
        # Valida dados
        if not self._validate_data(df):
            return all_divergences
        
        # Calcula indicadores necessários
        df_with_indicators = self._calculate_indicators(df, indicators)
        
        # Detecta divergências para cada indicador
        for indicator in indicators:
            if indicator in df_with_indicators.columns:
                divergences = self._detect_divergences_for_indicator(
                    df_with_indicators,
                    indicator,
                    timeframe
                )
                all_divergences.extend(divergences)
        
        # Ordena por confiança
        all_divergences.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_divergences
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Valida dados de entrada"""
        required_columns = ['high', 'low', 'close']
        
        if not all(col in df.columns for col in required_columns):
            logger.error("DataFrame deve conter colunas: high, low, close")
            return False
        
        if len(df) < self.lookback_periods:
            logger.warning(f"Dados insuficientes: {len(df)} < {self.lookback_periods}")
            return False
        
        return True
    
    def _calculate_indicators(self, df: pd.DataFrame, indicators: List[str]) -> pd.DataFrame:
        """Calcula todos os indicadores necessários"""
        df_copy = df.copy()
        
        for indicator in indicators:
            try:
                if indicator == 'rsi':
                    config = self.indicator_configs['rsi']
                    df_copy['rsi'] = talib.RSI(df['close'], timeperiod=config['period'])
                
                elif indicator == 'macd':
                    config = self.indicator_configs['macd']
                    macd, signal, hist = talib.MACD(
                        df['close'],
                        fastperiod=config['fast'],
                        slowperiod=config['slow'],
                        signalperiod=config['signal']
                    )
                    df_copy['macd'] = macd
                    df_copy['macd_signal'] = signal
                    df_copy['macd_histogram'] = hist
                
                elif indicator == 'stoch':
                    config = self.indicator_configs['stoch']
                    k, d = talib.STOCH(
                        df['high'],
                        df['low'], 
                        df['close'],
                        k_period=config['k_period'],
                        d_period=config['d_period']
                    )
                    df_copy['stoch_k'] = k
                    df_copy['stoch_d'] = d
                
                elif indicator == 'momentum':
                    config = self.indicator_configs['momentum']
                    df_copy['momentum'] = talib.MOM(df['close'], timeperiod=config['period'])
                
                elif indicator == 'williams_r':
                    config = self.indicator_configs['williams_r']
                    df_copy['williams_r'] = talib.WILLR(
                        df['high'],
                        df['low'],
                        df['close'],
                        timeperiod=config['period']
                    )
                
                elif indicator == 'cci':
                    config = self.indicator_configs['cci']
                    df_copy['cci'] = talib.CCI(
                        df['high'],
                        df['low'],
                        df['close'],
                        timeperiod=config['period']
                    )
                
            except Exception as e:
                logger.error(f"Erro ao calcular {indicator}: {e}")
        
        return df_copy
    
    def _detect_divergences_for_indicator(
        self,
        df: pd.DataFrame,
        indicator: str,
        timeframe: str
    ) -> List[DivergenceSignal]:
        """Detecta divergências para um indicador específico"""
        
        divergences = []
        
        # Obtém dados do indicador
        indicator_data = self._get_indicator_data(df, indicator)
        if indicator_data is None:
            return divergences
        
        price_data = df['close'].values
        
        # Encontra extremos no preço e indicador
        price_extremes = find_local_extremes(
            price_data[-self.lookback_periods:].tolist(),
            window=self.min_bars_between_peaks
        )
        
        indicator_extremes = find_local_extremes(
            indicator_data[-self.lookback_periods:].tolist(),
            window=self.min_bars_between_peaks
        )
        
        # Ajusta índices para o DataFrame completo
        offset = len(df) - self.lookback_periods
        price_highs = [i + offset for i in price_extremes['highs']]
        price_lows = [i + offset for i in price_extremes['lows']]
        indicator_highs = [i + offset for i in indicator_extremes['highs']]
        indicator_lows = [i + offset for i in indicator_extremes['lows']]
        
        # Detecta divergências bullish (regular e oculta)
        divergences.extend(self._find_bullish_divergences(
            df, price_lows, indicator_highs, price_data, indicator_data, 
            indicator, timeframe
        ))
        
        # Detecta divergências bearish (regular e oculta)
        divergences.extend(self._find_bearish_divergences(
            df, price_highs, indicator_lows, price_data, indicator_data,
            indicator, timeframe
        ))
        
        return divergences
    
    def _get_indicator_data(self, df: pd.DataFrame, indicator: str) -> Optional[np.ndarray]:
        """Obtém dados do indicador específico"""
        
        if indicator == 'rsi' and 'rsi' in df.columns:
            return df['rsi'].values
        
        elif indicator == 'macd' and 'macd' in df.columns:
            return df['macd'].values
        
        elif indicator == 'stoch' and 'stoch_k' in df.columns:
            return df['stoch_k'].values
        
        elif indicator == 'momentum' and 'momentum' in df.columns:
            return df['momentum'].values
        
        elif indicator == 'williams_r' and 'williams_r' in df.columns:
            return df['williams_r'].values
        
        elif indicator == 'cci' and 'cci' in df.columns:
            return df['cci'].values
        
        return None
    
    def _find_bullish_divergences(
        self,
        df: pd.DataFrame,
        price_lows: List[int],
        indicator_highs: List[int],
        price_data: np.ndarray,
        indicator_data: np.ndarray,
        indicator: str,
        timeframe: str
    ) -> List[DivergenceSignal]:
        """Encontra divergências bullish"""
        
        divergences = []
        
        # Divergência bullish regular: preço faz LL, indicador faz HL
        for i in range(1, len(price_lows)):
            current_low_idx = price_lows[i]
            previous_low_idx = price_lows[i-1]
            
            # Verifica se preço fez lower low
            if price_data[current_low_idx] < price_data[previous_low_idx]:
                
                # Procura highs do indicador entre os lows do preço
                relevant_highs = [
                    idx for idx in indicator_highs
                    if previous_low_idx < idx < current_low_idx
                ]
                
                if len(relevant_highs) >= 2:
                    # Verifica se indicador fez higher high
                    if indicator_data[relevant_highs[-1]] > indicator_data[relevant_highs[0]]:
                        
                        # Calcula parâmetros da divergência
                        price_change = ((price_data[current_low_idx] - price_data[previous_low_idx]) 
                                      / price_data[previous_low_idx]) * 100
                        
                        indicator_change = ((indicator_data[relevant_highs[-1]] - indicator_data[relevant_highs[0]]) 
                                          / indicator_data[relevant_highs[0]]) * 100
                        
                        # Verifica critérios mínimos
                        if (abs(price_change) >= self.min_price_change and 
                            abs(indicator_change) >= self.min_indicator_change):
                            
                            divergence = self._create_divergence_signal(
                                DivergenceType.BULLISH_REGULAR,
                                indicator,
                                timeframe,
                                [(previous_low_idx, price_data[previous_low_idx]),
                                 (current_low_idx, price_data[current_low_idx])],
                                [(relevant_highs[0], indicator_data[relevant_highs[0]]),
                                 (relevant_highs[-1], indicator_data[relevant_highs[-1]])],
                                current_low_idx,
                                price_change,
                                indicator_change
                            )
                            
                            divergences.append(divergence)
        
        return divergences
    
    def _find_bearish_divergences(
        self,
        df: pd.DataFrame,
        price_highs: List[int],
        indicator_lows: List[int],
        price_data: np.ndarray,
        indicator_data: np.ndarray,
        indicator: str,
        timeframe: str
    ) -> List[DivergenceSignal]:
        """Encontra divergências bearish"""
        
        divergences = []
        
        # Divergência bearish regular: preço faz HH, indicador faz LH
        for i in range(1, len(price_highs)):
            current_high_idx = price_highs[i]
            previous_high_idx = price_highs[i-1]
            
            # Verifica se preço fez higher high
            if price_data[current_high_idx] > price_data[previous_high_idx]:
                
                # Procura lows do indicador entre os highs do preço
                relevant_lows = [
                    idx for idx in indicator_lows
                    if previous_high_idx < idx < current_high_idx
                ]
                
                if len(relevant_lows) >= 2:
                    # Verifica se indicador fez lower low
                    if indicator_data[relevant_lows[-1]] < indicator_data[relevant_lows[0]]:
                        
                        # Calcula parâmetros da divergência
                        price_change = ((price_data[current_high_idx] - price_data[previous_high_idx]) 
                                      / price_data[previous_high_idx]) * 100
                        
                        indicator_change = ((indicator_data[relevant_lows[-1]] - indicator_data[relevant_lows[0]]) 
                                          / indicator_data[relevant_lows[0]]) * 100
                        
                        # Verifica critérios mínimos
                        if (abs(price_change) >= self.min_price_change and 
                            abs(indicator_change) >= self.min_indicator_change):
                            
                            divergence = self._create_divergence_signal(
                                DivergenceType.BEARISH_REGULAR,
                                indicator,
                                timeframe,
                                [(previous_high_idx, price_data[previous_high_idx]),
                                 (current_high_idx, price_data[current_high_idx])],
                                [(relevant_lows[0], indicator_data[relevant_lows[0]]),
                                 (relevant_lows[-1], indicator_data[relevant_lows[-1]])],
                                current_high_idx,
                                price_change,
                                indicator_change
                            )
                            
                            divergences.append(divergence)
        
        return divergences
    
    def _create_divergence_signal(
        self,
        div_type: DivergenceType,
        indicator: str,
        timeframe: str,
        price_points: List[Tuple[int, float]],
        indicator_points: List[Tuple[int, float]],
        detected_at: int,
        price_change: float,
        indicator_change: float
    ) -> DivergenceSignal:
        """Cria sinal de divergência com scores calculados"""
        
        # Calcula duração
        bars_duration = abs(price_points[-1][0] - price_points[0][0])
        
        # Calcula força da divergência
        strength = self._calculate_divergence_strength(
            abs(price_change),
            abs(indicator_change),
            bars_duration
        )
        
        # Calcula scores
        confidence = self._calculate_confidence_score(
            div_type, indicator, price_change, indicator_change, bars_duration
        )
        
        reliability = self._calculate_reliability_score(
            indicator, bars_duration, abs(price_change)
        )
        
        timing_score = self._calculate_timing_score(
            bars_duration, detected_at, len(price_points)
        )
        
        return DivergenceSignal(
            type=div_type,
            strength=strength,
            indicator=indicator,
            timeframe=timeframe,
            price_points=price_points,
            indicator_points=indicator_points,
            confidence=confidence,
            reliability=reliability,
            timing_score=timing_score,
            detected_at=detected_at,
            bars_duration=bars_duration,
            price_change=price_change,
            indicator_change=indicator_change
        )
    
    def _calculate_divergence_strength(
        self,
        price_change: float,
        indicator_change: float,
        bars_duration: int
    ) -> DivergenceStrength:
        """Calcula força da divergência"""
        
        # Score baseado na magnitude das mudanças
        magnitude_score = (price_change + indicator_change) / 2
        
        # Penaliza divergências muito longas
        duration_penalty = max(0, (bars_duration - 20) * 0.1)
        final_score = magnitude_score - duration_penalty
        
        if final_score >= 15:
            return DivergenceStrength.VERY_STRONG
        elif final_score >= 10:
            return DivergenceStrength.STRONG
        elif final_score >= 5:
            return DivergenceStrength.MODERATE
        else:
            return DivergenceStrength.WEAK
    
    def _calculate_confidence_score(
        self,
        div_type: DivergenceType,
        indicator: str,
        price_change: float,
        indicator_change: float,
        bars_duration: int
    ) -> float:
        """Calcula score de confiança (0-100)"""
        
        base_score = 50
        
        # Bonus por magnitude da divergência
        magnitude_bonus = min(30, (abs(price_change) + abs(indicator_change)) / 2)
        base_score += magnitude_bonus
        
        # Bonus por indicador confiável
        indicator_weight = self.indicator_configs.get(indicator, {}).get('weight', 0.5)
        indicator_bonus = indicator_weight * 20
        base_score += indicator_bonus
        
        # Penalidade por duração muito longa
        if bars_duration > 30:
            base_score -= (bars_duration - 30) * 0.5
        
        # Bonus para divergências regulares (mais confiáveis)
        if div_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BEARISH_REGULAR]:
            base_score += 10
        
        return max(0, min(100, base_score))
    
    def _calculate_reliability_score(
        self,
        indicator: str,
        bars_duration: int,
        price_change: float
    ) -> float:
        """Calcula score de confiabilidade (0-100)"""
        
        base_score = 60
        
        # Indicadores mais confiáveis para divergências
        reliability_weights = {
            'rsi': 0.9,
            'macd': 0.8,
            'stoch': 0.7,
            'momentum': 0.6,
            'williams_r': 0.6,
            'cci': 0.7
        }
        
        weight = reliability_weights.get(indicator, 0.5)
        base_score *= weight
        
        # Prefere divergências de duração média
        if 10 <= bars_duration <= 25:
            base_score += 15
        elif bars_duration < 10:
            base_score -= 10
        elif bars_duration > 35:
            base_score -= 20
        
        # Bonus por mudança significativa de preço
        if price_change >= 5:
            base_score += 15
        
        return max(0, min(100, base_score))
    
    def _calculate_timing_score(
        self,
        bars_duration: int,
        detected_at: int,
        num_points: int
    ) -> float:
        """Calcula score de timing (0-100)"""
        
        base_score = 50
        
        # Prefere divergências recentes
        if bars_duration <= 15:
            base_score += 25
        elif bars_duration <= 25:
            base_score += 15
        else:
            base_score -= 10
        
        # Bonus por mais pontos de confirmação
        if num_points >= 3:
            base_score += 15
        
        return max(0, min(100, base_score))
    
    def get_divergence_summary(self, divergences: List[DivergenceSignal]) -> Dict:
        """Obtém resumo das divergências detectadas"""
        
        if not divergences:
            return {
                'total_divergences': 0,
                'by_type': {},
                'by_indicator': {},
                'by_strength': {},
                'average_confidence': 0,
                'best_divergence': None
            }
        
        # Contagens por tipo
        by_type = {}
        for div in divergences:
            div_type = div.type.value
            by_type[div_type] = by_type.get(div_type, 0) + 1
        
        # Contagens por indicador
        by_indicator = {}
        for div in divergences:
            indicator = div.indicator
            by_indicator[indicator] = by_indicator.get(indicator, 0) + 1
        
        # Contagens por força
        by_strength = {}
        for div in divergences:
            strength = div.strength.name
            by_strength[strength] = by_strength.get(strength, 0) + 1
        
        # Confiança média
        avg_confidence = np.mean([div.confidence for div in divergences])
        
        # Melhor divergência
        best_div = max(divergences, key=lambda x: x.confidence)
        
        return {
            'total_divergences': len(divergences),
            'by_type': by_type,
            'by_indicator': by_indicator,
            'by_strength': by_strength,
            'average_confidence': avg_confidence,
            'best_divergence': {
                'type': best_div.type.value,
                'indicator': best_div.indicator,
                'confidence': best_div.confidence,
                'strength': best_div.strength.name
            }
        }


# Funções de conveniência
def detect_rsi_divergences(df: pd.DataFrame, timeframe: str = "1h") -> List[DivergenceSignal]:
    """Detecta divergências RSI especificamente"""
    detector = DivergenceDetector()
    return detector.detect_all_divergences(df, timeframe, ['rsi'])


def detect_macd_divergences(df: pd.DataFrame, timeframe: str = "1h") -> List[DivergenceSignal]:
    """Detecta divergências MACD especificamente"""
    detector = DivergenceDetector()
    return detector.detect_all_divergences(df, timeframe, ['macd'])


def get_strongest_divergences(
    divergences: List[DivergenceSignal],
    min_confidence: float = 70,
    limit: int = 5
) -> List[DivergenceSignal]:
    """Filtra divergências mais fortes"""
    filtered = [d for d in divergences if d.confidence >= min_confidence]
    filtered.sort(key=lambda x: (x.strength.value, x.confidence), reverse=True)
    return filtered[:limit]