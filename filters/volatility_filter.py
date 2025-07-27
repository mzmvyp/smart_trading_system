"""
⚡ VOLATILITY FILTER - Smart Trading System v2.0

Filtro avançado de volatilidade para otimizar timing de entrada:
- Regime detection (low/normal/high/extreme)
- Volatility clustering analysis  
- Mean reversion opportunities
- Breakout volatility confirmation
- Position sizing adjustments

Filosofia: Right Volatility = Right Strategy = Right Position Size
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """Regimes de volatilidade"""
    EXTREMELY_LOW = "extremely_low"      # < 10th percentile
    LOW = "low"                         # 10-30th percentile
    NORMAL = "normal"                   # 30-70th percentile
    HIGH = "high"                       # 70-90th percentile
    EXTREMELY_HIGH = "extremely_high"   # > 90th percentile


class VolatilityTrend(Enum):
    """Tendência da volatilidade"""
    DECREASING = "decreasing"           # Volatilidade diminuindo
    STABLE = "stable"                   # Volatilidade estável
    INCREASING = "increasing"           # Volatilidade aumentando
    EXPLOSIVE = "explosive"             # Volatilidade explodindo


class VolatilityPattern(Enum):
    """Padrões de volatilidade"""
    COMPRESSION = "compression"          # Compressão (baixa vol)
    EXPANSION = "expansion"             # Expansão (alta vol)
    MEAN_REVERSION = "mean_reversion"   # Reversão à média
    CLUSTERING = "clustering"           # Clustering (vol persistente)
    BREAKOUT = "breakout"               # Breakout de vol


@dataclass
class VolatilityMetrics:
    """Métricas de volatilidade"""
    current_atr_pct: float              # ATR atual (%)
    atr_percentile: float               # Percentil do ATR (0-100)
    realized_vol: float                 # Volatilidade realizada
    implied_vol: Optional[float]        # Volatilidade implícita (se disponível)
    vol_of_vol: float                   # Volatilidade da volatilidade
    garch_forecast: float               # Previsão GARCH (simplified)
    
    # Bollinger Bands
    bb_width: float                     # Largura das Bollinger Bands
    bb_position: float                  # Posição dentro das BBs (0-1)
    bb_squeeze: bool                    # Squeeze das BBs
    
    # Intraday patterns
    intraday_range_avg: float           # Range médio intraday
    gap_volatility: float               # Volatilidade de gaps
    overnight_vol: float                # Volatilidade overnight
    
    # Clustering metrics
    vol_persistence: float              # Persistência da volatilidade
    vol_autocorrelation: float          # Autocorrelação da vol


@dataclass
class VolatilitySignal:
    """Sinal de volatilidade"""
    signal_type: VolatilityPattern
    regime: VolatilityRegime
    trend: VolatilityTrend
    strength: float                     # 0-100 força do sinal
    confidence: float                   # 0-100 confiança
    expected_duration: int              # Duração esperada em períodos
    
    # Trading implications
    recommended_strategies: List[str]    # Estratégias recomendadas
    position_size_adjustment: float     # Ajuste de position size
    stop_loss_adjustment: float         # Ajuste de stop loss
    entry_timing: str                   # 'immediate', 'wait', 'avoid'
    
    # Details
    current_percentile: float
    mean_reversion_target: Optional[float]
    breakout_threshold: Optional[float]
    timestamp: pd.Timestamp


class VolatilityFilter:
    """
    ⚡ Filtro Principal de Volatilidade
    
    Analisa padrões de volatilidade para:
    1. Identificar regimes de volatilidade
    2. Detectar compressões/expansões
    3. Timing de breakouts
    4. Ajustes de position sizing
    5. Otimização de stop loss
    """
    
    def __init__(self,
                 atr_period: int = 14,
                 vol_lookback: int = 100,
                 bb_period: int = 20,
                 bb_std: float = 2.0):
        
        self.atr_period = atr_period
        self.vol_lookback = vol_lookback
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        self.logger = logging.getLogger(f"{__name__}.VolatilityFilter")
        
        # Thresholds para classificação
        self.regime_thresholds = {
            'extremely_low': 0.10,      # 10th percentile
            'low': 0.30,                # 30th percentile
            'high': 0.70,               # 70th percentile
            'extremely_high': 0.90      # 90th percentile
        }
        
        # Configurações de padrões
        self.pattern_config = {
            'compression_threshold': 0.5,    # 50% of normal width
            'expansion_threshold': 1.5,      # 150% of normal width
            'squeeze_periods': 20,           # Períodos para BB squeeze
            'breakout_confirmation': 1.2,    # 120% vol increase
            'mean_reversion_threshold': 2.0   # 2 std devs
        }
        
        # Cache para otimização
        self.vol_cache: Dict[str, VolatilityMetrics] = {}
        self.historical_percentiles: Dict[str, np.ndarray] = {}
    
    def analyze_volatility(self, 
                         data: pd.DataFrame,
                         symbol: str = "BTCUSDT") -> VolatilitySignal:
        """
        Análise principal de volatilidade
        
        Args:
            data: DataFrame com dados OHLCV
            symbol: Símbolo para análise
            
        Returns:
            VolatilitySignal com análise completa
        """
        try:
            self.logger.info(f"Analisando volatilidade para {symbol}")
            
            if len(data) < self.vol_lookback:
                raise ValueError("Dados insuficientes para análise de volatilidade")
            
            # 1. Calcular métricas de volatilidade
            vol_metrics = self._calculate_volatility_metrics(data, symbol)
            
            # 2. Determinar regime de volatilidade
            regime = self._determine_regime(vol_metrics, symbol)
            
            # 3. Analisar tendência da volatilidade
            trend = self._analyze_volatility_trend(data)
            
            # 4. Detectar padrões específicos
            pattern, pattern_strength = self._detect_volatility_patterns(data, vol_metrics)
            
            # 5. Calcular implicações para trading
            trading_implications = self._calculate_trading_implications(
                regime, trend, pattern, vol_metrics)
            
            # 6. Gerar recomendações
            recommendations = self._generate_volatility_recommendations(
                regime, trend, pattern, vol_metrics)
            
            signal = VolatilitySignal(
                signal_type=pattern,
                regime=regime,
                trend=trend,
                strength=pattern_strength,
                confidence=self._calculate_confidence(vol_metrics, regime, pattern),
                expected_duration=self._estimate_pattern_duration(pattern, trend),
                
                recommended_strategies=recommendations['strategies'],
                position_size_adjustment=recommendations['position_sizing'],
                stop_loss_adjustment=recommendations['stop_adjustment'],
                entry_timing=recommendations['timing'],
                
                current_percentile=vol_metrics.atr_percentile,
                mean_reversion_target=self._calculate_mean_reversion_target(vol_metrics),
                breakout_threshold=self._calculate_breakout_threshold(vol_metrics),
                timestamp=pd.Timestamp.now()
            )
            
            self.logger.info(f"Volatilidade analisada - Regime: {regime.value}, "
                           f"Padrão: {pattern.value}, Força: {pattern_strength:.1f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volatilidade: {e}")
            raise
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame, symbol: str) -> VolatilityMetrics:
        """Calcula métricas abrangentes de volatilidade"""
        try:
            # 1. ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.atr_period).mean()
            atr_pct = (atr / data['close']) * 100
            
            current_atr_pct = atr_pct.iloc[-1]
            
            # 2. ATR Percentile
            atr_historical = atr_pct.tail(self.vol_lookback).dropna()
            atr_percentile = stats.percentileofscore(atr_historical, current_atr_pct)
            
            # Cache historical percentiles
            self.historical_percentiles[symbol] = atr_historical.values
            
            # 3. Realized Volatility
            returns = data['close'].pct_change().dropna()
            realized_vol = returns.rolling(20).std() * np.sqrt(365) * 100  # Anualizada
            current_realized_vol = realized_vol.iloc[-1]
            
            # 4. Volatility of Volatility
            vol_series = atr_pct.rolling(20).std()
            vol_of_vol = vol_series.iloc[-1] if not pd.isna(vol_series.iloc[-1]) else 0
            
            # 5. GARCH Forecast (simplified)
            garch_forecast = self._simple_garch_forecast(returns.tail(50))
            
            # 6. Bollinger Bands
            bb_middle = data['close'].rolling(self.bb_period).mean()
            bb_std_dev = data['close'].rolling(self.bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * self.bb_std)
            bb_lower = bb_middle - (bb_std_dev * self.bb_std)
            
            bb_width = ((bb_upper - bb_lower) / bb_middle * 100).iloc[-1]
            current_price = data['close'].iloc[-1]
            bb_position = ((current_price - bb_lower.iloc[-1]) / 
                          (bb_upper.iloc[-1] - bb_lower.iloc[-1])).clip(0, 1)
            
            # BB Squeeze detection
            bb_width_ma = ((bb_upper - bb_lower) / bb_middle * 100).rolling(self.pattern_config['squeeze_periods']).mean()
            bb_squeeze = bb_width < bb_width_ma.iloc[-1] * 0.8  # 80% of average
            
            # 7. Intraday metrics
            intraday_ranges = (data['high'] - data['low']) / data['close'] * 100
            intraday_range_avg = intraday_ranges.rolling(20).mean().iloc[-1]
            
            # Gap volatility (open vs previous close)
            gaps = np.abs(data['open'] - data['close'].shift()) / data['close'].shift() * 100
            gap_volatility = gaps.rolling(20).mean().iloc[-1]
            
            # Overnight volatility (simplified - open to close)
            overnight_moves = np.abs(data['open'] - data['close'].shift()) / data['close'].shift() * 100
            overnight_vol = overnight_moves.rolling(20).std().iloc[-1]
            
            # 8. Clustering metrics
            vol_persistence = self._calculate_volatility_persistence(atr_pct.tail(50))
            vol_autocorr = self._calculate_volatility_autocorrelation(atr_pct.tail(30))
            
            return VolatilityMetrics(
                current_atr_pct=current_atr_pct,
                atr_percentile=atr_percentile,
                realized_vol=current_realized_vol,
                implied_vol=None,  # Não disponível para crypto spot
                vol_of_vol=vol_of_vol,
                garch_forecast=garch_forecast,
                
                bb_width=bb_width,
                bb_position=bb_position,
                bb_squeeze=bb_squeeze,
                
                intraday_range_avg=intraday_range_avg,
                gap_volatility=gap_volatility,
                overnight_vol=overnight_vol,
                
                vol_persistence=vol_persistence,
                vol_autocorrelation=vol_autocorr
            )
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de métricas: {e}")
            raise
    
    def _simple_garch_forecast(self, returns: pd.Series) -> float:
        """Previsão GARCH simplificada"""
        try:
            if len(returns) < 20:
                return returns.std() * 100
            
            # GARCH(1,1) simplificado
            returns_squared = returns ** 2
            
            # Parâmetros típicos
            alpha = 0.1  # ARCH term
            beta = 0.85  # GARCH term
            omega = 0.00001  # Long term variance
            
            # Calcular variância condicional
            h = returns_squared.rolling(10).mean().iloc[-1]  # Initial variance
            
            # One-step ahead forecast
            h_forecast = omega + alpha * returns_squared.iloc[-1] + beta * h
            
            return np.sqrt(h_forecast) * np.sqrt(365) * 100  # Anualizada
            
        except Exception:
            return returns.std() * np.sqrt(365) * 100
    
    def _calculate_volatility_persistence(self, vol_series: pd.Series) -> float:
        """Calcula persistência da volatilidade"""
        try:
            if len(vol_series) < 10:
                return 0.5
            
            # Autocorrelação lag-1
            correlation = vol_series.autocorr(lag=1)
            return correlation if not pd.isna(correlation) else 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_volatility_autocorrelation(self, vol_series: pd.Series) -> float:
        """Calcula autocorrelação da volatilidade"""
        try:
            if len(vol_series) < 5:
                return 0
            
            # Média das autocorrelações de 1-5 lags
            autocorrs = [vol_series.autocorr(lag=i) for i in range(1, 6)]
            autocorrs = [ac for ac in autocorrs if not pd.isna(ac)]
            
            return np.mean(autocorrs) if autocorrs else 0
            
        except Exception:
            return 0
    
    def _determine_regime(self, metrics: VolatilityMetrics, symbol: str) -> VolatilityRegime:
        """Determina regime de volatilidade atual"""
        try:
            percentile = metrics.atr_percentile
            
            if percentile <= self.regime_thresholds['extremely_low'] * 100:
                return VolatilityRegime.EXTREMELY_LOW
            elif percentile <= self.regime_thresholds['low'] * 100:
                return VolatilityRegime.LOW
            elif percentile >= self.regime_thresholds['extremely_high'] * 100:
                return VolatilityRegime.EXTREMELY_HIGH
            elif percentile >= self.regime_thresholds['high'] * 100:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.NORMAL
                
        except Exception:
            return VolatilityRegime.NORMAL
    
    def _analyze_volatility_trend(self, data: pd.DataFrame) -> VolatilityTrend:
        """Analisa tendência da volatilidade"""
        try:
            # Calcular ATR para análise de trend
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.atr_period).mean()
            atr_pct = (atr / data['close']) * 100
            
            # Analisar trend dos últimos 20 períodos
            recent_atr = atr_pct.tail(20)
            if len(recent_atr) < 10:
                return VolatilityTrend.STABLE
            
            # Linear regression slope
            x = np.arange(len(recent_atr))
            y = recent_atr.values
            
            slope, _, r_value, _, _ = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Interpretar slope
            if r_squared > 0.3:  # Trend significativo
                if slope > 0.1:  # Threshold for increasing
                    if slope > 0.3:
                        return VolatilityTrend.EXPLOSIVE
                    else:
                        return VolatilityTrend.INCREASING
                elif slope < -0.1:  # Threshold for decreasing
                    return VolatilityTrend.DECREASING
                else:
                    return VolatilityTrend.STABLE
            else:
                return VolatilityTrend.STABLE
                
        except Exception as e:
            self.logger.error(f"Erro na análise de trend: {e}")
            return VolatilityTrend.STABLE
    
    def _detect_volatility_patterns(self, data: pd.DataFrame, 
                                  metrics: VolatilityMetrics) -> Tuple[VolatilityPattern, float]:
        """Detecta padrões específicos de volatilidade"""
        try:
            patterns_detected = []
            
            # 1. Compression Pattern (BB Squeeze)
            if metrics.bb_squeeze and metrics.atr_percentile < 30:
                compression_strength = (30 - metrics.atr_percentile) * 2  # 0-60
                patterns_detected.append((VolatilityPattern.COMPRESSION, compression_strength))
            
            # 2. Expansion Pattern (High vol)
            if metrics.atr_percentile > 80:
                expansion_strength = (metrics.atr_percentile - 80) * 5  # 0-100
                patterns_detected.append((VolatilityPattern.EXPANSION, expansion_strength))
            
            # 3. Mean Reversion Pattern
            if metrics.atr_percentile > 85 or metrics.atr_percentile < 15:
                reversion_strength = max(85 - metrics.atr_percentile, metrics.atr_percentile - 85)
                if reversion_strength > 0:
                    patterns_detected.append((VolatilityPattern.MEAN_REVERSION, reversion_strength))
            
            # 4. Clustering Pattern (Persistent volatility)
            if metrics.vol_persistence > 0.6 and metrics.vol_autocorrelation > 0.4:
                clustering_strength = (metrics.vol_persistence + metrics.vol_autocorrelation) * 50
                patterns_detected.append((VolatilityPattern.CLUSTERING, clustering_strength))
            
            # 5. Breakout Pattern (Volatility expansion from compression)
            recent_atr = self._get_recent_atr(data, 10)
            if len(recent_atr) >= 5:
                recent_increase = recent_atr.iloc[-1] / recent_atr.iloc[-5]
                if recent_increase > self.pattern_config['breakout_confirmation']:
                    breakout_strength = min(100, (recent_increase - 1) * 100)
                    patterns_detected.append((VolatilityPattern.BREAKOUT, breakout_strength))
            
            # Selecionar padrão mais forte
            if patterns_detected:
                best_pattern, best_strength = max(patterns_detected, key=lambda x: x[1])
                return best_pattern, min(100, best_strength)
            else:
                return VolatilityPattern.CLUSTERING, 30  # Default pattern
                
        except Exception as e:
            self.logger.error(f"Erro na detecção de padrões: {e}")
            return VolatilityPattern.CLUSTERING, 30
    
    def _get_recent_atr(self, data: pd.DataFrame, periods: int) -> pd.Series:
        """Calcula ATR recente"""
        try:
            recent_data = data.tail(periods + self.atr_period)
            
            high_low = recent_data['high'] - recent_data['low']
            high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
            low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.atr_period).mean()
            atr_pct = (atr / recent_data['close']) * 100
            
            return atr_pct.tail(periods)
            
        except Exception:
            return pd.Series([2.0] * periods)  # Default values
    
    def _calculate_trading_implications(self, regime: VolatilityRegime, trend: VolatilityTrend,
                                      pattern: VolatilityPattern, metrics: VolatilityMetrics) -> Dict:
        """Calcula implicações para trading"""
        try:
            implications = {
                'position_sizing': 1.0,
                'stop_adjustment': 1.0,
                'entry_timing': 'immediate',
                'risk_level': 'normal'
            }
            
            # Ajustes baseados no regime
            if regime == VolatilityRegime.EXTREMELY_LOW:
                implications['position_sizing'] = 1.3  # Aumentar size
                implications['stop_adjustment'] = 0.7   # Stops mais apertados
                implications['entry_timing'] = 'immediate'
                implications['risk_level'] = 'low'
                
            elif regime == VolatilityRegime.LOW:
                implications['position_sizing'] = 1.1
                implications['stop_adjustment'] = 0.8
                implications['entry_timing'] = 'immediate'
                implications['risk_level'] = 'low'
                
            elif regime == VolatilityRegime.HIGH:
                implications['position_sizing'] = 0.7
                implications['stop_adjustment'] = 1.3  # Stops mais largos
                implications['entry_timing'] = 'wait'
                implications['risk_level'] = 'high'
                
            elif regime == VolatilityRegime.EXTREMELY_HIGH:
                implications['position_sizing'] = 0.4
                implications['stop_adjustment'] = 1.5
                implications['entry_timing'] = 'avoid'
                implications['risk_level'] = 'extreme'
            
            # Ajustes baseados no padrão
            if pattern == VolatilityPattern.COMPRESSION:
                implications['position_sizing'] *= 1.2  # Oportunidade
                implications['entry_timing'] = 'immediate'
                
            elif pattern == VolatilityPattern.EXPANSION:
                implications['position_sizing'] *= 0.6  # Reduzir risco
                implications['stop_adjustment'] *= 1.4
                
            elif pattern == VolatilityPattern.BREAKOUT:
                implications['position_sizing'] *= 0.8  # Cauteloso
                implications['entry_timing'] = 'wait'  # Aguardar confirmação
                
            elif pattern == VolatilityPattern.MEAN_REVERSION:
                if metrics.atr_percentile > 80:
                    implications['entry_timing'] = 'avoid'  # Muito volátil
                else:
                    implications['entry_timing'] = 'immediate'  # Oportunidade
            
            # Limitar ajustes
            implications['position_sizing'] = max(0.2, min(2.0, implications['position_sizing']))
            implications['stop_adjustment'] = max(0.5, min(2.0, implications['stop_adjustment']))
            
            return implications
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de implicações: {e}")
            return {'position_sizing': 1.0, 'stop_adjustment': 1.0, 'entry_timing': 'wait', 'risk_level': 'high'}
    
    def _generate_volatility_recommendations(self, regime: VolatilityRegime, trend: VolatilityTrend,
                                           pattern: VolatilityPattern, metrics: VolatilityMetrics) -> Dict:
        """Gera recomendações específicas baseadas na volatilidade"""
        try:
            recommendations = {
                'strategies': [],
                'position_sizing': 1.0,
                'stop_adjustment': 1.0,
                'timing': 'immediate'
            }
            
            # Estratégias baseadas no regime
            if regime in [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.LOW]:
                recommendations['strategies'].extend([
                    'scalping', 'range_trading', 'mean_reversion', 'grid_trading'
                ])
                recommendations['position_sizing'] = 1.2
                recommendations['timing'] = 'immediate'
                
            elif regime == VolatilityRegime.NORMAL:
                recommendations['strategies'].extend([
                    'swing_trading', 'trend_following', 'breakout_trading'
                ])
                recommendations['position_sizing'] = 1.0
                recommendations['timing'] = 'immediate'
                
            elif regime == VolatilityRegime.HIGH:
                recommendations['strategies'].extend([
                    'trend_following', 'momentum_trading'
                ])
                recommendations['position_sizing'] = 0.7
                recommendations['timing'] = 'wait'
                
            elif regime == VolatilityRegime.EXTREMELY_HIGH:
                recommendations['strategies'].extend([
                    'wait_for_calm', 'defensive_positions'
                ])
                recommendations['position_sizing'] = 0.4
                recommendations['timing'] = 'avoid'
            
            # Ajustes baseados no padrão
            if pattern == VolatilityPattern.COMPRESSION:
                recommendations['strategies'].append('breakout_preparation')
                recommendations['position_sizing'] *= 1.3
                
            elif pattern == VolatilityPattern.EXPANSION:
                recommendations['strategies'] = ['trend_following', 'momentum_trading']
                recommendations['position_sizing'] *= 0.6
                
            elif pattern == VolatilityPattern.BREAKOUT:
                recommendations['strategies'].append('breakout_confirmation')
                recommendations['timing'] = 'wait'
                
            elif pattern == VolatilityPattern.MEAN_REVERSION:
                recommendations['strategies'].append('volatility_mean_reversion')
                if metrics.atr_percentile > 85:
                    recommendations['timing'] = 'wait'
            
            # Ajustes de stop loss
            if regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH]:
                recommendations['stop_adjustment'] = 1.5
            elif regime in [VolatilityRegime.LOW, VolatilityRegime.EXTREMELY_LOW]:
                recommendations['stop_adjustment'] = 0.7
            else:
                recommendations['stop_adjustment'] = 1.0
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Erro nas recomendações: {e}")
            return {'strategies': ['wait'], 'position_sizing': 0.5, 'stop_adjustment': 1.0, 'timing': 'avoid'}
    
    def _calculate_confidence(self, metrics: VolatilityMetrics, regime: VolatilityRegime, 
                            pattern: VolatilityPattern) -> float:
        """Calcula confiança na análise"""
        try:
            confidence_factors = []
            
            # Data quality (número de observações)
            confidence_factors.append(15)  # Assumindo dados completos
            
            # Clarity of regime
            if regime in [VolatilityRegime.EXTREMELY_LOW, VolatilityRegime.EXTREMELY_HIGH]:
                confidence_factors.append(25)  # Regime claro
            elif regime in [VolatilityRegime.LOW, VolatilityRegime.HIGH]:
                confidence_factors.append(20)
            else:
                confidence_factors.append(10)  # Regime normal (menos claro)
            
            # Pattern strength
            if pattern in [VolatilityPattern.COMPRESSION, VolatilityPattern.EXPANSION]:
                confidence_factors.append(20)  # Padrões claros
            else:
                confidence_factors.append(15)
            
            # Statistical significance
            if metrics.vol_persistence > 0.5:
                confidence_factors.append(15)
            else:
                confidence_factors.append(10)
            
            # Historical context
            if 10 < metrics.atr_percentile < 90:
                confidence_factors.append(15)  # Dentro de range normal
            else:
                confidence_factors.append(20)  # Extremos (mais confiáveis)
            
            return min(100, sum(confidence_factors))
            
        except Exception:
            return 60  # Default moderate confidence
    
    def _estimate_pattern_duration(self, pattern: VolatilityPattern, trend: VolatilityTrend) -> int:
        """Estima duração esperada do padrão"""
        try:
            base_duration = {
                VolatilityPattern.COMPRESSION: 15,      # 15 períodos
                VolatilityPattern.EXPANSION: 8,         # 8 períodos
                VolatilityPattern.BREAKOUT: 5,          # 5 períodos
                VolatilityPattern.MEAN_REVERSION: 10,   # 10 períodos
                VolatilityPattern.CLUSTERING: 20        # 20 períodos
            }
            
            duration = base_duration.get(pattern, 10)
            
            # Ajustar baseado no trend
            if trend == VolatilityTrend.EXPLOSIVE:
                duration = int(duration * 0.5)  # Padrões explosivos são curtos
            elif trend == VolatilityTrend.STABLE:
                duration = int(duration * 1.5)  # Padrões estáveis duram mais
            
            return max(3, min(50, duration))
            
        except Exception:
            return 10
    
    def _calculate_mean_reversion_target(self, metrics: VolatilityMetrics) -> Optional[float]:
        """Calcula target para mean reversion de volatilidade"""
        try:
            if metrics.atr_percentile > 80 or metrics.atr_percentile < 20:
                # Target é a mediana histórica (aproximadamente percentil 50)
                return 50.0
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_breakout_threshold(self, metrics: VolatilityMetrics) -> Optional[float]:
        """Calcula threshold para breakout de volatilidade"""
        try:
            if metrics.atr_percentile < 30:  # Low volatility
                # Breakout seria movimento para percentil 70+
                return 70.0
            else:
                return None
                
        except Exception:
            return None
    
    def should_trade_with_volatility(self, signal: VolatilitySignal, strategy_type: str) -> Dict:
        """
        Determina se deve tradear considerando a volatilidade
        
        Args:
            signal: VolatilitySignal atual
            strategy_type: Tipo de estratégia
            
        Returns:
            Dict com decisão e ajustes
        """
        try:
            should_trade = True
            adjustments = {}
            reasons = []
            
            # Verificar se estratégia é apropriada
            if strategy_type not in signal.recommended_strategies:
                if signal.entry_timing == 'avoid':
                    should_trade = False
                    reasons.append(f"Volatilidade {signal.regime.value} - evitar trading")
                elif signal.entry_timing == 'wait':
                    should_trade = False
                    reasons.append(f"Volatilidade {signal.regime.value} - aguardar melhores condições")
            
            # Aplicar ajustes se trading for permitido
            if should_trade:
                adjustments = {
                    'position_size_multiplier': signal.position_size_adjustment,
                    'stop_loss_multiplier': signal.stop_loss_adjustment,
                    'confidence_adjustment': signal.confidence / 100
                }
                
                if signal.regime == VolatilityRegime.EXTREMELY_HIGH:
                    adjustments['position_size_multiplier'] *= 0.5
                    reasons.append("Volatilidade extrema - reduzir size significativamente")
                
                if signal.signal_type == VolatilityPattern.COMPRESSION:
                    reasons.append("Compressão de volatilidade - oportunidade favorável")
                elif signal.signal_type == VolatilityPattern.EXPANSION:
                    reasons.append("Expansão de volatilidade - reduzir exposição")
            
            return {
                'should_trade': should_trade,
                'adjustments': adjustments,
                'reasons': reasons,
                'volatility_regime': signal.regime.value,
                'pattern': signal.signal_type.value,
                'confidence': signal.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Erro na decisão de trading: {e}")
            return {
                'should_trade': False,
                'adjustments': {'position_size_multiplier': 0.5},
                'reasons': ['Erro na análise de volatilidade'],
                'confidence': 30
            }


def main():
    """Teste básico do filtro de volatilidade"""
    # Criar dados com diferentes regimes de volatilidade
    dates = pd.date_range(start='2024-01-01', periods=200, freq='4H')
    np.random.seed(42)
    
    # Simular dados com volatilidade variável
    volatility_regimes = [0.01, 0.01, 0.03, 0.05, 0.02]  # Low to high vol
    regime_length = 40
    
    all_data = []
    base_price = 50000
    
    for i, vol in enumerate(volatility_regimes):
        regime_dates = dates[i*regime_length:(i+1)*regime_length]
        regime_returns = np.random.normal(0, vol, len(regime_dates))
        
        # Generate OHLCV data
        for j, date in enumerate(regime_dates):
            if j == 0 and i == 0:
                price = base_price
            else:
                price = all_data[-1]['close'] * (1 + regime_returns[j])
            
            daily_vol = vol * 0.5
            high = price * (1 + abs(np.random.normal(0, daily_vol)))
            low = price * (1 - abs(np.random.normal(0, daily_vol)))
            open_price = price * (1 + np.random.normal(0, daily_vol * 0.3))
            
            all_data.append({
                'open': open_price,
                'high': max(open_price, price, high),
                'low': min(open_price, price, low),
                'close': price,
                'volume': np.random.exponential(1000000)
            })
    
    df = pd.DataFrame(all_data, index=dates)
    
    # Testar filtro
    vol_filter = VolatilityFilter()
    signal = vol_filter.analyze_volatility(df, "BTCUSDT")
    
    print(f"\n⚡ VOLATILITY ANALYSIS")
    print(f"Regime: {signal.regime.value}")
    print(f"Trend: {signal.trend.value}")
    print(f"Pattern: {signal.signal_type.value}")
    print(f"Strength: {signal.strength:.1f}")
    print(f"Confidence: {signal.confidence:.1f}%")
    print(f"Current Percentile: {signal.current_percentile:.1f}")
    print(f"Expected Duration: {signal.expected_duration} periods")
    
    print(f"\n📊 TRADING IMPLICATIONS")
    print(f"Recommended Strategies: {', '.join(signal.recommended_strategies)}")
    print(f"Position Size Adjustment: {signal.position_size_adjustment:.2f}x")
    print(f"Stop Loss Adjustment: {signal.stop_loss_adjustment:.2f}x")
    print(f"Entry Timing: {signal.entry_timing}")
    
    # Testar decisão de trading
    trade_decision = vol_filter.should_trade_with_volatility(signal, "swing_trading")
    print(f"\n🎯 TRADE DECISION (Swing Trading)")
    print(f"Should Trade: {trade_decision['should_trade']}")
    if trade_decision['should_trade']:
        print(f"Position Size Multiplier: {trade_decision['adjustments']['position_size_multiplier']:.2f}x")
        print(f"Stop Loss Multiplier: {trade_decision['adjustments']['stop_loss_multiplier']:.2f}x")
    print(f"Reasons: {', '.join(trade_decision['reasons'])}")


if __name__ == "__main__":
    main()