"""
🌊 MARKET CONDITION FILTER - Smart Trading System v2.0

Filtro inteligente de condições de mercado:
- Bull/Bear/Sideways detection
- Market regime classification
- Momentum analysis
- Trend strength assessment
- Risk-on/Risk-off sentiment

Filosofia: Trade With The Market, Not Against It
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Condições de mercado"""
    STRONG_BULL = "strong_bull"           # Bull market forte
    MODERATE_BULL = "moderate_bull"       # Bull market moderado
    WEAK_BULL = "weak_bull"              # Bull market fraco
    SIDEWAYS_BULL = "sideways_bull"      # Lateral com viés alta
    SIDEWAYS = "sideways"                # Completamente lateral
    SIDEWAYS_BEAR = "sideways_bear"      # Lateral com viés baixa
    WEAK_BEAR = "weak_bear"              # Bear market fraco
    MODERATE_BEAR = "moderate_bear"      # Bear market moderado
    STRONG_BEAR = "strong_bear"          # Bear market forte


class MarketSentiment(Enum):
    """Sentimento de mercado"""
    EXTREME_GREED = "extreme_greed"      # Ganância extrema
    GREED = "greed"                      # Ganância
    NEUTRAL = "neutral"                  # Neutro
    FEAR = "fear"                        # Medo
    EXTREME_FEAR = "extreme_fear"        # Medo extremo


class TrendStrength(Enum):
    """Força da tendência"""
    VERY_STRONG = "very_strong"          # 80-100
    STRONG = "strong"                    # 60-80
    MODERATE = "moderate"                # 40-60
    WEAK = "weak"                        # 20-40
    VERY_WEAK = "very_weak"              # 0-20


@dataclass
class MarketAnalysis:
    """Análise completa das condições de mercado"""
    condition: MarketCondition
    sentiment: MarketSentiment
    trend_strength: TrendStrength
    momentum_score: float                # 0-100
    volatility_regime: str               # 'low', 'normal', 'high', 'extreme'
    risk_on_off: str                     # 'risk_on', 'risk_off', 'neutral'
    confidence: float                    # 0-100 confiança na análise
    
    # Métricas detalhadas
    trend_direction: str                 # 'up', 'down', 'sideways'
    trend_duration_days: int             # Duração da tendência atual
    support_resistance_strength: float   # Força dos S/R
    volume_confirmation: bool            # Volume confirma movimento
    breadth_indicators: Dict             # Indicadores de amplitude
    
    # Recomendações de trading
    recommended_strategies: List[str]    # Estratégias recomendadas
    avoid_strategies: List[str]          # Estratégias a evitar
    position_sizing_modifier: float     # Modificador de position size
    
    timestamp: pd.Timestamp


class MarketConditionFilter:
    """
    🌊 Filtro Principal de Condições de Mercado
    
    Analisa múltiplos aspectos do mercado para determinar:
    1. Condição geral (Bull/Bear/Sideways)
    2. Sentimento (Fear/Greed)
    3. Força da tendência
    4. Regime de volatilidade
    5. Risk-on/Risk-off
    """
    
    def __init__(self,
                 trend_lookback: int = 50,
                 momentum_periods: int = 14,
                 volatility_periods: int = 20):
        
        self.trend_lookback = trend_lookback
        self.momentum_periods = momentum_periods
        self.volatility_periods = volatility_periods
        
        self.logger = logging.getLogger(f"{__name__}.MarketConditionFilter")
        
        # Configurações de thresholds
        self.thresholds = {
            'strong_trend': 0.7,           # 70% para trend forte
            'moderate_trend': 0.5,         # 50% para trend moderado
            'weak_trend': 0.3,             # 30% para trend fraco
            'sideways_threshold': 0.15,    # 15% para sideways
            'high_volatility': 0.04,       # 4% para alta volatilidade
            'low_volatility': 0.015,       # 1.5% para baixa volatilidade
            'momentum_overbought': 80,     # RSI overbought
            'momentum_oversold': 20,       # RSI oversold
        }
        
        # Cache para otimização
        self.analysis_cache: Dict[str, MarketAnalysis] = {}
        self.cache_timestamp: Optional[pd.Timestamp] = None
    
    def analyze_market_condition(self, 
                               market_data: Dict[str, pd.DataFrame],
                               symbol: str = "BTCUSDT") -> MarketAnalysis:
        """
        Análise principal das condições de mercado
        
        Args:
            market_data: Dict com DataFrames por timeframe
            symbol: Símbolo para análise específica
            
        Returns:
            MarketAnalysis com condições completas
        """
        try:
            self.logger.info(f"Analisando condições de mercado para {symbol}")
            
            # Verificar cache
            cache_key = f"{symbol}_{pd.Timestamp.now().floor('H')}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Usar dados 4H como principal para análise
            primary_data = market_data.get("4H")
            if primary_data is None or len(primary_data) < self.trend_lookback:
                raise ValueError("Dados insuficientes para análise")
            
            # 1. Análise de Tendência
            trend_analysis = self._analyze_trend(primary_data)
            
            # 2. Análise de Momentum
            momentum_analysis = self._analyze_momentum(primary_data)
            
            # 3. Análise de Volatilidade
            volatility_analysis = self._analyze_volatility(primary_data)
            
            # 4. Análise de Volume
            volume_analysis = self._analyze_volume(primary_data)
            
            # 5. Análise Multi-timeframe
            mtf_analysis = self._analyze_multi_timeframe(market_data)
            
            # 6. Combinar análises para determinar condição
            market_condition = self._determine_market_condition(
                trend_analysis, momentum_analysis, volatility_analysis, mtf_analysis)
            
            # 7. Determinar sentimento
            sentiment = self._determine_market_sentiment(
                momentum_analysis, volatility_analysis, trend_analysis)
            
            # 8. Calcular força da tendência
            trend_strength = self._calculate_trend_strength(trend_analysis, momentum_analysis)
            
            # 9. Análise Risk-on/Risk-off
            risk_sentiment = self._analyze_risk_sentiment(
                trend_analysis, volatility_analysis, momentum_analysis)
            
            # 10. Gerar recomendações
            recommendations = self._generate_trading_recommendations(
                market_condition, sentiment, trend_strength, volatility_analysis)
            
            # Criar análise final
            analysis = MarketAnalysis(
                condition=market_condition,
                sentiment=sentiment,
                trend_strength=trend_strength,
                momentum_score=momentum_analysis['momentum_score'],
                volatility_regime=volatility_analysis['regime'],
                risk_on_off=risk_sentiment,
                confidence=self._calculate_confidence(trend_analysis, momentum_analysis, volatility_analysis),
                
                # Métricas detalhadas
                trend_direction=trend_analysis['direction'],
                trend_duration_days=trend_analysis['duration_days'],
                support_resistance_strength=trend_analysis['sr_strength'],
                volume_confirmation=volume_analysis['trend_confirmation'],
                breadth_indicators=mtf_analysis,
                
                # Recomendações
                recommended_strategies=recommendations['recommended'],
                avoid_strategies=recommendations['avoid'],
                position_sizing_modifier=recommendations['sizing_modifier'],
                
                timestamp=pd.Timestamp.now()
            )
            
            # Cache resultado
            self.analysis_cache[cache_key] = analysis
            
            self.logger.info(f"Análise concluída - Condição: {market_condition.value}, "
                           f"Sentimento: {sentiment.value}, Força: {trend_strength.value}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise de condições: {e}")
            raise
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analisa tendência principal"""
        try:
            recent_data = data.tail(self.trend_lookback)
            
            # 1. Análise de preço (EMAs)
            ema_short = recent_data['close'].ewm(span=20).mean()
            ema_medium = recent_data['close'].ewm(span=50).mean()
            ema_long = recent_data['close'].ewm(span=100).mean() if len(recent_data) >= 100 else ema_medium
            
            current_price = recent_data['close'].iloc[-1]
            
            # 2. Posição relativa das EMAs
            ema_alignment_bull = (ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1])
            ema_alignment_bear = (ema_short.iloc[-1] < ema_medium.iloc[-1] < ema_long.iloc[-1])
            
            # 3. Slope das EMAs (direção)
            ema_short_slope = (ema_short.iloc[-1] - ema_short.iloc[-5]) / ema_short.iloc[-5]
            ema_medium_slope = (ema_medium.iloc[-1] - ema_medium.iloc[-5]) / ema_medium.iloc[-5]
            
            # 4. Highs/Lows analysis
            highs = recent_data['high'].rolling(10).max()
            lows = recent_data['low'].rolling(10).min()
            
            higher_highs = (highs.iloc[-1] > highs.iloc[-10:].iloc[:-1].max())
            higher_lows = (lows.iloc[-1] > lows.iloc[-10:].iloc[:-1].min())
            lower_highs = (highs.iloc[-1] < highs.iloc[-10:].iloc[:-1].max())
            lower_lows = (lows.iloc[-1] < lows.iloc[-10:].iloc[:-1].min())
            
            # 5. Determinar direção e força
            bullish_signals = sum([
                ema_alignment_bull,
                ema_short_slope > 0,
                ema_medium_slope > 0,
                higher_highs and higher_lows,
                current_price > ema_short.iloc[-1]
            ])
            
            bearish_signals = sum([
                ema_alignment_bear,
                ema_short_slope < 0,
                ema_medium_slope < 0,
                lower_highs and lower_lows,
                current_price < ema_short.iloc[-1]
            ])
            
            # 6. Calcular força da tendência
            if bullish_signals > bearish_signals:
                direction = "up"
                strength = bullish_signals / 5.0
            elif bearish_signals > bullish_signals:
                direction = "down"
                strength = bearish_signals / 5.0
            else:
                direction = "sideways"
                strength = 0.3
            
            # 7. Duração da tendência (simplificado)
            trend_duration = self._calculate_trend_duration(recent_data, direction)
            
            # 8. Força dos S/R (simplificado)
            sr_strength = self._calculate_sr_strength(recent_data)
            
            return {
                'direction': direction,
                'strength': strength,
                'duration_days': trend_duration,
                'sr_strength': sr_strength,
                'ema_alignment_bull': ema_alignment_bull,
                'ema_alignment_bear': ema_alignment_bear,
                'ema_short_slope': ema_short_slope,
                'ema_medium_slope': ema_medium_slope,
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'lower_highs': lower_highs,
                'lower_lows': lower_lows
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de tendência: {e}")
            return {'direction': 'sideways', 'strength': 0.3, 'duration_days': 0, 'sr_strength': 50}
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analisa momentum do mercado"""
        try:
            recent_data = data.tail(50)
            
            # 1. RSI
            delta = recent_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.momentum_periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.momentum_periods).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 2. MACD
            ema_12 = recent_data['close'].ewm(span=12).mean()
            ema_26 = recent_data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            macd_bullish = macd.iloc[-1] > macd_signal.iloc[-1] and macd_histogram.iloc[-1] > 0
            macd_bearish = macd.iloc[-1] < macd_signal.iloc[-1] and macd_histogram.iloc[-1] < 0
            
            # 3. Rate of Change
            roc_periods = 10
            roc = ((recent_data['close'] / recent_data['close'].shift(roc_periods)) - 1) * 100
            current_roc = roc.iloc[-1]
            
            # 4. Stochastic
            lowest_low = recent_data['low'].rolling(window=self.momentum_periods).min()
            highest_high = recent_data['high'].rolling(window=self.momentum_periods).max()
            k_percent = 100 * ((recent_data['close'] - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=3).mean()
            
            current_stoch_k = k_percent.iloc[-1]
            current_stoch_d = d_percent.iloc[-1]
            
            # 5. Calcular momentum score (0-100)
            momentum_factors = []
            
            # RSI contribution
            if current_rsi > 70:
                rsi_score = 80 + (current_rsi - 70) * 0.67  # 80-100
            elif current_rsi < 30:
                rsi_score = 20 - (30 - current_rsi) * 0.67  # 0-20
            else:
                rsi_score = 20 + (current_rsi - 30) * 1.5   # 20-80
            
            momentum_factors.append(rsi_score)
            
            # MACD contribution
            if macd_bullish:
                macd_score = 70
            elif macd_bearish:
                macd_score = 30
            else:
                macd_score = 50
            
            momentum_factors.append(macd_score)
            
            # ROC contribution
            roc_score = 50 + (current_roc * 2)  # Simplified
            roc_score = max(0, min(100, roc_score))
            momentum_factors.append(roc_score)
            
            # Stochastic contribution
            if current_stoch_k > 80:
                stoch_score = 85
            elif current_stoch_k < 20:
                stoch_score = 15
            else:
                stoch_score = current_stoch_k
            
            momentum_factors.append(stoch_score)
            
            # Calcular score final
            momentum_score = np.mean(momentum_factors)
            
            return {
                'momentum_score': momentum_score,
                'rsi': current_rsi,
                'macd_bullish': macd_bullish,
                'macd_bearish': macd_bearish,
                'roc': current_roc,
                'stoch_k': current_stoch_k,
                'stoch_d': current_stoch_d,
                'rsi_overbought': current_rsi > self.thresholds['momentum_overbought'],
                'rsi_oversold': current_rsi < self.thresholds['momentum_oversold']
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de momentum: {e}")
            return {'momentum_score': 50, 'rsi': 50, 'macd_bullish': False, 'macd_bearish': False}
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analisa regime de volatilidade"""
        try:
            recent_data = data.tail(50)
            
            # 1. ATR (Average True Range)
            high_low = recent_data['high'] - recent_data['low']
            high_close = np.abs(recent_data['high'] - recent_data['close'].shift())
            low_close = np.abs(recent_data['low'] - recent_data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.volatility_periods).mean()
            atr_pct = atr / recent_data['close'] * 100
            
            current_atr_pct = atr_pct.iloc[-1]
            avg_atr_pct = atr_pct.mean()
            
            # 2. Realized Volatility (returns)
            returns = recent_data['close'].pct_change().dropna()
            realized_vol = returns.std() * np.sqrt(365) * 100  # Anualizada
            
            # 3. Bollinger Bands Width
            bb_period = 20
            bb_std = 2
            bb_middle = recent_data['close'].rolling(bb_period).mean()
            bb_std_dev = recent_data['close'].rolling(bb_period).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            bb_width = (bb_upper - bb_lower) / bb_middle * 100
            
            current_bb_width = bb_width.iloc[-1]
            avg_bb_width = bb_width.mean()
            
            # 4. Determinar regime de volatilidade
            if current_atr_pct > self.thresholds['high_volatility']:
                if current_atr_pct > self.thresholds['high_volatility'] * 2:
                    regime = "extreme"
                else:
                    regime = "high"
            elif current_atr_pct < self.thresholds['low_volatility']:
                regime = "low"
            else:
                regime = "normal"
            
            # 5. Volatility trend (increasing/decreasing)
            vol_trend_periods = 10
            recent_atr = atr_pct.tail(vol_trend_periods)
            vol_trend = "increasing" if recent_atr.iloc[-1] > recent_atr.iloc[0] else "decreasing"
            
            return {
                'regime': regime,
                'current_atr_pct': current_atr_pct,
                'avg_atr_pct': avg_atr_pct,
                'realized_vol': realized_vol,
                'bb_width': current_bb_width,
                'avg_bb_width': avg_bb_width,
                'vol_trend': vol_trend,
                'vol_percentile': (current_atr_pct / atr_pct.quantile(0.9)) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volatilidade: {e}")
            return {'regime': 'normal', 'current_atr_pct': 2.0, 'vol_trend': 'stable'}
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analisa padrões de volume"""
        try:
            recent_data = data.tail(30)
            
            # 1. Volume médio
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # 2. Volume trend
            volume_ma_short = recent_data['volume'].rolling(5).mean()
            volume_ma_long = recent_data['volume'].rolling(15).mean()
            volume_trend = "increasing" if volume_ma_short.iloc[-1] > volume_ma_long.iloc[-1] else "decreasing"
            
            # 3. Price-Volume confirmation
            price_changes = recent_data['close'].pct_change().tail(5)
            volume_changes = recent_data['volume'].pct_change().tail(5)
            
            # Correlação entre price e volume (deve ser positiva em trends)
            price_vol_correlation = price_changes.corr(volume_changes)
            
            # 4. Trend confirmation
            last_5_moves = recent_data['close'].pct_change().tail(5)
            last_5_volumes = recent_data['volume'].tail(5)
            
            up_moves_volume = last_5_volumes[last_5_moves > 0].mean() if (last_5_moves > 0).any() else 0
            down_moves_volume = last_5_volumes[last_5_moves < 0].mean() if (last_5_moves < 0).any() else 0
            
            if up_moves_volume > 0 and down_moves_volume > 0:
                volume_bias = "bullish" if up_moves_volume > down_moves_volume else "bearish"
                trend_confirmation = abs(up_moves_volume - down_moves_volume) / max(up_moves_volume, down_moves_volume) > 0.2
            else:
                volume_bias = "neutral"
                trend_confirmation = False
            
            return {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'price_vol_correlation': price_vol_correlation,
                'volume_bias': volume_bias,
                'trend_confirmation': trend_confirmation,
                'avg_volume': avg_volume,
                'current_volume': current_volume
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volume: {e}")
            return {'volume_ratio': 1.0, 'volume_trend': 'stable', 'trend_confirmation': False}
    
    def _analyze_multi_timeframe(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analisa alinhamento multi-timeframe"""
        try:
            timeframe_trends = {}
            
            for tf, data in market_data.items():
                if len(data) < 20:
                    continue
                
                # Análise simples de trend por TF
                ema_20 = data['close'].ewm(span=20).mean()
                ema_50 = data['close'].ewm(span=50).mean() if len(data) >= 50 else ema_20
                
                current_price = data['close'].iloc[-1]
                
                if current_price > ema_20.iloc[-1] > ema_50.iloc[-1]:
                    tf_trend = "bullish"
                elif current_price < ema_20.iloc[-1] < ema_50.iloc[-1]:
                    tf_trend = "bearish"
                else:
                    tf_trend = "neutral"
                
                timeframe_trends[tf] = tf_trend
            
            # Calcular alinhamento
            total_tfs = len(timeframe_trends)
            bullish_tfs = sum(1 for trend in timeframe_trends.values() if trend == "bullish")
            bearish_tfs = sum(1 for trend in timeframe_trends.values() if trend == "bearish")
            
            alignment_score = max(bullish_tfs, bearish_tfs) / total_tfs if total_tfs > 0 else 0
            
            if bullish_tfs > bearish_tfs:
                alignment_direction = "bullish"
            elif bearish_tfs > bullish_tfs:
                alignment_direction = "bearish"
            else:
                alignment_direction = "neutral"
            
            return {
                'timeframe_trends': timeframe_trends,
                'alignment_score': alignment_score,
                'alignment_direction': alignment_direction,
                'bullish_tfs': bullish_tfs,
                'bearish_tfs': bearish_tfs,
                'total_tfs': total_tfs
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise multi-timeframe: {e}")
            return {'alignment_score': 0.5, 'alignment_direction': 'neutral'}
    
    def _calculate_trend_duration(self, data: pd.DataFrame, direction: str) -> int:
        """Calcula duração aproximada da tendência atual"""
        try:
            # Simplificado - conta períodos consecutivos na mesma direção
            ema_20 = data['close'].ewm(span=20).mean()
            
            duration = 0
            for i in range(len(data) - 1, 0, -1):
                if direction == "up":
                    if data['close'].iloc[i] > ema_20.iloc[i]:
                        duration += 1
                    else:
                        break
                elif direction == "down":
                    if data['close'].iloc[i] < ema_20.iloc[i]:
                        duration += 1
                    else:
                        break
                else:
                    break
            
            # Converter períodos para dias (assumindo 4H = 6 períodos/dia)
            return duration // 6
            
        except Exception:
            return 0
    
    def _calculate_sr_strength(self, data: pd.DataFrame) -> float:
        """Calcula força média dos suportes e resistências"""
        try:
            # Simplificado - baseado na frequência de toques em levels
            recent_data = data.tail(30)
            
            # Encontrar pivots
            highs = recent_data['high'].rolling(5, center=True).max()
            lows = recent_data['low'].rolling(5, center=True).min()
            
            # Contar toques em levels similares
            sr_touches = 0
            total_levels = 0
            
            for i in range(len(recent_data)):
                current_high = recent_data['high'].iloc[i]
                current_low = recent_data['low'].iloc[i]
                
                # Contar toques próximos
                high_touches = sum(1 for h in recent_data['high'] if abs(h - current_high) / current_high < 0.01)
                low_touches = sum(1 for l in recent_data['low'] if abs(l - current_low) / current_low < 0.01)
                
                if high_touches > 2:
                    sr_touches += high_touches
                    total_levels += 1
                if low_touches > 2:
                    sr_touches += low_touches
                    total_levels += 1
            
            return min(100, (sr_touches / max(1, total_levels)) * 20)
            
        except Exception:
            return 50
    
    def _determine_market_condition(self, trend_analysis: Dict, momentum_analysis: Dict, 
                                  volatility_analysis: Dict, mtf_analysis: Dict) -> MarketCondition:
        """Determina condição geral do mercado"""
        try:
            trend_direction = trend_analysis['direction']
            trend_strength = trend_analysis['strength']
            momentum_score = momentum_analysis['momentum_score']
            volatility_regime = volatility_analysis['regime']
            mtf_alignment = mtf_analysis['alignment_score']
            
            # Lógica de decisão baseada em múltiplos fatores
            if trend_direction == "up":
                if trend_strength > 0.7 and momentum_score > 70 and mtf_alignment > 0.7:
                    return MarketCondition.STRONG_BULL
                elif trend_strength > 0.5 and momentum_score > 55:
                    return MarketCondition.MODERATE_BULL
                elif trend_strength > 0.3:
                    return MarketCondition.WEAK_BULL
                else:
                    return MarketCondition.SIDEWAYS_BULL
                    
            elif trend_direction == "down":
                if trend_strength > 0.7 and momentum_score < 30 and mtf_alignment > 0.7:
                    return MarketCondition.STRONG_BEAR
                elif trend_strength > 0.5 and momentum_score < 45:
                    return MarketCondition.MODERATE_BEAR
                elif trend_strength > 0.3:
                    return MarketCondition.WEAK_BEAR
                else:
                    return MarketCondition.SIDEWAYS_BEAR
                    
            else:  # sideways
                if momentum_score > 60:
                    return MarketCondition.SIDEWAYS_BULL
                elif momentum_score < 40:
                    return MarketCondition.SIDEWAYS_BEAR
                else:
                    return MarketCondition.SIDEWAYS
                    
        except Exception as e:
            self.logger.error(f"Erro na determinação de condição: {e}")
            return MarketCondition.SIDEWAYS
    
    def _determine_market_sentiment(self, momentum_analysis: Dict, volatility_analysis: Dict, 
                                  trend_analysis: Dict) -> MarketSentiment:
        """Determina sentimento do mercado"""
        try:
            rsi = momentum_analysis.get('rsi', 50)
            volatility_regime = volatility_analysis['regime']
            trend_strength = trend_analysis['strength']
            
            # Lógica de sentimento
            if rsi > 80 and volatility_regime in ['high', 'extreme'] and trend_strength > 0.6:
                return MarketSentiment.EXTREME_GREED
            elif rsi > 70:
                return MarketSentiment.GREED
            elif rsi < 20 and volatility_regime in ['high', 'extreme']:
                return MarketSentiment.EXTREME_FEAR
            elif rsi < 30:
                return MarketSentiment.FEAR
            else:
                return MarketSentiment.NEUTRAL
                
        except Exception:
            return MarketSentiment.NEUTRAL
    
    def _calculate_trend_strength(self, trend_analysis: Dict, momentum_analysis: Dict) -> TrendStrength:
        """Calcula força da tendência"""
        try:
            trend_strength = trend_analysis['strength']
            momentum_score = momentum_analysis['momentum_score']
            
            # Combinar trend e momentum
            combined_strength = (trend_strength * 0.6 + (momentum_score / 100) * 0.4)
            
            if combined_strength > 0.8:
                return TrendStrength.VERY_STRONG
            elif combined_strength > 0.6:
                return TrendStrength.STRONG
            elif combined_strength > 0.4:
                return TrendStrength.MODERATE
            elif combined_strength > 0.2:
                return TrendStrength.WEAK
            else:
                return TrendStrength.VERY_WEAK
                
        except Exception:
            return TrendStrength.MODERATE
    
    def _analyze_risk_sentiment(self, trend_analysis: Dict, volatility_analysis: Dict, 
                              momentum_analysis: Dict) -> str:
        """Analisa sentimento risk-on/risk-off"""
        try:
            trend_direction = trend_analysis['direction']
            volatility_regime = volatility_analysis['regime']
            momentum_score = momentum_analysis['momentum_score']
            
            # Risk-on conditions: uptrend + normal/low vol + positive momentum
            risk_on_score = 0
            if trend_direction == "up":
                risk_on_score += 1
            if volatility_regime in ['low', 'normal']:
                risk_on_score += 1
            if momentum_score > 55:
                risk_on_score += 1
            
            if risk_on_score >= 2:
                return "risk_on"
            elif risk_on_score == 1:
                return "neutral"
            else:
                return "risk_off"
                
        except Exception:
            return "neutral"
    
    def _generate_trading_recommendations(self, condition: MarketCondition, sentiment: MarketSentiment,
                                        trend_strength: TrendStrength, volatility_analysis: Dict) -> Dict:
        """Gera recomendações de trading baseadas nas condições"""
        try:
            recommended = []
            avoid = []
            sizing_modifier = 1.0
            
            # Baseado na condição do mercado
            if condition in [MarketCondition.STRONG_BULL, MarketCondition.MODERATE_BULL]:
                recommended.extend(["trend_following", "breakout_long", "swing_long"])
                avoid.extend(["mean_reversion_short", "range_trading"])
                sizing_modifier = 1.2
                
            elif condition in [MarketCondition.STRONG_BEAR, MarketCondition.MODERATE_BEAR]:
                recommended.extend(["trend_following_short", "breakout_short", "swing_short"])
                avoid.extend(["trend_following_long", "breakout_long"])
                sizing_modifier = 1.1
                
            elif condition == MarketCondition.SIDEWAYS:
                recommended.extend(["range_trading", "mean_reversion", "scalping"])
                avoid.extend(["trend_following", "breakout"])
                sizing_modifier = 0.8
                
            # Ajustes baseados no sentimento
            if sentiment in [MarketSentiment.EXTREME_GREED, MarketSentiment.EXTREME_FEAR]:
                avoid.extend(["large_positions", "high_leverage"])
                sizing_modifier *= 0.7
                
            # Ajustes baseados na volatilidade
            if volatility_analysis['regime'] == 'extreme':
                avoid.extend(["tight_stops", "scalping"])
                sizing_modifier *= 0.6
            elif volatility_analysis['regime'] == 'low':
                recommended.extend(["scalping", "range_trading"])
                sizing_modifier *= 1.1
            
            return {
                'recommended': recommended,
                'avoid': avoid,
                'sizing_modifier': max(0.3, min(1.5, sizing_modifier))
            }
            
        except Exception as e:
            self.logger.error(f"Erro nas recomendações: {e}")
            return {'recommended': [], 'avoid': [], 'sizing_modifier': 1.0}
    
    def _calculate_confidence(self, trend_analysis: Dict, momentum_analysis: Dict, 
                            volatility_analysis: Dict) -> float:
        """Calcula confiança na análise"""
        try:
            # Fatores que aumentam confiança
            confidence_factors = []
            
            # Trend clarity
            if trend_analysis['strength'] > 0.7:
                confidence_factors.append(20)
            elif trend_analysis['strength'] > 0.5:
                confidence_factors.append(15)
            else:
                confidence_factors.append(5)
            
            # Momentum confirmation
            momentum_score = momentum_analysis['momentum_score']
            if momentum_score > 70 or momentum_score < 30:
                confidence_factors.append(20)  # Extreme momentum
            elif momentum_score > 60 or momentum_score < 40:
                confidence_factors.append(15)  # Moderate momentum
            else:
                confidence_factors.append(5)   # Weak momentum
            
            # Volatility regime
            vol_regime = volatility_analysis['regime']
            if vol_regime in ['normal', 'low']:
                confidence_factors.append(15)
            elif vol_regime == 'high':
                confidence_factors.append(10)
            else:  # extreme
                confidence_factors.append(5)
            
            # Data quality (assumindo sempre boa para simplificar)
            confidence_factors.append(15)
            
            return min(100, sum(confidence_factors))
            
        except Exception:
            return 60  # Default moderate confidence
    
    def should_trade(self, analysis: MarketAnalysis, strategy_type: str) -> Dict:
        """
        Determina se deve tradear baseado nas condições
        
        Args:
            analysis: MarketAnalysis atual
            strategy_type: Tipo de estratégia ("swing", "breakout", "scalping", etc.)
            
        Returns:
            Dict com decisão e reasoning
        """
        try:
            should_trade = True
            reasons = []
            risk_adjustment = 1.0
            
            # Verificar se estratégia é recomendada
            if strategy_type in analysis.avoid_strategies:
                should_trade = False
                reasons.append(f"Estratégia {strategy_type} não recomendada para {analysis.condition.value}")
            
            # Verificar volatilidade extrema
            if analysis.volatility_regime == "extreme":
                if strategy_type in ["scalping", "tight_stops"]:
                    should_trade = False
                    reasons.append("Volatilidade extrema - evitar estratégias de curto prazo")
                else:
                    risk_adjustment *= 0.6
                    reasons.append("Volatilidade extrema - reduzir position size")
            
            # Verificar sentimento extremo
            if analysis.sentiment in [MarketSentiment.EXTREME_GREED, MarketSentiment.EXTREME_FEAR]:
                risk_adjustment *= 0.7
                reasons.append(f"Sentimento extremo ({analysis.sentiment.value}) - reduzir exposição")
            
            # Verificar confiança na análise
            if analysis.confidence < 50:
                should_trade = False
                reasons.append("Baixa confiança na análise de mercado")
            elif analysis.confidence < 70:
                risk_adjustment *= 0.8
                reasons.append("Confiança moderada - reduzir position size")
            
            # Aplicar modificador de position sizing
            final_risk_adjustment = risk_adjustment * analysis.position_sizing_modifier
            
            return {
                'should_trade': should_trade,
                'risk_adjustment': final_risk_adjustment,
                'reasons': reasons,
                'confidence': analysis.confidence,
                'recommended_strategies': analysis.recommended_strategies
            }
            
        except Exception as e:
            self.logger.error(f"Erro na decisão de trade: {e}")
            return {'should_trade': False, 'risk_adjustment': 0.5, 'reasons': ['Erro na análise']}


def main():
    """Teste básico do filtro de condições"""
    # Dados de exemplo
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    def create_trending_data(dates, base_price=50000, trend="up"):
        if trend == "up":
            price_changes = np.random.randn(len(dates)).cumsum() * 50 + np.arange(len(dates)) * 20
        elif trend == "down":
            price_changes = np.random.randn(len(dates)).cumsum() * 50 - np.arange(len(dates)) * 15
        else:  # sideways
            price_changes = np.random.randn(len(dates)).cumsum() * 30
        
        opens = base_price + price_changes
        closes = opens + np.random.randn(len(dates)) * 25
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(len(dates)) * 30)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(len(dates)) * 30)
        volumes = np.random.exponential(1000000, len(dates))
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
    
    # Criar dados simulando bull market
    market_data = {
        "1H": create_trending_data(dates, trend="up"),
        "4H": create_trending_data(dates[::4], trend="up"),
        "1D": create_trending_data(dates[::24], trend="up")
    }
    
    # Testar filtro
    filter = MarketConditionFilter()
    analysis = filter.analyze_market_condition(market_data, "BTCUSDT")
    
    print(f"\n🌊 MARKET CONDITION ANALYSIS")
    print(f"Condition: {analysis.condition.value}")
    print(f"Sentiment: {analysis.sentiment.value}")
    print(f"Trend Strength: {analysis.trend_strength.value}")
    print(f"Momentum Score: {analysis.momentum_score:.1f}")
    print(f"Volatility Regime: {analysis.volatility_regime}")
    print(f"Risk Sentiment: {analysis.risk_on_off}")
    print(f"Confidence: {analysis.confidence:.1f}%")
    print(f"Trend Direction: {analysis.trend_direction}")
    print(f"Trend Duration: {analysis.trend_duration_days} days")
    
    print(f"\n📈 RECOMMENDATIONS")
    print(f"Recommended Strategies: {', '.join(analysis.recommended_strategies)}")
    print(f"Avoid Strategies: {', '.join(analysis.avoid_strategies)}")
    print(f"Position Sizing Modifier: {analysis.position_sizing_modifier:.2f}x")
    
    # Testar decisão de trade
    trade_decision = filter.should_trade(analysis, "swing")
    print(f"\n🎯 TRADE DECISION (Swing Strategy)")
    print(f"Should Trade: {trade_decision['should_trade']}")
    print(f"Risk Adjustment: {trade_decision['risk_adjustment']:.2f}x")
    print(f"Reasons: {', '.join(trade_decision['reasons'])}")


if __name__ == "__main__":
    main()