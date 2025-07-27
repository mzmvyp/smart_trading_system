"""
🕐 TIME FILTER - Smart Trading System v2.0

Filtro inteligente de timing baseado em:
- Trading sessions (Asian/European/US)
- Optimal trading hours por estratégia
- Volume and liquidity analysis
- Weekend/holiday effects
- Intraday patterns

Filosofia: Right Time = Right Liquidity = Better Execution = Higher Probability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, time, timezone
import pytz

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Sessões de trading"""
    ASIAN = "asian"           # Tokyo: 00:00-09:00 UTC
    EUROPEAN = "european"     # London: 08:00-17:00 UTC  
    US = "us"                # New York: 13:00-22:00 UTC
    OVERLAP_ASIAN_EU = "overlap_asian_eu"      # 08:00-09:00 UTC
    OVERLAP_EU_US = "overlap_eu_us"            # 13:00-17:00 UTC
    QUIET_HOURS = "quiet_hours"                # 22:00-00:00 UTC


class TimeOfDay(Enum):
    """Períodos do dia"""
    EARLY_MORNING = "early_morning"    # 00:00-06:00
    MORNING = "morning"                # 06:00-12:00
    AFTERNOON = "afternoon"            # 12:00-18:00
    EVENING = "evening"                # 18:00-24:00


class WeekPeriod(Enum):
    """Períodos da semana"""
    MONDAY_OPEN = "monday_open"        # Segunda abertura
    MIDWEEK = "midweek"               # Terça-Quinta
    FRIDAY_CLOSE = "friday_close"      # Sexta fechamento
    WEEKEND = "weekend"                # Sábado-Domingo


class LiquidityLevel(Enum):
    """Níveis de liquidez"""
    VERY_HIGH = "very_high"           # Overlaps de sessões
    HIGH = "high"                     # Sessões principais
    MODERATE = "moderate"             # Sessões individuais
    LOW = "low"                       # Transições
    VERY_LOW = "very_low"             # Weekend/quiet hours


@dataclass
class TimeMetrics:
    """Métricas temporais"""
    current_session: TradingSession
    time_of_day: TimeOfDay
    week_period: WeekPeriod
    liquidity_level: LiquidityLevel
    
    # Volume analysis
    current_volume: float
    session_avg_volume: float
    volume_percentile: float          # 0-100
    volume_trend: str                 # 'increasing', 'decreasing', 'stable'
    
    # Spread and execution metrics
    estimated_spread: float           # Estimated bid-ask spread
    execution_quality: float          # 0-100 execution quality score
    slippage_risk: str               # 'low', 'medium', 'high'
    
    # Historical performance
    session_performance: Dict[str, float]  # Win rate por sessão
    optimal_hours: List[int]          # Horas ótimas (UTC)
    
    # Market microstructure
    price_stability: float            # Volatilidade intraday
    mean_reversion_tendency: float    # Tendência de reversão


@dataclass
class TimeSignal:
    """Sinal de timing"""
    timing_score: float               # 0-100 qualidade do timing
    liquidity_score: float            # 0-100 qualidade da liquidez
    execution_score: float            # 0-100 qualidade de execução
    overall_score: float              # Score geral
    
    # Recommendations
    entry_timing: str                 # 'immediate', 'wait', 'avoid'
    optimal_timeframe: str            # Timeframe recomendado
    position_size_adjustment: float   # Ajuste de tamanho
    execution_strategy: str           # 'market', 'limit', 'twap'
    
    # Session info
    current_session: TradingSession
    session_ends_in: int              # Minutos até fim da sessão
    next_optimal_time: Optional[datetime]  # Próximo horário ótimo
    
    # Risk factors
    weekend_risk: bool                # Risco de weekend
    news_timing_risk: bool            # Risco de timing de news
    low_liquidity_warning: bool       # Aviso de baixa liquidez
    
    timestamp: pd.Timestamp


class TimeFilter:
    """
    🕐 Filtro Principal de Timing
    
    Analisa fatores temporais para otimizar timing:
    1. Trading sessions e overlaps
    2. Volume e liquidez intraday
    3. Padrões históricos de performance
    4. Microstructure do mercado
    5. Otimização de execução
    """
    
    def __init__(self,
                 min_liquidity_score: float = 40.0,
                 min_volume_percentile: float = 30.0):
        
        self.min_liquidity_score = min_liquidity_score
        self.min_volume_percentile = min_volume_percentile
        
        self.logger = logging.getLogger(f"{__name__}.TimeFilter")
        
        # Definição de sessões (UTC)
        self.session_times = {
            TradingSession.ASIAN: (0, 9),           # 00:00-09:00 UTC
            TradingSession.EUROPEAN: (8, 17),       # 08:00-17:00 UTC
            TradingSession.US: (13, 22),            # 13:00-22:00 UTC
            TradingSession.OVERLAP_ASIAN_EU: (8, 9), # 08:00-09:00 UTC
            TradingSession.OVERLAP_EU_US: (13, 17),  # 13:00-17:00 UTC
            TradingSession.QUIET_HOURS: (22, 24),   # 22:00-24:00 UTC
        }
        
        # Configurações de liquidez por sessão
        self.session_liquidity = {
            TradingSession.OVERLAP_EU_US: LiquidityLevel.VERY_HIGH,
            TradingSession.OVERLAP_ASIAN_EU: LiquidityLevel.HIGH,
            TradingSession.EUROPEAN: LiquidityLevel.HIGH,
            TradingSession.US: LiquidityLevel.HIGH,
            TradingSession.ASIAN: LiquidityLevel.MODERATE,
            TradingSession.QUIET_HOURS: LiquidityLevel.LOW
        }
        
        # Configurações de horários ótimos por estratégia
        self.optimal_hours_by_strategy = {
            'scalping': [8, 9, 13, 14, 15, 16, 17],     # High liquidity hours
            'swing': [9, 10, 14, 15, 16],               # Stable hours
            'breakout': [8, 9, 13, 14, 20, 21],         # Volatility hours
            'mean_reversion': [1, 2, 3, 22, 23],        # Quiet hours
            'trend_following': [9, 10, 14, 15, 16, 17], # Trend continuation hours
        }
        
        # Cache de dados históricos
        self.historical_cache: Dict[str, Dict] = {}
        self.volume_patterns: Dict[int, float] = {}  # Hourly volume patterns
    
    def analyze_timing(self, 
                      data: pd.DataFrame,
                      current_time: Optional[datetime] = None,
                      strategy_type: str = "swing") -> TimeSignal:
        """
        Análise principal de timing
        
        Args:
            data: DataFrame com dados OHLCV
            current_time: Timestamp atual (UTC)
            strategy_type: Tipo de estratégia
            
        Returns:
            TimeSignal com análise completa
        """
        try:
            if current_time is None:
                current_time = datetime.now(timezone.utc)
            
            self.logger.info(f"Analisando timing para estratégia {strategy_type}")
            
            # 1. Calcular métricas temporais
            time_metrics = self._calculate_time_metrics(data, current_time)
            
            # 2. Analisar padrões de volume intraday
            volume_analysis = self._analyze_volume_patterns(data, current_time)
            
            # 3. Avaliar qualidade de execução
            execution_analysis = self._analyze_execution_quality(data, current_time)
            
            # 4. Calcular scores de timing
            timing_scores = self._calculate_timing_scores(
                time_metrics, volume_analysis, execution_analysis, strategy_type)
            
            # 5. Determinar recomendações
            recommendations = self._generate_timing_recommendations(
                time_metrics, timing_scores, strategy_type, current_time)
            
            # 6. Identificar riscos temporais
            time_risks = self._identify_time_risks(current_time, time_metrics)
            
            signal = TimeSignal(
                timing_score=timing_scores['timing'],
                liquidity_score=timing_scores['liquidity'],
                execution_score=timing_scores['execution'],
                overall_score=timing_scores['overall'],
                
                entry_timing=recommendations['entry_timing'],
                optimal_timeframe=recommendations['timeframe'],
                position_size_adjustment=recommendations['position_adjustment'],
                execution_strategy=recommendations['execution_strategy'],
                
                current_session=time_metrics.current_session,
                session_ends_in=self._minutes_until_session_end(current_time, time_metrics.current_session),
                next_optimal_time=self._find_next_optimal_time(current_time, strategy_type),
                
                weekend_risk=time_risks['weekend_risk'],
                news_timing_risk=time_risks['news_risk'],
                low_liquidity_warning=time_risks['low_liquidity'],
                
                timestamp=pd.Timestamp.now()
            )
            
            self.logger.info(f"Timing analisado - Sessão: {time_metrics.current_session.value}, "
                           f"Score: {timing_scores['overall']:.1f}, "
                           f"Timing: {recommendations['entry_timing']}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erro na análise de timing: {e}")
            raise
    
    def _calculate_time_metrics(self, data: pd.DataFrame, current_time: datetime) -> TimeMetrics:
        """Calcula métricas temporais básicas"""
        try:
            # Determinar sessão atual
            current_session = self._determine_current_session(current_time)
            
            # Determinar período do dia
            time_of_day = self._determine_time_of_day(current_time)
            
            # Determinar período da semana
            week_period = self._determine_week_period(current_time)
            
            # Nível de liquidez baseado na sessão
            liquidity_level = self.session_liquidity.get(current_session, LiquidityLevel.MODERATE)
            
            # Análise de volume
            current_volume = data['volume'].iloc[-1] if len(data) > 0 else 0
            recent_volume = data['volume'].tail(24).mean() if len(data) >= 24 else current_volume
            session_avg_volume = self._calculate_session_average_volume(data, current_session)
            
            # Volume percentile
            if len(data) >= 100:
                volume_percentile = (data['volume'].tail(100).rank(pct=True).iloc[-1]) * 100
            else:
                volume_percentile = 50.0
            
            # Volume trend
            if len(data) >= 10:
                recent_volumes = data['volume'].tail(10)
                volume_trend = self._determine_volume_trend(recent_volumes)
            else:
                volume_trend = 'stable'
            
            # Estimativa de spread (simplificado)
            estimated_spread = self._estimate_spread(data, liquidity_level)
            
            # Qualidade de execução
            execution_quality = self._calculate_execution_quality(liquidity_level, volume_percentile)
            
            # Risco de slippage
            slippage_risk = self._assess_slippage_risk(liquidity_level, volume_percentile)
            
            # Performance histórica por sessão (simplificado)
            session_performance = self._calculate_session_performance(data)
            
            # Horas ótimas
            optimal_hours = self._get_optimal_hours_for_current_conditions(
                current_session, liquidity_level, volume_percentile)
            
            # Microstructure
            price_stability = self._calculate_price_stability(data)
            mean_reversion_tendency = self._calculate_mean_reversion_tendency(data)
            
            return TimeMetrics(
                current_session=current_session,
                time_of_day=time_of_day,
                week_period=week_period,
                liquidity_level=liquidity_level,
                
                current_volume=current_volume,
                session_avg_volume=session_avg_volume,
                volume_percentile=volume_percentile,
                volume_trend=volume_trend,
                
                estimated_spread=estimated_spread,
                execution_quality=execution_quality,
                slippage_risk=slippage_risk,
                
                session_performance=session_performance,
                optimal_hours=optimal_hours,
                
                price_stability=price_stability,
                mean_reversion_tendency=mean_reversion_tendency
            )
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de métricas temporais: {e}")
            raise
    
    def _determine_current_session(self, current_time: datetime) -> TradingSession:
        """Determina sessão de trading atual"""
        try:
            hour = current_time.hour
            
            # Verificar overlaps primeiro (mais específicos)
            if 8 <= hour < 9:
                return TradingSession.OVERLAP_ASIAN_EU
            elif 13 <= hour < 17:
                return TradingSession.OVERLAP_EU_US
            
            # Sessões individuais
            elif 0 <= hour < 8:
                return TradingSession.ASIAN
            elif 9 <= hour < 13:
                return TradingSession.EUROPEAN
            elif 17 <= hour < 22:
                return TradingSession.US
            else:  # 22-24
                return TradingSession.QUIET_HOURS
                
        except Exception:
            return TradingSession.QUIET_HOURS
    
    def _determine_time_of_day(self, current_time: datetime) -> TimeOfDay:
        """Determina período do dia"""
        hour = current_time.hour
        
        if 0 <= hour < 6:
            return TimeOfDay.EARLY_MORNING
        elif 6 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 18:
            return TimeOfDay.AFTERNOON
        else:
            return TimeOfDay.EVENING
    
    def _determine_week_period(self, current_time: datetime) -> WeekPeriod:
        """Determina período da semana"""
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        hour = current_time.hour
        
        if weekday == 0 and hour < 12:  # Monday morning
            return WeekPeriod.MONDAY_OPEN
        elif weekday in [1, 2, 3]:  # Tuesday-Thursday
            return WeekPeriod.MIDWEEK
        elif weekday == 4 and hour > 16:  # Friday afternoon
            return WeekPeriod.FRIDAY_CLOSE
        elif weekday in [5, 6]:  # Weekend
            return WeekPeriod.WEEKEND
        else:
            return WeekPeriod.MIDWEEK
    
    def _calculate_session_average_volume(self, data: pd.DataFrame, session: TradingSession) -> float:
        """Calcula volume médio da sessão"""
        try:
            if len(data) < 24:
                return data['volume'].mean()
            
            # Usar últimos 30 dias de dados
            recent_data = data.tail(30 * 24)  # Assumindo dados horários
            
            # Filtrar por horário da sessão
            start_hour, end_hour = self.session_times.get(session, (0, 24))
            
            session_volumes = []
            for idx, row in recent_data.iterrows():
                hour = idx.hour if hasattr(idx, 'hour') else 12  # Default
                if start_hour <= hour < end_hour:
                    session_volumes.append(row['volume'])
            
            return np.mean(session_volumes) if session_volumes else recent_data['volume'].mean()
            
        except Exception:
            return data['volume'].mean() if len(data) > 0 else 1000000
    
    def _determine_volume_trend(self, volumes: pd.Series) -> str:
        """Determina tendência de volume"""
        try:
            if len(volumes) < 5:
                return 'stable'
            
            # Comparar primeira e segunda metade
            first_half = volumes.head(len(volumes)//2).mean()
            second_half = volumes.tail(len(volumes)//2).mean()
            
            change_pct = (second_half - first_half) / first_half
            
            if change_pct > 0.2:
                return 'increasing'
            elif change_pct < -0.2:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'stable'
    
    def _estimate_spread(self, data: pd.DataFrame, liquidity_level: LiquidityLevel) -> float:
        """Estima spread bid-ask baseado na liquidez"""
        try:
            # Spread base em basis points
            base_spreads = {
                LiquidityLevel.VERY_HIGH: 0.02,   # 2 bps
                LiquidityLevel.HIGH: 0.05,        # 5 bps
                LiquidityLevel.MODERATE: 0.10,    # 10 bps
                LiquidityLevel.LOW: 0.20,         # 20 bps
                LiquidityLevel.VERY_LOW: 0.50     # 50 bps
            }
            
            base_spread = base_spreads.get(liquidity_level, 0.10)
            
            # Ajustar baseado na volatilidade recente
            if len(data) >= 20:
                recent_returns = data['close'].pct_change().tail(20)
                volatility = recent_returns.std()
                volatility_adjustment = min(2.0, volatility * 100)  # Cap at 2x
                return base_spread * (1 + volatility_adjustment)
            
            return base_spread
            
        except Exception:
            return 0.10  # Default 10 bps
    
    def _calculate_execution_quality(self, liquidity_level: LiquidityLevel, volume_percentile: float) -> float:
        """Calcula score de qualidade de execução"""
        try:
            # Base score por liquidez
            liquidity_scores = {
                LiquidityLevel.VERY_HIGH: 90,
                LiquidityLevel.HIGH: 75,
                LiquidityLevel.MODERATE: 60,
                LiquidityLevel.LOW: 40,
                LiquidityLevel.VERY_LOW: 20
            }
            
            base_score = liquidity_scores.get(liquidity_level, 50)
            
            # Ajuste por volume
            volume_adjustment = (volume_percentile - 50) * 0.3  # -15 to +15
            
            return max(0, min(100, base_score + volume_adjustment))
            
        except Exception:
            return 50
    
    def _assess_slippage_risk(self, liquidity_level: LiquidityLevel, volume_percentile: float) -> str:
        """Avalia risco de slippage"""
        try:
            if liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH] and volume_percentile > 40:
                return 'low'
            elif liquidity_level == LiquidityLevel.MODERATE and volume_percentile > 30:
                return 'medium'
            else:
                return 'high'
                
        except Exception:
            return 'medium'
    
    def _calculate_session_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcula performance histórica por sessão"""
        try:
            # Simplificado - retorna performance média por sessão
            session_performance = {}
            
            for session in TradingSession:
                if session in [TradingSession.OVERLAP_ASIAN_EU, TradingSession.OVERLAP_EU_US]:
                    session_performance[session.value] = 0.65  # Overlaps tendem a ser melhores
                elif session in [TradingSession.EUROPEAN, TradingSession.US]:
                    session_performance[session.value] = 0.60  # Sessões principais
                elif session == TradingSession.ASIAN:
                    session_performance[session.value] = 0.55  # Moderada
                else:
                    session_performance[session.value] = 0.45  # Quiet hours
            
            return session_performance
            
        except Exception:
            return {session.value: 0.50 for session in TradingSession}
    
    def _get_optimal_hours_for_current_conditions(self, session: TradingSession, 
                                                 liquidity_level: LiquidityLevel,
                                                 volume_percentile: float) -> List[int]:
        """Determina horas ótimas baseadas nas condições atuais"""
        try:
            # Base optimal hours
            if liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                optimal_hours = [8, 9, 13, 14, 15, 16, 17]  # High liquidity hours
            else:
                optimal_hours = [9, 10, 14, 15, 16]  # Conservative hours
            
            # Ajustar baseado no volume
            if volume_percentile > 70:
                optimal_hours.extend([10, 11, 18, 19])  # Extend hours during high volume
            
            return sorted(list(set(optimal_hours)))  # Remove duplicates and sort
            
        except Exception:
            return [9, 14, 15]  # Default safe hours
    
    def _calculate_price_stability(self, data: pd.DataFrame) -> float:
        """Calcula estabilidade de preço intraday"""
        try:
            if len(data) < 20:
                return 50.0
            
            # Usar últimos 20 períodos
            recent_data = data.tail(20)
            
            # Calcular volatilidade intraday
            intraday_ranges = (recent_data['high'] - recent_data['low']) / recent_data['close']
            avg_range = intraday_ranges.mean()
            
            # Converter para score (menor range = maior estabilidade)
            stability_score = max(0, min(100, (0.05 - avg_range) / 0.05 * 100))
            
            return stability_score
            
        except Exception:
            return 50.0
    
    def _calculate_mean_reversion_tendency(self, data: pd.DataFrame) -> float:
        """Calcula tendência de reversão à média"""
        try:
            if len(data) < 30:
                return 50.0
            
            # Usar últimos 30 períodos
            recent_data = data.tail(30)
            
            # Calcular retornos
            returns = recent_data['close'].pct_change().dropna()
            
            # Autocorrelação dos retornos (mean reversion = correlação negativa)
            if len(returns) > 10:
                autocorr = returns.autocorr(lag=1)
                # Converter para score (mais negativo = mais mean reversion)
                reversion_score = max(0, min(100, (-autocorr + 0.5) * 100))
                return reversion_score
            
            return 50.0
            
        except Exception:
            return 50.0
    
    def _analyze_volume_patterns(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """Analisa padrões de volume intraday"""
        try:
            analysis = {
                'current_vs_average': 1.0,
                'hourly_pattern_score': 50.0,
                'volume_momentum': 'stable',
                'liquidity_forecast': 'stable'
            }
            
            if len(data) < 24:
                return analysis
            
            # Current vs average volume
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].tail(24).mean()
            analysis['current_vs_average'] = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Hourly pattern score
            current_hour = current_time.hour
            if current_hour in [8, 9, 13, 14, 15, 16, 17]:  # High activity hours
                analysis['hourly_pattern_score'] = 80.0
            elif current_hour in [10, 11, 18, 19]:  # Moderate activity
                analysis['hourly_pattern_score'] = 60.0
            else:  # Low activity hours
                analysis['hourly_pattern_score'] = 30.0
            
            # Volume momentum
            if len(data) >= 6:
                recent_volumes = data['volume'].tail(6)
                if recent_volumes.iloc[-1] > recent_volumes.iloc[-3]:
                    analysis['volume_momentum'] = 'increasing'
                elif recent_volumes.iloc[-1] < recent_volumes.iloc[-3]:
                    analysis['volume_momentum'] = 'decreasing'
                else:
                    analysis['volume_momentum'] = 'stable'
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Erro na análise de volume: {e}")
            return {'current_vs_average': 1.0, 'hourly_pattern_score': 50.0}
    
    def _analyze_execution_quality(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """Analisa qualidade de execução esperada"""
        try:
            analysis = {
                'market_impact_score': 70.0,
                'fill_probability': 0.85,
                'execution_speed': 'fast',
                'recommended_order_type': 'limit'
            }
            
            # Determinar qualidade baseada no horário
            hour = current_time.hour
            
            if 8 <= hour <= 17:  # Business hours
                analysis['market_impact_score'] = 80.0
                analysis['fill_probability'] = 0.90
                analysis['execution_speed'] = 'fast'
                analysis['recommended_order_type'] = 'market'
            elif hour in [18, 19, 20, 7]:  # Transition hours
                analysis['market_impact_score'] = 60.0
                analysis['fill_probability'] = 0.75
                analysis['execution_speed'] = 'medium'
                analysis['recommended_order_type'] = 'limit'
            else:  # Off hours
                analysis['market_impact_score'] = 40.0
                analysis['fill_probability'] = 0.60
                analysis['execution_speed'] = 'slow'
                analysis['recommended_order_type'] = 'limit'
            
            return analysis
            
        except Exception:
            return {'market_impact_score': 50.0, 'fill_probability': 0.70}
    
    def _calculate_timing_scores(self, time_metrics: TimeMetrics, volume_analysis: Dict,
                               execution_analysis: Dict, strategy_type: str) -> Dict:
        """Calcula scores de timing"""
        try:
            # Timing score baseado na estratégia
            current_hour = time_metrics.optimal_hours
            strategy_optimal_hours = self.optimal_hours_by_strategy.get(strategy_type, [9, 14, 15])
            
            # Check if current hour is optimal
            now_hour = datetime.now(timezone.utc).hour
            if now_hour in strategy_optimal_hours:
                timing_score = 80.0
            elif any(abs(now_hour - h) <= 1 for h in strategy_optimal_hours):  # Within 1 hour
                timing_score = 60.0
            else:
                timing_score = 30.0
            
            # Liquidity score
            liquidity_scores = {
                LiquidityLevel.VERY_HIGH: 95,
                LiquidityLevel.HIGH: 80,
                LiquidityLevel.MODERATE: 60,
                LiquidityLevel.LOW: 35,
                LiquidityLevel.VERY_LOW: 15
            }
            liquidity_score = liquidity_scores.get(time_metrics.liquidity_level, 50)
            
            # Execution score
            execution_score = execution_analysis.get('market_impact_score', 50)
            
            # Volume adjustment
            volume_adjustment = min(20, max(-20, (time_metrics.volume_percentile - 50) * 0.4))
            
            # Apply adjustments
            timing_score += volume_adjustment * 0.3
            liquidity_score += volume_adjustment * 0.2
            execution_score += volume_adjustment * 0.5
            
            # Overall score
            overall_score = (timing_score * 0.4 + liquidity_score * 0.35 + execution_score * 0.25)
            
            # Cap scores
            scores = {
                'timing': max(0, min(100, timing_score)),
                'liquidity': max(0, min(100, liquidity_score)),
                'execution': max(0, min(100, execution_score)),
                'overall': max(0, min(100, overall_score))
            }
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de scores: {e}")
            return {'timing': 50, 'liquidity': 50, 'execution': 50, 'overall': 50}
    
    def _generate_timing_recommendations(self, time_metrics: TimeMetrics, timing_scores: Dict,
                                       strategy_type: str, current_time: datetime) -> Dict:
        """Gera recomendações de timing"""
        try:
            overall_score = timing_scores['overall']
            
            # Entry timing decision
            if overall_score >= 70:
                entry_timing = 'immediate'
                position_adjustment = 1.0
            elif overall_score >= 50:
                entry_timing = 'immediate'
                position_adjustment = 0.8
            elif overall_score >= 30:
                entry_timing = 'wait'
                position_adjustment = 0.6
            else:
                entry_timing = 'avoid'
                position_adjustment = 0.3
            
            # Timeframe recommendation
            if time_metrics.liquidity_level in [LiquidityLevel.VERY_HIGH, LiquidityLevel.HIGH]:
                timeframe = '1H'  # Can use shorter timeframes
            else:
                timeframe = '4H'  # Stick to longer timeframes
            
            # Execution strategy
            if timing_scores['execution'] > 70 and time_metrics.volume_percentile > 60:
                execution_strategy = 'market'
            elif timing_scores['execution'] > 50:
                execution_strategy = 'limit'
            else:
                execution_strategy = 'twap'  # Time-weighted average price
            
            # Weekend adjustment
            if time_metrics.week_period == WeekPeriod.WEEKEND:
                entry_timing = 'avoid'
                position_adjustment *= 0.5
            
            return {
                'entry_timing': entry_timing,
                'timeframe': timeframe,
                'position_adjustment': position_adjustment,
                'execution_strategy': execution_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Erro nas recomendações: {e}")
            return {
                'entry_timing': 'wait',
                'timeframe': '4H',
                'position_adjustment': 0.7,
                'execution_strategy': 'limit'
            }
    
    def _identify_time_risks(self, current_time: datetime, time_metrics: TimeMetrics) -> Dict:
        """Identifica riscos relacionados ao timing"""
        try:
            risks = {
                'weekend_risk': False,
                'news_risk': False,
                'low_liquidity': False
            }
            
            # Weekend risk
            if time_metrics.week_period == WeekPeriod.WEEKEND:
                risks['weekend_risk'] = True
            
            # Low liquidity risk
            if time_metrics.liquidity_level in [LiquidityLevel.LOW, LiquidityLevel.VERY_LOW]:
                risks['low_liquidity'] = True
            
            # News timing risk (simplified - major news typically at specific hours)
            hour = current_time.hour
            if hour in [14, 15, 16]:  # Common news release hours (UTC)
                risks['news_risk'] = True
            
            return risks
            
        except Exception:
            return {'weekend_risk': False, 'news_risk': False, 'low_liquidity': False}
    
    def _minutes_until_session_end(self, current_time: datetime, session: TradingSession) -> int:
        """Calcula minutos até o fim da sessão"""
        try:
            start_hour, end_hour = self.session_times.get(session, (0, 24))
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            if current_hour < end_hour:
                minutes_left = (end_hour - current_hour - 1) * 60 + (60 - current_minute)
            else:
                # Session ended or will end tomorrow
                minutes_left = 0
            
            return max(0, minutes_left)
            
        except Exception:
            return 60  # Default 1 hour
    
    def _find_next_optimal_time(self, current_time: datetime, strategy_type: str) -> Optional[datetime]:
        """Encontra próximo horário ótimo"""
        try:
            optimal_hours = self.optimal_hours_by_strategy.get(strategy_type, [9, 14, 15])
            current_hour = current_time.hour
            
            # Find next optimal hour today
            next_hour = None
            for hour in sorted(optimal_hours):
                if hour > current_hour:
                    next_hour = hour
                    break
            
            if next_hour is not None:
                # Today
                next_time = current_time.replace(hour=next_hour, minute=0, second=0, microsecond=0)
            else:
                # Tomorrow - first optimal hour
                next_time = current_time.replace(hour=min(optimal_hours), minute=0, second=0, microsecond=0)
                next_time = next_time + pd.Timedelta(days=1)
            
            return next_time
            
        except Exception:
            return None
    
    def should_trade_now(self, signal: TimeSignal, strategy_type: str) -> Dict:
        """
        Determina se deve tradear agora baseado no timing
        
        Args:
            signal: TimeSignal atual
            strategy_type: Tipo de estratégia
            
        Returns:
            Dict com decisão e ajustes
        """
        try:
            should_trade = True
            adjustments = {}
            reasons = []
            
            # Check weekend risk
            if signal.weekend_risk:
                should_trade = False
                reasons.append("Risco de weekend - mercado crypto pode ter gaps")
            
            # Check liquidity
            if signal.low_liquidity_warning and signal.liquidity_score < self.min_liquidity_score:
                should_trade = False
                reasons.append(f"Liquidez muito baixa ({signal.liquidity_score:.0f}) - evitar trading")
            
            # Check overall timing
            if signal.entry_timing == 'avoid':
                should_trade = False
                reasons.append("Timing não favorável - aguardar melhores condições")
            elif signal.entry_timing == 'wait':
                should_trade = False
                reasons.append(f"Aguardar - próximo horário ótimo: {signal.next_optimal_time}")
            
            # Apply adjustments if trading is allowed
            if should_trade:
                adjustments = {
                    'position_size_multiplier': signal.position_size_adjustment,
                    'execution_strategy': signal.execution_strategy,
                    'recommended_timeframe': signal.optimal_timeframe
                }
                
                if signal.overall_score > 80:
                    reasons.append("Timing excelente - condições ótimas")
                elif signal.overall_score > 60:
                    reasons.append("Timing bom - prosseguir com cuidado")
                else:
                    reasons.append("Timing moderado - reduzir position size")
                    adjustments['position_size_multiplier'] *= 0.8
            
            return {
                'should_trade': should_trade,
                'adjustments': adjustments,
                'reasons': reasons,
                'timing_score': signal.timing_score,
                'liquidity_score': signal.liquidity_score,
                'execution_score': signal.execution_score,
                'overall_score': signal.overall_score,
                'current_session': signal.current_session.value,
                'minutes_until_session_end': signal.session_ends_in
            }
            
        except Exception as e:
            self.logger.error(f"Erro na decisão de timing: {e}")
            return {
                'should_trade': False,
                'adjustments': {'position_size_multiplier': 0.5},
                'reasons': ['Erro na análise de timing'],
                'overall_score': 30
            }


def main():
    """Teste básico do filtro de timing"""
    # Criar dados de exemplo
    dates = pd.date_range(start='2024-01-01', periods=168, freq='1H')  # 1 week of hourly data
    np.random.seed(42)
    
    # Simular padrões de volume por hora
    base_volumes = []
    for date in dates:
        hour = date.hour
        if 8 <= hour <= 17:  # Business hours
            base_volume = 2000000
        elif hour in [18, 19, 20, 7]:  # Transition
            base_volume = 1500000
        else:  # Off hours
            base_volume = 800000
        
        # Add some randomness
        volume = base_volume * (0.7 + 0.6 * np.random.random())
        base_volumes.append(volume)
    
    # Create OHLCV data
    price_base = 50000
    prices = price_base + np.random.randn(len(dates)).cumsum() * 50
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(len(dates)) * 25),
        'low': prices - np.abs(np.random.randn(len(dates)) * 25),
        'close': prices,
        'volume': base_volumes
    }, index=dates)
    
    # Test filter
    time_filter = TimeFilter()
    
    # Test different times
    test_times = [
        datetime(2024, 1, 2, 9, 0, tzinfo=timezone.utc),   # Tuesday 9 AM UTC (EU open)
        datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc),  # Tuesday 2 PM UTC (EU-US overlap)
        datetime(2024, 1, 2, 23, 0, tzinfo=timezone.utc),  # Tuesday 11 PM UTC (quiet)
        datetime(2024, 1, 6, 15, 0, tzinfo=timezone.utc),  # Saturday 3 PM UTC (weekend)
    ]
    
    for i, test_time in enumerate(test_times, 1):
        print(f"\n🕐 TIME ANALYSIS {i} - {test_time.strftime('%A %H:%M UTC')}")
        
        signal = time_filter.analyze_timing(data, test_time, "swing")
        
        print(f"Session: {signal.current_session.value}")
        print(f"Timing Score: {signal.timing_score:.1f}")
        print(f"Liquidity Score: {signal.liquidity_score:.1f}")
        print(f"Execution Score: {signal.execution_score:.1f}")
        print(f"Overall Score: {signal.overall_score:.1f}")
        print(f"Entry Timing: {signal.entry_timing}")
        print(f"Position Adjustment: {signal.position_size_adjustment:.2f}x")
        print(f"Execution Strategy: {signal.execution_strategy}")
        print(f"Session Ends In: {signal.session_ends_in} minutes")
        
        if signal.weekend_risk:
            print("⚠️  Weekend Risk")
        if signal.low_liquidity_warning:
            print("⚠️  Low Liquidity Warning")
        
        # Test trading decision
        decision = time_filter.should_trade_now(signal, "swing")
        print(f"\n📊 TRADING DECISION")
        print(f"Should Trade: {decision['should_trade']}")
        if decision['should_trade']:
            print(f"Position Multiplier: {decision['adjustments']['position_size_multiplier']:.2f}x")
            print(f"Execution: {decision['adjustments']['execution_strategy']}")
        print(f"Reasons: {', '.join(decision['reasons'])}")
        print("-" * 60)


if __name__ == "__main__":
    main()