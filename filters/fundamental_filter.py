"""
📰 FUNDAMENTAL FILTER - Smart Trading System v2.0

Filtro de eventos fundamentais e notícias:
- Economic calendar events
- Crypto-specific news impact
- Regulatory announcements
- Market-moving events detection
- Sentiment analysis integration

Filosofia: Information is Edge - Trade the News, Not the Noise
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Tipos de eventos fundamentais"""
    ECONOMIC_DATA = "economic_data"           # Dados econômicos (CPI, GDP, etc.)
    CENTRAL_BANK = "central_bank"            # Decisões de bancos centrais
    REGULATORY = "regulatory"                # Anúncios regulatórios
    ADOPTION = "adoption"                    # Adoção institucional
    TECHNICAL = "technical"                  # Atualizações técnicas
    EARNINGS = "earnings"                    # Earnings de empresas relacionadas
    GEOPOLITICAL = "geopolitical"           # Eventos geopolíticos
    CRYPTO_SPECIFIC = "crypto_specific"      # Eventos específicos de crypto


class EventImpact(Enum):
    """Impacto esperado dos eventos"""
    VERY_HIGH = "very_high"      # Potencial de movimento >5%
    HIGH = "high"                # Potencial de movimento 2-5%
    MEDIUM = "medium"            # Potencial de movimento 1-2%
    LOW = "low"                  # Potencial de movimento <1%
    UNKNOWN = "unknown"          # Impacto incerto


class EventSentiment(Enum):
    """Sentimento do evento"""
    VERY_BULLISH = "very_bullish"    # Muito positivo
    BULLISH = "bullish"              # Positivo
    NEUTRAL = "neutral"              # Neutro
    BEARISH = "bearish"              # Negativo
    VERY_BEARISH = "very_bearish"    # Muito negativo


class NewsSource(Enum):
    """Fontes de notícias"""
    OFFICIAL = "official"            # Fontes oficiais (govt, exchanges)
    MAJOR_NEWS = "major_news"        # Grandes veículos de notícias
    CRYPTO_MEDIA = "crypto_media"    # Mídia especializada em crypto
    SOCIAL_MEDIA = "social_media"    # Redes sociais
    RUMORS = "rumors"                # Rumores não confirmados


@dataclass
class FundamentalEvent:
    """Evento fundamental"""
    event_id: str
    event_type: EventType
    title: str
    description: str
    impact: EventImpact
    sentiment: EventSentiment
    source: NewsSource
    
    # Timing
    event_time: datetime
    time_until_event: timedelta      # Tempo até o evento
    is_scheduled: bool               # Se é evento agendado
    
    # Market implications
    affected_assets: List[str]       # Assets afetados
    expected_direction: str          # 'bullish', 'bearish', 'neutral'
    confidence: float                # 0-100 confiança na análise
    
    # Historical context
    similar_events_impact: Optional[float]  # Impacto histórico médio
    market_preparation: float        # Nível de preparação do mercado (0-100)
    
    # Risk factors
    surprise_factor: float           # Fator surpresa (0-100)
    volatility_expected: float       # Volatilidade esperada


@dataclass
class NewsAnalysis:
    """Análise de notícias"""
    headline_sentiment: float       # -100 a +100
    content_sentiment: float        # -100 a +100
    keywords_found: List[str]        # Palavras-chave encontradas
    credibility_score: float        # 0-100 credibilidade da fonte
    market_relevance: float         # 0-100 relevância para o mercado
    urgency_level: float            # 0-100 urgência
    
    # Impact estimation
    estimated_price_impact: float   # % impacto estimado no preço
    time_horizon: str               # 'immediate', 'short_term', 'long_term'
    confidence_interval: Tuple[float, float]  # Min, max impact


@dataclass
class FundamentalSignal:
    """Sinal fundamental"""
    signal_strength: float          # 0-100 força do sinal
    direction: str                  # 'bullish', 'bearish', 'neutral'
    confidence: float               # 0-100 confiança
    time_horizon: str               # Horizonte temporal
    
    # Events summary
    upcoming_events: List[FundamentalEvent]
    recent_events: List[FundamentalEvent]
    high_impact_events: List[FundamentalEvent]
    
    # Market preparation
    market_positioned_for: str      # O que o mercado está precificando
    surprise_potential: float       # Potencial de surpresa (0-100)
    
    # Trading implications
    recommended_action: str         # 'buy', 'sell', 'hold', 'wait'
    position_size_adjustment: float # Multiplicador de position size
    risk_adjustment: float          # Ajuste de risco
    timing_recommendation: str      # 'before_event', 'after_event', 'avoid'
    
    # Risk warnings
    high_volatility_warning: bool
    regulatory_risk: bool
    black_swan_potential: bool
    
    timestamp: pd.Timestamp


class FundamentalFilter:
    """
    📰 Filtro Principal de Eventos Fundamentais
    
    Monitora e analisa:
    1. Calendar econômico
    2. Notícias crypto-específicas
    3. Anúncios regulatórios
    4. Eventos de adoção
    5. Sentiment de mercado
    """
    
    def __init__(self,
                 high_impact_threshold: float = 70.0,
                 news_lookback_hours: int = 24,
                 event_horizon_days: int = 7):
        
        self.high_impact_threshold = high_impact_threshold
        self.news_lookback_hours = news_lookback_hours
        self.event_horizon_days = event_horizon_days
        
        self.logger = logging.getLogger(f"{__name__}.FundamentalFilter")
        
        # Keywords para análise de sentiment
        self.bullish_keywords = [
            'adoption', 'approval', 'partnership', 'integration', 'bullish',
            'upgrade', 'positive', 'growth', 'increase', 'rally', 'surge',
            'institutional', 'etf', 'mainstream', 'breakthrough', 'innovation'
        ]
        
        self.bearish_keywords = [
            'ban', 'regulation', 'crash', 'fall', 'decline', 'bearish',
            'hack', 'security', 'concern', 'warning', 'investigation',
            'crackdown', 'restriction', 'prohibition', 'fraud', 'scam'
        ]
        
        self.high_impact_keywords = [
            'fed', 'federal reserve', 'interest rate', 'inflation', 'cpi',
            'gdp', 'unemployment', 'fomc', 'bitcoin etf', 'sec', 'cftc',
            'regulation', 'ban', 'approval', 'institutional adoption'
        ]
        
        # Event impact multipliers por tipo
        self.event_impact_multipliers = {
            EventType.CENTRAL_BANK: 1.5,
            EventType.REGULATORY: 1.3,
            EventType.ECONOMIC_DATA: 1.2,
            EventType.ADOPTION: 1.1,
            EventType.CRYPTO_SPECIFIC: 1.0,
            EventType.TECHNICAL: 0.8,
            EventType.EARNINGS: 0.7,
            EventType.GEOPOLITICAL: 0.9
        }
        
        # Cache de eventos e notícias
        self.events_cache: List[FundamentalEvent] = []
        self.news_cache: List[Dict] = []
        self.sentiment_cache: Dict[str, float] = {}
    
    def analyze_fundamentals(self, 
                           symbol: str = "BTCUSDT",
                           current_time: Optional[datetime] = None) -> FundamentalSignal:
        """
        Análise principal de fundamentals
        
        Args:
            symbol: Símbolo para análise
            current_time: Timestamp atual
            
        Returns:
            FundamentalSignal com análise completa
        """
        try:
            if current_time is None:
                current_time = datetime.now()
            
            self.logger.info(f"Analisando fundamentals para {symbol}")
            
            # 1. Carregar eventos do calendar econômico
            upcoming_events = self._load_economic_calendar(current_time)
            
            # 2. Analisar notícias recentes
            recent_news = self._analyze_recent_news(current_time)
            
            # 3. Avaliar eventos crypto-específicos
            crypto_events = self._analyze_crypto_events(symbol, current_time)
            
            # 4. Combinar todos os eventos
            all_events = upcoming_events + crypto_events
            
            # 5. Filtrar eventos de alto impacto
            high_impact_events = [e for e in all_events if self._calculate_event_impact_score(e) > self.high_impact_threshold]
            
            # 6. Calcular sentiment agregado
            overall_sentiment = self._calculate_overall_sentiment(all_events, recent_news)
            
            # 7. Avaliar preparação do mercado
            market_preparation = self._assess_market_preparation(all_events, recent_news)
            
            # 8. Gerar recomendações
            recommendations = self._generate_fundamental_recommendations(
                all_events, high_impact_events, overall_sentiment, market_preparation)
            
            # 9. Identificar riscos
            risk_assessment = self._assess_fundamental_risks(all_events, recent_news)
            
            signal = FundamentalSignal(
                signal_strength=recommendations['signal_strength'],
                direction=recommendations['direction'],
                confidence=recommendations['confidence'],
                time_horizon=recommendations['time_horizon'],
                
                upcoming_events=upcoming_events,
                recent_events=crypto_events,
                high_impact_events=high_impact_events,
                
                market_positioned_for=market_preparation['positioned_for'],
                surprise_potential=market_preparation['surprise_potential'],
                
                recommended_action=recommendations['action'],
                position_size_adjustment=recommendations['position_adjustment'],
                risk_adjustment=recommendations['risk_adjustment'],
                timing_recommendation=recommendations['timing'],
                
                high_volatility_warning=risk_assessment['high_volatility'],
                regulatory_risk=risk_assessment['regulatory_risk'],
                black_swan_potential=risk_assessment['black_swan'],
                
                timestamp=pd.Timestamp.now()
            )
            
            self.logger.info(f"Fundamentals analisados - Direção: {signal.direction}, "
                           f"Força: {signal.signal_strength:.1f}, "
                           f"Eventos: {len(high_impact_events)} alto impacto")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erro na análise de fundamentals: {e}")
            raise
    
    def _load_economic_calendar(self, current_time: datetime) -> List[FundamentalEvent]:
        """Carrega eventos do calendário econômico"""
        try:
            # Simular eventos econômicos importantes
            events = []
            
            # Eventos típicos que afetam crypto
            typical_events = [
                {
                    'title': 'Federal Reserve Interest Rate Decision',
                    'type': EventType.CENTRAL_BANK,
                    'impact': EventImpact.VERY_HIGH,
                    'days_ahead': 3,
                    'sentiment': EventSentiment.NEUTRAL
                },
                {
                    'title': 'US CPI Inflation Data',
                    'type': EventType.ECONOMIC_DATA,
                    'impact': EventImpact.HIGH,
                    'days_ahead': 1,
                    'sentiment': EventSentiment.NEUTRAL
                },
                {
                    'title': 'SEC Crypto Regulation Update',
                    'type': EventType.REGULATORY,
                    'impact': EventImpact.HIGH,
                    'days_ahead': 5,
                    'sentiment': EventSentiment.BEARISH
                },
                {
                    'title': 'Bitcoin ETF Decision',
                    'type': EventType.REGULATORY,
                    'impact': EventImpact.VERY_HIGH,
                    'days_ahead': 7,
                    'sentiment': EventSentiment.BULLISH
                }
            ]
            
            for event_data in typical_events:
                event_time = current_time + timedelta(days=event_data['days_ahead'])
                
                event = FundamentalEvent(
                    event_id=f"econ_{event_data['title'][:10]}_{event_time.strftime('%Y%m%d')}",
                    event_type=event_data['type'],
                    title=event_data['title'],
                    description=f"Scheduled {event_data['title'].lower()}",
                    impact=event_data['impact'],
                    sentiment=event_data['sentiment'],
                    source=NewsSource.OFFICIAL,
                    
                    event_time=event_time,
                    time_until_event=event_time - current_time,
                    is_scheduled=True,
                    
                    affected_assets=['BTCUSDT', 'ETHUSDT'],
                    expected_direction=self._sentiment_to_direction(event_data['sentiment']),
                    confidence=75.0,
                    
                    similar_events_impact=self._get_historical_impact(event_data['type']),
                    market_preparation=60.0,
                    
                    surprise_factor=30.0,
                    volatility_expected=self._impact_to_volatility(event_data['impact'])
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar calendar: {e}")
            return []
    
    def _analyze_recent_news(self, current_time: datetime) -> List[NewsAnalysis]:
        """Analisa notícias recentes"""
        try:
            # Simular análise de notícias recentes
            news_items = [
                {
                    'headline': 'Major Corporation Announces Bitcoin Treasury Adoption',
                    'content': 'Leading technology company adds BTC to balance sheet',
                    'source': NewsSource.MAJOR_NEWS,
                    'time_ago_hours': 2,
                    'bullish_score': 80
                },
                {
                    'headline': 'Regulatory Concerns Mount Over Crypto Derivatives',
                    'content': 'Financial regulators express concerns about crypto derivatives trading',
                    'source': NewsSource.OFFICIAL,
                    'time_ago_hours': 6,
                    'bullish_score': -60
                },
                {
                    'headline': 'DeFi Protocol Suffers $50M Hack',
                    'content': 'Major DeFi protocol exploited, funds drained',
                    'source': NewsSource.CRYPTO_MEDIA,
                    'time_ago_hours': 12,
                    'bullish_score': -70
                }
            ]
            
            analyses = []
            
            for news in news_items:
                # Análise de sentiment
                headline_sentiment = news['bullish_score']
                content_sentiment = news['bullish_score'] * 0.8  # Slightly less extreme
                
                # Keywords
                keywords = self._extract_keywords(news['headline'] + ' ' + news['content'])
                
                # Credibilidade
                credibility = self._assess_source_credibility(news['source'])
                
                # Relevância
                relevance = self._calculate_market_relevance(keywords)
                
                # Urgência
                urgency = max(0, 100 - (news['time_ago_hours'] * 5))  # Decays over time
                
                analysis = NewsAnalysis(
                    headline_sentiment=headline_sentiment,
                    content_sentiment=content_sentiment,
                    keywords_found=keywords,
                    credibility_score=credibility,
                    market_relevance=relevance,
                    urgency_level=urgency,
                    
                    estimated_price_impact=abs(headline_sentiment) * 0.05,  # 0-5% impact
                    time_horizon='short_term',
                    confidence_interval=(
                        abs(headline_sentiment) * 0.02,
                        abs(headline_sentiment) * 0.08
                    )
                )
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            self.logger.error(f"Erro na análise de notícias: {e}")
            return []
    
    def _analyze_crypto_events(self, symbol: str, current_time: datetime) -> List[FundamentalEvent]:
        """Analisa eventos específicos de crypto"""
        try:
            events = []
            
            # Eventos crypto típicos
            crypto_events_data = [
                {
                    'title': 'Major Exchange Listing Announcement',
                    'type': EventType.ADOPTION,
                    'impact': EventImpact.MEDIUM,
                    'sentiment': EventSentiment.BULLISH,
                    'hours_ago': 4
                },
                {
                    'title': 'Network Upgrade Completion',
                    'type': EventType.TECHNICAL,
                    'impact': EventImpact.LOW,
                    'sentiment': EventSentiment.BULLISH,
                    'hours_ago': 8
                },
                {
                    'title': 'Large Whale Movement Detected',
                    'type': EventType.CRYPTO_SPECIFIC,
                    'impact': EventImpact.MEDIUM,
                    'sentiment': EventSentiment.BEARISH,
                    'hours_ago': 1
                }
            ]
            
            for event_data in crypto_events_data:
                event_time = current_time - timedelta(hours=event_data['hours_ago'])
                
                event = FundamentalEvent(
                    event_id=f"crypto_{symbol}_{event_time.strftime('%Y%m%d%H')}",
                    event_type=event_data['type'],
                    title=event_data['title'],
                    description=f"Recent crypto event: {event_data['title']}",
                    impact=event_data['impact'],
                    sentiment=event_data['sentiment'],
                    source=NewsSource.CRYPTO_MEDIA,
                    
                    event_time=event_time,
                    time_until_event=timedelta(0),  # Already happened
                    is_scheduled=False,
                    
                    affected_assets=[symbol],
                    expected_direction=self._sentiment_to_direction(event_data['sentiment']),
                    confidence=65.0,
                    
                    similar_events_impact=self._get_historical_impact(event_data['type']),
                    market_preparation=40.0,  # Less preparation for unscheduled events
                    
                    surprise_factor=70.0,  # Higher surprise for crypto events
                    volatility_expected=self._impact_to_volatility(event_data['impact'])
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Erro na análise de eventos crypto: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave relevantes"""
        try:
            text_lower = text.lower()
            found_keywords = []
            
            # Check bullish keywords
            for keyword in self.bullish_keywords:
                if keyword in text_lower:
                    found_keywords.append(f"bullish:{keyword}")
            
            # Check bearish keywords
            for keyword in self.bearish_keywords:
                if keyword in text_lower:
                    found_keywords.append(f"bearish:{keyword}")
            
            # Check high impact keywords
            for keyword in self.high_impact_keywords:
                if keyword in text_lower:
                    found_keywords.append(f"high_impact:{keyword}")
            
            return found_keywords
            
        except Exception:
            return []
    
    def _assess_source_credibility(self, source: NewsSource) -> float:
        """Avalia credibilidade da fonte"""
        credibility_scores = {
            NewsSource.OFFICIAL: 95.0,
            NewsSource.MAJOR_NEWS: 80.0,
            NewsSource.CRYPTO_MEDIA: 70.0,
            NewsSource.SOCIAL_MEDIA: 40.0,
            NewsSource.RUMORS: 20.0
        }
        return credibility_scores.get(source, 50.0)
    
    def _calculate_market_relevance(self, keywords: List[str]) -> float:
        """Calcula relevância para o mercado"""
        try:
            relevance_score = 50.0  # Base score
            
            # High impact keywords add significant relevance
            high_impact_count = sum(1 for kw in keywords if kw.startswith('high_impact:'))
            relevance_score += high_impact_count * 20
            
            # Bullish/bearish keywords add moderate relevance
            sentiment_count = sum(1 for kw in keywords if kw.startswith(('bullish:', 'bearish:')))
            relevance_score += sentiment_count * 10
            
            return min(100.0, relevance_score)
            
        except Exception:
            return 50.0
    
    def _sentiment_to_direction(self, sentiment: EventSentiment) -> str:
        """Converte sentiment em direção"""
        if sentiment in [EventSentiment.VERY_BULLISH, EventSentiment.BULLISH]:
            return 'bullish'
        elif sentiment in [EventSentiment.VERY_BEARISH, EventSentiment.BEARISH]:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_historical_impact(self, event_type: EventType) -> float:
        """Retorna impacto histórico médio por tipo de evento"""
        historical_impacts = {
            EventType.CENTRAL_BANK: 3.5,     # 3.5% average impact
            EventType.REGULATORY: 4.2,       # 4.2% average impact
            EventType.ECONOMIC_DATA: 2.1,    # 2.1% average impact
            EventType.ADOPTION: 2.8,         # 2.8% average impact
            EventType.CRYPTO_SPECIFIC: 1.5,  # 1.5% average impact
            EventType.TECHNICAL: 0.8,        # 0.8% average impact
            EventType.EARNINGS: 1.2,         # 1.2% average impact
            EventType.GEOPOLITICAL: 2.5      # 2.5% average impact
        }
        return historical_impacts.get(event_type, 1.0)
    
    def _impact_to_volatility(self, impact: EventImpact) -> float:
        """Converte impact em volatilidade esperada"""
        volatility_mapping = {
            EventImpact.VERY_HIGH: 8.0,  # 8% expected volatility
            EventImpact.HIGH: 5.0,       # 5% expected volatility
            EventImpact.MEDIUM: 3.0,     # 3% expected volatility
            EventImpact.LOW: 1.5,        # 1.5% expected volatility
            EventImpact.UNKNOWN: 2.0     # 2% default
        }
        return volatility_mapping.get(impact, 2.0)
    
    def _calculate_event_impact_score(self, event: FundamentalEvent) -> float:
        """Calcula score de impacto de um evento"""
        try:
            # Base score por impact level
            impact_scores = {
                EventImpact.VERY_HIGH: 90,
                EventImpact.HIGH: 70,
                EventImpact.MEDIUM: 50,
                EventImpact.LOW: 30,
                EventImpact.UNKNOWN: 25
            }
            
            base_score = impact_scores.get(event.impact, 25)
            
            # Adjustments
            
            # Time decay (events lose impact over time)
            if event.time_until_event.total_seconds() > 0:
                # Future event
                days_until = event.time_until_event.days
                time_factor = max(0.5, 1 - (days_until * 0.1))  # Decay over time
            else:
                # Past event
                hours_ago = abs(event.time_until_event.total_seconds()) / 3600
                time_factor = max(0.3, 1 - (hours_ago * 0.05))  # Rapid decay for past events
            
            # Event type multiplier
            type_multiplier = self.event_impact_multipliers.get(event.event_type, 1.0)
            
            # Source credibility factor
            source_scores = {
                NewsSource.OFFICIAL: 1.2,
                NewsSource.MAJOR_NEWS: 1.0,
                NewsSource.CRYPTO_MEDIA: 0.9,
                NewsSource.SOCIAL_MEDIA: 0.6,
                NewsSource.RUMORS: 0.4
            }
            source_factor = source_scores.get(event.source, 0.8)
            
            # Surprise factor
            surprise_factor = 1 + (event.surprise_factor / 200)  # 0.5x to 1.5x
            
            final_score = (base_score * time_factor * type_multiplier * 
                          source_factor * surprise_factor)
            
            return min(100, max(0, final_score))
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de impact score: {e}")
            return 25
    
    def _calculate_overall_sentiment(self, events: List[FundamentalEvent], 
                                   news: List[NewsAnalysis]) -> Dict:
        """Calcula sentiment geral"""
        try:
            sentiment_scores = []
            weights = []
            
            # Events sentiment
            for event in events:
                event_score = self._calculate_event_impact_score(event)
                if event_score > 30:  # Only significant events
                    sentiment_value = self._sentiment_enum_to_value(event.sentiment)
                    sentiment_scores.append(sentiment_value)
                    weights.append(event_score)
            
            # News sentiment
            for news_item in news:
                if news_item.market_relevance > 50:
                    sentiment_scores.append(news_item.headline_sentiment)
                    weights.append(news_item.credibility_score)
            
            if sentiment_scores:
                # Weighted average
                weighted_sentiment = np.average(sentiment_scores, weights=weights)
            else:
                weighted_sentiment = 0  # Neutral
            
            # Classify sentiment
            if weighted_sentiment > 40:
                direction = 'bullish'
                strength = min(100, abs(weighted_sentiment))
            elif weighted_sentiment < -40:
                direction = 'bearish'
                strength = min(100, abs(weighted_sentiment))
            else:
                direction = 'neutral'
                strength = 30
            
            return {
                'sentiment_score': weighted_sentiment,
                'direction': direction,
                'strength': strength,
                'sample_size': len(sentiment_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de sentiment: {e}")
            return {'sentiment_score': 0, 'direction': 'neutral', 'strength': 30, 'sample_size': 0}
    
    def _sentiment_enum_to_value(self, sentiment: EventSentiment) -> float:
        """Converte enum de sentiment para valor numérico"""
        sentiment_values = {
            EventSentiment.VERY_BULLISH: 80,
            EventSentiment.BULLISH: 50,
            EventSentiment.NEUTRAL: 0,
            EventSentiment.BEARISH: -50,
            EventSentiment.VERY_BEARISH: -80
        }
        return sentiment_values.get(sentiment, 0)
    
    def _assess_market_preparation(self, events: List[FundamentalEvent], 
                                 news: List[NewsAnalysis]) -> Dict:
        """Avalia preparação do mercado"""
        try:
            # Analisar eventos agendados vs não agendados
            scheduled_events = [e for e in events if e.is_scheduled]
            unscheduled_events = [e for e in events if not e.is_scheduled]
            
            # Market preparation score
            if scheduled_events:
                avg_preparation = np.mean([e.market_preparation for e in scheduled_events])
            else:
                avg_preparation = 50.0
            
            # Surprise potential
            if unscheduled_events:
                avg_surprise = np.mean([e.surprise_factor for e in unscheduled_events])
            else:
                avg_surprise = 30.0
            
            # What market is positioned for
            upcoming_events = [e for e in events if e.time_until_event.total_seconds() > 0]
            if upcoming_events:
                # Find most impactful upcoming event
                most_impactful = max(upcoming_events, key=self._calculate_event_impact_score)
                positioned_for = most_impactful.expected_direction
            else:
                positioned_for = 'neutral'
            
            return {
                'preparation_score': avg_preparation,
                'surprise_potential': avg_surprise,
                'positioned_for': positioned_for,
                'scheduled_events_count': len(scheduled_events),
                'unscheduled_events_count': len(unscheduled_events)
            }
            
        except Exception as e:
            self.logger.error(f"Erro na avaliação de preparação: {e}")
            return {
                'preparation_score': 50.0,
                'surprise_potential': 30.0,
                'positioned_for': 'neutral',
                'scheduled_events_count': 0,
                'unscheduled_events_count': 0
            }
    
    def _generate_fundamental_recommendations(self, all_events: List[FundamentalEvent],
                                            high_impact_events: List[FundamentalEvent],
                                            sentiment: Dict, preparation: Dict) -> Dict:
        """Gera recomendações baseadas em fundamentals"""
        try:
            recommendations = {
                'signal_strength': 50.0,
                'direction': 'neutral',
                'confidence': 50.0,
                'time_horizon': 'medium_term',
                'action': 'hold',
                'position_adjustment': 1.0,
                'risk_adjustment': 1.0,
                'timing': 'normal'
            }
            
            # Signal strength baseado em eventos de alto impacto
            if high_impact_events:
                avg_impact_score = np.mean([self._calculate_event_impact_score(e) for e in high_impact_events])
                recommendations['signal_strength'] = min(95, avg_impact_score)
            else:
                recommendations['signal_strength'] = 30
            
            # Direction baseado em sentiment
            recommendations['direction'] = sentiment['direction']
            
            # Confidence
            confidence_factors = [
                sentiment['strength'] * 0.4,
                preparation['preparation_score'] * 0.3,
                min(100, len(all_events) * 10) * 0.3  # More events = more confidence
            ]
            recommendations['confidence'] = min(95, sum(confidence_factors))
            
            # Time horizon
            upcoming_events = [e for e in all_events if e.time_until_event.total_seconds() > 0]
            if upcoming_events:
                min_time_until = min(e.time_until_event.days for e in upcoming_events)
                if min_time_until <= 1:
                    recommendations['time_horizon'] = 'short_term'
                elif min_time_until <= 7:
                    recommendations['time_horizon'] = 'medium_term'
                else:
                    recommendations['time_horizon'] = 'long_term'
            
            # Action recommendation
            if recommendations['signal_strength'] > 70:
                if sentiment['direction'] == 'bullish':
                    recommendations['action'] = 'buy'
                elif sentiment['direction'] == 'bearish':
                    recommendations['action'] = 'sell'
                else:
                    recommendations['action'] = 'wait'
            elif recommendations['signal_strength'] > 40:
                recommendations['action'] = 'hold'
            else:
                recommendations['action'] = 'wait'
            
            # Position size adjustment
            if high_impact_events:
                # Reduce size before high impact events due to uncertainty
                recommendations['position_adjustment'] = 0.7
                recommendations['risk_adjustment'] = 1.3
            else:
                recommendations['position_adjustment'] = 1.0
                recommendations['risk_adjustment'] = 1.0
            
            # Timing
            immediate_events = [e for e in upcoming_events 
                              if e.time_until_event.total_seconds() < 24*3600]  # Next 24h
            if immediate_events:
                recommendations['timing'] = 'before_event'
            elif preparation['surprise_potential'] > 70:
                recommendations['timing'] = 'after_event'
            else:
                recommendations['timing'] = 'normal'
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Erro nas recomendações: {e}")
            return {
                'signal_strength': 30, 'direction': 'neutral', 'confidence': 40,
                'action': 'wait', 'position_adjustment': 0.7, 'timing': 'normal'
            }
    
    def _assess_fundamental_risks(self, events: List[FundamentalEvent], 
                                news: List[NewsAnalysis]) -> Dict:
        """Avalia riscos fundamentais"""
        try:
            risks = {
                'high_volatility': False,
                'regulatory_risk': False,
                'black_swan': False
            }
            
            # High volatility risk
            high_vol_events = [e for e in events if e.volatility_expected > 5.0]
            if high_vol_events:
                risks['high_volatility'] = True
            
            # Regulatory risk
            regulatory_events = [e for e in events if e.event_type == EventType.REGULATORY]
            if regulatory_events:
                risks['regulatory_risk'] = True
            
            # Black swan potential
            very_high_impact = [e for e in events 
                              if e.impact == EventImpact.VERY_HIGH and e.surprise_factor > 80]
            if very_high_impact:
                risks['black_swan'] = True
            
            return risks
            
        except Exception:
            return {'high_volatility': False, 'regulatory_risk': False, 'black_swan': False}
    
    def should_trade_with_fundamentals(self, signal: FundamentalSignal, 
                                     strategy_type: str) -> Dict:
        """
        Determina se deve tradear considerando fundamentals
        
        Args:
            signal: FundamentalSignal atual
            strategy_type: Tipo de estratégia
            
        Returns:
            Dict com decisão e ajustes
        """
        try:
            should_trade = True
            adjustments = {}
            reasons = []
            
            # Check for high impact events
            if signal.high_impact_events:
                next_event_hours = min(
                    e.time_until_event.total_seconds() / 3600 
                    for e in signal.high_impact_events 
                    if e.time_until_event.total_seconds() > 0
                )
                
                if next_event_hours < 24:  # Event in next 24 hours
                    if signal.timing_recommendation == 'avoid':
                        should_trade = False
                        reasons.append("Alto impacto evento nas próximas 24h - evitar trading")
                    else:
                        adjustments['position_size_multiplier'] = signal.position_size_adjustment
                        reasons.append(f"Evento de alto impacto próximo - ajustar posição para {signal.position_size_adjustment:.1f}x")
            
            # Check regulatory risk
            if signal.regulatory_risk:
                adjustments['position_size_multiplier'] = 0.6
                reasons.append("Risco regulatório detectado - reduzir exposição")
            
            # Check black swan potential
            if signal.black_swan_potential:
                should_trade = False
                reasons.append("Potencial black swan - evitar trading")
            
            # Direction alignment check
            if strategy_type in ['swing_long', 'trend_following_long'] and signal.direction == 'bearish':
                should_trade = False
                reasons.append("Fundamentals bearish conflitam com estratégia bullish")
            elif strategy_type in ['swing_short', 'trend_following_short'] and signal.direction == 'bullish':
                should_trade = False
                reasons.append("Fundamentals bullish conflitam com estratégia bearish")
            
            # Apply fundamental adjustments if trading
            if should_trade:
                if 'position_size_multiplier' not in adjustments:
                    adjustments['position_size_multiplier'] = signal.position_size_adjustment
                
                adjustments['risk_multiplier'] = signal.risk_adjustment
                adjustments['time_horizon'] = signal.time_horizon
                
                if signal.signal_strength > 70:
                    reasons.append(f"Fundamentals fortes ({signal.signal_strength:.0f}) suportam {signal.direction}")
                elif signal.signal_strength > 40:
                    reasons.append(f"Fundamentals moderados - proceder com cautela")
                else:
                    reasons.append(f"Fundamentals fracos - reduzir exposição")
                    adjustments['position_size_multiplier'] *= 0.7
            
            return {
                'should_trade': should_trade,
                'adjustments': adjustments,
                'reasons': reasons,
                'signal_strength': signal.signal_strength,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'upcoming_events': len(signal.upcoming_events),
                'high_impact_events': len(signal.high_impact_events),
                'recommended_action': signal.recommended_action
            }
            
        except Exception as e:
            self.logger.error(f"Erro na decisão fundamental: {e}")
            return {
                'should_trade': False,
                'adjustments': {'position_size_multiplier': 0.5},
                'reasons': ['Erro na análise fundamental'],
                'confidence': 30
            }


def main():
    """Teste básico do filtro fundamental"""
    # Testar filtro
    fundamental_filter = FundamentalFilter()
    
    # Simular análise para Bitcoin
    signal = fundamental_filter.analyze_fundamentals("BTCUSDT")
    
    print(f"\n📰 FUNDAMENTAL ANALYSIS")
    print(f"Signal Strength: {signal.signal_strength:.1f}")
    print(f"Direction: {signal.direction}")
    print(f"Confidence: {signal.confidence:.1f}%")
    print(f"Time Horizon: {signal.time_horizon}")
    print(f"Recommended Action: {signal.recommended_action}")
    
    print(f"\n📅 EVENTS SUMMARY")
    print(f"Upcoming Events: {len(signal.upcoming_events)}")
    print(f"Recent Events: {len(signal.recent_events)}")
    print(f"High Impact Events: {len(signal.high_impact_events)}")
    
    if signal.high_impact_events:
        print(f"\n⚠️  HIGH IMPACT EVENTS:")
        for event in signal.high_impact_events[:3]:
            print(f"   • {event.title}")
            print(f"     Impact: {event.impact.value}, Time: {event.time_until_event}")
    
    print(f"\n📊 MARKET POSITIONING")
    print(f"Market Positioned For: {signal.market_positioned_for}")
    print(f"Surprise Potential: {signal.surprise_potential:.1f}")
    
    print(f"\n🎯 TRADING IMPLICATIONS")
    print(f"Position Size Adjustment: {signal.position_size_adjustment:.2f}x")
    print(f"Risk Adjustment: {signal.risk_adjustment:.2f}x")
    print(f"Timing Recommendation: {signal.timing_recommendation}")
    
    print(f"\n⚠️  RISK WARNINGS")
    if signal.high_volatility_warning:
        print("   • High Volatility Warning")
    if signal.regulatory_risk:
        print("   • Regulatory Risk")
    if signal.black_swan_potential:
        print("   • Black Swan Potential")
    
    # Testar decisão de trading
    trade_decision = fundamental_filter.should_trade_with_fundamentals(signal, "swing")
    print(f"\n🎯 TRADE DECISION (Swing Strategy)")
    print(f"Should Trade: {trade_decision['should_trade']}")
    if trade_decision['should_trade']:
        print(f"Position Multiplier: {trade_decision['adjustments']['position_size_multiplier']:.2f}x")
        print(f"Risk Multiplier: {trade_decision['adjustments']['risk_multiplier']:.2f}x")
    print(f"Reasons: {', '.join(trade_decision['reasons'])}")


if __name__ == "__main__":
    main()