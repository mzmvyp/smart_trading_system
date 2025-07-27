"""
Strategies: Mean Reversion Strategy
Estratégia de reversão à média em extremos com filtros inteligentes
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.signal_manager import SignalType, SignalPriority
from indicators.confluence_analyzer import ConfluenceAnalyzer
from indicators.divergence_detector import DivergenceDetector, DivergenceType
from filters.volatility_filter import VolatilityFilter
from filters.market_condition import MarketConditionFilter
from utils.logger import get_logger
from utils.helpers import (
    calculate_percentage_change,
    safe_divide,
    normalize_value
)


logger = get_logger(__name__)


class MeanReversionSetup(Enum):
    """Tipos de setup de mean reversion"""
    OVERSOLD_BOUNCE = "oversold_bounce"
    OVERBOUGHT_DUMP = "overbought_dump"
    BOLLINGER_REVERSION = "bollinger_reversion"
    SUPPORT_BOUNCE = "support_bounce"
    RESISTANCE_REJECTION = "resistance_rejection"


@dataclass
class MeanReversionConfig:
    """Configurações da estratégia Mean Reversion"""
    # Indicadores principais
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    rsi_extreme_oversold: float = 20
    rsi_extreme_overbought: float = 80
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_squeeze_threshold: float = 0.1  # Quando as bands estão apertadas
    
    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_oversold: float = 20
    stoch_overbought: float = 80
    
    # Williams %R
    williams_period: int = 14
    williams_oversold: float = -80
    williams_overbought: float = -20
    
    # Filtros
    min_volume_ratio: float = 1.2  # Volume deve ser 20% acima da média
    max_volatility_percentile: float = 80  # Não opera em volatilidade extrema
    min_confluence_score: float = 65
    
    # Risk Management
    default_stop_distance: float = 3.0  # % de distância do stop
    profit_target_ratio: float = 2.0   # Risk:Reward ratio
    max_holding_period: int = 48        # Máximo de barras para manter posição


class MeanReversionStrategy:
    """Estratégia de Mean Reversion inteligente"""
    
    def __init__(self, config: MeanReversionConfig = None):
        self.config = config or MeanReversionConfig()
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.volatility_filter = VolatilityFilter()
        self.market_filter = MarketConditionFilter()
        
        self.name = "mean_reversion"
        self.timeframes = ["1h", "4h"]  # Timeframes preferidos
        
        # Cache para otimização
        self._indicator_cache = {}
    
    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Análise principal da estratégia"""
        
        if len(df) < 50:
            return {'signals': [], 'analysis': {'error': 'Dados insuficientes'}}
        
        try:
            # Calcula indicadores
            indicators = self._calculate_indicators(df)
            
            # Aplica filtros
            filters_result = self._apply_filters(df, symbol, timeframe, indicators)
            if not filters_result['passed']:
                return {
                    'signals': [],
                    'analysis': {
                        'filters_passed': False,
                        'filter_reasons': filters_result['reasons']
                    }
                }
            
            # Identifica setups
            setups = self._identify_setups(df, indicators, timeframe)
            
            # Gera sinais
            signals = []
            for setup in setups:
                signal = self._generate_signal(df, setup, indicators, symbol, timeframe)
                if signal:
                    signals.append(signal)
            
            # Análise de confluência
            confluence_analysis = self._analyze_confluence(df, indicators, timeframe)
            
            return {
                'signals': signals,
                'analysis': {
                    'filters_passed': True,
                    'setups_found': len(setups),
                    'confluence': confluence_analysis,
                    'market_condition': filters_result['market_condition'],
                    'volatility_state': filters_result['volatility_state']
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise mean reversion: {e}")
            return {'signals': [], 'analysis': {'error': str(e)}}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcula todos os indicadores necessários"""
        
        indicators = {}
        
        try:
            # RSI
            indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.config.rsi_period)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'],
                timeperiod=self.config.bb_period,
                nbdevup=self.config.bb_std,
                nbdevdn=self.config.bb_std
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                k_period=self.config.stoch_k,
                d_period=self.config.stoch_d
            )
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
            
            # Williams %R
            indicators['williams_r'] = talib.WILLR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.config.williams_period
            )
            
            # CCI (Commodity Channel Index)
            indicators['cci'] = talib.CCI(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=20
            )
            
            # Volume Analysis
            indicators['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            # Price position within Bollinger Bands
            indicators['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Distance from moving averages
            indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            indicators['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            indicators['distance_sma20'] = ((df['close'] - indicators['sma_20']) / indicators['sma_20']) * 100
            indicators['distance_sma50'] = ((df['close'] - indicators['sma_50']) / indicators['sma_50']) * 100
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores: {e}")
        
        return indicators
    
    def _apply_filters(self, df: pd.DataFrame, symbol: str, timeframe: str, indicators: Dict) -> Dict:
        """Aplica filtros da estratégia"""
        
        reasons = []
        
        # Filtro de volatilidade
        volatility_result = self.volatility_filter.analyze(df, timeframe)
        if volatility_result['volatility_percentile'] > self.config.max_volatility_percentile:
            reasons.append(f"Volatilidade muito alta: {volatility_result['volatility_percentile']:.1f}%")
        
        # Filtro de condição de mercado
        market_result = self.market_filter.analyze(df, symbol, timeframe)
        
        # Para mean reversion, preferimos mercados laterais ou com reversões
        if market_result['trend_strength'] > 80:  # Trend muito forte
            reasons.append(f"Trend muito forte para mean reversion: {market_result['trend_strength']:.1f}")
        
        # Filtro de volume
        current_volume_ratio = indicators['volume_ratio'].iloc[-1] if len(indicators['volume_ratio']) > 0 else 0
        if current_volume_ratio < self.config.min_volume_ratio:
            reasons.append(f"Volume insuficiente: {current_volume_ratio:.2f}")
        
        return {
            'passed': len(reasons) == 0,
            'reasons': reasons,
            'market_condition': market_result.get('condition', 'unknown'),
            'volatility_state': volatility_result.get('state', 'unknown')
        }
    
    def _identify_setups(self, df: pd.DataFrame, indicators: Dict, timeframe: str) -> List[Dict]:
        """Identifica setups de mean reversion"""
        
        setups = []
        current_idx = len(df) - 1
        
        # Setup 1: RSI Extremo com Divergência
        setups.extend(self._find_rsi_extreme_setups(df, indicators, current_idx))
        
        # Setup 2: Bollinger Bands Reversion
        setups.extend(self._find_bollinger_reversion_setups(df, indicators, current_idx))
        
        # Setup 3: Multi-Oscillator Oversold/Overbought
        setups.extend(self._find_multi_oscillator_setups(df, indicators, current_idx))
        
        # Setup 4: Support/Resistance Reversion
        setups.extend(self._find_support_resistance_setups(df, indicators, current_idx))
        
        return setups
    
    def _find_rsi_extreme_setups(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups baseados em RSI extremo"""
        
        setups = []
        
        current_rsi = indicators['rsi'].iloc[current_idx]
        
        # Oversold extremo
        if current_rsi <= self.config.rsi_extreme_oversold:
            
            # Verifica se RSI está saindo da zona oversold
            prev_rsi = indicators['rsi'].iloc[current_idx - 1] if current_idx > 0 else current_rsi
            
            if current_rsi > prev_rsi:  # RSI começando a subir
                setup = {
                    'type': MeanReversionSetup.OVERSOLD_BOUNCE,
                    'signal_type': SignalType.BUY,
                    'entry_price': df['close'].iloc[current_idx],
                    'confidence': self._calculate_rsi_setup_confidence(indicators, current_idx, 'bullish'),
                    'metadata': {
                        'rsi_value': current_rsi,
                        'rsi_divergence': self._check_rsi_divergence(df, indicators, 'bullish')
                    }
                }
                setups.append(setup)
        
        # Overbought extremo
        elif current_rsi >= self.config.rsi_extreme_overbought:
            
            prev_rsi = indicators['rsi'].iloc[current_idx - 1] if current_idx > 0 else current_rsi
            
            if current_rsi < prev_rsi:  # RSI começando a descer
                setup = {
                    'type': MeanReversionSetup.OVERBOUGHT_DUMP,
                    'signal_type': SignalType.SELL,
                    'entry_price': df['close'].iloc[current_idx],
                    'confidence': self._calculate_rsi_setup_confidence(indicators, current_idx, 'bearish'),
                    'metadata': {
                        'rsi_value': current_rsi,
                        'rsi_divergence': self._check_rsi_divergence(df, indicators, 'bearish')
                    }
                }
                setups.append(setup)
        
        return setups
    
    def _find_bollinger_reversion_setups(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de reversão nas Bollinger Bands"""
        
        setups = []
        
        current_bb_position = indicators['bb_position'].iloc[current_idx]
        current_price = df['close'].iloc[current_idx]
        bb_width = indicators['bb_width'].iloc[current_idx]
        
        # Verifica se as bands não estão muito apertadas (baixa volatilidade)
        if bb_width < self.config.bb_squeeze_threshold:
            return setups
        
        # Bounce da banda inferior
        if current_bb_position <= 0.1:  # Preço próximo da banda inferior
            
            # Verifica se houve toque na banda e agora está voltando
            prev_bb_position = indicators['bb_position'].iloc[current_idx - 1] if current_idx > 0 else current_bb_position
            
            if current_bb_position > prev_bb_position:  # Saindo da banda inferior
                setup = {
                    'type': MeanReversionSetup.BOLLINGER_REVERSION,
                    'signal_type': SignalType.BUY,
                    'entry_price': current_price,
                    'confidence': self._calculate_bollinger_setup_confidence(indicators, current_idx, 'bullish'),
                    'metadata': {
                        'bb_position': current_bb_position,
                        'bb_width': bb_width,
                        'support_level': indicators['bb_lower'].iloc[current_idx]
                    }
                }
                setups.append(setup)
        
        # Rejeição da banda superior
        elif current_bb_position >= 0.9:  # Preço próximo da banda superior
            
            prev_bb_position = indicators['bb_position'].iloc[current_idx - 1] if current_idx > 0 else current_bb_position
            
            if current_bb_position < prev_bb_position:  # Saindo da banda superior
                setup = {
                    'type': MeanReversionSetup.BOLLINGER_REVERSION,
                    'signal_type': SignalType.SELL,
                    'entry_price': current_price,
                    'confidence': self._calculate_bollinger_setup_confidence(indicators, current_idx, 'bearish'),
                    'metadata': {
                        'bb_position': current_bb_position,
                        'bb_width': bb_width,
                        'resistance_level': indicators['bb_upper'].iloc[current_idx]
                    }
                }
                setups.append(setup)
        
        return setups
    
    def _find_multi_oscillator_setups(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups baseados em múltiplos osciladores"""
        
        setups = []
        
        # Coleta valores atuais
        rsi = indicators['rsi'].iloc[current_idx]
        stoch_k = indicators['stoch_k'].iloc[current_idx]
        williams_r = indicators['williams_r'].iloc[current_idx]
        cci = indicators['cci'].iloc[current_idx]
        
        # Conta osciladores em zona oversold
        oversold_count = 0
        if rsi <= self.config.rsi_oversold:
            oversold_count += 1
        if stoch_k <= self.config.stoch_oversold:
            oversold_count += 1
        if williams_r <= self.config.williams_oversold:
            oversold_count += 1
        if cci <= -100:
            oversold_count += 1
        
        # Conta osciladores em zona overbought
        overbought_count = 0
        if rsi >= self.config.rsi_overbought:
            overbought_count += 1
        if stoch_k >= self.config.stoch_overbought:
            overbought_count += 1
        if williams_r >= self.config.williams_overbought:
            overbought_count += 1
        if cci >= 100:
            overbought_count += 1
        
        # Setup bullish: múltiplos osciladores oversold
        if oversold_count >= 3:
            setup = {
                'type': MeanReversionSetup.OVERSOLD_BOUNCE,
                'signal_type': SignalType.BUY,
                'entry_price': df['close'].iloc[current_idx],
                'confidence': min(95, 50 + (oversold_count * 10)),
                'metadata': {
                    'oversold_oscillators': oversold_count,
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'williams_r': williams_r,
                    'cci': cci
                }
            }
            setups.append(setup)
        
        # Setup bearish: múltiplos osciladores overbought
        elif overbought_count >= 3:
            setup = {
                'type': MeanReversionSetup.OVERBOUGHT_DUMP,
                'signal_type': SignalType.SELL,
                'entry_price': df['close'].iloc[current_idx],
                'confidence': min(95, 50 + (overbought_count * 10)),
                'metadata': {
                    'overbought_oscillators': overbought_count,
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'williams_r': williams_r,
                    'cci': cci
                }
            }
            setups.append(setup)
        
        return setups
    
    def _find_support_resistance_setups(self, df: pd.DataFrame, indicators: Dict, current_idx: int) -> List[Dict]:
        """Encontra setups de reversão em suporte/resistência"""
        
        setups = []
        
        # Simplificado: usa SMAs como proxy para S/R
        current_price = df['close'].iloc[current_idx]
        sma_20 = indicators['sma_20'].iloc[current_idx]
        sma_50 = indicators['sma_50'].iloc[current_idx]
        
        distance_sma20 = indicators['distance_sma20'].iloc[current_idx]
        distance_sma50 = indicators['distance_sma50'].iloc[current_idx]
        
        # Bounce do suporte (SMA50)
        if distance_sma50 <= -3 and distance_sma50 > -8:  # Próximo mas não muito longe da SMA50
            
            # Verifica se está voltando em direção à média
            prev_distance = indicators['distance_sma50'].iloc[current_idx - 1] if current_idx > 0 else distance_sma50
            
            if distance_sma50 > prev_distance:  # Voltando para a média
                setup = {
                    'type': MeanReversionSetup.SUPPORT_BOUNCE,
                    'signal_type': SignalType.BUY,
                    'entry_price': current_price,
                    'confidence': self._calculate_support_resistance_confidence(distance_sma50, 'support'),
                    'metadata': {
                        'support_level': sma_50,
                        'distance_from_support': distance_sma50,
                        'support_type': 'sma_50'
                    }
                }
                setups.append(setup)
        
        # Rejeição da resistência
        elif distance_sma50 >= 3 and distance_sma50 < 8:  # Próximo mas não muito longe da resistência
            
            prev_distance = indicators['distance_sma50'].iloc[current_idx - 1] if current_idx > 0 else distance_sma50
            
            if distance_sma50 < prev_distance:  # Voltando para a média
                setup = {
                    'type': MeanReversionSetup.RESISTANCE_REJECTION,
                    'signal_type': SignalType.SELL,
                    'entry_price': current_price,
                    'confidence': self._calculate_support_resistance_confidence(distance_sma50, 'resistance'),
                    'metadata': {
                        'resistance_level': sma_50,
                        'distance_from_resistance': distance_sma50,
                        'resistance_type': 'sma_50'
                    }
                }
                setups.append(setup)
        
        return setups
    
    def _generate_signal(self, df: pd.DataFrame, setup: Dict, indicators: Dict, symbol: str, timeframe: str) -> Optional[Dict]:
        """Gera sinal final baseado no setup"""
        
        try:
            entry_price = setup['entry_price']
            signal_type = setup['signal_type']
            
            # Calcula stop loss e take profit
            stop_loss, take_profit = self._calculate_stop_and_target(
                df, indicators, entry_price, signal_type, setup
            )
            
            # Calcula scores
            confluence_score = self._calculate_confluence_score(setup, indicators)
            risk_score = self._calculate_risk_score(entry_price, stop_loss, take_profit)
            timing_score = setup['confidence']
            
            # Verifica score mínimo
            if confluence_score < self.config.min_confluence_score:
                logger.debug(f"Setup rejeitado por confluence score baixo: {confluence_score}")
                return None
            
            # Determina prioridade
            priority = self._determine_priority(confluence_score, setup['confidence'])
            
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy': self.name,
                'timeframe': timeframe,
                'priority': priority,
                'scores': {
                    'confluence': confluence_score,
                    'risk': risk_score,
                    'timing': timing_score
                },
                'setup_type': setup['type'].value,
                'metadata': setup.get('metadata', {}),
                'expires_at': None  # Mean reversion é mais rápido, sem expiração específica
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return None
    
    def _calculate_stop_and_target(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        entry_price: float,
        signal_type: SignalType,
        setup: Dict
    ) -> Tuple[float, float]:
        """Calcula stop loss e take profit"""
        
        # Stop loss baseado no setup
        if setup['type'] == MeanReversionSetup.BOLLINGER_REVERSION:
            if signal_type == SignalType.BUY:
                # Stop abaixo da banda inferior
                stop_loss = setup['metadata']['support_level'] * 0.995
            else:
                # Stop acima da banda superior
                stop_loss = setup['metadata']['resistance_level'] * 1.005
        
        elif setup['type'] in [MeanReversionSetup.SUPPORT_BOUNCE, MeanReversionSetup.RESISTANCE_REJECTION]:
            if signal_type == SignalType.BUY:
                # Stop abaixo do suporte
                support_level = setup['metadata']['support_level']
                stop_loss = support_level * 0.98
            else:
                # Stop acima da resistência
                resistance_level = setup['metadata']['resistance_level']
                stop_loss = resistance_level * 1.02
        
        else:
            # Stop loss padrão baseado em ATR ou percentual
            atr = self._calculate_atr(df)
            if signal_type == SignalType.BUY:
                stop_loss = entry_price - (atr * 2)
            else:
                stop_loss = entry_price + (atr * 2)
        
        # Take profit baseado na relação risk:reward
        risk = abs(entry_price - stop_loss)
        reward = risk * self.config.profit_target_ratio
        
        if signal_type == SignalType.BUY:
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return stop_loss, take_profit
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula Average True Range"""
        try:
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            return atr.iloc[-1] if len(atr) > 0 else df['close'].iloc[-1] * 0.02
        except:
            return df['close'].iloc[-1] * 0.02  # Fallback para 2%
    
    def _calculate_confluence_score(self, setup: Dict, indicators: Dict) -> float:
        """Calcula score de confluência"""
        
        base_score = setup['confidence']
        
        # Bonus por tipo de setup
        setup_bonuses = {
            MeanReversionSetup.OVERSOLD_BOUNCE: 10,
            MeanReversionSetup.OVERBOUGHT_DUMP: 10,
            MeanReversionSetup.BOLLINGER_REVERSION: 15,
            MeanReversionSetup.SUPPORT_BOUNCE: 12,
            MeanReversionSetup.RESISTANCE_REJECTION: 12
        }
        
        base_score += setup_bonuses.get(setup['type'], 0)
        
        # Bonus por divergências
        if setup.get('metadata', {}).get('rsi_divergence'):
            base_score += 15
        
        return min(100, max(0, base_score))
    
    def _calculate_risk_score(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calcula score de risco"""
        
        risk = abs(entry_price - stop_loss) / entry_price
        reward = abs(take_profit - entry_price) / entry_price
        
        rr_ratio = safe_divide(reward, risk, 0)
        
        # Score baseado na relação risco/recompensa
        if rr_ratio >= 3:
            return 90
        elif rr_ratio >= 2:
            return 80
        elif rr_ratio >= 1.5:
            return 70
        elif rr_ratio >= 1:
            return 60
        else:
            return 40
    
    def _calculate_rsi_setup_confidence(self, indicators: Dict, current_idx: int, direction: str) -> float:
        """Calcula confiança do setup RSI"""
        
        rsi = indicators['rsi'].iloc[current_idx]
        
        if direction == 'bullish':
            # Quanto mais oversold, maior a confiança
            confidence = normalize_value(rsi, 0, 30) * 40 + 50
        else:
            # Quanto mais overbought, maior a confiança
            confidence = normalize_value(rsi, 70, 100) * 40 + 50
        
        return min(95, max(50, confidence))
    
    def _calculate_bollinger_setup_confidence(self, indicators: Dict, current_idx: int, direction: str) -> float:
        """Calcula confiança do setup Bollinger"""
        
        bb_position = indicators['bb_position'].iloc[current_idx]
        bb_width = indicators['bb_width'].iloc[current_idx]
        
        # Maior confiança quando está mais próximo das bandas e bands são mais largas
        if direction == 'bullish':
            position_score = (1 - bb_position) * 40  # Mais próximo da banda inferior
        else:
            position_score = bb_position * 40  # Mais próximo da banda superior
        
        width_score = min(20, bb_width * 1000)  # Bonus por volatilidade
        
        return min(95, max(50, 50 + position_score + width_score))
    
    def _calculate_support_resistance_confidence(self, distance: float, level_type: str) -> float:
        """Calcula confiança do setup S/R"""
        
        abs_distance = abs(distance)
        
        # Confiança inversamente proporcional à distância
        if abs_distance <= 2:
            base_confidence = 85
        elif abs_distance <= 4:
            base_confidence = 75
        elif abs_distance <= 6:
            base_confidence = 65
        else:
            base_confidence = 55
        
        return base_confidence
    
    def _check_rsi_divergence(self, df: pd.DataFrame, indicators: Dict, direction: str) -> bool:
        """Verifica se existe divergência RSI"""
        
        try:
            # Usa o detector de divergências
            divergences = self.divergence_detector.detect_all_divergences(
                df.tail(50),  # Últimas 50 barras
                indicators=['rsi']
            )
            
            if direction == 'bullish':
                return any(d.type == DivergenceType.BULLISH_REGULAR for d in divergences)
            else:
                return any(d.type == DivergenceType.BEARISH_REGULAR for d in divergences)
        
        except Exception as e:
            logger.error(f"Erro ao verificar divergência: {e}")
            return False
    
    def _analyze_confluence(self, df: pd.DataFrame, indicators: Dict, timeframe: str) -> Dict:
        """Análise de confluência para mean reversion"""
        
        try:
            # Usa o analisador de confluência
            confluence_result = self.confluence_analyzer.analyze(df, timeframe)
            
            # Adiciona fatores específicos de mean reversion
            mean_reversion_factors = []
            
            current_idx = len(df) - 1
            rsi = indicators['rsi'].iloc[current_idx]
            bb_position = indicators['bb_position'].iloc[current_idx]
            
            if rsi <= 30 or rsi >= 70:
                mean_reversion_factors.append({
                    'factor': 'rsi_extreme',
                    'value': rsi,
                    'weight': 20
                })
            
            if bb_position <= 0.1 or bb_position >= 0.9:
                mean_reversion_factors.append({
                    'factor': 'bollinger_extreme',
                    'value': bb_position,
                    'weight': 15
                })
            
            confluence_result['mean_reversion_factors'] = mean_reversion_factors
            
            return confluence_result
        
        except Exception as e:
            logger.error(f"Erro na análise de confluência: {e}")
            return {}
    
    def _determine_priority(self, confluence_score: float, setup_confidence: float) -> SignalPriority:
        """Determina prioridade do sinal"""
        
        avg_score = (confluence_score + setup_confidence) / 2
        
        if avg_score >= 85:
            return SignalPriority.CRITICAL
        elif avg_score >= 75:
            return SignalPriority.HIGH
        elif avg_score >= 65:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW


# Função de conveniência
def create_mean_reversion_strategy(custom_config: Dict = None) -> MeanReversionStrategy:
    """Cria estratégia Mean Reversion com configuração customizada"""
    
    if custom_config:
        config = MeanReversionConfig()
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return MeanReversionStrategy(config)
    
    return MeanReversionStrategy()