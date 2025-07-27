"""
🔮 LEADING INDICATORS - Smart Trading System v2.0

Indicadores que precedem movimentos de preço:
- Volume Profile (VPOC, Value Areas)
- Order Flow Analysis (Buy/Sell Pressure)
- Market Microstructure
- Liquidity Analysis

Filosofia: Leading > Lagging | Context > Signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileData:
    """Dados do Volume Profile"""
    poc: float                    # Point of Control
    value_area_high: float       # VA High (70% volume)
    value_area_low: float        # VA Low (70% volume)
    volume_nodes: Dict[float, float]  # Price -> Volume
    total_volume: float
    timeframe: str


@dataclass
class OrderFlowData:
    """Dados do Order Flow"""
    buy_pressure: float          # 0-100 (% compra)
    sell_pressure: float         # 0-100 (% venda)
    net_flow: float             # Buy - Sell pressure
    volume_ratio: float         # Buy Vol / Sell Vol
    momentum_score: float       # 0-100 momentum
    liquidity_score: float      # 0-100 liquidez


@dataclass
class LeadingSignal:
    """Sinal dos indicadores leading"""
    signal_type: str            # 'volume_breakout', 'order_flow', 'liquidity'
    strength: float             # 0-100 força do sinal
    direction: str              # 'bullish', 'bearish', 'neutral'
    confidence: float           # 0-100 confiança
    timeframe: str
    timestamp: pd.Timestamp
    details: Dict


class VolumeProfileAnalyzer:
    """
    🎯 Volume Profile Analysis
    
    Analisa distribuição de volume por preço para identificar:
    - Point of Control (POC) - preço com maior volume
    - Value Areas - zonas de 70% do volume
    - Volume Nodes - suporte/resistência por volume
    """
    
    def __init__(self, num_bins: int = 50, value_area_percent: float = 0.70):
        self.num_bins = num_bins
        self.value_area_percent = value_area_percent
        self.logger = logging.getLogger(f"{__name__}.VolumeProfile")
    
    def calculate_volume_profile(self, 
                               df: pd.DataFrame, 
                               timeframe: str = "4H") -> VolumeProfileData:
        """
        Calcula Volume Profile para o período
        
        Args:
            df: DataFrame com OHLCV data
            timeframe: Timeframe dos dados
            
        Returns:
            VolumeProfileData com métricas calculadas
        """
        try:
            if len(df) < 20:
                raise ValueError("Dados insuficientes para Volume Profile")
            
            # Criar bins de preço
            price_min = df['low'].min()
            price_max = df['high'].max()
            price_bins = np.linspace(price_min, price_max, self.num_bins)
            
            # Distribuir volume pelos bins
            volume_by_price = {}
            
            for _, row in df.iterrows():
                # Volume distribuído proporcionalmente no range OHLC
                prices_in_bar = np.linspace(row['low'], row['high'], 10)
                volume_per_price = row['volume'] / len(prices_in_bar)
                
                for price in prices_in_bar:
                    bin_idx = np.digitize(price, price_bins) - 1
                    bin_idx = max(0, min(bin_idx, len(price_bins) - 2))
                    bin_price = price_bins[bin_idx]
                    
                    if bin_price in volume_by_price:
                        volume_by_price[bin_price] += volume_per_price
                    else:
                        volume_by_price[bin_price] = volume_per_price
            
            # Point of Control (maior volume)
            poc = max(volume_by_price.keys(), key=lambda x: volume_by_price[x])
            
            # Value Area (70% do volume)
            total_volume = sum(volume_by_price.values())
            target_volume = total_volume * self.value_area_percent
            
            # Sorted prices by volume (desc)
            sorted_prices = sorted(volume_by_price.keys(), 
                                 key=lambda x: volume_by_price[x], 
                                 reverse=True)
            
            # Build value area
            va_volume = 0
            va_prices = []
            
            for price in sorted_prices:
                va_volume += volume_by_price[price]
                va_prices.append(price)
                if va_volume >= target_volume:
                    break
            
            value_area_high = max(va_prices)
            value_area_low = min(va_prices)
            
            self.logger.info(f"Volume Profile calculado - POC: {poc:.2f}, "
                           f"VA: {value_area_low:.2f}-{value_area_high:.2f}")
            
            return VolumeProfileData(
                poc=poc,
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                volume_nodes=volume_by_price,
                total_volume=total_volume,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.error(f"Erro no Volume Profile: {e}")
            raise
    
    def analyze_volume_breakout(self, 
                              current_price: float,
                              volume_profile: VolumeProfileData,
                              volume_threshold: float = 1.5) -> Optional[LeadingSignal]:
        """
        Analisa breakouts baseados em volume profile
        
        Args:
            current_price: Preço atual
            volume_profile: Dados do volume profile
            volume_threshold: Multiplicador para breakout
            
        Returns:
            LeadingSignal se houver breakout significativo
        """
        try:
            poc = volume_profile.poc
            va_high = volume_profile.value_area_high
            va_low = volume_profile.value_area_low
            
            # Detectar posição em relação ao Volume Profile
            if current_price > va_high:
                position = "above_va"
                direction = "bullish"
                distance = (current_price - va_high) / va_high
            elif current_price < va_low:
                position = "below_va"
                direction = "bearish"
                distance = (va_low - current_price) / va_low
            else:
                position = "inside_va"
                direction = "neutral"
                distance = 0
            
            # Calcular força do sinal
            if position == "inside_va":
                strength = 30  # Neutro dentro da VA
            else:
                # Força baseada na distância da VA
                strength = min(90, 50 + (distance * 200))
            
            # Confiança baseada no volume total
            confidence = min(85, volume_profile.total_volume / 1000000 * 10)
            
            if strength > 60:  # Sinal significativo
                return LeadingSignal(
                    signal_type="volume_breakout",
                    strength=strength,
                    direction=direction,
                    confidence=confidence,
                    timeframe=volume_profile.timeframe,
                    timestamp=pd.Timestamp.now(),
                    details={
                        "position": position,
                        "poc": poc,
                        "va_high": va_high,
                        "va_low": va_low,
                        "distance_from_va": distance
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de breakout: {e}")
            return None


class OrderFlowAnalyzer:
    """
    💹 Order Flow Analysis
    
    Analisa fluxo de ordens para detectar:
    - Pressão compradora vs vendedora
    - Momentum direcional
    - Mudanças de sentimento
    """
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(f"{__name__}.OrderFlow")
    
    def calculate_order_flow(self, df: pd.DataFrame) -> OrderFlowData:
        """
        Calcula métricas de order flow
        
        Args:
            df: DataFrame com OHLCV + tape data (se disponível)
            
        Returns:
            OrderFlowData com métricas calculadas
        """
        try:
            if len(df) < self.lookback_periods:
                raise ValueError("Dados insuficientes para Order Flow")
            
            # Usar últimos períodos
            recent_df = df.tail(self.lookback_periods).copy()
            
            # Estimar buy/sell pressure usando price action
            buy_volume = []
            sell_volume = []
            
            for _, row in recent_df.iterrows():
                # Heurística: se close > open, mais pressão compradora
                close_position = (row['close'] - row['low']) / (row['high'] - row['low'] + 1e-8)
                
                buy_vol = row['volume'] * close_position
                sell_vol = row['volume'] * (1 - close_position)
                
                buy_volume.append(buy_vol)
                sell_volume.append(sell_vol)
            
            total_buy = sum(buy_volume)
            total_sell = sum(sell_volume)
            total_volume = total_buy + total_sell
            
            # Calcular pressões
            buy_pressure = (total_buy / total_volume) * 100 if total_volume > 0 else 50
            sell_pressure = (total_sell / total_volume) * 100 if total_volume > 0 else 50
            net_flow = buy_pressure - sell_pressure
            volume_ratio = total_buy / (total_sell + 1e-8)
            
            # Momentum score baseado em consistência
            momentum_periods = min(10, len(recent_df))
            momentum_scores = []
            
            for i in range(len(recent_df) - momentum_periods + 1):
                period_df = recent_df.iloc[i:i+momentum_periods]
                period_returns = period_df['close'].pct_change().dropna()
                
                if len(period_returns) > 0:
                    # Consistência de direção
                    positive_returns = (period_returns > 0).sum()
                    consistency = positive_returns / len(period_returns)
                    momentum_scores.append(consistency)
            
            momentum_score = np.mean(momentum_scores) * 100 if momentum_scores else 50
            
            # Liquidity score baseado em volume e spread
            avg_volume = recent_df['volume'].mean()
            volume_std = recent_df['volume'].std()
            liquidity_score = min(90, (avg_volume / (volume_std + 1e-8)) * 10)
            
            self.logger.info(f"Order Flow - Buy: {buy_pressure:.1f}%, "
                           f"Sell: {sell_pressure:.1f}%, Net: {net_flow:.1f}")
            
            return OrderFlowData(
                buy_pressure=buy_pressure,
                sell_pressure=sell_pressure,
                net_flow=net_flow,
                volume_ratio=volume_ratio,
                momentum_score=momentum_score,
                liquidity_score=liquidity_score
            )
            
        except Exception as e:
            self.logger.error(f"Erro no Order Flow: {e}")
            raise
    
    def analyze_order_flow_signal(self, 
                                order_flow: OrderFlowData,
                                timeframe: str = "4H") -> Optional[LeadingSignal]:
        """
        Analisa sinais baseados em order flow
        
        Args:
            order_flow: Dados do order flow
            timeframe: Timeframe dos dados
            
        Returns:
            LeadingSignal se houver sinal significativo
        """
        try:
            net_flow = order_flow.net_flow
            momentum = order_flow.momentum_score
            liquidity = order_flow.liquidity_score
            
            # Determinar direção
            if net_flow > 15 and momentum > 60:
                direction = "bullish"
                strength = min(90, (net_flow + momentum) / 2)
            elif net_flow < -15 and momentum < 40:
                direction = "bearish"
                strength = min(90, (abs(net_flow) + (100-momentum)) / 2)
            else:
                direction = "neutral"
                strength = 30
            
            # Confiança baseada em liquidez e consistência
            confidence = min(85, (liquidity + abs(net_flow)) / 2)
            
            if strength > 55:  # Sinal significativo
                return LeadingSignal(
                    signal_type="order_flow",
                    strength=strength,
                    direction=direction,
                    confidence=confidence,
                    timeframe=timeframe,
                    timestamp=pd.Timestamp.now(),
                    details={
                        "net_flow": net_flow,
                        "buy_pressure": order_flow.buy_pressure,
                        "sell_pressure": order_flow.sell_pressure,
                        "momentum_score": momentum,
                        "liquidity_score": liquidity,
                        "volume_ratio": order_flow.volume_ratio
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de order flow: {e}")
            return None


class LiquidityAnalyzer:
    """
    💧 Liquidity Analysis
    
    Analisa liquidez do mercado para identificar:
    - Zonas de liquidez
    - Sweeps de liquidez
    - Areas de acumulação/distribuição
    """
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(f"{__name__}.Liquidity")
    
    def find_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identifica zonas de liquidez baseadas em:
        - Highs/Lows com alta rejeição
        - Áreas de alto volume
        - Support/Resistance significantes
        """
        try:
            if len(df) < self.lookback_periods:
                return []
            
            recent_df = df.tail(self.lookback_periods).copy()
            liquidity_zones = []
            
            # Identificar highs/lows significativos
            highs = recent_df['high'].rolling(5, center=True).max()
            lows = recent_df['low'].rolling(5, center=True).min()
            
            # Zones onde preço foi rejeitado múltiplas vezes
            for i in range(5, len(recent_df) - 5):
                current_high = recent_df.iloc[i]['high']
                current_low = recent_df.iloc[i]['low']
                current_volume = recent_df.iloc[i]['volume']
                
                # Check for liquidity at highs
                if highs.iloc[i] == current_high:
                    # Count touches within 1% range
                    price_range = current_high * 0.01
                    touches = 0
                    total_volume = 0
                    
                    for j in range(max(0, i-10), min(len(recent_df), i+10)):
                        if abs(recent_df.iloc[j]['high'] - current_high) <= price_range:
                            touches += 1
                            total_volume += recent_df.iloc[j]['volume']
                    
                    if touches >= 3:  # Múltiplos toques
                        liquidity_zones.append({
                            'type': 'resistance',
                            'price': current_high,
                            'strength': min(100, touches * 20),
                            'volume': total_volume,
                            'touches': touches,
                            'timestamp': recent_df.index[i]
                        })
                
                # Check for liquidity at lows
                if lows.iloc[i] == current_low:
                    price_range = current_low * 0.01
                    touches = 0
                    total_volume = 0
                    
                    for j in range(max(0, i-10), min(len(recent_df), i+10)):
                        if abs(recent_df.iloc[j]['low'] - current_low) <= price_range:
                            touches += 1
                            total_volume += recent_df.iloc[j]['volume']
                    
                    if touches >= 3:
                        liquidity_zones.append({
                            'type': 'support',
                            'price': current_low,
                            'strength': min(100, touches * 20),
                            'volume': total_volume,
                            'touches': touches,
                            'timestamp': recent_df.index[i]
                        })
            
            # Sort by strength
            liquidity_zones.sort(key=lambda x: x['strength'], reverse=True)
            
            self.logger.info(f"Encontradas {len(liquidity_zones)} zonas de liquidez")
            return liquidity_zones[:10]  # Top 10
            
        except Exception as e:
            self.logger.error(f"Erro na análise de liquidez: {e}")
            return []
    
    def analyze_liquidity_sweep(self, 
                              current_price: float,
                              liquidity_zones: List[Dict],
                              timeframe: str = "4H") -> Optional[LeadingSignal]:
        """
        Analisa sweeps de liquidez e retornos para zonas
        """
        try:
            if not liquidity_zones:
                return None
            
            # Encontrar zona mais próxima
            closest_zone = min(liquidity_zones, 
                             key=lambda x: abs(x['price'] - current_price))
            
            distance_pct = abs(closest_zone['price'] - current_price) / current_price
            
            # Se muito próximo de uma zona (< 2%)
            if distance_pct < 0.02:
                zone_type = closest_zone['type']
                strength = closest_zone['strength']
                
                # Determinar se é possível sweep ou bounce
                if zone_type == 'resistance' and current_price >= closest_zone['price'] * 0.995:
                    # Próximo de resistência - possível rejeição
                    direction = "bearish"
                    signal_strength = min(80, strength * 0.8)
                elif zone_type == 'support' and current_price <= closest_zone['price'] * 1.005:
                    # Próximo de suporte - possível bounce
                    direction = "bullish"
                    signal_strength = min(80, strength * 0.8)
                else:
                    return None
                
                confidence = min(75, strength * 0.7)
                
                return LeadingSignal(
                    signal_type="liquidity",
                    strength=signal_strength,
                    direction=direction,
                    confidence=confidence,
                    timeframe=timeframe,
                    timestamp=pd.Timestamp.now(),
                    details={
                        "zone_type": zone_type,
                        "zone_price": closest_zone['price'],
                        "zone_strength": strength,
                        "distance_pct": distance_pct,
                        "touches": closest_zone['touches']
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro na análise de sweep: {e}")
            return None


class LeadingIndicatorsSystem:
    """
    🎯 Sistema Principal de Leading Indicators
    
    Combina todos os indicadores leading:
    - Volume Profile
    - Order Flow
    - Liquidity Analysis
    """
    
    def __init__(self):
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.LeadingSystem")
    
    def analyze_all_leading(self, 
                          df: pd.DataFrame,
                          current_price: float,
                          timeframe: str = "4H") -> List[LeadingSignal]:
        """
        Analisa todos os indicadores leading
        
        Args:
            df: DataFrame com dados OHLCV
            current_price: Preço atual
            timeframe: Timeframe dos dados
            
        Returns:
            Lista de LeadingSignals encontrados
        """
        signals = []
        
        try:
            # Volume Profile Analysis
            volume_profile = self.volume_analyzer.calculate_volume_profile(df, timeframe)
            volume_signal = self.volume_analyzer.analyze_volume_breakout(
                current_price, volume_profile)
            if volume_signal:
                signals.append(volume_signal)
            
            # Order Flow Analysis  
            order_flow = self.order_flow_analyzer.calculate_order_flow(df)
            flow_signal = self.order_flow_analyzer.analyze_order_flow_signal(
                order_flow, timeframe)
            if flow_signal:
                signals.append(flow_signal)
            
            # Liquidity Analysis
            liquidity_zones = self.liquidity_analyzer.find_liquidity_zones(df)
            liquidity_signal = self.liquidity_analyzer.analyze_liquidity_sweep(
                current_price, liquidity_zones, timeframe)
            if liquidity_signal:
                signals.append(liquidity_signal)
            
            self.logger.info(f"Leading Analysis completa - {len(signals)} sinais encontrados")
            return signals
            
        except Exception as e:
            self.logger.error(f"Erro na análise leading: {e}")
            return signals
    
    def get_leading_score(self, signals: List[LeadingSignal]) -> Dict:
        """
        Calcula score agregado dos leading indicators
        
        Returns:
            Dict com scores e direção predominante
        """
        if not signals:
            return {
                'overall_score': 50,
                'direction': 'neutral',
                'confidence': 30,
                'signal_count': 0
            }
        
        # Calcular scores por direção
        bullish_scores = [s.strength for s in signals if s.direction == 'bullish']
        bearish_scores = [s.strength for s in signals if s.direction == 'bearish']
        neutral_scores = [s.strength for s in signals if s.direction == 'neutral']
        
        # Score agregado
        bullish_score = np.mean(bullish_scores) if bullish_scores else 0
        bearish_score = np.mean(bearish_scores) if bearish_scores else 0
        neutral_score = np.mean(neutral_scores) if neutral_scores else 0
        
        # Determinar direção predominante
        if bullish_score > bearish_score and bullish_score > 60:
            direction = 'bullish'
            overall_score = min(90, bullish_score)
        elif bearish_score > bullish_score and bearish_score > 60:
            direction = 'bearish'
            overall_score = min(90, bearish_score)
        else:
            direction = 'neutral'
            overall_score = max(30, neutral_score)
        
        # Confiança baseada em consenso
        total_signals = len(signals)
        direction_signals = len([s for s in signals if s.direction == direction])
        confidence = min(85, (direction_signals / total_signals) * 100)
        
        return {
            'overall_score': overall_score,
            'direction': direction,
            'confidence': confidence,
            'signal_count': total_signals,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'signals': signals
        }


def main():
    """Teste básico do sistema"""
    # Dados de exemplo
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
    np.random.seed(42)
    
    # Simular dados OHLCV
    data = {
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50500,
        'low': np.random.randn(100).cumsum() + 49500,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.exponential(1000000, 100)
    }
    
    df = pd.DataFrame(data, index=dates)
    df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
    df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
    
    # Testar sistema
    system = LeadingIndicatorsSystem()
    current_price = df['close'].iloc[-1]
    
    signals = system.analyze_all_leading(df, current_price, "4H")
    score = system.get_leading_score(signals)
    
    print(f"\n🔮 LEADING INDICATORS ANALYSIS")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Signals Found: {len(signals)}")
    print(f"Overall Score: {score['overall_score']:.1f}")
    print(f"Direction: {score['direction']}")
    print(f"Confidence: {score['confidence']:.1f}%")
    
    for signal in signals:
        print(f"\n📊 {signal.signal_type.upper()}")
        print(f"   Direction: {signal.direction}")
        print(f"   Strength: {signal.strength:.1f}")
        print(f"   Confidence: {signal.confidence:.1f}%")


if __name__ == "__main__":
    main()