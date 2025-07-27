"""
⚡ RISK MANAGER - Smart Trading System v2.0

Sistema avançado de gestão de risco adaptativo:
- Position Sizing baseado em volatilidade
- Stop Loss dinâmico por market structure
- Portfolio exposure management
- Correlation analysis
- Drawdown protection

Filosofia: Preserve Capital First, Profits Second = Long-term Survival
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Níveis de risco"""
    VERY_LOW = "very_low"      # 0.5-1% risk
    LOW = "low"               # 1-2% risk
    MEDIUM = "medium"         # 2-3% risk
    HIGH = "high"            # 3-4% risk
    VERY_HIGH = "very_high"  # 4-5% risk


class MarketRegime(Enum):
    """Regimes de mercado"""
    BULL_MARKET = "bull_market"         # Trending up
    BEAR_MARKET = "bear_market"         # Trending down
    RANGE_MARKET = "range_market"       # Sideways
    HIGH_VOLATILITY = "high_volatility" # Volatile
    LOW_VOLATILITY = "low_volatility"   # Low vol


@dataclass
class RiskMetrics:
    """Métricas de risco do portfolio"""
    total_exposure: float             # % exposição total
    max_single_position: float        # % maior posição
    correlation_risk: float           # Risco de correlação
    volatility_adjusted_risk: float   # Risco ajustado por vol
    drawdown_risk: float             # Risco de drawdown
    var_1d: float                    # Value at Risk 1 dia
    expected_shortfall: float        # Expected Shortfall
    sharpe_ratio: float              # Sharpe ratio atual
    max_drawdown: float              # Máximo drawdown
    winning_rate: float              # Taxa de acerto


@dataclass
class PositionRisk:
    """Risco de posição individual"""
    symbol: str
    entry_price: float
    current_price: float
    stop_loss: float
    position_size_usd: float
    position_size_pct: float
    risk_amount_usd: float
    risk_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    correlation_exposure: Dict[str, float]
    volatility_risk: float
    time_risk: float                 # Risco por tempo em posição


@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_portfolio_risk: float = 0.15        # 15% máx risco total
    max_single_position: float = 0.05       # 5% máx por posição
    max_correlation_exposure: float = 0.10   # 10% máx correlação
    max_daily_risk: float = 0.03            # 3% máx risco diário
    max_drawdown_limit: float = 0.20        # 20% máx drawdown
    max_positions: int = 10                 # Máx 10 posições
    correlation_threshold: float = 0.7       # Correlação limite
    volatility_multiplier: float = 2.0      # Multiplicador vol


@dataclass
class PositionSizingInput:
    """Input para cálculo de position sizing"""
    signal_strength: float           # 0-100 força do sinal
    confidence: float               # 0-100 confiança
    risk_reward_ratio: float        # R:R do setup
    entry_price: float
    stop_loss: float
    market_volatility: float        # ATR% atual
    portfolio_balance: float
    existing_exposure: float        # Exposição existente
    correlation_factor: float       # Fator de correlação


class RiskManager:
    """
    ⚡ Sistema Principal de Gestão de Risco
    
    Responsável por:
    1. Position Sizing adaptativo
    2. Stop Loss dinâmico
    3. Portfolio risk management
    4. Correlation analysis
    5. Drawdown protection
    """
    
    def __init__(self, 
                 initial_balance: float = 100000,
                 risk_limits: RiskLimits = None):
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_limits = risk_limits or RiskLimits()
        
        # Histórico de trades e performance
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.drawdown_periods: List[Dict] = []
        
        # Posições ativas
        self.active_positions: Dict[str, PositionRisk] = {}
        
        # Cache de volatilidade e correlações
        self.volatility_cache: Dict[str, float] = {}
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        self.logger = logging.getLogger(f"{__name__}.RiskManager")
        
        # Configurações adaptativas
        self.adaptive_config = {
            'volatility_lookback': 20,        # Períodos para ATR
            'correlation_lookback': 50,       # Períodos para correlação
            'performance_lookback': 100,      # Trades para performance
            'regime_detection_period': 30,    # Períodos para regime
            'max_heat': 0.06,                 # 6% máximo "heat"
            'cool_down_factor': 0.5           # Redução após loss
        }
    
    def calculate_position_size(self, 
                              sizing_input: PositionSizingInput,
                              symbol: str) -> Dict:
        """
        Calcula position size otimizado baseado em múltiplos fatores
        
        Args:
            sizing_input: Parâmetros para cálculo
            symbol: Símbolo da posição
            
        Returns:
            Dict com position size e detalhes do cálculo
        """
        try:
            self.logger.info(f"Calculando position size para {symbol}")
            
            # 1. Base risk baseado na força do sinal
            base_risk_pct = self._calculate_base_risk(sizing_input)
            
            # 2. Ajuste por volatilidade
            volatility_adjustment = self._calculate_volatility_adjustment(
                symbol, sizing_input.market_volatility)
            
            # 3. Ajuste por correlação
            correlation_adjustment = self._calculate_correlation_adjustment(
                symbol, sizing_input.correlation_factor)
            
            # 4. Ajuste por performance recente
            performance_adjustment = self._calculate_performance_adjustment()
            
            # 5. Ajuste por drawdown atual
            drawdown_adjustment = self._calculate_drawdown_adjustment()
            
            # 6. Verificar limites de portfolio
            portfolio_limits = self._check_portfolio_limits()
            
            # Combinar todos os ajustes
            adjusted_risk_pct = (base_risk_pct * 
                                volatility_adjustment * 
                                correlation_adjustment * 
                                performance_adjustment * 
                                drawdown_adjustment * 
                                portfolio_limits)
            
            # Aplicar limites absolutos
            final_risk_pct = min(adjusted_risk_pct, self.risk_limits.max_single_position)
            
            # Calcular position size em USD
            risk_amount = self.current_balance * final_risk_pct
            price_distance = abs(sizing_input.entry_price - sizing_input.stop_loss)
            position_size_usd = risk_amount / (price_distance / sizing_input.entry_price)
            
            # Position size como % do portfolio
            position_size_pct = position_size_usd / self.current_balance
            
            # Verificações finais
            if position_size_pct > self.risk_limits.max_single_position:
                position_size_pct = self.risk_limits.max_single_position
                position_size_usd = self.current_balance * position_size_pct
                risk_amount = position_size_usd * (price_distance / sizing_input.entry_price)
                final_risk_pct = risk_amount / self.current_balance
            
            self.logger.info(f"Position size calculado: {position_size_pct:.2%} "
                           f"(${position_size_usd:,.0f}) - Risk: {final_risk_pct:.2%}")
            
            return {
                'position_size_pct': position_size_pct,
                'position_size_usd': position_size_usd,
                'risk_amount_usd': risk_amount,
                'risk_pct': final_risk_pct,
                'base_risk': base_risk_pct,
                'volatility_adj': volatility_adjustment,
                'correlation_adj': correlation_adjustment,
                'performance_adj': performance_adjustment,
                'drawdown_adj': drawdown_adjustment,
                'portfolio_limits': portfolio_limits,
                'approved': risk_amount > 0 and position_size_pct > 0.001  # Min 0.1%
            }
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de position size: {e}")
            return {'approved': False, 'error': str(e)}
    
    def _calculate_base_risk(self, sizing_input: PositionSizingInput) -> float:
        """Calcula risco base baseado na qualidade do sinal"""
        try:
            # Base risk de 1-4% baseado na força do sinal
            signal_quality = (sizing_input.signal_strength + sizing_input.confidence) / 200
            
            # Ajuste pelo R:R (melhor R:R = mais risco)
            rr_multiplier = min(2.0, max(0.5, sizing_input.risk_reward_ratio / 2))
            
            # Base risk
            base_risk = 0.01 + (signal_quality * 0.03)  # 1-4%
            base_risk *= rr_multiplier
            
            return min(0.05, max(0.005, base_risk))  # Limitar entre 0.5-5%
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de base risk: {e}")
            return 0.02  # Default 2%
    
    def _calculate_volatility_adjustment(self, symbol: str, market_volatility: float) -> float:
        """Ajusta position size baseado na volatilidade"""
        try:
            # Volatilidade normal = 1.0x, alta vol = reduzir size
            normal_volatility = 0.02  # 2% volatilidade normal
            
            if market_volatility > normal_volatility:
                # Reduzir size em mercados voláteis
                vol_ratio = market_volatility / normal_volatility
                adjustment = 1 / (1 + (vol_ratio - 1) * 0.5)  # Redução gradual
            else:
                # Permitir size um pouco maior em baixa volatilidade
                vol_ratio = normal_volatility / market_volatility
                adjustment = min(1.3, 1 + (vol_ratio - 1) * 0.2)  # Aumento limitado
            
            self.volatility_cache[symbol] = market_volatility
            return max(0.3, min(1.5, adjustment))
            
        except Exception as e:
            self.logger.error(f"Erro no ajuste de volatilidade: {e}")
            return 1.0
    
    def _calculate_correlation_adjustment(self, symbol: str, correlation_factor: float) -> float:
        """Ajusta position size baseado na correlação com posições existentes"""
        try:
            if not self.active_positions:
                return 1.0  # Sem correlação se não há posições
            
            # Calcular correlação média com posições existentes
            total_correlation_exposure = 0
            
            for existing_symbol in self.active_positions.keys():
                if existing_symbol != symbol:
                    # Usar correlação estimada ou histórica
                    correlation = self.correlation_matrix.get((symbol, existing_symbol), 0)
                    existing_exposure = self.active_positions[existing_symbol].position_size_pct
                    total_correlation_exposure += abs(correlation) * existing_exposure
            
            # Reduzir size se alta correlação
            if total_correlation_exposure > self.risk_limits.correlation_threshold:
                adjustment = 1 - (total_correlation_exposure - self.risk_limits.correlation_threshold) * 2
                return max(0.2, adjustment)
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Erro no ajuste de correlação: {e}")
            return 1.0
    
    def _calculate_performance_adjustment(self) -> float:
        """Ajusta position size baseado na performance recente"""
        try:
            if len(self.trade_history) < 5:
                return 1.0  # Performance neutra sem histórico
            
            # Analisar últimos trades
            recent_trades = self.trade_history[-self.adaptive_config['performance_lookback']:]
            
            # Calcular métricas de performance
            wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            losses = len(recent_trades) - wins
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            
            # Calcular P&L recente
            recent_pnl = sum(trade.get('pnl_pct', 0) for trade in recent_trades[-10:])
            
            # Ajuste baseado na performance
            if win_rate > 0.6 and recent_pnl > 0.05:  # Boa performance
                adjustment = 1.2
            elif win_rate < 0.4 or recent_pnl < -0.05:  # Performance ruim
                adjustment = 0.7
            else:
                adjustment = 1.0
            
            # Ajuste adicional para streaks
            last_5_trades = recent_trades[-5:]
            consecutive_losses = 0
            for trade in reversed(last_5_trades):
                if trade.get('pnl', 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            
            if consecutive_losses >= 3:
                adjustment *= (0.8 ** (consecutive_losses - 2))  # Redução exponencial
            
            return max(0.3, min(1.5, adjustment))
            
        except Exception as e:
            self.logger.error(f"Erro no ajuste de performance: {e}")
            return 1.0
    
    def _calculate_drawdown_adjustment(self) -> float:
        """Ajusta position size baseado no drawdown atual"""
        try:
            current_drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
            
            if current_drawdown <= 0:  # Sem drawdown
                return 1.0
            
            # Reduzir size conforme drawdown aumenta
            if current_drawdown < 0.05:  # < 5% drawdown
                return 1.0
            elif current_drawdown < 0.10:  # 5-10% drawdown
                return 0.8
            elif current_drawdown < 0.15:  # 10-15% drawdown
                return 0.6
            else:  # > 15% drawdown
                return 0.4
                
        except Exception as e:
            self.logger.error(f"Erro no ajuste de drawdown: {e}")
            return 1.0
    
    def _check_portfolio_limits(self) -> float:
        """Verifica limites do portfolio"""
        try:
            # Verificar exposição total
            total_exposure = sum(pos.position_size_pct for pos in self.active_positions.values())
            
            if total_exposure >= self.risk_limits.max_portfolio_risk:
                return 0  # Não permitir novas posições
            
            # Verificar número de posições
            if len(self.active_positions) >= self.risk_limits.max_positions:
                return 0
            
            # Calcular espaço disponível
            available_exposure = self.risk_limits.max_portfolio_risk - total_exposure
            max_position_available = min(self.risk_limits.max_single_position, available_exposure)
            
            if max_position_available <= 0.005:  # Menos que 0.5%
                return 0
            
            return max_position_available / self.risk_limits.max_single_position
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de limites: {e}")
            return 0.5  # Conservative default
    
    def calculate_dynamic_stop_loss(self, 
                                  symbol: str,
                                  entry_price: float,
                                  initial_stop: float,
                                  current_price: float,
                                  market_data: pd.DataFrame) -> Dict:
        """
        Calcula stop loss dinâmico baseado em market structure
        
        Args:
            symbol: Símbolo da posição
            entry_price: Preço de entrada
            initial_stop: Stop loss inicial
            current_price: Preço atual
            market_data: Dados de mercado recentes
            
        Returns:
            Dict com novo stop loss e detalhes
        """
        try:
            direction = "long" if current_price > entry_price else "short"
            
            # 1. Stop loss baseado em ATR (volatilidade)
            atr_stop = self._calculate_atr_stop(market_data, current_price, direction)
            
            # 2. Stop loss baseado em structure (support/resistance)
            structure_stop = self._calculate_structure_stop(market_data, current_price, direction)
            
            # 3. Trailing stop baseado em swing highs/lows
            trailing_stop = self._calculate_trailing_stop(
                entry_price, current_price, initial_stop, direction, market_data)
            
            # 4. Time-based stop (se posição muito antiga)
            time_stop = self._calculate_time_stop(entry_price, current_price, direction)
            
            # Escolher o melhor stop (mais conservador)
            if direction == "long":
                candidate_stops = [s for s in [atr_stop, structure_stop, trailing_stop, time_stop] 
                                 if s and s < current_price]
                if candidate_stops:
                    new_stop = max(candidate_stops)  # Mais alto (menos risco)
                else:
                    new_stop = initial_stop
            else:  # short
                candidate_stops = [s for s in [atr_stop, structure_stop, trailing_stop, time_stop] 
                                 if s and s > current_price]
                if candidate_stops:
                    new_stop = min(candidate_stops)  # Mais baixo (menos risco)
                else:
                    new_stop = initial_stop
            
            # Nunca piorar o stop (só melhorar)
            if direction == "long":
                final_stop = max(initial_stop, new_stop) if new_stop else initial_stop
            else:
                final_stop = min(initial_stop, new_stop) if new_stop else initial_stop
            
            # Calcular novo risco
            new_risk_pct = abs(current_price - final_stop) / current_price
            
            return {
                'new_stop_loss': final_stop,
                'initial_stop': initial_stop,
                'stop_moved': abs(final_stop - initial_stop) > 0.001,
                'risk_reduction_pct': abs(initial_stop - final_stop) / current_price,
                'new_risk_pct': new_risk_pct,
                'atr_stop': atr_stop,
                'structure_stop': structure_stop,
                'trailing_stop': trailing_stop,
                'time_stop': time_stop
            }
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo de stop dinâmico: {e}")
            return {'new_stop_loss': initial_stop, 'error': str(e)}
    
    def _calculate_atr_stop(self, data: pd.DataFrame, current_price: float, direction: str) -> Optional[float]:
        """Calcula stop baseado em ATR"""
        try:
            if len(data) < 20:
                return None
            
            # Calcular ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Stop a 2x ATR
            atr_multiplier = 2.0
            
            if direction == "long":
                return current_price - (atr * atr_multiplier)
            else:
                return current_price + (atr * atr_multiplier)
                
        except Exception as e:
            self.logger.error(f"Erro no cálculo ATR stop: {e}")
            return None
    
    def _calculate_structure_stop(self, data: pd.DataFrame, current_price: float, direction: str) -> Optional[float]:
        """Calcula stop baseado em structure (swing highs/lows)"""
        try:
            if len(data) < 20:
                return None
            
            recent_data = data.tail(20)
            
            if direction == "long":
                # Encontrar swing low mais próximo
                swing_lows = []
                for i in range(5, len(recent_data) - 5):
                    if (recent_data.iloc[i]['low'] <= recent_data.iloc[i-5:i]['low'].min() and
                        recent_data.iloc[i]['low'] <= recent_data.iloc[i+1:i+6]['low'].min()):
                        swing_lows.append(recent_data.iloc[i]['low'])
                
                if swing_lows:
                    return max(swing_lows) * 0.995  # 0.5% abaixo do swing low
                    
            else:  # short
                # Encontrar swing high mais próximo
                swing_highs = []
                for i in range(5, len(recent_data) - 5):
                    if (recent_data.iloc[i]['high'] >= recent_data.iloc[i-5:i]['high'].max() and
                        recent_data.iloc[i]['high'] >= recent_data.iloc[i+1:i+6]['high'].max()):
                        swing_highs.append(recent_data.iloc[i]['high'])
                
                if swing_highs:
                    return min(swing_highs) * 1.005  # 0.5% acima do swing high
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo structure stop: {e}")
            return None
    
    def _calculate_trailing_stop(self, entry_price: float, current_price: float, 
                               initial_stop: float, direction: str, data: pd.DataFrame) -> Optional[float]:
        """Calcula trailing stop"""
        try:
            if direction == "long":
                # Se em lucro, ajustar stop para breakeven ou melhor
                if current_price > entry_price * 1.02:  # 2% lucro
                    return max(initial_stop, entry_price * 1.001)  # Breakeven + spread
                elif current_price > entry_price * 1.05:  # 5% lucro
                    return max(initial_stop, entry_price * 1.02)   # 2% lucro protegido
                    
            else:  # short
                if current_price < entry_price * 0.98:  # 2% lucro
                    return min(initial_stop, entry_price * 0.999)  # Breakeven - spread
                elif current_price < entry_price * 0.95:  # 5% lucro
                    return min(initial_stop, entry_price * 0.98)   # 2% lucro protegido
            
            return None
            
        except Exception as e:
            self.logger.error(f"Erro no trailing stop: {e}")
            return None
    
    def _calculate_time_stop(self, entry_price: float, current_price: float, direction: str) -> Optional[float]:
        """Stop baseado no tempo (se posição muito antiga sem progresso)"""
        # Simplificado - pode ser expandido com lógica temporal real
        return None
    
    def update_portfolio_metrics(self) -> RiskMetrics:
        """Atualiza e calcula métricas de risco do portfolio"""
        try:
            # Calcular exposição total
            total_exposure = sum(pos.position_size_pct for pos in self.active_positions.values())
            
            # Maior posição individual
            max_single_position = max(pos.position_size_pct for pos in self.active_positions.values()) if self.active_positions else 0
            
            # Risco de correlação (simplificado)
            correlation_risk = self._calculate_portfolio_correlation_risk()
            
            # Drawdown atual
            max_balance = max([h.get('balance', self.initial_balance) for h in self.portfolio_history] + [self.initial_balance])
            current_drawdown = (max_balance - self.current_balance) / max_balance
            
            # Calcular outras métricas
            var_1d = self._calculate_var()
            sharpe_ratio = self._calculate_sharpe_ratio()
            win_rate = self._calculate_win_rate()
            
            return RiskMetrics(
                total_exposure=total_exposure,
                max_single_position=max_single_position,
                correlation_risk=correlation_risk,
                volatility_adjusted_risk=total_exposure * 1.2,  # Simplified
                drawdown_risk=current_drawdown,
                var_1d=var_1d,
                expected_shortfall=var_1d * 1.3,  # Aproximação
                sharpe_ratio=sharpe_ratio,
                max_drawdown=current_drawdown,
                winning_rate=win_rate
            )
            
        except Exception as e:
            self.logger.error(f"Erro na atualização de métricas: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_portfolio_correlation_risk(self) -> float:
        """Calcula risco de correlação do portfolio"""
        if len(self.active_positions) < 2:
            return 0
        
        # Simplified correlation risk calculation
        symbols = list(self.active_positions.keys())
        total_correlation_risk = 0
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = self.correlation_matrix.get((symbol1, symbol2), 0.3)  # Default correlation
                pos1_weight = self.active_positions[symbol1].position_size_pct
                pos2_weight = self.active_positions[symbol2].position_size_pct
                
                total_correlation_risk += abs(correlation) * pos1_weight * pos2_weight
        
        return total_correlation_risk
    
    def _calculate_var(self) -> float:
        """Calcula Value at Risk (VaR) 1 dia"""
        if len(self.trade_history) < 30:
            return 0.02  # Default 2%
        
        # Usar returns dos últimos trades para estimar VaR
        returns = [trade.get('pnl_pct', 0) for trade in self.trade_history[-100:]]
        if returns:
            return abs(np.percentile(returns, 5))  # 5% worst case
        return 0.02
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calcula Sharpe ratio"""
        if len(self.trade_history) < 10:
            return 0
        
        returns = [trade.get('pnl_pct', 0) for trade in self.trade_history[-50:]]
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            return avg_return / std_return if std_return > 0 else 0
        return 0
    
    def _calculate_win_rate(self) -> float:
        """Calcula taxa de acerto"""
        if not self.trade_history:
            return 0.5
        
        recent_trades = self.trade_history[-50:]
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        return wins / len(recent_trades)
    
    def add_position(self, symbol: str, entry_price: float, stop_loss: float, position_size_usd: float):
        """Adiciona nova posição ao tracking"""
        position_size_pct = position_size_usd / self.current_balance
        risk_amount = position_size_usd * abs(entry_price - stop_loss) / entry_price
        
        self.active_positions[symbol] = PositionRisk(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            position_size_usd=position_size_usd,
            position_size_pct=position_size_pct,
            risk_amount_usd=risk_amount,
            risk_pct=risk_amount / self.current_balance,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            correlation_exposure={},
            volatility_risk=0,
            time_risk=0
        )
    
    def update_position(self, symbol: str, current_price: float):
        """Atualiza posição existente"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.entry_price) * (position.position_size_usd / position.entry_price)
            position.unrealized_pnl_pct = position.unrealized_pnl / position.position_size_usd
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "manual"):
        """Fecha posição e registra no histórico"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            
            # Calcular P&L final
            pnl = (exit_price - position.entry_price) * (position.position_size_usd / position.entry_price)
            pnl_pct = pnl / position.position_size_usd
            
            # Atualizar balance
            self.current_balance += pnl
            
            # Registrar trade
            trade_record = {
                'symbol': symbol,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'position_size_usd': position.position_size_usd,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': reason,
                'timestamp': pd.Timestamp.now()
            }
            self.trade_history.append(trade_record)
            
            # Remover posição ativa
            del self.active_positions[symbol]
            
            self.logger.info(f"Posição {symbol} fechada - P&L: ${pnl:,.2f} ({pnl_pct:.2%})")


def main():
    """Teste básico do risk manager"""
    # Inicializar risk manager
    risk_manager = RiskManager(initial_balance=100000)
    
    # Simular sizing input
    sizing_input = PositionSizingInput(
        signal_strength=80,
        confidence=75,
        risk_reward_ratio=3.0,
        entry_price=50000,
        stop_loss=48500,
        market_volatility=0.025,
        portfolio_balance=100000,
        existing_exposure=0.05,
        correlation_factor=0.3
    )
    
    # Calcular position size
    result = risk_manager.calculate_position_size(sizing_input, "BTCUSDT")
    
    print(f"\n⚡ RISK MANAGER TEST")
    print(f"Signal Strength: {sizing_input.signal_strength}")
    print(f"Confidence: {sizing_input.confidence}")
    print(f"Risk/Reward: {sizing_input.risk_reward_ratio}")
    print(f"Market Volatility: {sizing_input.market_volatility:.1%}")
    
    if result.get('approved'):
        print(f"\n✅ POSITION APPROVED")
        print(f"Position Size: {result['position_size_pct']:.2%}")
        print(f"Position Value: ${result['position_size_usd']:,.0f}")
        print(f"Risk Amount: ${result['risk_amount_usd']:,.0f}")
        print(f"Risk %: {result['risk_pct']:.2%}")
        print(f"Volatility Adj: {result['volatility_adj']:.2f}x")
        print(f"Performance Adj: {result['performance_adj']:.2f}x")
    else:
        print(f"\n❌ POSITION REJECTED")
        print(f"Error: {result.get('error', 'Unknown')}")
    
    # Simular dados para stop dinâmico
    dates = pd.date_range(start='2024-01-01', periods=50, freq='4H')
    np.random.seed(42)
    price_data = 50000 + np.random.randn(50).cumsum() * 100
    
    market_data = pd.DataFrame({
        'high': price_data + np.abs(np.random.randn(50) * 50),
        'low': price_data - np.abs(np.random.randn(50) * 50),
        'close': price_data,
        'volume': np.random.exponential(1000000, 50)
    }, index=dates)
    
    # Testar stop dinâmico
    stop_result = risk_manager.calculate_dynamic_stop_loss(
        "BTCUSDT", 49000, 47500, 50500, market_data)
    
    print(f"\n🛡️ DYNAMIC STOP LOSS")
    print(f"Entry: $49,000")
    print(f"Initial Stop: $47,500")
    print(f"Current Price: $50,500")
    print(f"New Stop: ${stop_result['new_stop_loss']:,.0f}")
    print(f"Stop Moved: {stop_result['stop_moved']}")
    print(f"Risk Reduction: {stop_result.get('risk_reduction_pct', 0):.2%}")


if __name__ == "__main__":
    main()