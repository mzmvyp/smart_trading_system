"""
Core: Signal Manager
Gerenciamento completo do ciclo de vida dos sinais de trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
import uuid

from database.database import db_manager
from utils.logger import get_logger
from utils.helpers import (
    calculate_risk_reward_ratio,
    calculate_percentage_change,
    sanitize_numeric_input
)


logger = get_logger(__name__)


class SignalStatus(Enum):
    """Estados possíveis de um sinal"""
    PENDING = "pending"        # Aguardando condições
    ACTIVE = "active"          # Ativo e válido
    TRIGGERED = "triggered"    # Entrada executada
    CANCELLED = "cancelled"    # Cancelado
    EXPIRED = "expired"        # Expirou
    FILLED = "filled"          # Completamente executado


class SignalType(Enum):
    """Tipos de sinal"""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


class SignalPriority(Enum):
    """Prioridades de sinal"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SignalData:
    """Estrutura de dados do sinal"""
    # Identificação
    id: Optional[str] = None
    symbol: str = ""
    signal_type: SignalType = SignalType.BUY
    strategy: str = ""
    
    # Preços e timing
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    current_price: float = 0.0
    
    # Scores e métricas
    confluence_score: float = 0.0
    risk_score: float = 0.0
    timing_score: float = 0.0
    final_score: float = 0.0
    
    # Configurações
    position_size: float = 0.0
    risk_percentage: float = 2.0
    max_slippage: float = 0.1
    
    # Estado e timing
    status: SignalStatus = SignalStatus.PENDING
    priority: SignalPriority = SignalPriority.MEDIUM
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    triggered_at: Optional[datetime] = None
    
    # Metadata
    timeframe: str = "1h"
    conditions: Dict = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.conditions is None:
            self.conditions = {}
        if self.metadata is None:
            self.metadata = {}


class SignalValidator:
    """Validador de sinais"""
    
    @staticmethod
    def validate_signal(signal: SignalData) -> Tuple[bool, List[str]]:
        """Valida um sinal e retorna erros encontrados"""
        errors = []
        
        # Validações básicas
        if not signal.symbol:
            errors.append("Símbolo não pode estar vazio")
        
        if signal.entry_price <= 0:
            errors.append("Preço de entrada deve ser positivo")
        
        if signal.stop_loss <= 0:
            errors.append("Stop loss deve ser positivo")
        
        if signal.position_size <= 0:
            errors.append("Tamanho da posição deve ser positivo")
        
        # Validações de lógica
        if signal.signal_type in [SignalType.BUY]:
            if signal.stop_loss >= signal.entry_price:
                errors.append("Stop loss deve ser menor que preço de entrada para compra")
            
            if signal.take_profit > 0 and signal.take_profit <= signal.entry_price:
                errors.append("Take profit deve ser maior que preço de entrada para compra")
        
        elif signal.signal_type in [SignalType.SELL]:
            if signal.stop_loss <= signal.entry_price:
                errors.append("Stop loss deve ser maior que preço de entrada para venda")
            
            if signal.take_profit > 0 and signal.take_profit >= signal.entry_price:
                errors.append("Take profit deve ser menor que preço de entrada para venda")
        
        # Validações de risk/reward
        if signal.take_profit > 0:
            rr_ratio = calculate_risk_reward_ratio(
                signal.entry_price, 
                signal.stop_loss, 
                signal.take_profit
            )
            if rr_ratio < 1.0:
                errors.append(f"Relação risco/recompensa muito baixa: {rr_ratio:.2f}")
        
        # Validações de scores
        if signal.final_score < 50:
            errors.append(f"Score final muito baixo: {signal.final_score}")
        
        return len(errors) == 0, errors


class SignalManager:
    """Gerenciador principal de sinais"""
    
    def __init__(self):
        self._active_signals: Dict[str, SignalData] = {}
        self._signal_history: List[SignalData] = []
        self._lock = Lock()
        self._max_active_signals = 50
        self._max_history_size = 1000
        self._validator = SignalValidator()
        
        # Configurações
        self.default_expiry_hours = 24
        self.min_score_threshold = 60
        self.max_signals_per_symbol = 3
        
        # Carrega sinais ativos do banco
        self._load_active_signals()
    
    def _load_active_signals(self):
        """Carrega sinais ativos do banco de dados"""
        try:
            active_signals = db_manager.get_active_signals()
            
            for signal_data in active_signals:
                signal = self._dict_to_signal(signal_data)
                if signal:
                    self._active_signals[signal.id] = signal
            
            logger.info(f"Carregados {len(self._active_signals)} sinais ativos")
        except Exception as e:
            logger.error(f"Erro ao carregar sinais ativos: {e}")
    
    def _dict_to_signal(self, data: Dict) -> Optional[SignalData]:
        """Converte dicionário para SignalData"""
        try:
            # Converte strings para enums
            if isinstance(data.get('signal_type'), str):
                data['signal_type'] = SignalType(data['signal_type'])
            
            if isinstance(data.get('status'), str):
                data['status'] = SignalStatus(data['status'])
            
            if isinstance(data.get('priority'), int):
                data['priority'] = SignalPriority(data['priority'])
            
            # Converte strings para datetime
            for field in ['created_at', 'expires_at', 'triggered_at']:
                if data.get(field) and isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
            
            # Sanitiza valores numéricos
            numeric_fields = [
                'entry_price', 'stop_loss', 'take_profit', 'current_price',
                'confluence_score', 'risk_score', 'timing_score', 'final_score',
                'position_size', 'risk_percentage', 'max_slippage'
            ]
            
            for field in numeric_fields:
                if field in data:
                    data[field] = sanitize_numeric_input(data[field], 0.0)
            
            return SignalData(**data)
        except Exception as e:
            logger.error(f"Erro ao converter dados para SignalData: {e}")
            return None
    
    def create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        entry_price: float,
        stop_loss: float,
        take_profit: float = 0.0,
        strategy: str = "",
        scores: Dict[str, float] = None,
        **kwargs
    ) -> Optional[str]:
        """Cria um novo sinal"""
        
        with self._lock:
            try:
                # Verifica limite de sinais por símbolo
                symbol_signals = [
                    s for s in self._active_signals.values() 
                    if s.symbol == symbol and s.status == SignalStatus.ACTIVE
                ]
                
                if len(symbol_signals) >= self.max_signals_per_symbol:
                    logger.warning(f"Limite de sinais para {symbol} atingido")
                    return None
                
                # Cria estrutura do sinal
                signal_data = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'strategy': strategy,
                    'current_price': entry_price,
                    **kwargs
                }
                
                # Aplica scores se fornecidos
                if scores:
                    signal_data.update({
                        'confluence_score': scores.get('confluence', 0),
                        'risk_score': scores.get('risk', 0),
                        'timing_score': scores.get('timing', 0)
                    })
                    
                    # Calcula score final
                    signal_data['final_score'] = self._calculate_final_score(scores)
                
                # Cria sinal
                signal = SignalData(**signal_data)
                
                # Define expiração se não fornecida
                if not signal.expires_at:
                    signal.expires_at = datetime.now() + timedelta(hours=self.default_expiry_hours)
                
                # Valida sinal
                is_valid, errors = self._validator.validate_signal(signal)
                if not is_valid:
                    logger.warning(f"Sinal inválido: {errors}")
                    return None
                
                # Verifica score mínimo
                if signal.final_score < self.min_score_threshold:
                    logger.info(f"Sinal rejeitado por score baixo: {signal.final_score}")
                    return None
                
                # Ativa sinal
                signal.status = SignalStatus.ACTIVE
                
                # Salva no banco
                signal_dict = asdict(signal)
                signal_dict['signal_type'] = signal.signal_type.value
                signal_dict['status'] = signal.status.value
                signal_dict['priority'] = signal.priority.value
                
                db_manager.save_signal(signal_dict)
                
                # Adiciona aos sinais ativos
                self._active_signals[signal.id] = signal
                
                logger.info(f"Sinal criado: {signal.id} - {symbol} {signal_type.value}")
                return signal.id
                
            except Exception as e:
                logger.error(f"Erro ao criar sinal: {e}")
                return None
    
    def _calculate_final_score(self, scores: Dict[str, float]) -> float:
        """Calcula score final ponderado"""
        weights = {
            'confluence': 0.4,
            'risk': 0.3,
            'timing': 0.3
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for score_type, score_value in scores.items():
            if score_type in weights:
                weight = weights[score_type]
                total_score += score_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def update_signal(
        self,
        signal_id: str,
        current_price: float = None,
        status: SignalStatus = None,
        **updates
    ) -> bool:
        """Atualiza um sinal existente"""
        
        with self._lock:
            try:
                if signal_id not in self._active_signals:
                    logger.warning(f"Sinal não encontrado: {signal_id}")
                    return False
                
                signal = self._active_signals[signal_id]
                
                # Atualiza campos
                if current_price is not None:
                    signal.current_price = current_price
                
                if status is not None:
                    old_status = signal.status
                    signal.status = status
                    
                    # Registra timestamp de trigger
                    if status == SignalStatus.TRIGGERED and old_status != SignalStatus.TRIGGERED:
                        signal.triggered_at = datetime.now()
                
                # Aplica outras atualizações
                for key, value in updates.items():
                    if hasattr(signal, key):
                        setattr(signal, key, value)
                
                # Atualiza no banco
                update_dict = {'current_price': signal.current_price}
                if status:
                    update_dict['status'] = status.value
                if signal.triggered_at:
                    update_dict['triggered_at'] = signal.triggered_at
                
                db_manager.update_signal(signal_id, update_dict)
                
                # Remove de ativos se não for mais ativo
                if status in [SignalStatus.FILLED, SignalStatus.CANCELLED, SignalStatus.EXPIRED]:
                    self._move_to_history(signal_id)
                
                return True
                
            except Exception as e:
                logger.error(f"Erro ao atualizar sinal {signal_id}: {e}")
                return False
    
    def cancel_signal(self, signal_id: str, reason: str = "") -> bool:
        """Cancela um sinal"""
        return self.update_signal(
            signal_id,
            status=SignalStatus.CANCELLED,
            metadata={'cancel_reason': reason}
        )
    
    def trigger_signal(self, signal_id: str, execution_price: float) -> bool:
        """Marca sinal como triggered (entrada executada)"""
        return self.update_signal(
            signal_id,
            status=SignalStatus.TRIGGERED,
            current_price=execution_price,
            triggered_at=datetime.now()
        )
    
    def get_active_signals(self, symbol: str = None) -> List[SignalData]:
        """Obtém sinais ativos"""
        with self._lock:
            signals = list(self._active_signals.values())
            
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Ordena por prioridade e score
            signals.sort(key=lambda x: (x.priority.value, x.final_score), reverse=True)
            
            return signals
    
    def get_signal_by_id(self, signal_id: str) -> Optional[SignalData]:
        """Obtém sinal por ID"""
        return self._active_signals.get(signal_id)
    
    def get_signals_by_strategy(self, strategy: str) -> List[SignalData]:
        """Obtém sinais por estratégia"""
        with self._lock:
            return [
                s for s in self._active_signals.values()
                if s.strategy == strategy
            ]
    
    def get_signals_by_symbol(self, symbol: str) -> List[SignalData]:
        """Obtém sinais por símbolo"""
        with self._lock:
            return [
                s for s in self._active_signals.values()
                if s.symbol == symbol
            ]
    
    def check_signal_conditions(self, signal_id: str, market_data: Dict) -> bool:
        """Verifica se condições do sinal foram atendidas"""
        signal = self.get_signal_by_id(signal_id)
        if not signal:
            return False
        
        current_price = market_data.get('price', 0)
        
        # Atualiza preço atual
        self.update_signal(signal_id, current_price=current_price)
        
        # Verifica condições de entrada
        if signal.signal_type == SignalType.BUY:
            # Para compra, preço deve estar próximo ou abaixo da entrada
            price_diff = abs(current_price - signal.entry_price) / signal.entry_price
            return price_diff <= signal.max_slippage / 100
        
        elif signal.signal_type == SignalType.SELL:
            # Para venda, preço deve estar próximo ou acima da entrada
            price_diff = abs(current_price - signal.entry_price) / signal.entry_price
            return price_diff <= signal.max_slippage / 100
        
        return False
    
    def cleanup_expired_signals(self):
        """Remove sinais expirados"""
        with self._lock:
            now = datetime.now()
            expired_signals = []
            
            for signal_id, signal in self._active_signals.items():
                if signal.expires_at and now > signal.expires_at:
                    expired_signals.append(signal_id)
            
            for signal_id in expired_signals:
                self.update_signal(signal_id, status=SignalStatus.EXPIRED)
                logger.info(f"Sinal expirado: {signal_id}")
    
    def _move_to_history(self, signal_id: str):
        """Move sinal para histórico"""
        if signal_id in self._active_signals:
            signal = self._active_signals.pop(signal_id)
            
            # Adiciona ao histórico
            self._signal_history.append(signal)
            
            # Limita tamanho do histórico
            if len(self._signal_history) > self._max_history_size:
                self._signal_history = self._signal_history[-self._max_history_size:]
    
    def get_signal_statistics(self) -> Dict:
        """Obtém estatísticas dos sinais"""
        with self._lock:
            stats = {
                'active_signals': len(self._active_signals),
                'total_history': len(self._signal_history),
                'by_status': {},
                'by_symbol': {},
                'by_strategy': {},
                'average_score': 0.0
            }
            
            # Estatísticas por status
            for signal in self._active_signals.values():
                status = signal.status.value
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                
                # Por símbolo
                symbol = signal.symbol
                stats['by_symbol'][symbol] = stats['by_symbol'].get(symbol, 0) + 1
                
                # Por estratégia
                strategy = signal.strategy
                stats['by_strategy'][strategy] = stats['by_strategy'].get(strategy, 0) + 1
            
            # Score médio
            if self._active_signals:
                total_score = sum(s.final_score for s in self._active_signals.values())
                stats['average_score'] = total_score / len(self._active_signals)
            
            return stats
    
    def clear_all_signals(self):
        """Remove todos os sinais (usar com cuidado)"""
        with self._lock:
            for signal_id in list(self._active_signals.keys()):
                self.cancel_signal(signal_id, "Clear all signals")
            
            logger.warning("Todos os sinais foram cancelados")


# Instância global do gerenciador
signal_manager = SignalManager()