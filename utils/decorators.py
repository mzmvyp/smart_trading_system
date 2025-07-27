# utils/decorators.py
"""
🎯 DECORADORES AVANÇADOS - SMART TRADING SYSTEM v2.0
Decoradores para rate limiting, cache, timing, retry e monitoramento
"""

import asyncio
import functools
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, Union, List
from collections import defaultdict, deque
import hashlib
import json

from utils.logger import get_logger

logger = get_logger(__name__)

# === RATE LIMITING ===

class RateLimiter:
    """🚦 Rate limiter thread-safe"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Verifica se chamada é permitida"""
        with self.lock:
            now = time.time()
            
            # Remove chamadas antigas
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            # Verifica limite
            if len(self.calls) >= self.max_calls:
                return False
            
            # Adiciona chamada atual
            self.calls.append(now)
            return True
    
    def wait_time(self) -> float:
        """Retorna tempo de espera necessário"""
        with self.lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            oldest_call = self.calls[0]
            return max(0.0, self.time_window - (time.time() - oldest_call))

def rate_limit(max_calls: int, time_window: int = 60, wait: bool = True):
    """
    🚦 Decorator para rate limiting
    
    Args:
        max_calls: Máximo de chamadas permitidas
        time_window: Janela de tempo em segundos
        wait: Se True, aguarda. Se False, levanta exceção
    """
    limiter = RateLimiter(max_calls, time_window)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            while not limiter.is_allowed():
                if not wait:
                    wait_time = limiter.wait_time()
                    raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f}s")
                
                wait_time = limiter.wait_time()
                if wait_time > 0:
                    logger.debug(f"⏳ Rate limit: waiting {wait_time:.1f}s for {func.__name__}")
                    await asyncio.sleep(wait_time)
            
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            while not limiter.is_allowed():
                if not wait:
                    wait_time = limiter.wait_time()
                    raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f}s")
                
                wait_time = limiter.wait_time()
                if wait_time > 0:
                    logger.debug(f"⏳ Rate limit: waiting {wait_time:.1f}s for {func.__name__}")
                    time.sleep(wait_time)
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === CACHING AVANÇADO ===

class TTLCache:
    """💾 Cache com Time-To-Live"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> tuple[bool, Any]:
        """Retorna (found, value)"""
        with self.lock:
            if key in self.cache:
                expiry_time, value = self.cache[key]
                if time.time() < expiry_time:
                    self.access_times[key] = time.time()
                    return True, value
                else:
                    # Expirado
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            return False, None
    
    def set(self, key: str, value: Any, ttl: int):
        """Define valor com TTL em segundos"""
        with self.lock:
            # Limpeza se necessário
            if len(self.cache) >= self.max_size:
                self._cleanup()
            
            expiry_time = time.time() + ttl
            self.cache[key] = (expiry_time, value)
            self.access_times[key] = time.time()
    
    def _cleanup(self):
        """Remove itens expirados e menos usados"""
        current_time = time.time()
        
        # Remove expirados
        expired_keys = []
        for key, (expiry_time, _) in self.cache.items():
            if current_time >= expiry_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        # Se ainda precisa de espaço, remove os menos acessados
        if len(self.cache) >= self.max_size:
            sorted_by_access = sorted(self.access_times.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_by_access[:len(sorted_by_access)//4]]
            
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                del self.access_times[key]

# Cache global
_global_cache = TTLCache()

def cache(ttl: int = 300, key_func: Optional[Callable] = None, use_global: bool = True):
    """
    💾 Decorator para cache com TTL
    
    Args:
        ttl: Time-to-live em segundos
        key_func: Função para gerar chave personalizada
        use_global: Usar cache global ou criar instância
    """
    cache_instance = _global_cache if use_global else TTLCache()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Gera chave do cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Verifica cache
            found, cached_result = cache_instance.get(cache_key)
            if found:
                logger.debug(f"💾 Cache hit: {func.__name__}")
                return cached_result
            
            # Executa função
            logger.debug(f"💨 Cache miss: {func.__name__}")
            result = await func(*args, **kwargs)
            
            # Salva no cache
            cache_instance.set(cache_key, result, ttl)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Gera chave do cache
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Verifica cache
            found, cached_result = cache_instance.get(cache_key)
            if found:
                logger.debug(f"💾 Cache hit: {func.__name__}")
                return cached_result
            
            # Executa função
            logger.debug(f"💨 Cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Salva no cache
            cache_instance.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === TIMING E PERFORMANCE ===

def timing(log_level: str = 'debug', include_args: bool = False, threshold_seconds: float = 1.0):
    """
    ⏱️ Decorator para medir e logar tempo de execução
    
    Args:
        log_level: Nível de log (debug, info, warning, error)
        include_args: Incluir argumentos no log
        threshold_seconds: Log apenas se execução > threshold
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time >= threshold_seconds:
                    args_str = ""
                    if include_args and args:
                        args_preview = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
                        args_str = f" | Args: {args_preview}"
                    
                    log_message = f"⏱️ {function_name} | {execution_time:.3f}s{args_str}"
                    
                    if execution_time > 10.0:
                        logger.error(f"🐌 MUITO LENTO: {log_message}")
                    elif execution_time > 5.0:
                        logger.warning(f"🐢 LENTO: {log_message}")
                    else:
                        getattr(logger, log_level)(log_message)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {function_name} falhou após {execution_time:.3f}s: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if execution_time >= threshold_seconds:
                    args_str = ""
                    if include_args and args:
                        args_preview = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
                        args_str = f" | Args: {args_preview}"
                    
                    log_message = f"⏱️ {function_name} | {execution_time:.3f}s{args_str}"
                    
                    if execution_time > 10.0:
                        logger.error(f"🐌 MUITO LENTO: {log_message}")
                    elif execution_time > 5.0:
                        logger.warning(f"🐢 LENTO: {log_message}")
                    else:
                        getattr(logger, log_level)(log_message)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"❌ {function_name} falhou após {execution_time:.3f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === RETRY AVANÇADO ===

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
         max_delay: float = 60.0, exceptions: tuple = (Exception,),
         on_retry: Optional[Callable] = None):
    """
    🔄 Decorator para retry com backoff exponencial
    
    Args:
        max_attempts: Máximo de tentativas
        delay: Delay inicial em segundos
        backoff: Multiplicador para backoff exponencial
        max_delay: Delay máximo
        exceptions: Tupla de exceções para retry
        on_retry: Callback chamado a cada retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"❌ {func.__name__} falhou definitivamente após {max_attempts} tentativas")
                        raise
                    
                    logger.warning(f"⚠️ {func.__name__} falhou (tentativa {attempt + 1}/{max_attempts}): {e}")
                    
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, *args, **kwargs)
                        except Exception as retry_error:
                            logger.error(f"Erro no callback on_retry: {retry_error}")
                    
                    await asyncio.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)
                
                except Exception as e:
                    # Exceção não listada para retry
                    logger.error(f"❌ {func.__name__} falhou com exceção não recuperável: {e}")
                    raise
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"❌ {func.__name__} falhou definitivamente após {max_attempts} tentativas")
                        raise
                    
                    logger.warning(f"⚠️ {func.__name__} falhou (tentativa {attempt + 1}/{max_attempts}): {e}")
                    
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, *args, **kwargs)
                        except Exception as retry_error:
                            logger.error(f"Erro no callback on_retry: {retry_error}")
                    
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff, max_delay)
                
                except Exception as e:
                    # Exceção não listada para retry
                    logger.error(f"❌ {func.__name__} falhou com exceção não recuperável: {e}")
                    raise
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === VALIDATION ===

def validate_input(**validators):
    """
    ✅ Decorator para validação de inputs
    
    Args:
        **validators: Dict com nome do parâmetro e função validadora
    
    Example:
        @validate_input(price=lambda x: x > 0, symbol=lambda x: len(x) > 0)
        def place_order(symbol, price):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Obtém nomes dos parâmetros
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Valida cada parâmetro
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    try:
                        if not validator(value):
                            raise ValueError(f"Validação falhou para {param_name}: {value}")
                    except Exception as e:
                        raise ValueError(f"Erro na validação de {param_name}: {e}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# === CIRCUIT BREAKER ===

class CircuitBreaker:
    """🔌 Circuit breaker para proteção contra falhas consecutivas"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Executa função com circuit breaker"""
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info(f"🔌 Circuit breaker para {func.__name__}: HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker OPEN para {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                # Sucesso - reset circuit breaker
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info(f"🔌 Circuit breaker para {func.__name__}: CLOSED (recuperado)")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.error(f"🔌 Circuit breaker para {func.__name__}: OPEN ({self.failure_count} falhas)")
                
                raise

def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60):
    """
    🔌 Decorator para circuit breaker
    
    Args:
        failure_threshold: Número de falhas para abrir circuito
        recovery_timeout: Tempo em segundos para tentar recuperação
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === MONITORING ===

class FunctionMonitor:
    """📊 Monitor de funções para métricas"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'calls': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'errors': 0,
            'last_call': None
        })
        self.lock = threading.Lock()
    
    def record_call(self, func_name: str, execution_time: float, success: bool):
        """Registra chamada de função"""
        with self.lock:
            stats = self.stats[func_name]
            stats['calls'] += 1
            stats['last_call'] = datetime.now()
            
            if success:
                stats['total_time'] += execution_time
                stats['avg_time'] = stats['total_time'] / stats['calls']
                stats['min_time'] = min(stats['min_time'], execution_time)
                stats['max_time'] = max(stats['max_time'], execution_time)
            else:
                stats['errors'] += 1
    
    def get_stats(self, func_name: str = None) -> Dict:
        """Retorna estatísticas"""
        with self.lock:
            if func_name:
                return dict(self.stats.get(func_name, {}))
            else:
                return {name: dict(stats) for name, stats in self.stats.items()}

# Monitor global
_global_monitor = FunctionMonitor()

def monitor(use_global: bool = True):
    """
    📊 Decorator para monitoramento de funções
    
    Args:
        use_global: Usar monitor global
    """
    monitor_instance = _global_monitor if use_global else FunctionMonitor()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                monitor_instance.record_call(func.__name__, execution_time, success)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                monitor_instance.record_call(func.__name__, execution_time, success)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === TRADING ESPECÍFICOS ===

def trading_session_only(sessions: List[str] = None):
    """
    🕒 Decorator que só executa em sessões específicas de trading
    
    Args:
        sessions: Lista de sessões permitidas ('asian', 'london', 'new_york', 'overlap')
    """
    if sessions is None:
        sessions = ['london', 'new_york', 'overlap']
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from utils.helpers import get_trading_session
            
            current_session = get_trading_session()
            
            if current_session not in sessions:
                logger.warning(f"🕒 {func.__name__} chamado fora das sessões permitidas. "
                             f"Atual: {current_session}, Permitidas: {sessions}")
                return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def market_data_required(symbols: List[str] = None, timeframes: List[str] = None):
    """
    📊 Decorator que verifica disponibilidade de dados de mercado
    
    Args:
        symbols: Símbolos obrigatórios
        timeframes: Timeframes obrigatórios
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # TODO: Implementar verificação de dados de mercado
            # Por enquanto, apenas log
            logger.debug(f"📊 Verificando dados para {func.__name__}: {symbols}, {timeframes}")
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"📊 Verificando dados para {func.__name__}: {symbols}, {timeframes}")
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# === FUNÇÕES UTILITÁRIAS ===

def get_monitor_stats(func_name: str = None) -> Dict:
    """Retorna estatísticas do monitor global"""
    return _global_monitor.get_stats(func_name)

def clear_cache():
    """Limpa cache global"""
    _global_cache.cache.clear()
    _global_cache.access_times.clear()
    logger.info("🧹 Cache global limpo")

def reset_monitor():
    """Reseta estatísticas do monitor global"""
    _global_monitor.stats.clear()
    logger.info("📊 Monitor global resetado")