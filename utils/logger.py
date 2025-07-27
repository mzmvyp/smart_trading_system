# utils/logger.py
"""
📝 SISTEMA DE LOGGING AVANÇADO - SMART TRADING SYSTEM v2.0
Logging estruturado, colorido e otimizado para trading
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.traceback import install
    install()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from config.settings import settings

class ColoredFormatter(logging.Formatter):
    """🎨 Formatter com cores para diferentes níveis de log"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if COLORAMA_AVAILABLE:
            self.colors = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
            }
        else:
            self.colors = {}
    
    def format(self, record):
        if COLORAMA_AVAILABLE and record.levelname in self.colors:
            record.levelname = f"{self.colors[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        return super().format(record)

class JsonFormatter(logging.Formatter):
    """📄 Formatter JSON estruturado"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Adiciona campos extras se disponíveis
        if hasattr(record, 'symbol'):
            log_data['symbol'] = record.symbol
        if hasattr(record, 'timeframe'):
            log_data['timeframe'] = record.timeframe
        if hasattr(record, 'strategy'):
            log_data['strategy'] = record.strategy
        if hasattr(record, 'signal_id'):
            log_data['signal_id'] = record.signal_id
        if hasattr(record, 'trade_id'):
            log_data['trade_id'] = record.trade_id
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time
        
        # Adiciona traceback se for exceção
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

class TradingLoggerAdapter(logging.LoggerAdapter):
    """🎯 Adapter para adicionar contexto de trading aos logs"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """Adiciona contexto extra ao log"""
        # Merge do contexto extra
        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra.copy()
        
        return msg, kwargs
    
    def signal(self, msg, signal_id=None, symbol=None, strategy=None, **kwargs):
        """Log específico para sinais"""
        extra = {
            'signal_id': signal_id,
            'symbol': symbol,
            'strategy': strategy,
            'category': 'signal'
        }
        self.info(f"📈 {msg}", extra={**extra, **kwargs})
    
    def trade(self, msg, trade_id=None, symbol=None, action=None, **kwargs):
        """Log específico para trades"""
        extra = {
            'trade_id': trade_id,
            'symbol': symbol,
            'action': action,
            'category': 'trade'
        }
        self.info(f"💼 {msg}", extra={**extra, **kwargs})
    
    def performance(self, msg, metric=None, value=None, **kwargs):
        """Log específico para performance"""
        extra = {
            'metric': metric,
            'value': value,
            'category': 'performance'
        }
        self.info(f"📊 {msg}", extra={**extra, **kwargs})
    
    def timing(self, msg, execution_time=None, operation=None, **kwargs):
        """Log específico para timing/performance"""
        extra = {
            'execution_time': execution_time,
            'operation': operation,
            'category': 'timing'
        }
        
        if execution_time:
            if execution_time > 5.0:
                self.warning(f"⏱️ SLOW: {msg} ({execution_time:.2f}s)", extra={**extra, **kwargs})
            elif execution_time > 1.0:
                self.info(f"⏱️ {msg} ({execution_time:.2f}s)", extra={**extra, **kwargs})
            else:
                self.debug(f"⏱️ {msg} ({execution_time:.3f}s)", extra={**extra, **kwargs})
        else:
            self.info(f"⏱️ {msg}", extra={**extra, **kwargs})

def setup_logger(name: str = None, 
                level: str = None,
                use_json: bool = None,
                include_console: bool = True,
                include_file: bool = True) -> TradingLoggerAdapter:
    """
    🚀 Configura e retorna logger otimizado para trading
    
    Args:
        name: Nome do logger (default: nome do módulo)
        level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        use_json: Usar formato JSON (default: configuração)
        include_console: Incluir handler de console
        include_file: Incluir handler de arquivo
    
    Returns:
        TradingLoggerAdapter configurado
    """
    
    # Configurações padrão
    if name is None:
        name = __name__
    if level is None:
        level = settings.logging.CONSOLE_LEVEL
    if use_json is None:
        use_json = settings.logging.USE_JSON_FORMAT
    
    # Cria logger base
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove handlers existentes para evitar duplicatas
    logger.handlers.clear()
    
    # === CONSOLE HANDLER ===
    if include_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        if RICH_AVAILABLE and not use_json:
            # Rich handler (mais bonito)
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True
            )
            console_handler.setLevel(getattr(logging, level.upper()))
        else:
            # Handler padrão
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            
            if use_json:
                console_handler.setFormatter(JsonFormatter())
            else:
                formatter = ColoredFormatter(
                    '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    # === FILE HANDLERS ===
    if include_file:
        # Cria diretório de logs se não existir
        log_dir = Path(settings.logging.MAIN_LOG_FILE).parent
        log_dir.mkdir(exist_ok=True)
        
        # Handler principal (rotativo por tamanho)
        main_handler = RotatingFileHandler(
            settings.logging.MAIN_LOG_FILE,
            maxBytes=settings.logging.MAX_FILE_SIZE_MB * 1024 * 1024,
            backupCount=settings.logging.BACKUP_COUNT,
            encoding='utf-8'
        )
        main_handler.setLevel(getattr(logging, settings.logging.FILE_LEVEL.upper()))
        
        if use_json:
            main_handler.setFormatter(JsonFormatter())
        else:
            main_formatter = logging.Formatter(
                '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            main_handler.setFormatter(main_formatter)
        
        logger.addHandler(main_handler)
        
        # Handler de erros separado
        error_handler = RotatingFileHandler(
            settings.logging.ERROR_LOG_FILE,
            maxBytes=settings.logging.MAX_FILE_SIZE_MB * 1024 * 1024,
            backupCount=settings.logging.BACKUP_COUNT,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        error_formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s | %(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
        
        # Handler específico para trades (rotativo diário)
        trades_handler = TimedRotatingFileHandler(
            settings.logging.TRADES_LOG_FILE,
            when='midnight',
            interval=1,
            backupCount=30,  # 30 dias de histórico
            encoding='utf-8'
        )
        trades_handler.setLevel(logging.INFO)
        
        # Filtro para apenas logs de trade
        class TradeFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'category') and record.category == 'trade'
        
        trades_handler.addFilter(TradeFilter())
        
        if use_json:
            trades_handler.setFormatter(JsonFormatter())
        else:
            trades_formatter = logging.Formatter(
                '%(asctime)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            trades_handler.setFormatter(trades_formatter)
        
        logger.addHandler(trades_handler)
    
    # Cria adapter com contexto base
    adapter = TradingLoggerAdapter(logger, {
        'system': 'smart_trading_system',
        'version': '2.0.0',
        'mode': getattr(settings, 'MODE', 'unknown')
    })
    
    return adapter

def get_logger(name: str = None, **kwargs) -> TradingLoggerAdapter:
    """
    🎯 Função de conveniência para obter logger configurado
    """
    if name is None:
        # Obtém nome do módulo que está chamando
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return setup_logger(name, **kwargs)

def log_function_call(func_name: str = None, include_args: bool = False):
    """
    🎯 Decorator para log automático de chamadas de função
    
    Usage:
        @log_function_call()
        def my_function(arg1, arg2):
            return result
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            func_name_final = func_name or func.__name__
            
            start_time = time.time()
            
            if include_args:
                args_str = ', '.join([repr(arg) for arg in args[:3]])  # Máximo 3 args
                if len(args) > 3:
                    args_str += '...'
                logger.debug(f"🔧 Calling {func_name_final}({args_str})")
            else:
                logger.debug(f"🔧 Calling {func_name_final}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.timing(
                    f"Executed {func_name_final}",
                    execution_time=execution_time,
                    operation=func_name_final
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"❌ Error in {func_name_final}: {e}",
                    extra={
                        'execution_time': execution_time,
                        'operation': func_name_final,
                        'error_type': type(e).__name__
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator

def log_market_event(event_type: str, symbol: str = None, timeframe: str = None, **kwargs):
    """
    📈 Log específico para eventos de mercado
    
    Args:
        event_type: Tipo do evento (signal_generated, trade_executed, etc.)
        symbol: Símbolo relacionado
        timeframe: Timeframe relacionado
        **kwargs: Dados adicionais
    """
    logger = get_logger('market_events')
    
    extra = {
        'event_type': event_type,
        'symbol': symbol,
        'timeframe': timeframe,
        'category': 'market_event',
        **kwargs
    }
    
    emojis = {
        'signal_generated': '📈',
        'signal_validated': '✅',
        'signal_rejected': '❌',
        'trade_executed': '💼',
        'trade_closed': '🔒',
        'stop_loss_hit': '🛑',
        'take_profit_hit': '🎯',
        'data_fetched': '📊',
        'analysis_completed': '🔍',
        'system_started': '🚀',
        'system_stopped': '🛑',
        'error_occurred': '💥'
    }
    
    emoji = emojis.get(event_type, '📌')
    
    if event_type in ['signal_generated', 'trade_executed']:
        logger.info(f"{emoji} {event_type.upper()}: {symbol} {timeframe}", extra=extra)
    elif event_type in ['signal_rejected', 'error_occurred']:
        logger.warning(f"{emoji} {event_type.upper()}: {symbol} {timeframe}", extra=extra)
    else:
        logger.debug(f"{emoji} {event_type}: {symbol} {timeframe}", extra=extra)

class LogContext:
    """🎯 Context manager para adicionar contexto temporário aos logs"""
    
    def __init__(self, logger: TradingLoggerAdapter, **context):
        self.logger = logger
        self.context = context
        self.original_extra = logger.extra.copy()
    
    def __enter__(self):
        self.logger.extra.update(self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.extra = self.original_extra

# Configuração global do sistema
def configure_global_logging():
    """🌍 Configuração global do sistema de logging"""
    
    # Configura logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)  # Apenas warnings e erros de libs externas
    
    # Suprime logs verbose de bibliotecas externas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Logger principal do sistema
    main_logger = get_logger('smart_trading_system')
    main_logger.info("🚀 Sistema de logging configurado")
    main_logger.info(f"📁 Logs salvos em: {settings.logging.MAIN_LOG_FILE}")
    main_logger.info(f"🎨 Rich formatting: {'✅' if RICH_AVAILABLE else '❌'}")
    main_logger.info(f"🌈 Color support: {'✅' if COLORAMA_AVAILABLE else '❌'}")
    
    return main_logger

# Auto-configuração na importação
if not logging.getLogger().handlers:
    configure_global_logging()