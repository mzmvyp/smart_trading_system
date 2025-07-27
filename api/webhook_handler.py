"""
API: Webhook Handler
Servidor de webhooks para receber alertas externos (TradingView, etc.)
"""
import json
import asyncio
import hashlib
import hmac
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import aiohttp
from aiohttp import web, ClientSession
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from sqlalchemy import Tuple

from core.signal_manager import signal_manager, SignalType, SignalPriority
from core.portfolio_manager import portfolio_manager
from utils.logger import get_logger
from utils.helpers import validate_symbol, sanitize_numeric_input


logger = get_logger(__name__)


class WebhookSource(Enum):
    """Fontes de webhook suportadas"""
    TRADINGVIEW = "tradingview"
    CUSTOM = "custom"
    TELEGRAM = "telegram"
    DISCORD = "discord"


@dataclass
class WebhookConfig:
    """Configurações do webhook"""
    enabled: bool = True
    secret_key: Optional[str] = None
    allowed_ips: List[str] = None
    max_requests_per_minute: int = 60
    require_authentication: bool = True
    log_all_requests: bool = True


class WebhookValidator:
    """Validador de webhooks"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key
    
    def validate_signature(self, payload: str, signature: str) -> bool:
        """Valida assinatura HMAC do webhook"""
        
        if not self.secret_key or not signature:
            return False
        
        try:
            # Remove prefixo se existir (ex: "sha256=")
            if "=" in signature:
                signature = signature.split("=", 1)[1]
            
            # Calcula hash esperado
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Compara com segurança
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Erro na validação de assinatura: {e}")
            return False
    
    def validate_tradingview_payload(self, payload: Dict) -> Tuple[bool, str]:
        """Valida payload específico do TradingView"""
        
        required_fields = ['symbol', 'action', 'price']
        
        for field in required_fields:
            if field not in payload:
                return False, f"Campo obrigatório ausente: {field}"
        
        # Valida símbolo
        try:
            validate_symbol(payload['symbol'])
        except ValueError as e:
            return False, str(e)
        
        # Valida ação
        valid_actions = ['buy', 'sell', 'close', 'cancel']
        if payload['action'].lower() not in valid_actions:
            return False, f"Ação inválida: {payload['action']}"
        
        # Valida preço
        try:
            price = sanitize_numeric_input(payload['price'])
            if price <= 0:
                return False, "Preço deve ser positivo"
        except:
            return False, "Preço inválido"
        
        return True, "OK"


class RateLimiter:
    """Rate limiter para webhooks"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # IP -> lista de timestamps
    
    def is_allowed(self, client_ip: str) -> bool:
        """Verifica se IP pode fazer request"""
        
        now = datetime.now().timestamp()
        
        # Inicializa lista para novo IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Remove requests antigas
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window_seconds
        ]
        
        # Verifica limite
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Adiciona request atual
        self.requests[client_ip].append(now)
        return True


class WebhookProcessor:
    """Processador de webhooks"""
    
    def __init__(self):
        self.handlers = {
            WebhookSource.TRADINGVIEW: self._process_tradingview_webhook,
            WebhookSource.CUSTOM: self._process_custom_webhook,
            WebhookSource.TELEGRAM: self._process_telegram_webhook
        }
    
    async def process_webhook(
        self,
        source: WebhookSource,
        payload: Dict,
        headers: Dict = None
    ) -> Dict[str, Any]:
        """Processa webhook baseado na fonte"""
        
        try:
            if source not in self.handlers:
                raise ValueError(f"Fonte não suportada: {source}")
            
            handler = self.handlers[source]
            result = await handler(payload, headers or {})
            
            logger.info(f"Webhook {source.value} processado com sucesso")
            return {'status': 'success', 'result': result}
            
        except Exception as e:
            logger.error(f"Erro ao processar webhook {source.value}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _process_tradingview_webhook(self, payload: Dict, headers: Dict) -> Dict:
        """Processa webhook do TradingView"""
        
        try:
            # Extrai dados do payload
            symbol = validate_symbol(payload['symbol'])
            action = payload['action'].lower()
            price = sanitize_numeric_input(payload['price'])
            
            # Dados opcionais
            stop_loss = sanitize_numeric_input(payload.get('stop_loss', 0))
            take_profit = sanitize_numeric_input(payload.get('take_profit', 0))
            quantity = sanitize_numeric_input(payload.get('quantity', 0))
            strategy = payload.get('strategy', 'tradingview')
            timeframe = payload.get('timeframe', '1h')
            
            # Processa ação
            if action in ['buy', 'sell']:
                # Cria sinal
                signal_type = SignalType.BUY if action == 'buy' else SignalType.SELL
                
                signal_id = signal_manager.create_signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    entry_price=price,
                    stop_loss=stop_loss if stop_loss > 0 else None,
                    take_profit=take_profit if take_profit > 0 else None,
                    strategy=strategy,
                    scores={
                        'confluence': 75,  # Score padrão para TradingView
                        'risk': 70,
                        'timing': 80
                    },
                    position_size=quantity,
                    timeframe=timeframe,
                    metadata={
                        'source': 'tradingview',
                        'original_payload': payload
                    }
                )
                
                return {
                    'action': 'signal_created',
                    'signal_id': signal_id,
                    'symbol': symbol,
                    'type': action
                }
            
            elif action == 'close':
                # Fecha posição
                if symbol in portfolio_manager.positions:
                    success = portfolio_manager.close_position(symbol, None, price)
                    
                    return {
                        'action': 'position_closed',
                        'symbol': symbol,
                        'success': success
                    }
                else:
                    return {
                        'action': 'no_position',
                        'symbol': symbol,
                        'message': 'Nenhuma posição encontrada'
                    }
            
            elif action == 'cancel':
                # Cancela sinais ativos
                active_signals = signal_manager.get_signals_by_symbol(symbol)
                canceled = 0
                
                for signal in active_signals:
                    signal_manager.cancel_signal(signal.id, "Cancelado via webhook")
                    canceled += 1
                
                return {
                    'action': 'signals_canceled',
                    'symbol': symbol,
                    'count': canceled
                }
            
            else:
                raise ValueError(f"Ação não reconhecida: {action}")
            
        except Exception as e:
            logger.error(f"Erro no webhook TradingView: {e}")
            raise
    
    async def _process_custom_webhook(self, payload: Dict, headers: Dict) -> Dict:
        """Processa webhook customizado"""
        
        # Implementação flexível para webhooks customizados
        webhook_type = payload.get('type', 'unknown')
        
        if webhook_type == 'signal':
            return await self._process_signal_webhook(payload)
        elif webhook_type == 'alert':
            return await self._process_alert_webhook(payload)
        elif webhook_type == 'status':
            return await self._process_status_webhook(payload)
        else:
            return {'message': f'Webhook customizado recebido: {webhook_type}'}
    
    async def _process_signal_webhook(self, payload: Dict) -> Dict:
        """Processa webhook de sinal customizado"""
        
        # Similar ao TradingView mas com mais flexibilidade
        signal_data = payload.get('signal', {})
        
        signal_id = signal_manager.create_signal(
            symbol=validate_symbol(signal_data['symbol']),
            signal_type=SignalType(signal_data['type']),
            entry_price=sanitize_numeric_input(signal_data['entry_price']),
            stop_loss=sanitize_numeric_input(signal_data.get('stop_loss')),
            take_profit=sanitize_numeric_input(signal_data.get('take_profit')),
            strategy=signal_data.get('strategy', 'custom'),
            scores=signal_data.get('scores', {'confluence': 70, 'risk': 70, 'timing': 70}),
            timeframe=signal_data.get('timeframe', '1h'),
            metadata={'source': 'custom_webhook', 'payload': payload}
        )
        
        return {'signal_id': signal_id}
    
    async def _process_alert_webhook(self, payload: Dict) -> Dict:
        """Processa webhook de alerta"""
        
        alert_data = payload.get('alert', {})
        
        logger.info(f"🚨 Alerta recebido: {alert_data.get('message', 'Sem mensagem')}")
        
        # Pode implementar notificações, logs especiais, etc.
        return {'alert_processed': True}
    
    async def _process_status_webhook(self, payload: Dict) -> Dict:
        """Processa webhook de status"""
        
        # Retorna status do sistema
        portfolio_summary = portfolio_manager.get_portfolio_summary()
        active_signals = signal_manager.get_active_signals()
        
        return {
            'system_status': 'active',
            'timestamp': datetime.now().isoformat(),
            'portfolio_equity': portfolio_summary['balance']['total_equity'],
            'active_signals': len(active_signals),
            'open_positions': len(portfolio_manager.positions)
        }
    
    async def _process_telegram_webhook(self, payload: Dict, headers: Dict) -> Dict:
        """Processa webhook do Telegram"""
        
        # Implementação básica para bot do Telegram
        message = payload.get('message', {})
        text = message.get('text', '').strip()
        
        if text.startswith('/status'):
            return await self._process_status_webhook({})
        elif text.startswith('/balance'):
            portfolio_summary = portfolio_manager.get_portfolio_summary()
            return {
                'response': f"Balance: ${portfolio_summary['balance']['total_equity']:.2f}"
            }
        else:
            return {'response': 'Comando não reconhecido'}


class WebhookServer:
    """Servidor principal de webhooks"""
    
    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 8080,
        config: WebhookConfig = None
    ):
        self.host = host
        self.port = port
        self.config = config or WebhookConfig()
        
        self.app = web.Application(middlewares=[self._logging_middleware])
        self.validator = WebhookValidator(self.config.secret_key)
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
        self.processor = WebhookProcessor()
        
        self._setup_routes()
        self._runner = None
        self._site = None
    
    def _setup_routes(self):
        """Configura rotas do servidor"""
        
        # Rotas principais
        self.app.router.add_post('/webhook/tradingview', self._handle_tradingview)
        self.app.router.add_post('/webhook/custom', self._handle_custom)
        self.app.router.add_post('/webhook/telegram', self._handle_telegram)
        
        # Rotas de status
        self.app.router.add_get('/health', self._handle_health)
        self.app.router.add_get('/status', self._handle_status)
        
        # Catch-all para webhooks genéricos
        self.app.router.add_post('/webhook', self._handle_generic)
    
    async def _logging_middleware(self, request: Request, handler: Callable) -> Response:
        """Middleware de logging"""
        
        start_time = datetime.now()
        client_ip = request.remote
        
        if self.config.log_all_requests:
            logger.info(f"📥 Webhook request: {request.method} {request.path} from {client_ip}")
        
        try:
            response = await handler(request)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"📤 Response: {response.status} in {elapsed:.3f}s")
            
            return response
            
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Error in {elapsed:.3f}s: {e}")
            raise
    
    async def _validate_request(self, request: Request) -> Tuple[bool, Optional[str]]:
        """Valida request de webhook"""
        
        client_ip = request.remote
        
        # Verifica IP permitido
        if self.config.allowed_ips and client_ip not in self.config.allowed_ips:
            return False, f"IP não autorizado: {client_ip}"
        
        # Verifica rate limiting
        if not self.rate_limiter.is_allowed(client_ip):
            return False, "Rate limit excedido"
        
        return True, None
    
    async def _handle_tradingview(self, request: Request) -> Response:
        """Handler para webhooks do TradingView"""
        
        try:
            # Validações básicas
            valid, error = await self._validate_request(request)
            if not valid:
                return web.json_response({'error': error}, status=403)
            
            # Lê payload
            body = await request.text()
            
            # Valida assinatura se configurada
            if self.config.require_authentication:
                signature = request.headers.get('X-Webhook-Signature', '')
                if not self.validator.validate_signature(body, signature):
                    return web.json_response({'error': 'Assinatura inválida'}, status=401)
            
            # Parse JSON
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return web.json_response({'error': 'JSON inválido'}, status=400)
            
            # Valida payload específico do TradingView
            valid, error = self.validator.validate_tradingview_payload(payload)
            if not valid:
                return web.json_response({'error': error}, status=400)
            
            # Processa webhook
            result = await self.processor.process_webhook(
                WebhookSource.TRADINGVIEW,
                payload,
                dict(request.headers)
            )
            
            if result['status'] == 'success':
                return web.json_response(result, status=200)
            else:
                return web.json_response(result, status=400)
            
        except Exception as e:
            logger.error(f"Erro no handler TradingView: {e}")
            return web.json_response({'error': 'Erro interno'}, status=500)
    
    async def _handle_custom(self, request: Request) -> Response:
        """Handler para webhooks customizados"""
        
        try:
            valid, error = await self._validate_request(request)
            if not valid:
                return web.json_response({'error': error}, status=403)
            
            body = await request.text()
            payload = json.loads(body)
            
            result = await self.processor.process_webhook(
                WebhookSource.CUSTOM,
                payload,
                dict(request.headers)
            )
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Erro no handler custom: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_telegram(self, request: Request) -> Response:
        """Handler para webhooks do Telegram"""
        
        try:
            valid, error = await self._validate_request(request)
            if not valid:
                return web.json_response({'error': error}, status=403)
            
            body = await request.text()
            payload = json.loads(body)
            
            result = await self.processor.process_webhook(
                WebhookSource.TELEGRAM,
                payload,
                dict(request.headers)
            )
            
            return web.json_response(result)
            
        except Exception as e:
            logger.error(f"Erro no handler Telegram: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_generic(self, request: Request) -> Response:
        """Handler genérico para webhooks"""
        
        try:
            valid, error = await self._validate_request(request)
            if not valid:
                return web.json_response({'error': error}, status=403)
            
            body = await request.text()
            
            logger.info(f"📨 Webhook genérico recebido: {body[:200]}...")
            
            return web.json_response({
                'status': 'received',
                'message': 'Webhook genérico processado',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erro no handler genérico: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_health(self, request: Request) -> Response:
        """Health check endpoint"""
        
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'server': 'webhook_handler',
            'version': '2.0'
        })
    
    async def _handle_status(self, request: Request) -> Response:
        """Status endpoint detalhado"""
        
        try:
            portfolio_summary = portfolio_manager.get_portfolio_summary()
            active_signals = signal_manager.get_active_signals()
            
            status = {
                'server_status': 'running',
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'active_signals': len(active_signals),
                    'open_positions': len(portfolio_manager.positions),
                    'total_equity': portfolio_summary['balance']['total_equity']
                },
                'webhook_config': {
                    'authentication_required': self.config.require_authentication,
                    'rate_limit': self.config.max_requests_per_minute
                }
            }
            
            return web.json_response(status)
            
        except Exception as e:
            logger.error(f"Erro no status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def start(self):
        """Inicia o servidor"""
        
        try:
            self._runner = web.AppRunner(self.app)
            await self._runner.setup()
            
            self._site = web.TCPSite(self._runner, self.host, self.port)
            await self._site.start()
            
            logger.info(f"🌐 Webhook server iniciado em http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Erro ao iniciar webhook server: {e}")
            raise
    
    async def stop(self):
        """Para o servidor"""
        
        try:
            if self._site:
                await self._site.stop()
            
            if self._runner:
                await self._runner.cleanup()
            
            logger.info("🛑 Webhook server parado")
            
        except Exception as e:
            logger.error(f"Erro ao parar webhook server: {e}")


# Função de conveniência para criar servidor
def create_webhook_server(
    host: str = '0.0.0.0',
    port: int = 8080,
    secret_key: str = None,
    allowed_ips: List[str] = None
) -> WebhookServer:
    """Cria servidor de webhook com configuração padrão"""
    
    config = WebhookConfig(
        secret_key=secret_key,
        allowed_ips=allowed_ips or [],
        require_authentication=secret_key is not None
    )
    
    return WebhookServer(host, port, config)


# Exemplo de uso
if __name__ == "__main__":
    import asyncio
    
    async def main():
        server = create_webhook_server(port=8080)
        await server.start()
        
        try:
            # Mantém servidor rodando
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await server.stop()
    
    asyncio.run(main())