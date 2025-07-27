"""
SMART TRADING SYSTEM v2.0
Aplicação principal do sistema de trading inteligente
"""
import asyncio
import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import signal
import json

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports do sistema
from config.settings import TradingConfig, load_config
from config.symbols import get_enabled_symbols
from config.timeframes import timeframe_manager
from core.signal_generator import SignalGenerator
from core.signal_manager import signal_manager
from core.portfolio_manager import portfolio_manager
from core.risk_manager import RiskManager
from api.data_provider import data_provider
from api.webhook_handler import WebhookServer
from database.database import db_manager
from utils.logger import get_logger, setup_logging
from web.dashboard import create_dashboard


logger = get_logger(__name__)


class TradingSystemApp:
    """Aplicação principal do sistema de trading"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = load_config(config_path)
        self.running = False
        self.tasks = []
        
        # Componentes principais
        self.signal_generator = None
        self.risk_manager = None
        self.webhook_server = None
        
        # Estado
        self.last_update = None
        self.update_interval = 60  # segundos
        
        # Setup do sistema
        setup_logging(self.config.logging)
        
    async def initialize(self):
        """Inicializa todos os componentes do sistema"""
        
        logger.info("🚀 Inicializando Smart Trading System v2.0...")
        
        try:
            # 1. Inicializa data provider
            await data_provider.initialize_all()
            logger.info("✅ Data providers inicializados")
            
            # 2. Inicializa componentes principais
            self.signal_generator = SignalGenerator(self.config)
            self.risk_manager = RiskManager(self.config.risk_management)
            
            # 3. Carrega símbolos ativos
            symbols = get_enabled_symbols()
            logger.info(f"✅ Carregados {len(symbols)} símbolos para trading")
            
            # 4. Inicializa webhook server se habilitado
            if self.config.api.enable_webhooks:
                self.webhook_server = WebhookServer(
                    host=self.config.api.webhook_host,
                    port=self.config.api.webhook_port
                )
                await self.webhook_server.start()
                logger.info(f"✅ Webhook server iniciado em {self.config.api.webhook_host}:{self.config.api.webhook_port}")
            
            # 5. Verifica conexões
            await self._health_check()
            
            logger.info("🎯 Sistema inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            raise
    
    async def _health_check(self):
        """Verifica saúde dos componentes"""
        
        try:
            # Testa conexão com dados
            test_symbol = 'BTCUSDT'
            test_data = await data_provider.get_current_prices([test_symbol])
            
            if not test_data or test_symbol not in test_data:
                raise Exception("Falha ao obter dados de teste")
            
            # Testa banco de dados
            db_stats = db_manager.get_database_stats()
            logger.info(f"📊 Database: {db_stats}")
            
            # Testa portfolio
            portfolio_summary = portfolio_manager.get_portfolio_summary()
            logger.info(f"💰 Portfolio: ${portfolio_summary['balance']['total_equity']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Health check falhou: {e}")
            raise
    
    async def run(self):
        """Executa o loop principal do sistema"""
        
        logger.info("🔄 Iniciando loop principal...")
        self.running = True
        
        try:
            # Cria tasks principais
            self.tasks = [
                asyncio.create_task(self._market_analysis_loop()),
                asyncio.create_task(self._signal_processing_loop()),
                asyncio.create_task(self._risk_monitoring_loop()),
                asyncio.create_task(self._portfolio_update_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            # Aguarda todas as tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"❌ Erro no loop principal: {e}")
            await self.shutdown()
    
    async def _market_analysis_loop(self):
        """Loop de análise de mercado e geração de sinais"""
        
        while self.running:
            try:
                start_time = datetime.now()
                
                # Obtém símbolos habilitados
                symbols = get_enabled_symbols()
                timeframes = timeframe_manager.get_analysis_timeframes()
                
                logger.info(f"📈 Analisando {len(symbols)} símbolos em {len(timeframes)} timeframes...")
                
                # Processa cada símbolo
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    try:
                        await self._analyze_symbol(symbol, timeframes)
                    except Exception as e:
                        logger.error(f"Erro ao analisar {symbol}: {e}")
                        continue
                
                # Log de performance
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"⏱️ Análise concluída em {elapsed:.2f}s")
                
                # Aguarda próximo ciclo
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop de análise: {e}")
                await asyncio.sleep(30)  # Pausa em caso de erro
    
    async def _analyze_symbol(self, symbol: str, timeframes: List[str]):
        """Analisa um símbolo específico"""
        
        try:
            # Obtém dados de mercado para todos os timeframes
            market_data = {}
            
            for timeframe in timeframes:
                try:
                    df = await data_provider.get_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=500
                    )
                    
                    if not df.empty:
                        market_data[timeframe] = df
                    
                except Exception as e:
                    logger.warning(f"Erro ao obter dados {symbol} {timeframe}: {e}")
            
            if not market_data:
                return
            
            # Gera sinais usando o signal generator
            signals = await self.signal_generator.generate_signals(
                symbol=symbol,
                market_data=market_data
            )
            
            # Processa sinais gerados
            for signal_data in signals:
                await self._process_new_signal(signal_data)
                
        except Exception as e:
            logger.error(f"Erro ao analisar símbolo {symbol}: {e}")
    
    async def _process_new_signal(self, signal_data: Dict):
        """Processa um novo sinal gerado"""
        
        try:
            # Valida sinal com risk manager
            risk_analysis = self.risk_manager.analyze_signal_risk(signal_data)
            
            if not risk_analysis['approved']:
                logger.info(f"🚫 Sinal rejeitado: {risk_analysis['reason']}")
                return
            
            # Cria sinal no signal manager
            signal_id = signal_manager.create_signal(
                symbol=signal_data['symbol'],
                signal_type=signal_data['signal_type'],
                entry_price=signal_data['entry_price'],
                stop_loss=signal_data['stop_loss'],
                take_profit=signal_data['take_profit'],
                strategy=signal_data['strategy'],
                scores={
                    'confluence': signal_data['scores']['confluence'],
                    'risk': signal_data['scores']['risk'],
                    'timing': signal_data['scores']['timing']
                },
                position_size=risk_analysis['position_size'],
                timeframe=signal_data['timeframe'],
                metadata=signal_data.get('metadata', {})
            )
            
            if signal_id:
                logger.info(f"✅ Novo sinal criado: {signal_id} - {signal_data['symbol']} {signal_data['signal_type'].value}")
            
        except Exception as e:
            logger.error(f"Erro ao processar sinal: {e}")
    
    async def _signal_processing_loop(self):
        """Loop de processamento de sinais ativos"""
        
        while self.running:
            try:
                # Obtém sinais ativos
                active_signals = signal_manager.get_active_signals()
                
                if active_signals:
                    logger.debug(f"🎯 Processando {len(active_signals)} sinais ativos")
                    
                    # Obtém preços atuais
                    symbols = list(set(signal.symbol for signal in active_signals))
                    current_prices = await data_provider.get_current_prices(symbols)
                    
                    # Atualiza cada sinal
                    for signal in active_signals:
                        try:
                            current_price = current_prices.get(signal.symbol, 0)
                            if current_price > 0:
                                signal_manager.update_signal(
                                    signal.id,
                                    current_price=current_price
                                )
                                
                                # Verifica condições de entrada
                                market_data = {
                                    'price': current_price,
                                    'timestamp': datetime.now()
                                }
                                
                                if signal_manager.check_signal_conditions(signal.id, market_data):
                                    await self._execute_signal(signal, current_price)
                        
                        except Exception as e:
                            logger.error(f"Erro ao processar sinal {signal.id}: {e}")
                
                # Limpa sinais expirados
                signal_manager.cleanup_expired_signals()
                
                await asyncio.sleep(10)  # Verifica a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro no loop de sinais: {e}")
                await asyncio.sleep(30)
    
    async def _execute_signal(self, signal, current_price: float):
        """Executa um sinal (simulação)"""
        
        try:
            # Em ambiente de produção, aqui seria feita a ordem real na exchange
            # Por enquanto, simula a execução
            
            success = portfolio_manager.add_position(
                symbol=signal.symbol,
                side='long' if signal.signal_type.value == 'buy' else 'short',
                size=signal.position_size,
                entry_price=current_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )
            
            if success:
                # Marca sinal como triggered
                signal_manager.trigger_signal(signal.id, current_price)
                
                logger.info(f"🎯 Sinal executado: {signal.symbol} {signal.signal_type.value} "
                           f"@ ${current_price:.4f}")
            else:
                logger.warning(f"❌ Falha ao executar sinal: {signal.id}")
                
        except Exception as e:
            logger.error(f"Erro ao executar sinal {signal.id}: {e}")
    
    async def _risk_monitoring_loop(self):
        """Loop de monitoramento de risco"""
        
        while self.running:
            try:
                # Monitora exposição do portfolio
                risk_metrics = self.risk_manager.calculate_portfolio_risk()
                
                if risk_metrics['total_risk'] > self.config.risk_management.max_portfolio_risk:
                    logger.warning(f"⚠️ Risco do portfolio elevado: {risk_metrics['total_risk']:.2f}%")
                    
                    # Pode implementar ações automáticas de redução de risco
                    await self._reduce_portfolio_risk()
                
                # Monitora drawdown
                portfolio_summary = portfolio_manager.get_portfolio_summary()
                current_dd = portfolio_summary.get('performance', {}).get('max_drawdown', 0)
                
                if current_dd > self.config.risk_management.max_drawdown:
                    logger.warning(f"⚠️ Drawdown crítico: {current_dd:.2f}%")
                    # Implementar ações de emergência
                
                await asyncio.sleep(30)  # Monitora a cada 30 segundos
                
            except Exception as e:
                logger.error(f"Erro no monitoramento de risco: {e}")
                await asyncio.sleep(60)
    
    async def _reduce_portfolio_risk(self):
        """Reduz risco do portfolio automaticamente"""
        
        try:
            # Implementa lógica de redução de risco
            # Ex: fechar posições com maior risco, reduzir tamanhos, etc.
            
            logger.info("🛡️ Implementando medidas de redução de risco...")
            
            # Por enquanto, apenas cancela sinais pendentes de baixa prioridade
            active_signals = signal_manager.get_active_signals()
            
            for signal in active_signals:
                if signal.priority.value <= 2:  # Low/Medium priority
                    signal_manager.cancel_signal(signal.id, "Risk reduction")
                    logger.info(f"Sinal cancelado por redução de risco: {signal.id}")
            
        except Exception as e:
            logger.error(f"Erro ao reduzir risco: {e}")
    
    async def _portfolio_update_loop(self):
        """Loop de atualização do portfolio"""
        
        while self.running:
            try:
                # Obtém preços atuais de todas as posições
                positions = portfolio_manager.positions
                
                if positions:
                    symbols = list(positions.keys())
                    current_prices = await data_provider.get_current_prices(symbols)
                    
                    # Atualiza preços no portfolio
                    portfolio_manager.update_prices(current_prices)
                    
                    # Verifica stops e targets
                    for symbol, position in positions.items():
                        current_price = current_prices.get(symbol, 0)
                        if current_price > 0:
                            await self._check_position_exit(position, current_price)
                
                # Salva snapshot do portfolio
                portfolio_manager.save_snapshot()
                
                await asyncio.sleep(15)  # Atualiza a cada 15 segundos
                
            except Exception as e:
                logger.error(f"Erro na atualização do portfolio: {e}")
                await asyncio.sleep(30)
    
    async def _check_position_exit(self, position, current_price: float):
        """Verifica se posição deve ser fechada"""
        
        try:
            should_close = False
            close_reason = ""
            
            # Verifica stop loss
            if position.stop_loss:
                if position.side == 'long' and current_price <= position.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
                elif position.side == 'short' and current_price >= position.stop_loss:
                    should_close = True
                    close_reason = "stop_loss"
            
            # Verifica take profit
            if not should_close and position.take_profit:
                if position.side == 'long' and current_price >= position.take_profit:
                    should_close = True
                    close_reason = "take_profit"
                elif position.side == 'short' and current_price <= position.take_profit:
                    should_close = True
                    close_reason = "take_profit"
            
            if should_close:
                success = portfolio_manager.close_position(
                    position.symbol,
                    position.size,
                    current_price
                )
                
                if success:
                    logger.info(f"🔴 Posição fechada: {position.symbol} @ ${current_price:.4f} ({close_reason})")
                
        except Exception as e:
            logger.error(f"Erro ao verificar saída da posição {position.symbol}: {e}")
    
    async def _cleanup_loop(self):
        """Loop de limpeza e manutenção"""
        
        while self.running:
            try:
                # Limpa cache do data provider
                data_provider.cache.clear()
                
                # Limpa cache do database
                db_manager.clear_cache()
                
                # Log de estatísticas
                stats = {
                    'active_signals': len(signal_manager.get_active_signals()),
                    'open_positions': len(portfolio_manager.positions),
                    'db_cache_size': len(db_manager._cache),
                    'total_equity': portfolio_manager.get_total_equity()
                }
                
                logger.info(f"📊 Stats: {stats}")
                
                await asyncio.sleep(300)  # Executa a cada 5 minutos
                
            except Exception as e:
                logger.error(f"Erro na limpeza: {e}")
                await asyncio.sleep(300)
    
    async def shutdown(self):
        """Para o sistema graciosamente"""
        
        logger.info("🛑 Iniciando shutdown do sistema...")
        
        self.running = False
        
        try:
            # Cancela todas as tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Para webhook server
            if self.webhook_server:
                await self.webhook_server.stop()
            
            # Fecha conexões do banco
            db_manager.close()
            
            logger.info("✅ Shutdown concluído")
            
        except Exception as e:
            logger.error(f"Erro no shutdown: {e}")


async def run_backtest_mode(args):
    """Executa modo de backtesting"""
    
    from backtesting.backtest_engine import BacktestEngine, BacktestConfig
    from backtesting.reports import generate_backtest_report
    
    logger.info("📊 Iniciando modo backtesting...")
    
    try:
        # Configura backtesting
        config = BacktestConfig(
            initial_capital=args.capital,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Obtém dados históricos
        symbols = args.symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        market_data = {}
        for symbol in symbols:
            df = await data_provider.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=2000
            )
            if not df.empty:
                market_data[symbol] = df
        
        if not market_data:
            logger.error("❌ Nenhum dado de mercado obtido")
            return
        
        # Executa backtesting
        engine = BacktestEngine(config)
        result = await engine.run_backtest(
            market_data=market_data,
            symbols=symbols,
            timeframes=['1h', '4h'],
            strategies=args.strategies or ['swing', 'breakout']
        )
        
        # Gera relatório
        report_path = generate_backtest_report(result)
        logger.info(f"📈 Relatório gerado: {report_path}")
        
        # Mostra resumo
        logger.info(f"🎯 Resultado: {result.total_return:.2f}% | "
                   f"Sharpe: {result.sharpe_ratio:.2f} | "
                   f"Trades: {result.total_trades}")
        
    except Exception as e:
        logger.error(f"❌ Erro no backtesting: {e}")


async def run_dashboard_mode(args):
    """Executa modo dashboard web"""
    
    logger.info("🌐 Iniciando dashboard web...")
    
    try:
        dashboard = create_dashboard()
        dashboard.run(
            host=args.host or '0.0.0.0',
            port=args.port or 8501,
            debug=args.debug
        )
    except Exception as e:
        logger.error(f"❌ Erro no dashboard: {e}")


def setup_signal_handlers(app: TradingSystemApp):
    """Configura handlers para sinais do sistema"""
    
    def signal_handler(signum, frame):
        logger.info(f"📡 Sinal recebido: {signum}")
        asyncio.create_task(app.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments():
    """Parse dos argumentos da linha de comando"""
    
    parser = argparse.ArgumentParser(
        description='Smart Trading System v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py                              # Modo trading normal
  python main.py --mode backtest --capital 10000  # Backtesting
  python main.py --mode dashboard --port 8080     # Dashboard web
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['trading', 'backtest', 'dashboard'],
        default='trading',
        help='Modo de operação'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.json',
        help='Arquivo de configuração'
    )
    
    # Argumentos para backtesting
    parser.add_argument(
        '--capital',
        type=float,
        default=10000,
        help='Capital inicial para backtesting'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='*',
        help='Símbolos para análise'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='*',
        help='Estratégias para usar'
    )
    
    # Argumentos para dashboard
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host para dashboard'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Porta para dashboard'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Modo debug'
    )
    
    return parser.parse_args()


async def main():
    """Função principal"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║               🚀 SMART TRADING SYSTEM v2.0 🚀                   ║
    ║                                                                  ║
    ║         Sistema de Trading Inteligente com IA e Confluência     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    args = parse_arguments()
    
    try:
        if args.mode == 'trading':
            # Modo trading normal
            app = TradingSystemApp(args.config)
            setup_signal_handlers(app)
            
            await app.initialize()
            await app.run()
            
        elif args.mode == 'backtest':
            # Modo backtesting
            await data_provider.initialize_all()
            await run_backtest_mode(args)
            
        elif args.mode == 'dashboard':
            # Modo dashboard
            await run_dashboard_mode(args)
        
    except KeyboardInterrupt:
        logger.info("👋 Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Verifica versão do Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requerido")
        sys.exit(1)
    
    # Executa aplicação
    asyncio.run(main())