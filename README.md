# 🎯 Smart Trading System v2.0

**Sistema de Trading Inteligente** - Arquitetura modular completa para análise de mercado, geração de sinais e execução automatizada.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-2.0-red.svg)

---

## 🌟 **VISÃO GERAL**

O Smart Trading System v2.0 é um sistema completo de trading algorítmico que combina:

- **📊 Análise Multi-Componente**: Market Structure + Trend + Leading Indicators + Confluence
- **🎯 Estratégias Avançadas**: Swing Trading + Breakout + Mean Reversion
- **🛡️ Risk Management**: Position Sizing adaptativo + Stop Loss dinâmico
- **🔍 Filtros Inteligentes**: Market Condition + Volatility + Time + Fundamental
- **🔬 Backtesting**: Engine completo com métricas avançadas
- **🖥️ Dashboard Web**: Interface interativa em tempo real

---

## 🏗️ **ARQUITETURA DO SISTEMA**

```
smart_trading_system/
├── 📁 config/                  # Configurações
│   ├── settings.py
│   ├── symbols.py
│   └── timeframes.py
├── 📁 core/                    # Sistema Central
│   ├── market_data.py          # Coleta dados OHLCV
│   ├── market_structure.py     # Análise HH/HL/LH/LL
│   ├── signal_generator.py     # 🎯 MOTOR PRINCIPAL
│   ├── signal_manager.py       # Gestão de sinais
│   ├── risk_manager.py         # ⚡ Risk Management
│   └── portfolio_manager.py    # Gestão portfólio
├── 📁 indicators/              # Indicadores Inteligentes
│   ├── leading_indicators.py   # Volume Profile + Order Flow
│   ├── confluence_analyzer.py  # 🎯 Sistema Confluência
│   ├── trend_analyzer.py       # Multi-timeframe
│   └── divergence_detector.py  # Divergências
├── 📁 strategies/              # Estratégias Core
│   ├── swing_strategy.py       # 🎯 Swing Trading
│   ├── breakout_strategy.py    # 💥 Breakout + Retest
│   ├── mean_reversion.py       # Mean Reversion
│   └── trend_following.py      # Trend Following
├── 📁 filters/                 # Filtros de Qualidade
│   ├── market_condition.py     # 🌊 Bull/Bear/Sideways
│   ├── volatility_filter.py    # ⚡ Regime de Vol
│   ├── time_filter.py          # 🕐 Trading Sessions
│   └── fundamental_filter.py   # 📰 News + Events
├── 📁 database/                # Persistência
│   ├── models.py               # 💾 SQLAlchemy Models
│   ├── database.py             # Operações DB
│   └── migrations/             # Schema versioning
├── 📁 backtesting/             # Validação
│   ├── backtest_engine.py      # 🔬 Engine Completo
│   ├── performance_analyzer.py # Métricas avançadas
│   └── reports.py              # Relatórios HTML
├── 📁 web/                     # Interface
│   ├── dashboard.py            # 🖥️ Dashboard Streamlit
│   ├── templates/              # Templates HTML
│   └── static/                 # CSS/JS/Images
└── 📁 tests/                   # Testes
    ├── test_indicators.py
    ├── test_strategies.py
    └── test_risk_manager.py
```

---

## 🎯 **COMPONENTES PRINCIPAIS**

### **1. 🧠 Signal Generator** (Motor Principal)
Orquestra todos os componentes para gerar sinais de alta qualidade:

```python
from core.signal_generator import SignalGenerator

# Inicializar
generator = SignalGenerator(
    market_data_provider=data_provider,
    risk_manager=risk_manager,
    min_signal_score=70.0
)

# Gerar sinais
signals = generator.generate_signals(["BTCUSDT", "ETHUSDT"])

for signal in signals:
    print(f"Signal: {signal.symbol} - Score: {signal.final_score}")
```

### **2. ⚡ Risk Manager** (Gestão de Risco)
Position sizing adaptativo baseado em volatilidade:

```python
from core.risk_manager import RiskManager, PositionSizingInput

risk_manager = RiskManager(initial_balance=100000)

# Calcular position size
sizing_input = PositionSizingInput(
    signal_strength=85,
    confidence=78,
    risk_reward_ratio=3.2,
    entry_price=50000,
    stop_loss=48500,
    market_volatility=0.025
)

result = risk_manager.calculate_position_size(sizing_input, "BTCUSDT")
```

### **3. 🎯 Confluence Analyzer** (Sistema de Confluência)
Combina múltiplos fatores para sinais de alta probabilidade:

```python
from indicators.confluence_analyzer import ConfluenceAnalyzer

analyzer = ConfluenceAnalyzer()
signals = analyzer.analyze_confluence(market_data, "BTCUSDT")

for signal in signals:
    print(f"Confluence Score: {signal.confluence_score}")
    print(f"Factors: {len(signal.factors)}")
```

---

## 🎯 **ESTRATÉGIAS IMPLEMENTADAS**

### **1. Swing Strategy** 🎯
- **Timeframes**: 4H/1D para setup, 1H para entrada
- **Lógica**: HH/HL para bull, LH/LL para bear
- **Confluência**: Multiple confirmations required
- **Risk**: Structure-based stops

### **2. Breakout Strategy** 💥
- **Padrões**: Range, Triangle, Flag, Pennant
- **Confirmação**: Volume 2x+ average
- **Entrada**: Breakout + Retest logic
- **Filtros**: False breakout filtering

### **3. Mean Reversion** 🔄
- **Detecção**: Extremos de volatilidade
- **Entrada**: Support/Resistance bounces
- **Saída**: Return to mean

### **4. Trend Following** 📈
- **Alinhamento**: Multi-timeframe trends
- **Entrada**: Pullback entries
- **Gestão**: Trailing stops

---

## 🔍 **SISTEMA DE FILTROS**

### **1. Market Condition Filter** 🌊
9 condições de mercado: `strong_bull` → `strong_bear`

```python
from filters.market_condition import MarketConditionFilter

filter = MarketConditionFilter()
analysis = filter.analyze_market_condition(market_data)

print(f"Condition: {analysis.condition.value}")
print(f"Recommended: {analysis.recommended_strategies}")
```

### **2. Volatility Filter** ⚡
5 regimes: `extremely_low` → `extremely_high`

### **3. Time Filter** 🕐
Trading sessions: Asian/European/US + Overlaps

### **4. Fundamental Filter** 📰
Economic calendar + Crypto news + Regulatory events

---

## 📊 **SCORING SYSTEM**

### **Confluence Score (0-100)**
Combina múltiplos fatores:
- **Market Structure**: 25% weight
- **Trend Alignment**: 25% weight  
- **Leading Indicators**: 20% weight
- **Strategy Signals**: 20% weight
- **Support/Resistance**: 10% weight

### **Signal Quality**
- **Exceptional**: 90-100 score
- **Excellent**: 80-90 score
- **Good**: 70-80 score
- **Moderate**: 60-70 score
- **Weak**: 50-60 score

---

## 🔬 **BACKTESTING ENGINE**

Engine completo com simulação realística:

```python
from backtesting.backtest_engine import BacktestEngine, BacktestConfig

config = BacktestConfig(
    name="Strategy Test",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    symbols=["BTCUSDT", "ETHUSDT"],
    initial_balance=100000,
    execution_model=ExecutionModel.REALISTIC
)

engine = BacktestEngine(signal_generator, risk_manager, data_provider)
results = engine.run_backtest(config)

print(f"Total Return: {results.total_return_pct:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown_pct:.2%}")
```

---

## 🖥️ **DASHBOARD WEB**

Interface completa em Streamlit:

```bash
streamlit run web/dashboard.py
```

**Features do Dashboard:**
- 📊 **Live Signals**: Monitoramento em tempo real
- 📈 **Performance**: Analytics completos
- ⚙️ **System Status**: Health monitoring
- 🔬 **Backtesting**: Interface interativa
- 🎛️ **Configuration**: Settings management

---

## 💾 **BANCO DE DADOS**

Sistema completo de persistência com SQLAlchemy:

### **Tabelas Principais:**
- **`signals`**: Sinais gerados
- **`trades`**: Execuções de trade
- **`market_data`**: Cache de dados
- **`performance_metrics`**: Métricas de performance
- **`backtest_results`**: Resultados de backtesting

```python
from database.models import DatabaseManager

db = DatabaseManager("postgresql://user:pass@host/db")
db.create_tables()

# Salvar sinal
signal_record = db.save_signal(master_signal)
```

---

## 🚀 **QUICK START**

### **1. Instalação**
```bash
git clone https://github.com/your-repo/smart-trading-system
cd smart-trading-system
pip install -r requirements.txt
```

### **2. Configuração**
```python
# config/settings.py
API_KEY = "your_binance_api_key"
API_SECRET = "your_binance_secret"
DATABASE_URL = "postgresql://user:pass@host/db"
```

### **3. Executar Sistema**
```python
from main import SmartTradingSystem

system = SmartTradingSystem()
system.start()
```

### **4. Dashboard**
```bash
streamlit run web/dashboard.py
```

---

## 📋 **REQUIREMENTS**

```txt
pandas>=1.5.0
numpy>=1.21.0
sqlalchemy>=1.4.0
streamlit>=1.28.0
plotly>=5.15.0
scikit-learn>=1.1.0
requests>=2.28.0
python-binance>=1.0.16
ta-lib>=0.4.25
```

---

## 🎯 **FILOSOFIA DO SISTEMA**

### **Princípios Core:**
1. **Qualidade > Quantidade** - Poucos sinais excelentes
2. **Contexto > Indicadores** - Market structure primeiro  
3. **Adaptabilidade** - Sistema aprende e se ajusta
4. **Simplicidade Elegante** - Complexo por dentro, simples por fora

### **Hierarchy de Timeframes:**
- **1D**: Bias e contexto macro
- **4H**: Setup e confirmação
- **1H**: Entrada e gestão
- **15m**: Apenas para stop loss dinâmico

---

## 📊 **PERFORMANCE ESPERADA**

### **Backtesting Results** (Mock)
- **Total Return**: 34.7% (6 meses)
- **Sharpe Ratio**: 2.15
- **Max Drawdown**: 8.3%
- **Win Rate**: 72.4%
- **Profit Factor**: 2.87

### **Risk Metrics**
- **Max Portfolio Risk**: 15%
- **Max Single Position**: 5%
- **Typical Position Size**: 2-3%
- **Stop Loss**: Structure-based

---

## 🔧 **CUSTOMIZAÇÃO**

### **Adicionar Nova Estratégia:**
```python
# strategies/custom_strategy.py
from strategies.base_strategy import BaseStrategy

class CustomStrategy(BaseStrategy):
    def analyze_opportunity(self, data):
        # Sua lógica aqui
        return signals
```

### **Adicionar Novo Filtro:**
```python
# filters/custom_filter.py
from filters.base_filter import BaseFilter

class CustomFilter(BaseFilter):
    def analyze(self, data):
        # Sua lógica aqui
        return filter_result
```

---

## 🧪 **TESTING**

```bash
# Executar todos os testes
pytest tests/

# Teste específico
pytest tests/test_signal_generator.py -v

# Coverage
pytest --cov=smart_trading_system tests/
```

---

## 📈 **ROADMAP**

### **v2.1 (Próxima Release)**
- [ ] Machine Learning integration
- [ ] Sentiment analysis avançado
- [ ] Multi-exchange support
- [ ] Mobile app

### **v2.2 (Futuro)**
- [ ] Portfolio optimization
- [ ] Advanced order types
- [ ] Social trading features
- [ ] Cloud deployment

---

## 🤝 **CONTRIBUIÇÃO**

1. Fork o repositório
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## ⚖️ **LICENÇA**

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

## 🙏 **AGRADECIMENTOS**

- **Binance API** - Dados de mercado
- **TA-Lib** - Indicadores técnicos
- **Streamlit** - Dashboard framework
- **SQLAlchemy** - ORM
- **Plotly** - Visualizações

---

## 📞 **CONTATO**

**Projeto**: Smart Trading System v2.0  
**Email**: trading@smartsystem.com  
**Discord**: [Trading Community](https://discord.gg/trading)  
**Documentação**: [Wiki completa](https://wiki.smartsystem.com)

---

## ⚠️ **DISCLAIMER**

Este sistema é para fins educacionais e de pesquisa. Trading envolve riscos significativos. Sempre teste extensivamente antes de usar capital real. Não nos responsabilizamos por perdas.

---

<div align="center">

**🎯 Smart Trading System v2.0**  
*"Intelligence Meets Markets"*

[![GitHub stars](https://img.shields.io/github/stars/smart-trading/system.svg?style=social&label=Star)](https://github.com/smart-trading/system)
[![Twitter Follow](https://img.shields.io/twitter/follow/smarttrading?style=social)](https://twitter.com/smarttrading)

</div>