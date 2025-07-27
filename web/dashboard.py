"""
🖥️ WEB DASHBOARD - Smart Trading System v2.0

Dashboard interativo completo usando Streamlit:
- Real-time signals monitoring
- Performance analytics
- Risk metrics visualization
- Component analysis
- Backtesting interface
- System configuration

Filosofia: Visualization is Understanding - See the System, Control the System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import json

# Configurar página
st.set_page_config(
    page_title="Smart Trading System v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .signal-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .bullish-signal {
        border-left-color: #2ca02c !important;
    }
    .bearish-signal {
        border-left-color: #d62728 !important;
    }
    .neutral-signal {
        border-left-color: #ff7f0e !important;
    }
</style>
""", unsafe_allow_html=True)

class DashboardManager:
    """Gerenciador principal do dashboard"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializa estado da sessão"""
        if 'signals_data' not in st.session_state:
            st.session_state.signals_data = self.generate_mock_signals()
        
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = self.generate_mock_performance()
        
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = '7D'
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
    
    def generate_mock_signals(self) -> List[Dict]:
        """Gera sinais mock para demonstração"""
        np.random.seed(42)
        signals = []
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
        signal_types = ["swing_long", "swing_short", "breakout_long", "breakout_short"]
        qualities = ["exceptional", "excellent", "good", "moderate"]
        
        for i in range(12):
            signal = {
                'signal_id': f"SIG_{i+1:03d}",
                'symbol': np.random.choice(symbols),
                'signal_type': np.random.choice(signal_types),
                'direction': 'bullish' if 'long' in np.random.choice(signal_types) else 'bearish',
                'final_score': np.random.uniform(65, 95),
                'confidence': np.random.uniform(60, 90),
                'quality': np.random.choice(qualities),
                'entry_price': np.random.uniform(45000, 55000),
                'stop_loss': np.random.uniform(42000, 48000),
                'take_profit_1': np.random.uniform(56000, 65000),
                'position_size_pct': np.random.uniform(0.02, 0.05),
                'risk_reward_ratio': np.random.uniform(2.0, 4.5),
                'generated_at': datetime.now(timezone.utc) - timedelta(hours=np.random.randint(0, 48)),
                'status': np.random.choice(['active', 'pending', 'executed']),
                'reasoning': f"Strong {np.random.choice(['trend', 'structure', 'momentum'])} signal with high confluence",
                'warnings': ['High volatility'] if np.random.random() > 0.7 else []
            }
            signals.append(signal)
        
        return signals
    
    def generate_mock_performance(self) -> Dict:
        """Gera dados de performance mock"""
        np.random.seed(42)
        
        # Equity curve
        dates = pd.date_range(start='2024-01-01', end='2024-07-27', freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
        cumulative_returns = np.cumsum(returns)
        equity_curve = 100000 * (1 + cumulative_returns)
        
        # Trade history
        trade_dates = pd.date_range(start='2024-01-01', end='2024-07-27', freq='3D')
        trades = []
        
        for i, date in enumerate(trade_dates):
            trade = {
                'date': date,
                'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT']),
                'side': np.random.choice(['long', 'short']),
                'pnl_pct': np.random.normal(0.02, 0.08),  # 2% avg, 8% vol
                'duration_hours': np.random.exponential(24),
                'exit_reason': np.random.choice(['take_profit', 'stop_loss', 'manual'])
            }
            trades.append(trade)
        
        return {
            'equity_curve': {
                'dates': dates,
                'values': equity_curve
            },
            'trades': trades,
            'metrics': {
                'total_return': 0.247,
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.08,
                'win_rate': 0.68,
                'total_trades': len(trades),
                'avg_trade_duration': 28.5
            }
        }
    
    def render_dashboard(self):
        """Renderiza dashboard principal"""
        st.markdown('<h1 class="main-header">🎯 Smart Trading System v2.0</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Signals", 
            "📈 Performance", 
            "⚙️ System Status", 
            "🔬 Backtesting", 
            "🎛️ Configuration"
        ])
        
        with tab1:
            self.render_signals_tab()
        
        with tab2:
            self.render_performance_tab()
        
        with tab3:
            self.render_system_status_tab()
        
        with tab4:
            self.render_backtesting_tab()
        
        with tab5:
            self.render_configuration_tab()
    
    def render_sidebar(self):
        """Renderiza sidebar"""
        with st.sidebar:
            st.markdown("### 🎛️ Controls")
            
            # Auto refresh
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh", 
                value=st.session_state.auto_refresh
            )
            
            if st.button("🔄 Refresh Now"):
                st.session_state.signals_data = self.generate_mock_signals()
                st.rerun()
            
            st.markdown("---")
            
            # Timeframe selector
            st.markdown("### 📅 Timeframe")
            st.session_state.selected_timeframe = st.selectbox(
                "Select Period",
                ["1D", "7D", "30D", "90D", "1Y"],
                index=1
            )
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            
            signals = st.session_state.signals_data
            active_signals = len([s for s in signals if s['status'] == 'active'])
            avg_score = np.mean([s['final_score'] for s in signals])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Signals", active_signals)
            with col2:
                st.metric("Avg Score", f"{avg_score:.1f}")
            
            performance = st.session_state.performance_data['metrics']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Return", f"{performance['total_return']:.1%}")
            with col2:
                st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            # System health
            st.markdown("### 🏥 System Health")
            st.success("🟢 All systems operational")
            st.info("📡 Data feed: Connected")
            st.info("🎯 Signal generation: Active")
    
    def render_signals_tab(self):
        """Renderiza tab de sinais"""
        st.markdown("## 📊 Live Trading Signals")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol_filter = st.selectbox(
                "Symbol",
                ["All"] + list(set([s['symbol'] for s in st.session_state.signals_data]))
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status",
                ["All", "active", "pending", "executed"]
            )
        
        with col3:
            min_score = st.slider("Min Score", 0, 100, 70)
        
        with col4:
            quality_filter = st.selectbox(
                "Quality",
                ["All", "exceptional", "excellent", "good", "moderate"]
            )
        
        # Filter signals
        filtered_signals = self.filter_signals(
            st.session_state.signals_data,
            symbol_filter, status_filter, min_score, quality_filter
        )
        
        if not filtered_signals:
            st.warning("No signals match the current filters.")
            return
        
        # Signals overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Signals", len(filtered_signals))
        
        with col2:
            active_count = len([s for s in filtered_signals if s['status'] == 'active'])
            st.metric("Active", active_count, delta=f"{active_count-2}")
        
        with col3:
            avg_score = np.mean([s['final_score'] for s in filtered_signals])
            st.metric("Avg Score", f"{avg_score:.1f}", delta="2.3")
        
        with col4:
            bullish_count = len([s for s in filtered_signals if s['direction'] == 'bullish'])
            st.metric("Bullish", f"{bullish_count}/{len(filtered_signals)}")
        
        st.markdown("---")
        
        # Signals grid
        self.render_signals_grid(filtered_signals)
        
        st.markdown("---")
        
        # Detailed signals table
        self.render_signals_table(filtered_signals)
    
    def filter_signals(self, signals, symbol_filter, status_filter, min_score, quality_filter):
        """Filtra sinais baseado nos critérios"""
        filtered = signals
        
        if symbol_filter != "All":
            filtered = [s for s in filtered if s['symbol'] == symbol_filter]
        
        if status_filter != "All":
            filtered = [s for s in filtered if s['status'] == status_filter]
        
        filtered = [s for s in filtered if s['final_score'] >= min_score]
        
        if quality_filter != "All":
            filtered = [s for s in filtered if s['quality'] == quality_filter]
        
        return filtered
    
    def render_signals_grid(self, signals):
        """Renderiza grid de sinais"""
        # Organizar em colunas
        cols = st.columns(3)
        
        for i, signal in enumerate(signals[:9]):  # Mostrar top 9
            with cols[i % 3]:
                self.render_signal_card(signal)
    
    def render_signal_card(self, signal):
        """Renderiza card individual de sinal"""
        direction_class = f"{signal['direction']}-signal"
        direction_emoji = "🟢" if signal['direction'] == 'bullish' else "🔴"
        
        card_html = f"""
        <div class="signal-card {direction_class}">
            <h4>{direction_emoji} {signal['symbol']}</h4>
            <p><strong>Type:</strong> {signal['signal_type'].replace('_', ' ').title()}</p>
            <p><strong>Score:</strong> {signal['final_score']:.1f}/100</p>
            <p><strong>Quality:</strong> {signal['quality'].title()}</p>
            <p><strong>Entry:</strong> ${signal['entry_price']:,.2f}</p>
            <p><strong>R:R:</strong> {signal['risk_reward_ratio']:.1f}</p>
            <p><strong>Status:</strong> {signal['status'].title()}</p>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Execute {signal['signal_id']}", key=f"exec_{signal['signal_id']}"):
                st.success("Signal executed!")
        with col2:
            if st.button(f"Details {signal['signal_id']}", key=f"details_{signal['signal_id']}"):
                self.show_signal_details(signal)
    
    def render_signals_table(self, signals):
        """Renderiza tabela detalhada de sinais"""
        st.markdown("### 📋 Detailed Signals")
        
        # Preparar dados para tabela
        table_data = []
        for signal in signals:
            table_data.append({
                'Signal ID': signal['signal_id'],
                'Symbol': signal['symbol'],
                'Type': signal['signal_type'],
                'Direction': signal['direction'],
                'Score': f"{signal['final_score']:.1f}",
                'Confidence': f"{signal['confidence']:.1f}%",
                'Entry': f"${signal['entry_price']:,.0f}",
                'Stop Loss': f"${signal['stop_loss']:,.0f}",
                'Take Profit': f"${signal['take_profit_1']:,.0f}",
                'R:R': f"{signal['risk_reward_ratio']:.1f}",
                'Position Size': f"{signal['position_size_pct']:.1%}",
                'Status': signal['status'],
                'Generated': signal['generated_at'].strftime('%Y-%m-%d %H:%M')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    def show_signal_details(self, signal):
        """Mostra detalhes do sinal em modal"""
        with st.expander(f"📊 Signal Details: {signal['signal_id']}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Info**")
                st.write(f"Symbol: {signal['symbol']}")
                st.write(f"Type: {signal['signal_type']}")
                st.write(f"Direction: {signal['direction']}")
                st.write(f"Status: {signal['status']}")
                st.write(f"Generated: {signal['generated_at'].strftime('%Y-%m-%d %H:%M')}")
                
                st.markdown("**Scores**")
                st.write(f"Final Score: {signal['final_score']:.1f}/100")
                st.write(f"Confidence: {signal['confidence']:.1f}%")
                st.write(f"Quality: {signal['quality']}")
            
            with col2:
                st.markdown("**Trading Parameters**")
                st.write(f"Entry Price: ${signal['entry_price']:,.2f}")
                st.write(f"Stop Loss: ${signal['stop_loss']:,.2f}")
                st.write(f"Take Profit: ${signal['take_profit_1']:,.2f}")
                st.write(f"Risk/Reward: {signal['risk_reward_ratio']:.1f}")
                st.write(f"Position Size: {signal['position_size_pct']:.1%}")
                
                if signal['warnings']:
                    st.markdown("**⚠️ Warnings**")
                    for warning in signal['warnings']:
                        st.warning(warning)
            
            st.markdown("**Reasoning**")
            st.write(signal['reasoning'])
    
    def render_performance_tab(self):
        """Renderiza tab de performance"""
        st.markdown("## 📈 Performance Analytics")
        
        performance = st.session_state.performance_data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{performance['metrics']['total_return']:.1%}",
                delta="2.3%"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                f"{performance['metrics']['sharpe_ratio']:.2f}",
                delta="0.15"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{performance['metrics']['max_drawdown']:.1%}",
                delta="-1.2%"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{performance['metrics']['win_rate']:.1%}",
                delta="3.5%"
            )
        
        st.markdown("---")
        
        # Equity curve
        st.markdown("### 📊 Equity Curve")
        self.render_equity_curve(performance['equity_curve'])
        
        st.markdown("---")
        
        # Performance breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Monthly Returns")
            self.render_monthly_returns()
        
        with col2:
            st.markdown("### 🎯 Trade Analysis")
            self.render_trade_analysis(performance['trades'])
        
        st.markdown("---")
        
        # Detailed metrics
        st.markdown("### 📋 Detailed Metrics")
        self.render_detailed_metrics(performance['metrics'])
    
    def render_equity_curve(self, equity_data):
        """Renderiza curva de equity"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=equity_data['dates'],
            y=equity_data['values'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Adicionar linha de benchmark (buy and hold)
        benchmark_values = equity_data['values'][0] * (1 + np.random.normal(0.0008, 0.025, len(equity_data['dates'])).cumsum())
        fig.add_trace(go.Scatter(
            x=equity_data['dates'],
            y=benchmark_values,
            mode='lines',
            name='Buy & Hold Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Performance vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_monthly_returns(self):
        """Renderiza returns mensais"""
        # Gerar dados mock de returns mensais
        months = pd.date_range(start='2024-01', end='2024-07', freq='M')
        returns = np.random.normal(0.02, 0.08, len(months))
        
        fig = go.Figure(data=[
            go.Bar(
                x=[month.strftime('%Y-%m') for month in months],
                y=returns,
                marker_color=['green' if r > 0 else 'red' for r in returns]
            )
        ])
        
        fig.update_layout(
            title="Monthly Returns",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_trade_analysis(self, trades):
        """Renderiza análise de trades"""
        df_trades = pd.DataFrame(trades)
        
        # PnL distribution
        fig = go.Figure(data=[
            go.Histogram(
                x=df_trades['pnl_pct'],
                nbinsx=20,
                name="Trade PnL Distribution"
            )
        ])
        
        fig.update_layout(
            title="Trade PnL Distribution",
            xaxis_title="PnL (%)",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade duration vs PnL
        fig = go.Figure(data=[
            go.Scatter(
                x=df_trades['duration_hours'],
                y=df_trades['pnl_pct'],
                mode='markers',
                marker=dict(
                    color=df_trades['pnl_pct'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                text=df_trades['symbol'],
                hovertemplate='<b>%{text}</b><br>Duration: %{x:.1f}h<br>PnL: %{y:.1%}'
            )
        ])
        
        fig.update_layout(
            title="Trade Duration vs PnL",
            xaxis_title="Duration (hours)",
            yaxis_title="PnL (%)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_detailed_metrics(self, metrics):
        """Renderiza métricas detalhadas"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Trading Metrics**")
            st.write(f"Total Trades: {metrics['total_trades']}")
            st.write(f"Win Rate: {metrics['win_rate']:.1%}")
            st.write(f"Avg Trade Duration: {metrics['avg_trade_duration']:.1f}h")
        
        with col2:
            st.markdown("**Risk Metrics**")
            st.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            st.write(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
            st.write(f"Volatility: 15.2%")  # Mock
        
        with col3:
            st.markdown("**Return Metrics**")
            st.write(f"Total Return: {metrics['total_return']:.1%}")
            st.write(f"CAGR: 28.5%")  # Mock
            st.write(f"Profit Factor: 1.85")  # Mock
    
    def render_system_status_tab(self):
        """Renderiza tab de status do sistema"""
        st.markdown("## ⚙️ System Status")
        
        # System health overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Uptime", "99.8%", delta="0.1%")
        
        with col2:
            st.metric("Signal Generation Rate", "18/day", delta="2")
        
        with col3:
            st.metric("Execution Success Rate", "94.5%", delta="1.2%")
        
        with col4:
            st.metric("Data Feed Status", "🟢 Online")
        
        st.markdown("---")
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Core Components")
            
            components = [
                ("Market Data Provider", "🟢", "Operational"),
                ("Signal Generator", "🟢", "Active"),
                ("Risk Manager", "🟢", "Monitoring"),
                ("Database", "🟢", "Connected"),
                ("Web Dashboard", "🟢", "Running")
            ]
            
            for name, status, description in components:
                st.write(f"{status} **{name}**: {description}")
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            
            # Component performance chart
            components = ['Market Structure', 'Trend Analysis', 'Leading Indicators', 'Strategies', 'Filters']
            scores = [85, 78, 82, 88, 75]
            
            fig = go.Figure(data=[
                go.Bar(x=components, y=scores, marker_color='lightblue')
            ])
            
            fig.update_layout(
                title="Component Performance Scores",
                yaxis_title="Score",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Recent activity log
        st.markdown("### 📝 Recent Activity")
        
        activity_log = [
            {"time": "14:32", "event": "New signal generated", "details": "BTCUSDT swing_long (Score: 87.3)"},
            {"time": "14:15", "event": "Position closed", "details": "ETHUSDT +3.2% (take_profit)"},
            {"time": "13:45", "event": "Risk limit adjusted", "details": "Max exposure reduced to 12%"},
            {"time": "13:20", "event": "Market condition change", "details": "Bull -> Sideways Bull"},
            {"time": "12:55", "event": "Signal executed", "details": "ADAUSDT breakout_long"},
        ]
        
        for log in activity_log:
            col1, col2, col3 = st.columns([1, 3, 6])
            with col1:
                st.write(log["time"])
            with col2:
                st.write(log["event"])
            with col3:
                st.write(log["details"])
    
    def render_backtesting_tab(self):
        """Renderiza tab de backtesting"""
        st.markdown("## 🔬 Backtesting Interface")
        
        # Backtest configuration
        st.markdown("### ⚙️ Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
            end_date = st.date_input("End Date", value=datetime(2024, 6, 30))
        
        with col2:
            symbols = st.multiselect(
                "Symbols",
                ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"],
                default=["BTCUSDT", "ETHUSDT"]
            )
            initial_balance = st.number_input("Initial Balance ($)", value=100000, step=1000)
        
        with col3:
            execution_model = st.selectbox(
                "Execution Model",
                ["Realistic", "Perfect", "Conservative"]
            )
            max_positions = st.slider("Max Positions", 1, 10, 5)
        
        # Advanced settings
        with st.expander("🎛️ Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                slippage_bps = st.number_input("Slippage (bps)", value=5.0, step=0.5)
                fees_pct = st.number_input("Fees (%)", value=0.1, step=0.01)
            
            with col2:
                min_signal_score = st.slider("Min Signal Score", 50, 95, 70)
                risk_free_rate = st.number_input("Risk-free Rate (%)", value=2.0, step=0.1)
        
        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                # Simular execução do backtest
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.success("Backtest completed successfully!")
                
                # Mock results
                self.render_backtest_results()
    
    def render_backtest_results(self):
        """Renderiza resultados do backtest"""
        st.markdown("### 📊 Backtest Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", "34.7%", delta="34.7%")
        
        with col2:
            st.metric("Sharpe Ratio", "2.15", delta="2.15")
        
        with col3:
            st.metric("Max Drawdown", "8.3%", delta="-8.3%")
        
        with col4:
            st.metric("Win Rate", "72.4%", delta="72.4%")
        
        # Results table
        results_data = {
            "Metric": [
                "Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Sortino Ratio",
                "Max Drawdown", "Calmar Ratio", "Total Trades", "Win Rate", "Profit Factor"
            ],
            "Value": [
                "34.7%", "28.5%", "18.2%", "2.15", "3.24",
                "8.3%", "3.43", "156", "72.4%", "2.87"
            ],
            "Benchmark": [
                "22.1%", "18.3%", "25.6%", "1.42", "2.01",
                "15.2%", "1.20", "1", "100%", "∞"
            ]
        }
        
        st.dataframe(pd.DataFrame(results_data), hide_index=True)
    
    def render_configuration_tab(self):
        """Renderiza tab de configuração"""
        st.markdown("## 🎛️ System Configuration")
        
        # Signal Generation Settings
        st.markdown("### 🎯 Signal Generation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Scoring Thresholds**")
            min_signal_score = st.slider("Minimum Signal Score", 50, 95, 70, help="Minimum score for signal generation")
            min_confluence_factors = st.slider("Min Confluence Factors", 2, 8, 4)
            
            st.markdown("**Component Weights**")
            market_structure_weight = st.slider("Market Structure", 0.0, 1.0, 0.25, step=0.05)
            trend_analysis_weight = st.slider("Trend Analysis", 0.0, 1.0, 0.25, step=0.05)
            leading_indicators_weight = st.slider("Leading Indicators", 0.0, 1.0, 0.20, step=0.05)
            strategy_signals_weight = st.slider("Strategy Signals", 0.0, 1.0, 0.20, step=0.05)
            confluence_weight = st.slider("Confluence Analysis", 0.0, 1.0, 0.10, step=0.05)
        
        with col2:
            st.markdown("**Timeframes**")
            enabled_timeframes = st.multiselect(
                "Enabled Timeframes",
                ["1H", "4H", "1D", "3D", "1W"],
                default=["1H", "4H", "1D"]
            )
            
            primary_timeframe = st.selectbox("Primary Timeframe", ["4H", "1D"], index=0)
            
            st.markdown("**Generation Frequency**")
            generation_frequency = st.selectbox(
                "Signal Generation Frequency",
                ["1H", "4H", "8H", "12H", "1D"],
                index=1
            )
        
        # Risk Management Settings
        st.markdown("### ⚡ Risk Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Position Sizing**")
            max_portfolio_risk = st.slider("Max Portfolio Risk (%)", 5, 25, 15)
            max_single_position = st.slider("Max Single Position (%)", 1, 10, 5)
            max_correlation_exposure = st.slider("Max Correlation Exposure (%)", 5, 20, 10)
            
        with col2:
            st.markdown("**Risk Limits**")
            max_daily_risk = st.slider("Max Daily Risk (%)", 1, 10, 3)
            max_drawdown_limit = st.slider("Max Drawdown Limit (%)", 10, 30, 20)
            max_positions = st.slider("Max Concurrent Positions", 3, 15, 10)
        
        # Filter Settings
        st.markdown("### 🔍 Filters Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Market Condition Filter**")
            market_condition_enabled = st.checkbox("Enable Market Condition Filter", value=True)
            min_market_confidence = st.slider("Min Market Confidence", 30, 90, 50)
            
            st.markdown("**Volatility Filter**")
            volatility_filter_enabled = st.checkbox("Enable Volatility Filter", value=True)
            min_liquidity_score = st.slider("Min Liquidity Score", 20, 80, 40)
        
        with col2:
            st.markdown("**Time Filter**")
            time_filter_enabled = st.checkbox("Enable Time Filter", value=True)
            avoid_weekend_gaps = st.checkbox("Avoid Weekend Gaps", value=True)
            
            st.markdown("**Fundamental Filter**")
            fundamental_filter_enabled = st.checkbox("Enable Fundamental Filter", value=True)
            avoid_high_impact_news = st.checkbox("Avoid High Impact News", value=True)
        
        # Save configuration
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Configuration", type="primary"):
                st.success("Configuration saved successfully!")
        
        with col2:
            if st.button("🔄 Reset to Defaults"):
                st.info("Configuration reset to defaults")
        
        with col3:
            if st.button("📤 Export Config"):
                config = {
                    "signal_generation": {
                        "min_signal_score": min_signal_score,
                        "min_confluence_factors": min_confluence_factors,
                        "enabled_timeframes": enabled_timeframes,
                        "primary_timeframe": primary_timeframe
                    },
                    "risk_management": {
                        "max_portfolio_risk": max_portfolio_risk,
                        "max_single_position": max_single_position,
                        "max_positions": max_positions
                    }
                }
                st.download_button(
                    "Download config.json",
                    json.dumps(config, indent=2),
                    "config.json",
                    "application/json"
                )


def main():
    """Função principal do dashboard"""
    dashboard = DashboardManager()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()