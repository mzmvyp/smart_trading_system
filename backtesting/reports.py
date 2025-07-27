"""
Backtesting: Reports Generator
Gerador de relatórios detalhados com gráficos e análises
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from jinja2 import Template

from .backtest_engine import BacktestResult
from utils.logger import get_logger
from utils.helpers import format_currency, format_percentage, format_number


logger = get_logger(__name__)

# Configure matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ReportGenerator:
    """Gerador de relatórios de backtesting"""
    
    def __init__(self):
        self.charts_data = {}
        self.report_style = {
            'primary_color': '#2E86C1',
            'success_color': '#28B463',
            'danger_color': '#E74C3C',
            'warning_color': '#F39C12',
            'info_color': '#85C1E9',
            'background_color': '#F8F9FA',
            'text_color': '#2C3E50'
        }
    
    def generate_full_report(
        self,
        result: BacktestResult,
        output_path: str = "backtest_report.html",
        include_charts: bool = True
    ) -> str:
        """Gera relatório completo HTML"""
        
        try:
            logger.info("Gerando relatório de backtesting...")
            
            # Gera componentes do relatório
            summary_data = self._prepare_summary_data(result)
            performance_data = self._prepare_performance_data(result)
            trade_analysis = self._prepare_trade_analysis(result)
            strategy_analysis = self._prepare_strategy_analysis(result)
            
            # Gera gráficos se solicitado
            charts = {}
            if include_charts:
                charts = self._generate_all_charts(result)
            
            # Gera HTML
            html_content = self._generate_html_report(
                summary_data,
                performance_data,
                trade_analysis,
                strategy_analysis,
                charts
            )
            
            # Salva arquivo
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Relatório salvo em: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            raise
    
    def _prepare_summary_data(self, result: BacktestResult) -> Dict:
        """Prepara dados do resumo executivo"""
        
        # Classificação de performance
        if result.total_return >= 50:
            performance_rating = "Excelente"
            performance_class = "success"
        elif result.total_return >= 20:
            performance_rating = "Bom"
            performance_class = "info"
        elif result.total_return >= 0:
            performance_rating = "Moderado"
            performance_class = "warning"
        else:
            performance_rating = "Ruim"
            performance_class = "danger"
        
        # Classificação do Sharpe
        if result.sharpe_ratio >= 1.5:
            sharpe_rating = "Excelente"
        elif result.sharpe_ratio >= 1.0:
            sharpe_rating = "Bom"
        elif result.sharpe_ratio >= 0.5:
            sharpe_rating = "Aceitável"
        else:
            sharpe_rating = "Ruim"
        
        return {
            'period': {
                'start': result.start_date.strftime('%d/%m/%Y') if result.start_date else 'N/A',
                'end': result.end_date.strftime('%d/%m/%Y') if result.end_date else 'N/A',
                'days': result.trading_days
            },
            'performance': {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'rating': performance_rating,
                'class': performance_class
            },
            'risk': {
                'sharpe_ratio': result.sharpe_ratio,
                'sharpe_rating': sharpe_rating,
                'max_drawdown': result.max_drawdown,
                'volatility': self._calculate_volatility(result.equity_curve)
            },
            'trades': {
                'total': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }
        }
    
    def _prepare_performance_data(self, result: BacktestResult) -> Dict:
        """Prepara dados detalhados de performance"""
        
        return {
            'returns': {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'monthly_avg': np.mean(list(result.monthly_returns.values())) if result.monthly_returns else 0,
                'best_month': max(result.monthly_returns.values()) if result.monthly_returns else 0,
                'worst_month': min(result.monthly_returns.values()) if result.monthly_returns else 0
            },
            'risk_metrics': {
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': self._calculate_volatility(result.equity_curve),
                'var_95': self._calculate_var(result.equity_curve),
                'calmar_ratio': self._calculate_calmar_ratio(result.annual_return, result.max_drawdown)
            },
            'profit_loss': {
                'net_profit': result.net_profit,
                'gross_profit': result.gross_profit,
                'gross_loss': result.gross_loss,
                'profit_factor': result.profit_factor,
                'recovery_factor': self._calculate_recovery_factor(result.net_profit, result.max_drawdown)
            }
        }
    
    def _prepare_trade_analysis(self, result: BacktestResult) -> Dict:
        """Prepara análise detalhada dos trades"""
        
        if not result.trade_history:
            return {
                'summary': {'total_trades': 0},
                'wins_vs_losses': {},
                'duration_analysis': {},
                'monthly_distribution': {}
            }
        
        trades_df = pd.DataFrame(result.trade_history)
        
        # Análise básica
        winning_trades = trades_df[trades_df['realized_pnl'] > 0]
        losing_trades = trades_df[trades_df['realized_pnl'] < 0]
        
        summary = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': result.win_rate,
            'avg_win': result.avg_win,
            'avg_loss': result.avg_loss,
            'largest_win': result.largest_win,
            'largest_loss': result.largest_loss,
            'avg_trade': trades_df['realized_pnl'].mean(),
            'consecutive_wins': self._calculate_consecutive_wins(trades_df),
            'consecutive_losses': self._calculate_consecutive_losses(trades_df)
        }
        
        # Análise de duração
        if 'duration_hours' in trades_df.columns:
            duration_analysis = {
                'avg_duration_hours': trades_df['duration_hours'].mean(),
                'avg_winning_duration': winning_trades['duration_hours'].mean() if len(winning_trades) > 0 else 0,
                'avg_losing_duration': losing_trades['duration_hours'].mean() if len(losing_trades) > 0 else 0,
                'shortest_trade': trades_df['duration_hours'].min(),
                'longest_trade': trades_df['duration_hours'].max()
            }
        else:
            duration_analysis = {}
        
        # Distribuição mensal
        if 'timestamp' in trades_df.columns:
            trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
            monthly_stats = trades_df.groupby('month').agg({
                'realized_pnl': ['count', 'sum', 'mean'],
            }).round(2)
            monthly_distribution = monthly_stats.to_dict()
        else:
            monthly_distribution = {}
        
        return {
            'summary': summary,
            'wins_vs_losses': {
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss,
                'win_loss_ratio': abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0
            },
            'duration_analysis': duration_analysis,
            'monthly_distribution': monthly_distribution
        }
    
    def _prepare_strategy_analysis(self, result: BacktestResult) -> Dict:
        """Prepara análise por estratégia"""
        
        strategy_data = {}
        
        for strategy, performance in result.strategy_performance.items():
            strategy_data[strategy] = {
                'trades': performance['total_trades'],
                'win_rate': performance['win_rate'],
                'total_pnl': performance['total_pnl'],
                'avg_pnl': performance['avg_pnl'],
                'profit_factor': performance['profit_factor'],
                'best_trade': performance['best_trade'],
                'worst_trade': performance['worst_trade'],
                'contribution': (performance['total_pnl'] / result.net_profit * 100) if result.net_profit != 0 else 0
            }
        
        return strategy_data
    
    def _generate_all_charts(self, result: BacktestResult) -> Dict[str, str]:
        """Gera todos os gráficos necessários"""
        
        charts = {}
        
        try:
            # Gráfico da curva de equity
            charts['equity_curve'] = self._generate_equity_curve_chart(result)
            
            # Gráfico de drawdown
            charts['drawdown_chart'] = self._generate_drawdown_chart(result)
            
            # Distribuição de retornos
            charts['returns_distribution'] = self._generate_returns_distribution(result)
            
            # Retornos mensais
            charts['monthly_returns'] = self._generate_monthly_returns_chart(result)
            
            # Performance por estratégia
            charts['strategy_performance'] = self._generate_strategy_performance_chart(result)
            
            # Análise de trades
            charts['trade_analysis'] = self._generate_trade_analysis_chart(result)
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos: {e}")
        
        return charts
    
    def _generate_equity_curve_chart(self, result: BacktestResult) -> str:
        """Gera gráfico da curva de equity"""
        
        if not result.equity_curve:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Cria datas para o eixo X
        if result.start_date and result.end_date:
            dates = pd.date_range(
                start=result.start_date,
                end=result.end_date,
                periods=len(result.equity_curve)
            )
        else:
            dates = range(len(result.equity_curve))
        
        # Plota curva de equity
        ax.plot(dates, result.equity_curve, linewidth=2, color=self.report_style['primary_color'])
        ax.fill_between(dates, result.equity_curve, alpha=0.3, color=self.report_style['primary_color'])
        
        # Linha de referência (capital inicial)
        if result.equity_curve:
            initial_value = result.equity_curve[0]
            ax.axhline(y=initial_value, color=self.report_style['danger_color'], 
                      linestyle='--', alpha=0.7, label='Capital Inicial')
        
        ax.set_title('Curva de Equity', fontsize=16, fontweight='bold')
        ax.set_xlabel('Período')
        ax.set_ylabel('Valor do Portfolio')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Formata eixo Y como moeda
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_currency(x)))
        
        # Formata eixo X se são datas
        if isinstance(dates[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _generate_drawdown_chart(self, result: BacktestResult) -> str:
        """Gera gráfico de drawdown"""
        
        if not result.equity_curve:
            return ""
        
        # Calcula drawdown
        equity_series = pd.Series(result.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = ((equity_series - running_max) / running_max) * 100
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Cria datas para o eixo X
        if result.start_date and result.end_date:
            dates = pd.date_range(
                start=result.start_date,
                end=result.end_date,
                periods=len(drawdown)
            )
        else:
            dates = range(len(drawdown))
        
        # Plota drawdown
        ax.fill_between(dates, drawdown, 0, alpha=0.7, color=self.report_style['danger_color'])
        ax.plot(dates, drawdown, linewidth=1, color=self.report_style['danger_color'])
        
        ax.set_title('Drawdown', fontsize=16, fontweight='bold')
        ax.set_xlabel('Período')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Linha no máximo drawdown
        max_dd = drawdown.min()
        ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.8, 
                  label=f'Max DD: {max_dd:.2f}%')
        ax.legend()
        
        # Formata eixo X se são datas
        if isinstance(dates[0], datetime):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _generate_returns_distribution(self, result: BacktestResult) -> str:
        """Gera distribuição de retornos"""
        
        if not result.trade_history:
            return ""
        
        trades_df = pd.DataFrame(result.trade_history)
        returns = trades_df['realized_pnl']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma
        ax1.hist(returns, bins=30, alpha=0.7, color=self.report_style['primary_color'], edgecolor='black')
        ax1.axvline(returns.mean(), color=self.report_style['danger_color'], 
                   linestyle='--', label=f'Média: {format_currency(returns.mean())}')
        ax1.set_title('Distribuição dos Retornos')
        ax1.set_xlabel('P&L por Trade')
        ax1.set_ylabel('Frequência')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        box_data = [returns[returns > 0], returns[returns < 0]]
        labels = ['Trades Vencedores', 'Trades Perdedores']
        colors = [self.report_style['success_color'], self.report_style['danger_color']]
        
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('Box Plot: Wins vs Losses')
        ax2.set_ylabel('P&L')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _generate_monthly_returns_chart(self, result: BacktestResult) -> str:
        """Gera gráfico de retornos mensais"""
        
        if not result.monthly_returns:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        months = list(result.monthly_returns.keys())
        returns = list(result.monthly_returns.values())
        
        # Cores baseadas no sinal do retorno
        colors = [self.report_style['success_color'] if r >= 0 
                 else self.report_style['danger_color'] for r in returns]
        
        bars = ax.bar(months, returns, color=colors, alpha=0.7, edgecolor='black')
        
        # Linha de referência em zero
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_title('Retornos Mensais', fontsize=16, fontweight='bold')
        ax.set_xlabel('Mês')
        ax.set_ylabel('Retorno (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                   f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _generate_strategy_performance_chart(self, result: BacktestResult) -> str:
        """Gera gráfico de performance por estratégia"""
        
        if not result.strategy_performance:
            return ""
        
        strategies = list(result.strategy_performance.keys())
        pnls = [result.strategy_performance[s]['total_pnl'] for s in strategies]
        win_rates = [result.strategy_performance[s]['win_rate'] for s in strategies]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # P&L por estratégia
        colors = [self.report_style['success_color'] if p >= 0 
                 else self.report_style['danger_color'] for p in pnls]
        
        bars1 = ax1.bar(strategies, pnls, color=colors, alpha=0.7)
        ax1.set_title('P&L por Estratégia')
        ax1.set_ylabel('P&L Total')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars1, pnls):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (height*0.01 if height >= 0 else height*0.01),
                    format_currency(value), ha='center', va='bottom' if height >= 0 else 'top')
        
        # Win rate por estratégia
        bars2 = ax2.bar(strategies, win_rates, color=self.report_style['info_color'], alpha=0.7)
        ax2.set_title('Win Rate por Estratégia')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adiciona valores nas barras
        for bar, value in zip(bars2, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _generate_trade_analysis_chart(self, result: BacktestResult) -> str:
        """Gera análise visual dos trades"""
        
        if not result.trade_history:
            return ""
        
        trades_df = pd.DataFrame(result.trade_history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. P&L cumulativo por trade
        cumulative_pnl = trades_df['realized_pnl'].cumsum()
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                color=self.report_style['primary_color'], linewidth=2)
        ax1.set_title('P&L Cumulativo por Trade')
        ax1.set_xlabel('Número do Trade')
        ax1.set_ylabel('P&L Cumulativo')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuição de P&L por símbolo (se disponível)
        if 'symbol' in trades_df.columns:
            symbol_pnl = trades_df.groupby('symbol')['realized_pnl'].sum().sort_values(ascending=False)
            colors = [self.report_style['success_color'] if p >= 0 
                     else self.report_style['danger_color'] for p in symbol_pnl.values]
            ax2.bar(range(len(symbol_pnl)), symbol_pnl.values, color=colors, alpha=0.7)
            ax2.set_title('P&L por Símbolo')
            ax2.set_xlabel('Símbolos')
            ax2.set_ylabel('P&L Total')
            ax2.set_xticks(range(len(symbol_pnl)))
            ax2.set_xticklabels(symbol_pnl.index, rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Dados de símbolo\nnão disponíveis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('P&L por Símbolo')
        
        # 3. Distribuição de duração dos trades (se disponível)
        if 'duration_hours' in trades_df.columns:
            ax3.hist(trades_df['duration_hours'], bins=20, alpha=0.7, 
                    color=self.report_style['info_color'], edgecolor='black')
            ax3.set_title('Distribuição de Duração dos Trades')
            ax3.set_xlabel('Duração (horas)')
            ax3.set_ylabel('Frequência')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Dados de duração\nnão disponíveis', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Duração dos Trades')
        
        # 4. Scatter plot: Duração vs P&L (se disponível)
        if 'duration_hours' in trades_df.columns:
            wins = trades_df[trades_df['realized_pnl'] > 0]
            losses = trades_df[trades_df['realized_pnl'] <= 0]
            
            if len(wins) > 0:
                ax4.scatter(wins['duration_hours'], wins['realized_pnl'], 
                           color=self.report_style['success_color'], alpha=0.6, label='Wins')
            if len(losses) > 0:
                ax4.scatter(losses['duration_hours'], losses['realized_pnl'], 
                           color=self.report_style['danger_color'], alpha=0.6, label='Losses')
            
            ax4.set_title('Duração vs P&L')
            ax4.set_xlabel('Duração (horas)')
            ax4.set_ylabel('P&L')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Análise de duração\nnão disponível', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Duração vs P&L')
        
        plt.tight_layout()
        
        return self._chart_to_base64(fig)
    
    def _chart_to_base64(self, fig) -> str:
        """Converte gráfico matplotlib para base64"""
        
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            plt.close(fig)
            
            return f"data:image/png;base64,{image_base64}"
        
        except Exception as e:
            logger.error(f"Erro ao converter gráfico: {e}")
            plt.close(fig)
            return ""
    
    def _calculate_volatility(self, equity_curve: List[float]) -> float:
        """Calcula volatilidade da curva de equity"""
        
        if len(equity_curve) < 2:
            return 0.0
        
        returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            for i in range(1, len(equity_curve))
        ]
        
        return np.std(returns) * np.sqrt(252) * 100  # Anualizada em %
    
    def _calculate_var(self, equity_curve: List[float], confidence: float = 0.05) -> float:
        """Calcula Value at Risk"""
        
        if len(equity_curve) < 2:
            return 0.0
        
        returns = [
            (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            for i in range(1, len(equity_curve))
        ]
        
        return np.percentile(returns, confidence * 100) * 100  # Em %
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calcula Calmar Ratio"""
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0
        
        return annual_return / abs(max_drawdown)
    
    def _calculate_recovery_factor(self, net_profit: float, max_drawdown: float) -> float:
        """Calcula Recovery Factor"""
        
        if max_drawdown == 0:
            return float('inf') if net_profit > 0 else 0
        
        return abs(net_profit / max_drawdown)
    
    def _calculate_consecutive_wins(self, trades_df: pd.DataFrame) -> int:
        """Calcula máximo de vitórias consecutivas"""
        
        if trades_df.empty:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for pnl in trades_df['realized_pnl']:
            if pnl > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calcula máximo de perdas consecutivas"""
        
        if trades_df.empty:
            return 0
        
        consecutive = 0
        max_consecutive = 0
        
        for pnl in trades_df['realized_pnl']:
            if pnl < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def _generate_html_report(
        self,
        summary_data: Dict,
        performance_data: Dict,
        trade_analysis: Dict,
        strategy_analysis: Dict,
        charts: Dict
    ) -> str:
        """Gera relatório HTML final"""
        
        html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Backtesting</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: {{ style.background_color }}; }
        .header { background: linear-gradient(135deg, {{ style.primary_color }}, {{ style.info_color }}); color: white; padding: 2rem 0; }
        .metric-card { background: white; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2rem; font-weight: bold; }
        .chart-container { background: white; border-radius: 10px; padding: 1rem; margin-bottom: 2rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .performance-{{ summary.performance.class }} { color: white; }
        .bg-{{ summary.performance.class }} { background-color: {{ style.success_color if summary.performance.class == 'success' else style.danger_color if summary.performance.class == 'danger' else style.warning_color }}!important; }
        .table-hover tbody tr:hover { background-color: {{ style.info_color }}20; }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1 class="mb-0">📊 Relatório de Backtesting</h1>
                    <p class="mb-0">Análise Completa de Performance</p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="bg-{{ summary.performance.class }} p-3 rounded">
                        <h3 class="mb-0 performance-{{ summary.performance.class }}">{{ "%.2f"|format(summary.performance.total_return) }}%</h3>
                        <small class="performance-{{ summary.performance.class }}">Retorno Total</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container my-5">
        <!-- Resumo Executivo -->
        <div class="row mb-4">
            <div class="col-12">
                <h2>📋 Resumo Executivo</h2>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-primary">{{ "%.2f"|format(summary.performance.total_return) }}%</div>
                    <div class="text-muted">Retorno Total</div>
                    <small class="badge bg-{{ summary.performance.class }}">{{ summary.performance.rating }}</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-info">{{ "%.2f"|format(summary.risk.sharpe_ratio) }}</div>
                    <div class="text-muted">Sharpe Ratio</div>
                    <small class="text-secondary">{{ summary.risk.sharpe_rating }}</small>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-danger">{{ "%.2f"|format(summary.risk.max_drawdown) }}%</div>
                    <div class="text-muted">Max Drawdown</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value text-success">{{ "%.1f"|format(summary.trades.win_rate) }}%</div>
                    <div class="text-muted">Win Rate</div>
                    <small class="text-secondary">{{ summary.trades.total }} trades</small>
                </div>
            </div>
        </div>

        <!-- Período e Informações Básicas -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="metric-card">
                    <h5>📅 Período de Análise</h5>
                    <div class="row">
                        <div class="col-md-4">
                            <strong>Início:</strong> {{ summary.period.start }}
                        </div>
                        <div class="col-md-4">
                            <strong>Fim:</strong> {{ summary.period.end }}
                        </div>
                        <div class="col-md-4">
                            <strong>Dias:</strong> {{ summary.period.days }}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Gráficos Principais -->
        {% if charts.equity_curve %}
        <div class="chart-container">
            <h5>📈 Curva de Equity</h5>
            <img src="{{ charts.equity_curve }}" class="img-fluid" alt="Curva de Equity">
        </div>
        {% endif %}

        {% if charts.drawdown_chart %}
        <div class="chart-container">
            <h5>📉 Análise de Drawdown</h5>
            <img src="{{ charts.drawdown_chart }}" class="img-fluid" alt="Drawdown">
        </div>
        {% endif %}

        <!-- Métricas Detalhadas -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="metric-card">
                    <h5>💰 Métricas de Retorno</h5>
                    <table class="table table-sm">
                        <tr><td>Retorno Total</td><td class="text-end">{{ "%.2f"|format(performance.returns.total_return) }}%</td></tr>
                        <tr><td>Retorno Anualizado</td><td class="text-end">{{ "%.2f"|format(performance.returns.annual_return) }}%</td></tr>
                        <tr><td>Retorno Médio Mensal</td><td class="text-end">{{ "%.2f"|format(performance.returns.monthly_avg) }}%</td></tr>
                        <tr><td>Melhor Mês</td><td class="text-end text-success">{{ "%.2f"|format(performance.returns.best_month) }}%</td></tr>
                        <tr><td>Pior Mês</td><td class="text-end text-danger">{{ "%.2f"|format(performance.returns.worst_month) }}%</td></tr>
                    </table>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <h5>⚠️ Métricas de Risco</h5>
                    <table class="table table-sm">
                        <tr><td>Sharpe Ratio</td><td class="text-end">{{ "%.3f"|format(performance.risk_metrics.sharpe_ratio) }}</td></tr>
                        <tr><td>Sortino Ratio</td><td class="text-end">{{ "%.3f"|format(performance.risk_metrics.sortino_ratio) }}</td></tr>
                        <tr><td>Máximo Drawdown</td><td class="text-end text-danger">{{ "%.2f"|format(performance.risk_metrics.max_drawdown) }}%</td></tr>
                        <tr><td>Volatilidade</td><td class="text-end">{{ "%.2f"|format(performance.risk_metrics.volatility) }}%</td></tr>
                        <tr><td>Calmar Ratio</td><td class="text-end">{{ "%.3f"|format(performance.risk_metrics.calmar_ratio) }}</td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- Análise de Trades -->
        <div class="row mb-4">
            <div class="col-12">
                <h3>🎯 Análise de Trades</h3>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-4">
                <div class="metric-card">
                    <h6>📊 Estatísticas Gerais</h6>
                    <table class="table table-sm">
                        <tr><td>Total de Trades</td><td class="text-end">{{ trade_analysis.summary.total_trades }}</td></tr>
                        <tr><td>Trades Vencedores</td><td class="text-end text-success">{{ trade_analysis.summary.winning_trades }}</td></tr>
                        <tr><td>Trades Perdedores</td><td class="text-end text-danger">{{ trade_analysis.summary.losing_trades }}</td></tr>
                        <tr><td>Win Rate</td><td class="text-end">{{ "%.1f"|format(trade_analysis.summary.win_rate) }}%</td></tr>
                    </table>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h6>💵 P&L Médio</h6>
                    <table class="table table-sm">
                        <tr><td>Trade Médio</td><td class="text-end">{{ "${:,.2f}"|format(trade_analysis.summary.avg_trade) }}</td></tr>
                        <tr><td>Vitória Média</td><td class="text-end text-success">{{ "${:,.2f}"|format(trade_analysis.summary.avg_win) }}</td></tr>
                        <tr><td>Perda Média</td><td class="text-end text-danger">{{ "${:,.2f}"|format(trade_analysis.summary.avg_loss) }}</td></tr>
                        <tr><td>Maior Vitória</td><td class="text-end text-success">{{ "${:,.2f}"|format(trade_analysis.summary.largest_win) }}</td></tr>
                        <tr><td>Maior Perda</td><td class="text-end text-danger">{{ "${:,.2f}"|format(trade_analysis.summary.largest_loss) }}</td></tr>
                    </table>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <h6>🔄 Sequências</h6>
                    <table class="table table-sm">
                        <tr><td>Vitórias Consecutivas</td><td class="text-end">{{ trade_analysis.summary.consecutive_wins }}</td></tr>
                        <tr><td>Perdas Consecutivas</td><td class="text-end">{{ trade_analysis.summary.consecutive_losses }}</td></tr>
                        {% if trade_analysis.duration_analysis %}
                        <tr><td>Duração Média</td><td class="text-end">{{ "%.1f"|format(trade_analysis.duration_analysis.avg_duration_hours) }}h</td></tr>
                        {% endif %}
                    </table>
                </div>
            </div>
        </div>

        <!-- Performance por Estratégia -->
        {% if strategy_analysis %}
        <div class="row mb-4">
            <div class="col-12">
                <h3>🎲 Performance por Estratégia</h3>
                <div class="metric-card">
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th>Estratégia</th>
                                    <th class="text-center">Trades</th>
                                    <th class="text-center">Win Rate</th>
                                    <th class="text-center">P&L Total</th>
                                    <th class="text-center">P&L Médio</th>
                                    <th class="text-center">Profit Factor</th>
                                    <th class="text-center">Contribuição</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for strategy, data in strategy_analysis.items() %}
                                <tr>
                                    <td><strong>{{ strategy.title() }}</strong></td>
                                    <td class="text-center">{{ data.trades }}</td>
                                    <td class="text-center">{{ "%.1f"|format(data.win_rate) }}%</td>
                                    <td class="text-center {% if data.total_pnl >= 0 %}text-success{% else %}text-danger{% endif %}">
                                        {{ "${:,.2f}"|format(data.total_pnl) }}
                                    </td>
                                    <td class="text-center">{{ "${:,.2f}"|format(data.avg_pnl) }}</td>
                                    <td class="text-center">{{ "%.2f"|format(data.profit_factor) }}</td>
                                    <td class="text-center">{{ "%.1f"|format(data.contribution) }}%</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        {% endif %}

        <!-- Gráficos Adicionais -->
        <div class="row">
            {% if charts.monthly_returns %}
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>📅 Retornos Mensais</h5>
                    <img src="{{ charts.monthly_returns }}" class="img-fluid" alt="Retornos Mensais">
                </div>
            </div>
            {% endif %}
            
            {% if charts.returns_distribution %}
            <div class="col-md-6">
                <div class="chart-container">
                    <h5>📊 Distribuição de Retornos</h5>
                    <img src="{{ charts.returns_distribution }}" class="img-fluid" alt="Distribuição de Retornos">
                </div>
            </div>
            {% endif %}
        </div>

        {% if charts.strategy_performance %}
        <div class="chart-container">
            <h5>🎲 Performance das Estratégias</h5>
            <img src="{{ charts.strategy_performance }}" class="img-fluid" alt="Performance das Estratégias">
        </div>
        {% endif %}

        {% if charts.trade_analysis %}
        <div class="chart-container">
            <h5>🔍 Análise Detalhada dos Trades</h5>
            <img src="{{ charts.trade_analysis }}" class="img-fluid" alt="Análise de Trades">
        </div>
        {% endif %}

        <!-- Footer -->
        <div class="text-center mt-5 py-4 border-top">
            <small class="text-muted">
                Relatório gerado automaticamente pelo Smart Trading System v2.0<br>
                Data: {{ datetime.now().strftime('%d/%m/%Y %H:%M:%S') }}
            </small>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        
        template = Template(html_template)
        
        return template.render(
            summary=summary_data,
            performance=performance_data,
            trade_analysis=trade_analysis,
            strategy_analysis=strategy_analysis,
            charts=charts,
            style=self.report_style,
            datetime=datetime
        )


# Função de conveniência
def generate_backtest_report(
    result: BacktestResult,
    output_path: str = "backtest_report.html",
    include_charts: bool = True
) -> str:
    """Gera relatório de backtesting completo"""
    
    generator = ReportGenerator()
    return generator.generate_full_report(result, output_path, include_charts)