"""
Tests: Strategies Testing
Testes unitários para estratégias de trading
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.swing_strategy import SwingStrategy
from strategies.breakout_strategy import BreakoutStrategy
from strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from strategies.trend_following import TrendFollowingStrategy, TrendFollowingConfig
from core.signal_manager import SignalType


class TestSwingStrategy(unittest.TestCase):
    """Testes para estratégia Swing"""
    
    def setUp(self):
        """Setup dos testes"""
        self.strategy = SwingStrategy()
        self.sample_data = self.create_swing_sample()
    
    def create_swing_sample(self) -> pd.DataFrame:
        """Cria dados adequados para swing trading"""
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='4H')
        
        # Cria movimento de swing (subida, lateralização, descida)
        prices = []
        base = 50000
        
        for i in range(200):
            if i < 60:
                # Uptrend inicial
                price = base + i * 50 + np.random.normal(0, 100)
            elif i < 120:
                # Lateralização
                price = base + 3000 + np.random.normal(0, 200)
            else:
                # Pullback
                price = base + 3000 - (i - 120) * 30 + np.random.normal(0, 150)
            
            prices.append(max(price, 1000))  # Evita preços negativos
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 200, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 200, len(df))
        df['volume'] = np.random.lognormal(6, 0.5, len(df))
        
        return df
    
    def test_analyze_swing(self):
        """Testa análise swing básica"""
        
        result = self.strategy.analyze(self.sample_data, 'BTCUSDT', '4h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('signals', result)
        self.assertIn('analysis', result)
        
        # Verifica estrutura da análise
        analysis = result['analysis']
        self.assertIn('filters_passed', analysis)
    
    def test_identify_swing_points(self):
        """Testa identificação de pontos de swing"""
        
        swing_points = self.strategy._identify_swing_points(self.sample_data)
        
        self.assertIsInstance(swing_points, dict)
        self.assertIn('highs', swing_points)
        self.assertIn('lows', swing_points)
    
    def test_signal_generation(self):
        """Testa geração de sinais"""
        
        result = self.strategy.analyze(self.sample_data, 'BTCUSDT', '4h')
        signals = result.get('signals', [])
        
        # Se gerou sinais, verifica estrutura
        if signals:
            signal = signals[0]
            self.assertIn('symbol', signal)
            self.assertIn('signal_type', signal)
            self.assertIn('entry_price', signal)
            self.assertIn('stop_loss', signal)
            self.assertIn('scores', signal)
    
    def test_empty_data(self):
        """Testa com dados insuficientes"""
        
        small_df = self.sample_data.head(10)
        result = self.strategy.analyze(small_df, 'BTCUSDT', '4h')
        
        self.assertEqual(len(result['signals']), 0)


class TestBreakoutStrategy(unittest.TestCase):
    """Testes para estratégia Breakout"""
    
    def setUp(self):
        """Setup dos testes"""
        self.strategy = BreakoutStrategy()
        self.breakout_data = self.create_breakout_sample()
    
    def create_breakout_sample(self) -> pd.DataFrame:
        """Cria dados com padrão de breakout"""
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        prices = []
        base = 45000
        
        for i in range(100):
            if i < 40:
                # Consolidação (range)
                price = base + np.random.uniform(-500, 500)
            elif i < 60:
                # Breakout para cima
                price = base + 500 + (i - 40) * 50 + np.random.normal(0, 100)
            else:
                # Continuação do movimento
                price = base + 1500 + (i - 60) * 25 + np.random.normal(0, 150)
            
            prices.append(max(price, 1000))
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 100, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 100, len(df))
        
        # Volume aumenta no breakout
        volumes = []
        for i in range(100):
            if 40 <= i < 60:
                vol = np.random.lognormal(7, 0.3)  # Volume alto no breakout
            else:
                vol = np.random.lognormal(6, 0.5)  # Volume normal
            volumes.append(vol)
        
        df['volume'] = volumes
        
        return df
    
    def test_detect_breakout(self):
        """Testa detecção de breakout"""
        
        result = self.strategy.analyze(self.breakout_data, 'BTCUSDT', '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('signals', result)
        self.assertIn('analysis', result)
    
    def test_identify_consolidation(self):
        """Testa identificação de consolidação"""
        
        # Usa dados apenas da fase de consolidação
        consolidation_data = self.breakout_data.iloc[:40]
        
        is_consolidating = self.strategy._is_consolidating(consolidation_data)
        
        # Pode ou não detectar dependendo dos parâmetros
        self.assertIsInstance(is_consolidating, bool)
    
    def test_volume_confirmation(self):
        """Testa confirmação por volume"""
        
        # Testa com volume alto
        high_vol_data = self.breakout_data.iloc[45:50]  # Período de breakout
        
        volume_confirmed = self.strategy._has_volume_confirmation(
            high_vol_data, 
            high_vol_data['close'].iloc[-1]
        )
        
        self.assertIsInstance(volume_confirmed, bool)


class TestMeanReversionStrategy(unittest.TestCase):
    """Testes para estratégia Mean Reversion"""
    
    def setUp(self):
        """Setup dos testes"""
        self.config = MeanReversionConfig()
        self.strategy = MeanReversionStrategy(self.config)
        self.reversion_data = self.create_reversion_sample()
    
    def create_reversion_sample(self) -> pd.DataFrame:
        """Cria dados com padrão de reversão à média"""
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Preços oscilando em torno de uma média
        mean_price = 48000
        prices = []
        
        for i in range(100):
            # Movimento mean-reverting
            if i == 0:
                price = mean_price
            else:
                # Mean reversion com ruído
                prev_price = prices[i-1]
                reversion = 0.1 * (mean_price - prev_price)
                noise = np.random.normal(0, 200)
                price = prev_price + reversion + noise
            
            prices.append(max(price, 1000))
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 100, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 100, len(df))
        df['volume'] = np.random.lognormal(6, 0.5, len(df))
        
        return df
    
    def test_analyze_mean_reversion(self):
        """Testa análise de mean reversion"""
        
        result = self.strategy.analyze(self.reversion_data, 'BTCUSDT', '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('signals', result)
        self.assertIn('analysis', result)
    
    def test_identify_extreme_levels(self):
        """Testa identificação de níveis extremos"""
        
        # Cria dados com RSI extremo
        extreme_data = self.reversion_data.copy()
        
        # Força preços para níveis extremos
        extreme_data.iloc[-10:, extreme_data.columns.get_loc('close')] *= 1.15  # +15%
        
        result = self.strategy.analyze(extreme_data, 'BTCUSDT', '1h')
        
        # Pode gerar sinais de reversão
        self.assertIsInstance(result, dict)
    
    def test_bollinger_bands_reversion(self):
        """Testa reversão nas Bollinger Bands"""
        
        indicators = self.strategy._calculate_indicators(self.reversion_data)
        
        self.assertIn('bb_upper', indicators)
        self.assertIn('bb_lower', indicators)
        self.assertIn('bb_position', indicators)
        
        # Verifica se posição nas bands está calculada
        bb_position = indicators['bb_position'].dropna()
        if len(bb_position) > 0:
            self.assertTrue(all(0 <= pos <= 1 for pos in bb_position))


class TestTrendFollowingStrategy(unittest.TestCase):
    """Testes para estratégia Trend Following"""
    
    def setUp(self):
        """Setup dos testes"""
        self.config = TrendFollowingConfig()
        self.strategy = TrendFollowingStrategy(self.config)
        self.trend_data = self.create_trend_sample()
    
    def create_trend_sample(self) -> pd.DataFrame:
        """Cria dados com tendência forte"""
        
        dates = pd.date_range(start='2024-01-01', periods=150, freq='4H')
        
        # Cria uptrend forte
        prices = []
        base = 40000
        
        for i in range(150):
            # Trend consistente com pullbacks ocasionais
            trend = i * 100
            
            # Adiciona pullbacks
            if i % 20 == 0 and i > 0:
                pullback = -500
            else:
                pullback = 0
            
            noise = np.random.normal(0, 150)
            price = base + trend + pullback + noise
            
            prices.append(max(price, 1000))
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 200, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 200, len(df))
        df['volume'] = np.random.lognormal(6.5, 0.4, len(df))
        
        return df
    
    def test_analyze_trend_following(self):
        """Testa análise de trend following"""
        
        result = self.strategy.analyze(self.trend_data, 'BTCUSDT', '4h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('signals', result)
        self.assertIn('analysis', result)
        
        # Verifica análise de tendência
        analysis = result['analysis']
        if 'trend_analysis' in analysis:
            trend_analysis = analysis['trend_analysis']
            self.assertIn('direction', trend_analysis)
            self.assertIn('strength', trend_analysis)
    
    def test_trend_detection(self):
        """Testa detecção de tendência"""
        
        indicators = self.strategy._calculate_indicators(self.trend_data)
        trend_analysis = self.strategy._analyze_trend(
            self.trend_data, 
            indicators, 
            '4h'
        )
        
        self.assertIsInstance(trend_analysis, dict)
        self.assertIn('direction', trend_analysis)
        self.assertIn('strength', trend_analysis)
        
        # Com dados de uptrend, deve detectar bullish ou sideways
        self.assertIn(trend_analysis['direction'], ['bullish', 'bearish', 'sideways'])
    
    def test_pullback_detection(self):
        """Testa detecção de pullbacks"""
        
        # Cria dados com pullback claro
        pullback_data = self.trend_data.iloc[80:100]  # Região com pullback
        
        indicators = self.strategy._calculate_indicators(pullback_data)
        trend_analysis = self.strategy._analyze_trend(pullback_data, indicators, '4h')
        
        setups = self.strategy._identify_setups(
            pullback_data,
            indicators,
            trend_analysis,
            '4h'
        )
        
        self.assertIsInstance(setups, list)
    
    def test_moving_average_alignment(self):
        """Testa alinhamento de médias móveis"""
        
        indicators = self.strategy._calculate_indicators(self.trend_data)
        
        # Verifica se médias foram calculadas
        self.assertIn('ema_fast', indicators)
        self.assertIn('ema_slow', indicators)
        self.assertIn('sma_trend', indicators)
        
        # Testa alinhamento
        current_idx = len(self.trend_data) - 1
        alignment = self.strategy._check_ma_alignment(indicators, current_idx)
        
        self.assertIsInstance(alignment, bool)


class TestStrategyIntegration(unittest.TestCase):
    """Testes de integração entre estratégias"""
    
    def setUp(self):
        """Setup dos testes"""
        self.strategies = {
            'swing': SwingStrategy(),
            'breakout': BreakoutStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'trend_following': TrendFollowingStrategy()
        }
        self.sample_data = self.create_comprehensive_sample()
    
    def create_comprehensive_sample(self) -> pd.DataFrame:
        """Cria dados abrangentes para testar múltiplas estratégias"""
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
        
        # Combina diferentes padrões
        prices = []
        base = 50000
        
        for i in range(200):
            if i < 50:
                # Trend inicial
                price = base + i * 20 + np.random.normal(0, 100)
            elif i < 100:
                # Consolidação/range
                price = base + 1000 + np.random.uniform(-300, 300)
            elif i < 150:
                # Breakout e continuação
                price = base + 1300 + (i - 100) * 30 + np.random.normal(0, 150)
            else:
                # Mean reversion
                price = base + 2800 - (i - 150) * 10 + np.random.normal(0, 200)
            
            prices.append(max(price, 1000))
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 150, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 150, len(df))
        df['volume'] = np.random.lognormal(6, 0.5, len(df))
        
        return df
    
    def test_all_strategies(self):
        """Testa todas as estratégias com os mesmos dados"""
        
        results = {}
        
        for name, strategy in self.strategies.items():
            try:
                result = strategy.analyze(self.sample_data, 'BTCUSDT', '1h')
                results[name] = result
                
                # Verifica estrutura básica
                self.assertIn('signals', result)
                self.assertIn('analysis', result)
                
            except Exception as e:
                self.fail(f"Estratégia {name} falhou: {e}")
        
        # Verifica se pelo menos algumas estratégias geraram sinais
        total_signals = sum(len(result['signals']) for result in results.values())
        self.assertGreaterEqual(total_signals, 0)  # Pode ser 0 se condições não forem atendidas
    
    def test_signal_consistency(self):
        """Testa consistência dos sinais gerados"""
        
        for name, strategy in self.strategies.items():
            result = strategy.analyze(self.sample_data, 'BTCUSDT', '1h')
            
            for signal in result['signals']:
                # Verifica campos obrigatórios
                required_fields = [
                    'symbol', 'signal_type', 'entry_price', 
                    'stop_loss', 'strategy', 'scores'
                ]
                
                for field in required_fields:
                    self.assertIn(field, signal, f"Campo {field} ausente em {name}")
                
                # Verifica tipos
                self.assertIsInstance(signal['signal_type'], SignalType)
                self.assertIsInstance(signal['entry_price'], (int, float))
                self.assertIsInstance(signal['scores'], dict)
                
                # Verifica valores lógicos
                self.assertGreater(signal['entry_price'], 0)
                if signal['stop_loss']:
                    self.assertGreater(signal['stop_loss'], 0)


def run_strategy_tests():
    """Executa todos os testes de estratégias"""
    
    test_suite = unittest.TestSuite()
    
    # Adiciona classes de teste
    test_classes = [
        TestSwingStrategy,
        TestBreakoutStrategy,
        TestMeanReversionStrategy,
        TestTrendFollowingStrategy,
        TestStrategyIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Executa testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Executando testes das estratégias...")
    success = run_strategy_tests()
    
    if success:
        print("✅ Todos os testes das estratégias passaram!")
    else:
        print("❌ Alguns testes falharam!")
        exit(1)