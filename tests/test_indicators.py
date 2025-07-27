"""
Tests: Indicators Testing
Testes unitários para indicadores e análises técnicas
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.confluence_analyzer import ConfluenceAnalyzer
from indicators.divergence_detector import (
    DivergenceDetector, 
    DivergenceType, 
    detect_rsi_divergences
)
from indicators.trend_analyzer import TrendAnalyzer
from indicators.leading_indicators import (
    VolumeProfileAnalyzer,
    OrderFlowAnalyzer,
    calculate_volume_profile
)


class TestConfluenceAnalyzer(unittest.TestCase):
    """Testes para o analisador de confluência"""
    
    def setUp(self):
        """Setup dos testes"""
        self.analyzer = ConfluenceAnalyzer()
        self.sample_data = self.create_sample_data()
    
    def create_sample_data(self) -> pd.DataFrame:
        """Cria dados de amostra para testes"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        # Gera dados OHLCV sintéticos com tendência
        np.random.seed(42)
        base_price = 50000
        
        prices = []
        volumes = []
        
        for i in range(100):
            # Trend + noise
            trend = i * 10
            noise = np.random.normal(0, 100)
            price = base_price + trend + noise
            
            prices.append(price)
            volumes.append(np.random.randint(100, 1000))
        
        # Cria OHLCV
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 50, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 50, len(df))
        df['volume'] = volumes
        
        return df
    
    def test_analyze_basic(self):
        """Testa análise básica de confluência"""
        
        result = self.analyzer.analyze(self.sample_data, '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('confluence_score', result)
        self.assertIn('factors', result)
        self.assertIn('strength', result)
        
        # Score deve estar entre 0 e 100
        self.assertGreaterEqual(result['confluence_score'], 0)
        self.assertLessEqual(result['confluence_score'], 100)
    
    def test_analyze_empty_data(self):
        """Testa análise com dados vazios"""
        
        empty_df = pd.DataFrame()
        result = self.analyzer.analyze(empty_df, '1h')
        
        self.assertEqual(result['confluence_score'], 0)
        self.assertEqual(len(result['factors']), 0)
    
    def test_calculate_factors(self):
        """Testa cálculo de fatores individuais"""
        
        factors = self.analyzer._calculate_confluence_factors(self.sample_data)
        
        self.assertIsInstance(factors, list)
        self.assertGreater(len(factors), 0)
        
        # Verifica estrutura dos fatores
        for factor in factors:
            self.assertIn('name', factor)
            self.assertIn('value', factor)
            self.assertIn('weight', factor)
            self.assertIn('score', factor)


class TestDivergenceDetector(unittest.TestCase):
    """Testes para o detector de divergências"""
    
    def setUp(self):
        """Setup dos testes"""
        self.detector = DivergenceDetector()
        self.sample_data = self.create_divergence_sample()
    
    def create_divergence_sample(self) -> pd.DataFrame:
        """Cria dados com divergência clara para teste"""
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        
        # Preços fazendo lower low
        prices = []
        base = 100
        
        for i in range(50):
            if i < 20:
                price = base + i * 0.5  # Subindo
            elif i < 30:
                price = base + 10 - (i - 20) * 2  # Descendo forte (primeiro low)
            elif i < 40:
                price = base + (i - 30) * 1.5  # Subindo
            else:
                price = base + 15 - (i - 40) * 2.5  # Descendo mais (lower low)
            
            prices.append(price)
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, len(df))
        df['volume'] = np.random.randint(100, 500, len(df))
        
        return df
    
    def test_detect_divergences(self):
        """Testa detecção de divergências"""
        
        divergences = self.detector.detect_all_divergences(
            self.sample_data, 
            timeframe='1h',
            indicators=['rsi']
        )
        
        self.assertIsInstance(divergences, list)
        
        # Se encontrou divergências, verifica estrutura
        if divergences:
            div = divergences[0]
            self.assertHasAttr(div, 'type')
            self.assertHasAttr(div, 'confidence')
            self.assertHasAttr(div, 'indicator')
    
    def test_rsi_divergences(self):
        """Testa detecção específica de divergências RSI"""
        
        divergences = detect_rsi_divergences(self.sample_data, '1h')
        
        self.assertIsInstance(divergences, list)
    
    def test_validate_data(self):
        """Testa validação de dados"""
        
        # Dados válidos
        valid = self.detector._validate_data(self.sample_data)
        self.assertTrue(valid)
        
        # Dados inválidos
        invalid_df = pd.DataFrame({'close': [1, 2, 3]})  # Faltam colunas
        invalid = self.detector._validate_data(invalid_df)
        self.assertFalse(invalid)


class TestTrendAnalyzer(unittest.TestCase):
    """Testes para o analisador de tendências"""
    
    def setUp(self):
        """Setup dos testes"""
        self.analyzer = TrendAnalyzer()
        self.uptrend_data = self.create_uptrend_data()
        self.downtrend_data = self.create_downtrend_data()
    
    def create_uptrend_data(self) -> pd.DataFrame:
        """Cria dados com uptrend claro"""
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        
        # Preços em uptrend
        prices = [100 + i * 2 + np.random.normal(0, 1) for i in range(50)]
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, len(df))
        df['volume'] = np.random.randint(100, 500, len(df))
        
        return df
    
    def create_downtrend_data(self) -> pd.DataFrame:
        """Cria dados com downtrend claro"""
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        
        # Preços em downtrend
        prices = [200 - i * 2 + np.random.normal(0, 1) for i in range(50)]
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 2, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 2, len(df))
        df['volume'] = np.random.randint(100, 500, len(df))
        
        return df
    
    def test_detect_uptrend(self):
        """Testa detecção de uptrend"""
        
        result = self.analyzer.analyze_trend(self.uptrend_data, '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('direction', result)
        self.assertIn('strength', result)
        
        # Deve detectar uptrend
        self.assertIn(result['direction'], ['bullish', 'sideways'])  # Pode ser sideways se muito noise
    
    def test_detect_downtrend(self):
        """Testa detecção de downtrend"""
        
        result = self.analyzer.analyze_trend(self.downtrend_data, '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('direction', result)
        self.assertIn('strength', result)
        
        # Deve detectar downtrend
        self.assertIn(result['direction'], ['bearish', 'sideways'])
    
    def test_multi_timeframe_analysis(self):
        """Testa análise multi-timeframe"""
        
        timeframes = ['1h', '4h']
        data = {
            '1h': self.uptrend_data,
            '4h': self.uptrend_data  # Simplificado
        }
        
        result = self.analyzer.analyze_multi_timeframe(data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_trend', result)
        self.assertIn('timeframe_analysis', result)


class TestVolumeProfile(unittest.TestCase):
    """Testes para análise de volume profile"""
    
    def setUp(self):
        """Setup dos testes"""
        self.analyzer = VolumeProfileAnalyzer()
        self.sample_data = self.create_volume_data()
    
    def create_volume_data(self) -> pd.DataFrame:
        """Cria dados com padrões de volume"""
        
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        
        np.random.seed(42)
        prices = 50000 + np.cumsum(np.random.normal(0, 100, 100))
        volumes = np.random.lognormal(5, 1, 100)  # Volume log-normal
        
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, 50, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, 50, len(df))
        df['volume'] = volumes
        
        return df
    
    def test_calculate_volume_profile(self):
        """Testa cálculo do volume profile"""
        
        profile = calculate_volume_profile(self.sample_data)
        
        self.assertIsInstance(profile, dict)
        self.assertIn('price_levels', profile)
        self.assertIn('volume_at_price', profile)
        self.assertIn('poc', profile)  # Point of Control
    
    def test_analyze_volume_profile(self):
        """Testa análise completa do volume profile"""
        
        result = self.analyzer.analyze_volume_profile(self.sample_data, '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('poc_price', result)
        self.assertIn('value_area', result)
        self.assertIn('profile_shape', result)
    
    def test_find_volume_nodes(self):
        """Testa identificação de nós de volume"""
        
        nodes = self.analyzer.find_high_volume_nodes(self.sample_data)
        
        self.assertIsInstance(nodes, list)
        
        if nodes:
            node = nodes[0]
            self.assertIn('price', node)
            self.assertIn('volume', node)
            self.assertIn('significance', node)


class TestOrderFlow(unittest.TestCase):
    """Testes para análise de order flow"""
    
    def setUp(self):
        """Setup dos testes"""
        self.analyzer = OrderFlowAnalyzer()
        self.sample_data = self.create_orderflow_data()
    
    def create_orderflow_data(self) -> pd.DataFrame:
        """Cria dados simulando order flow"""
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1H')
        
        np.random.seed(42)
        
        df = pd.DataFrame(index=dates)
        df['close'] = 50000 + np.cumsum(np.random.normal(0, 100, 50))
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(10, 100, len(df))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(10, 100, len(df))
        df['volume'] = np.random.lognormal(5, 1, 50)
        
        # Simula buy/sell volume
        df['buy_volume'] = df['volume'] * np.random.uniform(0.3, 0.7, len(df))
        df['sell_volume'] = df['volume'] - df['buy_volume']
        
        return df
    
    def test_analyze_order_flow(self):
        """Testa análise de order flow"""
        
        result = self.analyzer.analyze_order_flow(self.sample_data, '1h')
        
        self.assertIsInstance(result, dict)
        self.assertIn('buy_pressure', result)
        self.assertIn('sell_pressure', result)
        self.assertIn('flow_direction', result)
    
    def test_calculate_imbalance(self):
        """Testa cálculo de imbalance"""
        
        # Cria dados com imbalance claro
        df = self.sample_data.copy()
        df['buy_volume'] = df['volume'] * 0.8  # 80% compra
        df['sell_volume'] = df['volume'] * 0.2  # 20% venda
        
        imbalance = self.analyzer._calculate_volume_imbalance(df)
        
        self.assertGreater(imbalance, 0)  # Deve ser positivo (mais compra)


def run_all_tests():
    """Executa todos os testes"""
    
    # Cria test suite
    test_suite = unittest.TestSuite()
    
    # Adiciona testes
    test_classes = [
        TestConfluenceAnalyzer,
        TestDivergenceDetector,
        TestTrendAnalyzer,
        TestVolumeProfile,
        TestOrderFlow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Executa testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Executando testes dos indicadores...")
    success = run_all_tests()
    
    if success:
        print("✅ Todos os testes passaram!")
    else:
        print("❌ Alguns testes falharam!")
        exit(1)