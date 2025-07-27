# utils/validators.py 
# utils/validators.py
"""
🔍 VALIDADORES DE DADOS - SMART TRADING SYSTEM v2.0
Validações robustas para dados de mercado, sinais e configurações
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """📋 Resultado de validação"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0-100
    details: Dict[str, Any]

class DataQualityValidator:
    """🏆 Validador de qualidade de dados OHLCV"""
    
    @staticmethod
    def validate_ohlcv_dataframe(df: pd.DataFrame, symbol: str = None) -> ValidationResult:
        """Validação completa de DataFrame OHLCV"""
        errors = []
        warnings = []
        details = {}
        score = 100.0
        
        if df is None or df.empty:
            return ValidationResult(
                is_valid=False,
                errors=["DataFrame vazio ou None"],
                warnings=[],
                score=0.0,
                details={}
            )
        
        # === VALIDAÇÕES ESTRUTURAIS ===
        
        # Colunas obrigatórias
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Colunas faltantes: {missing_columns}")
            score -= 30
        
        # Timestamp (opcional, mas recomendado)
        if 'timestamp' not in df.columns:
            warnings.append("Coluna 'timestamp' não encontrada")
            score -= 5
        
        if errors:  # Se há erros estruturais, não continua
            return ValidationResult(False, errors, warnings, score, details)
        
        # === VALIDAÇÕES DE DADOS ===
        
        # Dados nulos
        null_counts = df[required_columns].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            null_pct = (total_nulls / (len(df) * len(required_columns))) * 100
            details['null_percentage'] = round(null_pct, 2)
            
            if null_pct > 10:
                errors.append(f"Muitos dados nulos: {null_pct:.1f}%")
                score -= 25
            elif null_pct > 5:
                warnings.append(f"Dados nulos detectados: {null_pct:.1f}%")
                score -= 10
            elif null_pct > 0:
                warnings.append(f"Alguns dados nulos: {null_pct:.1f}%")
                score -= 5
        
        # Valores negativos (não permitidos)
        negative_values = (df[required_columns] < 0).sum().sum()
        if negative_values > 0:
            errors.append(f"Valores negativos encontrados: {negative_values}")
            score -= 20
        
        # === VALIDAÇÕES LÓGICAS OHLC ===
        
        # High >= Open, Close
        high_violations = ((df['high'] < df['open']) | (df['high'] < df['close'])).sum()
        if high_violations > 0:
            pct = (high_violations / len(df)) * 100
            details['high_violations_pct'] = round(pct, 2)
            
            if pct > 1:
                errors.append(f"High < Open/Close em {pct:.1f}% dos candles")
                score -= 20
            else:
                warnings.append(f"High < Open/Close em {high_violations} candles")
                score -= 5
        
        # Low <= Open, Close
        low_violations = ((df['low'] > df['open']) | (df['low'] > df['close'])).sum()
        if low_violations > 0:
            pct = (low_violations / len(df)) * 100
            details['low_violations_pct'] = round(pct, 2)
            
            if pct > 1:
                errors.append(f"Low > Open/Close em {pct:.1f}% dos candles")
                score -= 20
            else:
                warnings.append(f"Low > Open/Close em {low_violations} candles")
                score -= 5
        
        # High >= Low
        high_low_violations = (df['high'] < df['low']).sum()
        if high_low_violations > 0:
            errors.append(f"High < Low em {high_low_violations} candles")
            score -= 30
        
        # === VALIDAÇÕES DE VOLUME ===
        
        # Volume zero
        zero_volume = (df['volume'] == 0).sum()
        if zero_volume > 0:
            zero_pct = (zero_volume / len(df)) * 100
            details['zero_volume_pct'] = round(zero_pct, 2)
            
            if zero_pct > 10:
                warnings.append(f"Volume zero em {zero_pct:.1f}% dos candles")
                score -= 15
            elif zero_pct > 0:
                warnings.append(f"Volume zero em {zero_volume} candles")
                score -= 5
        
        # === VALIDAÇÕES TEMPORAIS ===
        
        if 'timestamp' in df.columns:
            # Timestamps duplicados
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                errors.append(f"Timestamps duplicados: {duplicate_timestamps}")
                score -= 15
            
            # Ordem temporal
            if not df['timestamp'].is_monotonic_increasing:
                warnings.append("Timestamps fora de ordem")
                score -= 10
            
            # Gaps temporais grandes
            if len(df) > 1:
                time_diffs = df['timestamp'].diff().dt.total_seconds()
                # Remove o primeiro NaN
                time_diffs = time_diffs.dropna()
                
                if len(time_diffs) > 0:
                    median_interval = time_diffs.median()
                    large_gaps = (time_diffs > median_interval * 3).sum()
                    
                    if large_gaps > 0:
                        gap_pct = (large_gaps / len(time_diffs)) * 100
                        details['large_gaps_pct'] = round(gap_pct, 2)
                        
                        if gap_pct > 5:
                            warnings.append(f"Gaps temporais grandes: {gap_pct:.1f}%")
                            score -= 10
        
        # === VALIDAÇÕES ESTATÍSTICAS ===
        
        # Outliers extremos (preços)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # Outliers (> Q3 + 3*IQR ou < Q1 - 3*IQR)
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(df)) * 100
                    if outlier_pct > 2:
                        warnings.append(f"Outliers em {col}: {outlier_pct:.1f}%")
                        score -= 5
        
        # === DETALHES FINAIS ===
        
        details.update({
            'total_candles': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else 'N/A',
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else 'N/A'
            },
            'price_range': {
                'min': float(df[price_columns].min().min()),
                'max': float(df[price_columns].max().max())
            },
            'volume_stats': {
                'min': float(df['volume'].min()),
                'max': float(df['volume'].max()),
                'avg': float(df['volume'].mean())
            }
        })
        
        # Score final
        score = max(0.0, min(100.0, score))
        is_valid = score >= 70.0 and len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            score=score,
            details=details
        )

class SymbolValidator:
    """💰 Validador de símbolos de trading"""
    
    # Padrões de símbolos válidos
    SYMBOL_PATTERNS = {
        'binance': r'^[A-Z]{2,10}USDT?$',  # BTCUSDT, ETHUSDT, etc.
        'coinbase': r'^[A-Z]{2,10}-USD$',   # BTC-USD, ETH-USD, etc.
        'generic': r'^[A-Z]{2,10}[A-Z]{3,4}$'  # Genérico
    }
    
    @staticmethod
    def validate_symbol(symbol: str, exchange: str = 'binance') -> ValidationResult:
        """Valida formato de símbolo"""
        errors = []
        warnings = []
        score = 100.0
        
        if not symbol:
            return ValidationResult(False, ["Símbolo vazio"], [], 0.0, {})
        
        # Normalização
        symbol = symbol.upper().strip()
        
        # Padrão do exchange
        pattern = SymbolValidator.SYMBOL_PATTERNS.get(exchange, 
                                                     SymbolValidator.SYMBOL_PATTERNS['generic'])
        
        if not re.match(pattern, symbol):
            errors.append(f"Formato inválido para {exchange}: {symbol}")
            score -= 50
        
        # Tamanho
        if len(symbol) < 6:
            errors.append(f"Símbolo muito curto: {symbol}")
            score -= 30
        elif len(symbol) > 12:
            warnings.append(f"Símbolo longo: {symbol}")
            score -= 10
        
        # Caracteres válidos
        if not symbol.isalpha():
            invalid_chars = [c for c in symbol if not c.isalpha()]
            if exchange != 'coinbase' or '-' not in symbol:
                errors.append(f"Caracteres inválidos: {invalid_chars}")
                score -= 20
        
        details = {
            'symbol': symbol,
            'exchange': exchange,
            'length': len(symbol),
            'pattern_used': pattern
        }
        
        is_valid = len(errors) == 0 and score >= 70.0
        
        return ValidationResult(is_valid, errors, warnings, score, details)

class TimeframeValidator:
    """⏰ Validador de timeframes"""
    
    VALID_TIMEFRAMES = {
        '1m': 60,
        '3m': 180,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '2h': 7200,
        '4h': 14400,
        '6h': 21600,
        '8h': 28800,
        '12h': 43200,
        '1d': 86400,
        '3d': 259200,
        '1w': 604800
    }
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> ValidationResult:
        """Valida timeframe"""
        errors = []
        warnings = []
        score = 100.0
        
        if not timeframe:
            return ValidationResult(False, ["Timeframe vazio"], [], 0.0, {})
        
        timeframe = timeframe.lower().strip()
        
        if timeframe not in TimeframeValidator.VALID_TIMEFRAMES:
            errors.append(f"Timeframe inválido: {timeframe}")
            score = 0
        
        # Recomendações baseadas no sistema
        recommended_tfs = ['1h', '4h', '1d']
        if timeframe not in recommended_tfs:
            warnings.append(f"Timeframe não recomendado para este sistema: {timeframe}")
            score -= 20
        
        details = {
            'timeframe': timeframe,
            'seconds': TimeframeValidator.VALID_TIMEFRAMES.get(timeframe, 0),
            'is_recommended': timeframe in recommended_tfs,
            'valid_timeframes': list(TimeframeValidator.VALID_TIMEFRAMES.keys())
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, score, details)

class SignalValidator:
    """📈 Validador de sinais de trading"""
    
    @staticmethod
    def validate_trading_signal(signal_data: Dict[str, Any]) -> ValidationResult:
        """Valida estrutura e valores de um sinal de trading"""
        errors = []
        warnings = []
        score = 100.0
        
        # Campos obrigatórios
        required_fields = [
            'symbol', 'direction', 'entry_price', 'stop_loss', 
            'take_profit', 'confidence_score', 'strategy'
        ]
        
        missing_fields = [field for field in required_fields if field not in signal_data]
        if missing_fields:
            errors.append(f"Campos obrigatórios faltando: {missing_fields}")
            score -= 40
        
        if errors:  # Se campos obrigatórios faltam, não continua
            return ValidationResult(False, errors, warnings, score, {})
        
        # === VALIDAÇÕES DE VALORES ===
        
        # Direção
        valid_directions = ['long', 'short', 'buy', 'sell']
        direction = str(signal_data.get('direction', '')).lower()
        if direction not in valid_directions:
            errors.append(f"Direção inválida: {direction}")
            score -= 20
        
        # Preços
        entry_price = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', [])
        
        # Preços positivos
        if entry_price <= 0:
            errors.append("Entry price deve ser positivo")
            score -= 30
        
        if stop_loss <= 0:
            errors.append("Stop loss deve ser positivo")
            score -= 30
        
        # Lógica de preços para LONG
        if direction in ['long', 'buy'] and entry_price > 0 and stop_loss > 0:
            if stop_loss >= entry_price:
                errors.append("Stop loss deve ser menor que entry para LONG")
                score -= 25
        
        # Lógica de preços para SHORT
        elif direction in ['short', 'sell'] and entry_price > 0 and stop_loss > 0:
            if stop_loss <= entry_price:
                errors.append("Stop loss deve ser maior que entry para SHORT")
                score -= 25
        
        # Take profit
        if isinstance(take_profit, (list, tuple)):
            if len(take_profit) == 0:
                warnings.append("Nenhum take profit definido")
                score -= 10
            
            for i, tp in enumerate(take_profit):
                if tp <= 0:
                    errors.append(f"Take profit {i+1} deve ser positivo")
                    score -= 15
                
                # Validação lógica de TP
                if direction in ['long', 'buy'] and entry_price > 0:
                    if tp <= entry_price:
                        errors.append(f"Take profit {i+1} deve ser maior que entry para LONG")
                        score -= 15
                elif direction in ['short', 'sell'] and entry_price > 0:
                    if tp >= entry_price:
                        errors.append(f"Take profit {i+1} deve ser menor que entry para SHORT")
                        score -= 15
        
        # Confidence score
        confidence = signal_data.get('confidence_score', 0)
        if not 0 <= confidence <= 1:
            if 0 <= confidence <= 100:
                warnings.append("Confidence parece estar em % (0-100), esperado 0-1")
                score -= 5
            else:
                errors.append(f"Confidence inválido: {confidence} (esperado 0-1)")
                score -= 20
        
        # === VALIDAÇÕES DE RISK/REWARD ===
        
        if entry_price > 0 and stop_loss > 0 and take_profit:
            risk = abs(entry_price - stop_loss)
            
            if isinstance(take_profit, (list, tuple)) and len(take_profit) > 0:
                first_tp = take_profit[0]
                reward = abs(first_tp - entry_price)
                
                if risk > 0:
                    rr_ratio = reward / risk
                    
                    if rr_ratio < 1.0:
                        warnings.append(f"Risk/Reward baixo: {rr_ratio:.2f}")
                        score -= 15
                    elif rr_ratio < 2.0:
                        warnings.append(f"Risk/Reward moderado: {rr_ratio:.2f}")
                        score -= 5
                    
                    # Risk muito alto (> 5% do entry)
                    risk_pct = (risk / entry_price) * 100
                    if risk_pct > 5:
                        errors.append(f"Risk muito alto: {risk_pct:.1f}%")
                        score -= 20
                    elif risk_pct > 3:
                        warnings.append(f"Risk alto: {risk_pct:.1f}%")
                        score -= 10
        
        # === VALIDAÇÕES DE SÍMBOLO E ESTRATÉGIA ===
        
        # Símbolo
        symbol_result = SymbolValidator.validate_symbol(signal_data.get('symbol', ''))
        if not symbol_result.is_valid:
            errors.extend([f"Símbolo: {err}" for err in symbol_result.errors])
            score -= 15
        
        # Estratégia
        strategy = signal_data.get('strategy', '')
        if not strategy or len(strategy) < 3:
            warnings.append("Nome da estratégia muito curto")
            score -= 5
        
        # === DETALHES FINAIS ===
        
        details = {
            'signal_data': signal_data,
            'direction_normalized': direction,
            'risk_reward_ratio': None,
            'risk_percentage': None
        }
        
        if entry_price > 0 and stop_loss > 0:
            risk = abs(entry_price - stop_loss)
            details['risk_percentage'] = (risk / entry_price) * 100
            
            if take_profit and isinstance(take_profit, (list, tuple)) and len(take_profit) > 0:
                reward = abs(take_profit[0] - entry_price)
                if risk > 0:
                    details['risk_reward_ratio'] = reward / risk
        
        score = max(0.0, min(100.0, score))
        is_valid = len(errors) == 0 and score >= 60.0
        
        return ValidationResult(is_valid, errors, warnings, score, details)

class ConfigValidator:
    """⚙️ Validador de configurações do sistema"""
    
    @staticmethod
    def validate_trading_config(config: Dict[str, Any]) -> ValidationResult:
        """Valida configurações de trading"""
        errors = []
        warnings = []
        score = 100.0
        
        # Risk Management
        max_portfolio_risk = config.get('max_portfolio_risk', 0)
        max_position_risk = config.get('max_position_risk', 0)
        
        if max_portfolio_risk <= 0 or max_portfolio_risk > 1:
            errors.append(f"Max portfolio risk inválido: {max_portfolio_risk}")
            score -= 25
        
        if max_position_risk <= 0 or max_position_risk > 0.1:
            errors.append(f"Max position risk inválido: {max_position_risk}")
            score -= 25
        
        if max_position_risk >= max_portfolio_risk:
            errors.append("Position risk >= Portfolio risk")
            score -= 20
        
        # Timeframes
        timeframes = config.get('timeframes', [])
        if not timeframes:
            errors.append("Nenhum timeframe configurado")
            score -= 30
        
        recommended_tfs = ['1h', '4h', '1d']
        if not any(tf in recommended_tfs for tf in timeframes):
            warnings.append("Nenhum timeframe recomendado configurado")
            score -= 15
        
        # Strategies
        strategies = config.get('strategies', [])
        if not strategies:
            warnings.append("Nenhuma estratégia configurada")
            score -= 20
        
        is_valid = len(errors) == 0 and score >= 60.0
        
        return ValidationResult(is_valid, errors, warnings, score, {'config': config})

# === FUNÇÕES DE CONVENIÊNCIA ===

def validate_ohlcv_data(df: pd.DataFrame, symbol: str = None) -> bool:
    """Função rápida para validar dados OHLCV"""
    result = DataQualityValidator.validate_ohlcv_dataframe(df, symbol)
    
    if not result.is_valid:
        logger.warning(f"Dados OHLCV inválidos para {symbol}: {result.errors}")
    
    return result.is_valid

def validate_symbol_format(symbol: str, exchange: str = 'binance') -> bool:
    """Função rápida para validar formato de símbolo"""
    result = SymbolValidator.validate_symbol(symbol, exchange)
    return result.is_valid

def validate_timeframe_format(timeframe: str) -> bool:
    """Função rápida para validar timeframe"""
    result = TimeframeValidator.validate_timeframe(timeframe)
    return result.is_valid

def validate_signal_data(signal_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Função rápida para validar dados de sinal"""
    result = SignalValidator.validate_trading_signal(signal_data)
    return result.is_valid, result.errors

def get_data_quality_score(df: pd.DataFrame) -> float:
    """Retorna apenas o score de qualidade dos dados (0-100)"""
    result = DataQualityValidator.validate_ohlcv_dataframe(df)
    return result.score

# === VALIDADORES ESPECÍFICOS ===

def validate_price_data(prices: Dict[str, float]) -> ValidationResult:
    """Valida dicionário de preços"""
    errors = []
    warnings = []
    score = 100.0
    
    if not prices:
        return ValidationResult(False, ["Dicionário de preços vazio"], [], 0.0, {})
    
    for symbol, price in prices.items():
        if not isinstance(price, (int, float)):
            errors.append(f"Preço inválido para {symbol}: {price}")
            score -= 20
        elif price <= 0:
            errors.append(f"Preço negativo para {symbol}: {price}")
            score -= 20
        elif price > 1000000:
            warnings.append(f"Preço muito alto para {symbol}: {price}")
            score -= 5
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, score, {'prices': prices})

def validate_volume_data(volumes: Dict[str, float]) -> ValidationResult:
    """Valida dicionário de volumes"""
    errors = []
    warnings = []
    score = 100.0
    
    if not volumes:
        return ValidationResult(False, ["Dicionário de volumes vazio"], [], 0.0, {})
    
    for symbol, volume in volumes.items():
        if not isinstance(volume, (int, float)):
            errors.append(f"Volume inválido para {symbol}: {volume}")
            score -= 20
        elif volume < 0:
            errors.append(f"Volume negativo para {symbol}: {volume}")
            score -= 20
        elif volume == 0:
            warnings.append(f"Volume zero para {symbol}")
            score -= 5
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, score, {'volumes': volumes})