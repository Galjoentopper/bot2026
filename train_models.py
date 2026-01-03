#!/usr/bin/env python3
"""
Cryptocurrency Price Prediction Training Script - Enhanced Version
===================================================================
Implements LSTM, GRU, BiLSTM, and DLSTM models for cryptocurrency price prediction.

Improvements based on research findings:
- Proper DLSTM with moving average decomposition (Imperial College)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Class weights for imbalanced datasets
- Dynamic delta threshold calculation
- Multiple smoothing k values
- Ensemble prediction

Usage:
    python train_models.py                          # Train all models (regression)
    python train_models.py --task classification    # Train with classification
    python train_models.py --model dlstm            # Train specific model
    python train_models.py --dataset ETH-EUR        # Train on specific dataset
    python train_models.py --ensemble               # Use ensemble prediction
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# CuDNN for RNNs: TensorFlow 2.15.0 should work with CuDNN 9.1.0
# If you still get errors, set TF_USE_CUDNN_RNN=0 to disable
# For TensorFlow 2.16+, you may need CuDNN 9.3.0+
warnings.filterwarnings('ignore')

import tensorflow as tf
import keras
from keras.models import Model, Sequential

# ============================================================================
# GPU Configuration for Maximum Utilization
# ============================================================================

def configure_gpu():
    """Configure TensorFlow for optimal GPU utilization."""
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠️  No GPU detected. Training will use CPU.")
        print("   💡 In Google Colab: Runtime > Change runtime type > Hardware accelerator > GPU")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s)")
    
    # Check CuDNN compatibility
    # TensorFlow 2.20+ requires CuDNN 9.3.0+, but many systems have 9.1.0
    # If mismatch detected, disable CuDNN for RNN operations
    try:
        # Try to detect CuDNN version mismatch
        # This is a heuristic - if CuDNN fails, we'll catch it during training
        cudnn_enabled = os.environ.get('TF_USE_CUDNN_RNN', '1')
        if cudnn_enabled == '0':
            print("⚠️  CuDNN for RNNs disabled via TF_USE_CUDNN_RNN=0")
            print("   Using standard RNN implementation (slower but compatible)")
        else:
            print("ℹ️  CuDNN for RNNs enabled (will auto-fallback if version mismatch)")
    except:
        pass
    
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        # This allows TensorFlow to allocate memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set mixed precision policy for faster training (if supported)
        # This can provide 2-3x speedup on modern GPUs
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled (float16) - ~2-3x speedup")
        except Exception as e:
            print(f"⚠️  Mixed precision not available: {e}")
        
        # Enable XLA (Accelerated Linear Algebra) for faster execution
        # XLA compiles operations for better GPU utilization
        # NOTE: Disabled by default due to CuDNN version compatibility issues
        # Uncomment below if you have matching CuDNN versions
        try:
            # Check CuDNN version compatibility first
            # XLA requires matching CuDNN versions, so we disable it to avoid errors
            # tf.config.optimizer.set_jit(True)
            # print("✅ XLA JIT compilation enabled - better GPU utilization")
            print("⚠️  XLA JIT compilation disabled - using standard execution (CuDNN compatibility)")
        except Exception as e:
            print(f"⚠️  XLA not available: {e}")
        
        # Additional GPU optimizations
        try:
            # Enable cuDNN auto-tuning for optimal performance
            # This allows cuDNN to find the fastest algorithms
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'
            
            # Enable TensorFloat-32 (TF32) for faster training on Ampere+ GPUs
            # Provides ~1.2x speedup with minimal accuracy loss
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("✅ TensorFloat-32 (TF32) enabled - faster training on modern GPUs")
        except Exception as e:
            # TF32 not available on older GPUs or TensorFlow versions
            pass
        
        # Set GPU device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✅ Using GPU: {gpus[0].name}")
        
        # Print GPU info
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    print(f"    Compute Capability: {details.get('compute_capability', 'Unknown')}")
            except:
                pass
        
        # Print memory info if available
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            if gpu_details:
                device_name = gpu_details.get('device_name', 'Unknown')
                print(f"    Device: {device_name}")
        except:
            pass
        
        return True
        
    except RuntimeError as e:
        print(f"⚠️  GPU configuration error: {e}")
        print("   💡 Try restarting the runtime in Colab")
        return False

# Configure GPU at import time
GPU_AVAILABLE = configure_gpu()
from keras.layers import (
    Input, LSTM, GRU, Bidirectional, Dense, Dropout,
    Subtract, Add, Lambda, Concatenate, Layer,
    AvgPool1D, ZeroPadding1D
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import pickle


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str = 'training_config.txt') -> ConfigParser:
    """Load configuration from file."""
    config = ConfigParser()
    
    # Default configuration
    defaults = {
        'DATA': {
            'sequence_length': '60',
            'prediction_horizon': '1',
            'features': 'open,high,low,close,volume',
            'target': 'close',
            'train_test_split': '0.8',
            'use_technical_indicators': 'true'
        },
        'MODEL': {
            'units': '100',
            'layers': '2',
            'dropout': '0.2',
            'learning_rate': '0.0001',
            'optimizer': 'adam'
        },
        'TRAINING': {
            'batch_size': '128',  # Increased for better GPU utilization
            'epochs': '100',
            'early_stopping_patience': '10',
            'validation_split': '0.1',
            'use_class_weights': 'true'
        },
        'CLASSIFICATION': {
            'smoothing_k': '20',
            'threshold_delta': 'auto',
            'enable_classification': 'true'
        },
        'DLSTM': {
            'moving_average_window': '10',
            'trend_units': '64',
            'remainder_units': '64'
        },
        'ENSEMBLE': {
            'enable_ensemble': 'false',
            'models': 'lstm,gru,bilstm,dlstm',
            'voting': 'soft'
        },
        'OUTPUT': {
            'models_dir': 'models',
            'scalers_dir': 'scalers',
            'results_dir': 'results',
            'save_history': 'true',
            'generate_plots': 'true'
        }
    }
    
    # Set defaults
    for section, values in defaults.items():
        config[section] = values
    
    # Load from file if exists
    if os.path.exists(config_path):
        config.read(config_path)
    
    return config


# ============================================================================
# Technical Indicators
# ============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, 
                               num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # Percent B - where price is relative to bands
    percent_b = (prices - lower_band) / (upper_band - lower_band + 1e-10)
    
    return upper_band.fillna(prices), lower_band.fillna(prices), percent_b.fillna(0.5)


def calculate_volume_indicators(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate volume-based indicators."""
    result = df.copy()
    
    # Volume Moving Average
    result['volume_ma'] = df['volume'].rolling(window=window, min_periods=1).mean()
    
    # Volume Ratio (current / average)
    result['volume_ratio'] = df['volume'] / (result['volume_ma'] + 1e-10)
    
    # On-Balance Volume (OBV) - simplified
    price_change = df['close'].diff()
    obv = np.where(price_change > 0, df['volume'], 
                   np.where(price_change < 0, -df['volume'], 0))
    result['obv'] = pd.Series(obv).cumsum()
    
    return result


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    result = df.copy()
    
    # RSI
    result['rsi'] = calculate_rsi(df['close'])
    
    # MACD
    macd, signal, hist = calculate_macd(df['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist
    
    # Bollinger Bands
    upper, lower, percent_b = calculate_bollinger_bands(df['close'])
    result['bb_upper'] = upper
    result['bb_lower'] = lower
    result['bb_percent'] = percent_b
    
    # Volume indicators
    result = calculate_volume_indicators(result)
    
    # Price-based features
    result['price_change'] = df['close'].pct_change().fillna(0)
    result['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
    result['close_open_range'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        result[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
        result[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    return result


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load cryptocurrency dataset from CSV file."""
    df = pd.read_csv(filepath)
    
    # Convert timestamp from milliseconds to datetime
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df


def create_sequences(data: np.ndarray, target: np.ndarray, 
                     sequence_length: int, prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series prediction."""
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length + prediction_horizon - 1])
    
    return np.array(X), np.array(y)


def calculate_dynamic_delta(prices: np.ndarray, k: int = 20, 
                            percentile: float = 50) -> float:
    """
    Calculate dynamic delta threshold based on price volatility.
    
    This improves classification by adapting to the specific asset's volatility.
    """
    # Calculate smoothed changes
    changes = []
    for t in range(k, len(prices) - k):
        m_minus = np.mean(prices[t-k:t])
        m_plus = np.mean(prices[t+1:t+k+1])
        if m_minus != 0:
            l_t = abs((m_plus - m_minus) / m_minus)
            changes.append(l_t)
    
    if not changes:
        return 0.0002  # Default fallback
    
    # Use percentile to set threshold (aim for balanced classes)
    delta = np.percentile(changes, percentile)
    
    return max(delta, 0.0001)  # Minimum threshold


def create_movement_labels(prices: np.ndarray, k: int = 20, 
                           delta: float = None) -> Tuple[np.ndarray, float]:
    """
    Create price movement labels for classification.
    
    Based on Imperial College paper methodology:
    - 0: Fall (price going down)
    - 1: Stationary (no significant change)
    - 2: Rise (price going up)
    
    Uses smoothing: compare average of past k prices vs next k prices
    """
    # Calculate dynamic delta if not provided
    if delta is None or delta == 'auto':
        delta = calculate_dynamic_delta(prices, k)
        print(f"  Using dynamic delta threshold: {delta:.6f}")
    
    labels = np.zeros(len(prices), dtype=int)
    
    for t in range(k, len(prices) - k):
        m_minus = np.mean(prices[t-k:t])  # Past k average
        m_plus = np.mean(prices[t+1:t+k+1])  # Future k average
        
        if m_minus == 0:
            l_t = 0
        else:
            l_t = (m_plus - m_minus) / m_minus
        
        if l_t > delta:
            labels[t] = 2  # Rise
        elif l_t < -delta:
            labels[t] = 0  # Fall
        else:
            labels[t] = 1  # Stationary
    
    return labels, delta


def calculate_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


def prepare_data(df: pd.DataFrame, config: ConfigParser, 
                 task: str = 'regression') -> Dict:
    """Prepare data for training with enhanced features."""
    # Get configuration
    sequence_length = config.getint('DATA', 'sequence_length')
    prediction_horizon = config.getint('DATA', 'prediction_horizon')
    features = config.get('DATA', 'features').split(',')
    target_col = config.get('DATA', 'target')
    train_test_split = config.getfloat('DATA', 'train_test_split')
    use_technical_indicators = config.getboolean('DATA', 'use_technical_indicators')
    
    # Add technical indicators if enabled
    if use_technical_indicators:
        print("  Adding technical indicators...")
        df = add_technical_indicators(df)
        # Use all available features
        feature_cols = [col for col in df.columns if col != 'datetime']
    else:
        feature_cols = [col for col in features if col in df.columns]
    
    print(f"  Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    data = df[feature_cols].values
    target = df[target_col].values
    
    # Handle any remaining NaN values
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize data
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    data_scaled = feature_scaler.fit_transform(data)
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Class weights (for classification)
    class_weights = None
    delta_used = None
    
    # Create sequences
    if task == 'classification':
        # Create movement labels
        k = config.getint('CLASSIFICATION', 'smoothing_k')
        delta_str = config.get('CLASSIFICATION', 'threshold_delta')
        
        if delta_str == 'auto':
            delta = None
        else:
            delta = float(delta_str)
        
        labels, delta_used = create_movement_labels(target, k, delta)
        
        # Adjust data to match labels (skip first and last k samples)
        X, y = create_sequences(data_scaled[k:-k], labels[k:-k], 
                                sequence_length, prediction_horizon)
        
        # Calculate class weights if enabled
        if config.getboolean('TRAINING', 'use_class_weights'):
            class_weights = calculate_class_weights(y)
            print(f"  Class distribution: {Counter(y)}")
            print(f"  Class weights: {class_weights}")
    else:
        X, y = create_sequences(data_scaled, target_scaled, 
                                sequence_length, prediction_horizon)
    
    # Split data chronologically
    split_idx = int(len(X) * train_test_split)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'sequence_length': sequence_length,
        'class_weights': class_weights,
        'delta_used': delta_used
    }


# ============================================================================
# Model Architectures
# ============================================================================

def build_lstm_model(input_shape: Tuple, output_units: int = 1, 
                     units: int = 100, dropout: float = 0.2,
                     task: str = 'regression') -> Model:
    """Build LSTM model."""
    # TensorFlow 2.15.0 should work with CuDNN 9.1.0
    # Use CuDNN (implementation=2) for optimal performance, fallback to standard (1) if needed
    # The implementation parameter is deprecated in TF 2.15+, but we keep it for compatibility
    # If CuDNN fails, TensorFlow will auto-fallback to standard implementation
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
    ])
    
    if task == 'classification':
        # Use float32 for output layer when using mixed precision
        output_layer = Dense(output_units, activation='softmax', dtype='float32')
        model.add(output_layer)
    else:
        output_layer = Dense(output_units, activation='linear', dtype='float32')
        model.add(output_layer)
    
    return model


def build_gru_model(input_shape: Tuple, output_units: int = 1,
                    units: int = 100, dropout: float = 0.2,
                    task: str = 'regression') -> Model:
    """Build GRU model."""
    # TensorFlow 2.15.0 should work with CuDNN 9.1.0
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units, return_sequences=False),
        Dropout(dropout),
    ])
    
    if task == 'classification':
        output_layer = Dense(output_units, activation='softmax', dtype='float32')
        model.add(output_layer)
    else:
        output_layer = Dense(output_units, activation='linear', dtype='float32')
        model.add(output_layer)
    
    return model


def build_bilstm_model(input_shape: Tuple, output_units: int = 1,
                       units: int = 100, dropout: float = 0.2,
                       task: str = 'regression') -> Model:
    """Build Bidirectional LSTM model."""
    # TensorFlow 2.15.0 should work with CuDNN 9.1.0
    model = Sequential([
        Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
        Dropout(dropout),
        Bidirectional(LSTM(units, return_sequences=False)),
        Dropout(dropout),
    ])
    
    if task == 'classification':
        output_layer = Dense(output_units, activation='softmax', dtype='float32')
        model.add(output_layer)
    else:
        output_layer = Dense(output_units, activation='linear', dtype='float32')
        model.add(output_layer)
    
    return model


def build_dlstm_model(input_shape: Tuple, output_units: int = 1,
                      units: int = 100, dropout: float = 0.2,
                      task: str = 'regression', 
                      ma_window: int = 10) -> Model:
    """
    Build DLSTM (Decomposition LSTM) model.
    
    Based on Imperial College paper - best trading profitability.
    Uses proper time series decomposition: Trend (AvgPool) + Remainder
    
    Key insight: Decomposing the series allows the model to learn
    different patterns from smooth trends vs volatile residuals.
    
    Architecture matches Imperial College paper:
    - Trend extraction using AvgPool1D (not Conv1D)
    - Full units (100) for both trend and remainder branches
    - Simple Add fusion (no extra Dense layers)
    """
    inputs = Input(shape=input_shape)
    
    # Trend extraction using Average Pooling (as per Imperial College paper)
    # AvgPool1D with padding='same' maintains sequence length
    trend = AvgPool1D(pool_size=ma_window, strides=1, padding='same')(inputs)
    
    # Remainder: Original - Trend
    remainder = Subtract()([inputs, trend])
    
    # LSTM on Trend (captures long-term patterns)
    # Use full units (100) as per research paper
    # TensorFlow 2.15.0 should work with CuDNN 9.1.0
    trend_lstm = LSTM(units, return_sequences=True, name='trend_lstm_1')(trend)
    trend_lstm = Dropout(dropout)(trend_lstm)
    trend_lstm = LSTM(units, return_sequences=False, name='trend_lstm_2')(trend_lstm)
    trend_lstm = Dropout(dropout)(trend_lstm)
    
    # LSTM on Remainder (captures short-term fluctuations)
    # Use full units (100) as per research paper
    remainder_lstm = LSTM(units, return_sequences=True, name='remainder_lstm_1')(remainder)
    remainder_lstm = Dropout(dropout)(remainder_lstm)
    remainder_lstm = LSTM(units, return_sequences=False, name='remainder_lstm_2')(remainder_lstm)
    remainder_lstm = Dropout(dropout)(remainder_lstm)
    
    # Merge hidden states (simple additive fusion as per paper)
    merged = Add()([trend_lstm, remainder_lstm])
    
    # Output layer (direct connection, no extra Dense layers)
    # Use float32 for output layer when using mixed precision
    if task == 'classification':
        outputs = Dense(output_units, activation='softmax', dtype='float32')(merged)
    else:
        outputs = Dense(output_units, activation='linear', dtype='float32')(merged)
    
    model = Model(inputs=inputs, outputs=outputs, name='DLSTM')
    return model


def get_model(model_name: str, input_shape: Tuple, output_units: int,
              config: ConfigParser, task: str = 'regression') -> Model:
    """Get model by name."""
    units = config.getint('MODEL', 'units')
    dropout = config.getfloat('MODEL', 'dropout')
    
    if model_name.lower() == 'dlstm':
        ma_window = config.getint('DLSTM', 'moving_average_window')
        return build_dlstm_model(
            input_shape=input_shape,
            output_units=output_units,
            units=units,
            dropout=dropout,
            task=task,
            ma_window=ma_window
        )
    
    builders = {
        'lstm': build_lstm_model,
        'gru': build_gru_model,
        'bilstm': build_bilstm_model,
    }
    
    if model_name.lower() not in builders:
        raise ValueError(f"Unknown model: {model_name}. Available: lstm, gru, bilstm, dlstm")
    
    return builders[model_name.lower()](
        input_shape=input_shape,
        output_units=output_units,
        units=units,
        dropout=dropout,
        task=task
    )


# ============================================================================
# Training
# ============================================================================

def compile_model(model: Model, config: ConfigParser, task: str = 'regression'):
    """Compile model with appropriate loss and optimizer."""
    learning_rate = config.getfloat('MODEL', 'learning_rate')
    optimizer = Adam(learning_rate=learning_rate)
    
    if task == 'classification':
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    else:
        loss = 'mse'
        metrics = ['mae']
    
    # Compile with XLA JIT for MAXIMUM GPU performance (if available)
    # This enables just-in-time compilation for faster execution
    compile_kwargs = {
        'optimizer': optimizer,
        'loss': loss,
        'metrics': metrics
    }
    
    # Add jit_compile for XLA optimization (TF 2.7+)
    # NOTE: Disabled due to CuDNN version compatibility issues
    # Uncomment below if you have matching CuDNN versions
    if GPU_AVAILABLE:
        try:
            # XLA JIT compilation can provide 10-30% speedup
            # But requires matching CuDNN versions (compiled vs runtime)
            # compile_kwargs['jit_compile'] = True
            pass
        except TypeError:
            # jit_compile not available in older TF versions
            pass
    
    model.compile(**compile_kwargs)


def create_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int, 
                      shuffle: bool = True, buffer_size: int = None) -> tf.data.Dataset:
    """Create optimized TensorFlow dataset for MAXIMUM GPU utilization."""
    if buffer_size is None:
        # Optimize buffer size: use min of dataset size or 10k for shuffle efficiency
        buffer_size = min(len(X), 10000) if shuffle else 1
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Cache first (before shuffle) if dataset fits in memory - MAJOR speedup
    # Increased threshold for caching (more aggressive caching)
    if len(X) < 500000:  # Cache if less than 500k samples (was 100k)
        dataset = dataset.cache()
        print(f"  💾 Dataset cached in memory ({len(X)} samples) - faster training")
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # Batch with optimized settings
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Aggressive prefetching for MAXIMUM GPU utilization
    # Prefetch multiple batches to keep GPU busy
    try:
        # Use AUTOTUNE for optimal prefetch size
        prefetch_size = tf.data.AUTOTUNE
    except AttributeError:
        # Fallback: prefetch 2-4 batches
        prefetch_size = min(4, max(2, 1024 // batch_size)) if batch_size > 0 else 2
    
    dataset = dataset.prefetch(prefetch_size)
    
    # Enable parallel processing for data transformations (if available)
    try:
        # Use num_parallel_calls for any map operations (none here, but ready for future)
        pass
    except:
        pass
    
    return dataset


def train_model(model: Model, X_train: np.ndarray, y_train: np.ndarray,
                config: ConfigParser, model_path: str = None,
                class_weights: Dict = None) -> Dict:
    """Train model with early stopping and checkpointing."""
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    patience = config.getint('TRAINING', 'early_stopping_patience')
    validation_split = config.getfloat('TRAINING', 'validation_split')
    
    # Calculate training size first to determine optimal batch size
    train_size = int(len(X_train) * (1 - validation_split))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val_split = X_train[train_size:]
    y_val_split = y_train[train_size:]
    
    # Increase batch size for GPU if available - TARGET: 95% GPU utilization (~14GB/15GB)
    # BUT: Ensure we have at least 50-100 steps per epoch for stable training
    # Too few steps per epoch = poor learning and unstable gradients
    min_steps_per_epoch = 50  # Minimum steps for stable training
    max_steps_per_epoch = 200  # Maximum steps (beyond this, increase batch size)
    
    if GPU_AVAILABLE:
        # Calculate what batch size would give us good steps per epoch
        # Target: 50-100 steps per epoch for optimal balance
        target_steps = 75  # Sweet spot for training stability
        optimal_batch_for_steps = max(32, train_size // target_steps)
        
        # Now try to increase batch size for GPU, but respect minimum steps requirement
        if batch_size < 512:
            # Auto-adjust batch size for MAXIMUM GPU utilization (target: 95%)
            # But don't exceed what would give us < min_steps_per_epoch
            proposed_batch_size = min(1536, max(512, batch_size * 4))
            # Cap at batch size that gives us at least min_steps_per_epoch
            max_batch_for_min_steps = train_size // min_steps_per_epoch
            optimal_batch_size = min(proposed_batch_size, max_batch_for_min_steps)
            
            if optimal_batch_size > batch_size:
                steps_per_epoch = train_size // optimal_batch_size
                print(f"  📈 GPU detected: Increasing batch size from {batch_size} to {optimal_batch_size}")
                print(f"     This gives {steps_per_epoch} steps per epoch (target: {target_steps}, min: {min_steps_per_epoch})")
                batch_size = optimal_batch_size
            else:
                steps_per_epoch = train_size // batch_size
                print(f"  📈 GPU detected: Batch size {batch_size} gives {steps_per_epoch} steps per epoch")
                print(f"     (Limited by dataset size to maintain {min_steps_per_epoch}+ steps per epoch)")
        elif batch_size < 1024:
            # Check if we can increase further without dropping below min_steps
            max_batch_for_min_steps = train_size // min_steps_per_epoch
            if batch_size < max_batch_for_min_steps:
                proposed_batch_size = min(1536, batch_size * 2)
                optimal_batch_size = min(proposed_batch_size, max_batch_for_min_steps)
                steps_per_epoch = train_size // optimal_batch_size
                print(f"  📈 GPU detected: Increasing batch size from {batch_size} to {optimal_batch_size}")
                print(f"     This gives {steps_per_epoch} steps per epoch")
                batch_size = optimal_batch_size
            else:
                steps_per_epoch = train_size // batch_size
                print(f"  📈 GPU detected: Batch size {batch_size} gives {steps_per_epoch} steps per epoch")
                print(f"     (At maximum for dataset size to maintain {min_steps_per_epoch}+ steps per epoch)")
        else:
            steps_per_epoch = train_size // batch_size
            if steps_per_epoch < min_steps_per_epoch:
                # Batch size too large! Reduce it
                optimal_batch_size = train_size // min_steps_per_epoch
                print(f"  ⚠️  Batch size {batch_size} too large (only {steps_per_epoch} steps per epoch)")
                print(f"     Reducing to {optimal_batch_size} for {min_steps_per_epoch} steps per epoch")
                batch_size = optimal_batch_size
                steps_per_epoch = train_size // batch_size
            print(f"  📈 GPU detected: Using batch size {batch_size} ({steps_per_epoch} steps per epoch)")
    else:
        steps_per_epoch = train_size // batch_size
        print(f"  📊 Using batch size {batch_size} ({steps_per_epoch} steps per epoch)")
    
    # Convert to TensorFlow datasets for better GPU utilization
    print("  🔄 Creating optimized TensorFlow datasets...")
    
    train_dataset = create_tf_dataset(X_train_split, y_train_split, batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_val_split, y_val_split, batch_size, shuffle=False)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    if model_path:
        callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        )
    
    # Use dataset API for better GPU utilization
    # Recalculate to ensure accuracy (should match what we calculated above)
    steps_per_epoch = len(X_train_split) // batch_size
    validation_steps = len(X_val_split) // batch_size if len(X_val_split) > 0 else None
    
    print(f"  🚀 Training with batch size: {batch_size}, steps per epoch: {steps_per_epoch}")
    if steps_per_epoch < 20:
        print(f"  ⚠️  WARNING: Very few steps per epoch ({steps_per_epoch}). Consider:")
        print(f"     - Using a larger dataset")
        print(f"     - Reducing batch size in training_config.txt")
        print(f"     - Or the model may not train effectively")
    
    # Optimize fit() for MAXIMUM GPU utilization
    # All optimizations are already in place via dataset and model compilation
    history = model.fit(
        train_dataset,
        batch_size=None,  # Batch size is handled by dataset (optimized)
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
        # Note: XLA JIT, mixed precision, and dataset optimizations already enabled
    )
    
    return history.history


# ============================================================================
# Ensemble Methods
# ============================================================================

def build_ensemble(models: List[Model], input_shape: Tuple, 
                   output_units: int, task: str = 'regression',
                   voting: str = 'soft') -> Model:
    """
    Build an ensemble model combining multiple base models.
    
    For trading, ensemble often outperforms individual models.
    """
    inputs = Input(shape=input_shape)
    
    # Get outputs from all models
    outputs = []
    for i, model in enumerate(models):
        # Freeze the base model weights
        model.trainable = False
        model._name = f'base_model_{i}'
        output = model(inputs)
        outputs.append(output)
    
    if len(outputs) == 1:
        ensemble_output = outputs[0]
    else:
        if task == 'classification' and voting == 'soft':
            # Average probabilities (soft voting)
            stacked = tf.stack(outputs, axis=0)
            ensemble_output = tf.reduce_mean(stacked, axis=0)
        else:
            # Average predictions
            stacked = tf.stack(outputs, axis=0)
            ensemble_output = tf.reduce_mean(stacked, axis=0)
    
    return Model(inputs=inputs, outputs=ensemble_output, name='Ensemble')


def ensemble_predict(models: List[Model], X: np.ndarray, 
                     task: str = 'regression', voting: str = 'soft') -> np.ndarray:
    """Make ensemble predictions from multiple models."""
    predictions = []
    
    # Use MAXIMUM batch size for prediction if GPU available (target: 95% utilization)
    predict_batch_size = 1024 if GPU_AVAILABLE else 32
    
    for model in models:
        pred = model.predict(X, batch_size=predict_batch_size, verbose=0)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if task == 'classification' and voting == 'soft':
        # Average probabilities, then argmax
        avg_probs = np.mean(predictions, axis=0)
        return np.argmax(avg_probs, axis=1)
    elif task == 'classification' and voting == 'hard':
        # Majority voting
        class_preds = np.argmax(predictions, axis=-1)
        from scipy import stats
        return stats.mode(class_preds, axis=0)[0].flatten()
    else:
        # Average regression predictions
        return np.mean(predictions, axis=0).flatten()


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray,
                        scaler: MinMaxScaler = None) -> Dict:
    """Evaluate regression model."""
    # Inverse transform if scaler provided
    if scaler:
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # R-squared
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate classification model."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Precision_Fall': per_class_precision[0] if len(per_class_precision) > 0 else 0,
        'Precision_Stationary': per_class_precision[1] if len(per_class_precision) > 1 else 0,
        'Precision_Rise': per_class_precision[2] if len(per_class_precision) > 2 else 0,
    }


def calculate_trading_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              prices: np.ndarray = None,
                              transaction_cost: float = 0.001) -> Dict:
    """
    Calculate trading-specific metrics.
    
    Based on Imperial College paper findings about profitability.
    """
    # Direction accuracy (most important for trading)
    true_direction = np.sign(np.diff(np.concatenate([[0], y_true])))
    pred_direction = np.sign(np.diff(np.concatenate([[0], y_pred])))
    direction_accuracy = np.mean(true_direction == pred_direction)
    
    # Simulated trading returns (simplified)
    if prices is not None:
        # Long when predicting up, short when predicting down
        positions = np.sign(y_pred - 0.5) if len(np.unique(y_pred)) <= 3 else np.sign(y_pred)
        returns = np.diff(prices) / prices[:-1]
        
        # Apply positions and transaction costs
        strategy_returns = positions[:-1] * returns
        
        # Transaction costs when position changes
        position_changes = np.diff(np.concatenate([[0], positions]))
        costs = np.abs(position_changes[:-1]) * transaction_cost
        
        net_returns = strategy_returns - costs
        
        total_return = np.sum(net_returns)
        sharpe = np.mean(net_returns) / (np.std(net_returns) + 1e-10) * np.sqrt(252 * 24)  # Hourly data
        
        return {
            'direction_accuracy': direction_accuracy,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'avg_return_per_trade': np.mean(net_returns)
        }
    
    return {'direction_accuracy': direction_accuracy}


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric plot
    metric_key = [k for k in history.keys() if k not in ['loss', 'val_loss', 'lr']][0]
    val_metric_key = f'val_{metric_key}'
    
    axes[1].plot(history[metric_key], label=f'Training {metric_key}', linewidth=2)
    if val_metric_key in history:
        axes[1].plot(history[val_metric_key], label=f'Validation {metric_key}', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_key.upper())
    axes[1].set_title(f'Model {metric_key.upper()}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                     title: str = 'Predictions vs Actual',
                     save_path: str = None):
    """Plot predictions vs actual values."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(y_true, label='Actual', alpha=0.8, linewidth=1)
    ax.plot(y_pred, label='Predicted', alpha=0.8, linewidth=1)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: List[str] = None,
                          save_path: str = None):
    """Plot confusion matrix for classification."""
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['Fall', 'Stationary', 'Rise']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(model: Model, feature_names: List[str],
                            X_sample: np.ndarray, save_path: str = None):
    """Plot feature importance using gradient-based attribution."""
    try:
        # Get the last timestep features
        if len(X_sample.shape) == 3:
            X_sample = X_sample[:100]  # Use first 100 samples
        
        # Calculate gradient-based importance
        with tf.GradientTape() as tape:
            X_tensor = tf.Variable(X_sample, dtype=tf.float32)
            predictions = model(X_tensor)
        
        gradients = tape.gradient(predictions, X_tensor)
        importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:20]  # Top 20
        
        ax.barh(range(len(indices)), importance[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                           for i in indices])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Gradient-based)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_single_model(model_name: str, dataset_name: str, df: pd.DataFrame,
                       config: ConfigParser, task: str = 'regression',
                       output_dir: str = '.') -> Dict:
    """Train a single model on a dataset."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset_name} ({task})")
    print('='*60)
    
    # Prepare data
    data = prepare_data(df, config, task)
    
    # Determine output units
    if task == 'classification':
        output_units = 3  # Fall, Stationary, Rise
    else:
        output_units = 1
    
    # Build model
    input_shape = (data['sequence_length'], data['n_features'])
    model = get_model(model_name, input_shape, output_units, config, task)
    
    # Compile model
    compile_model(model, config, task)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create output directories
    models_dir = os.path.join(output_dir, config.get('OUTPUT', 'models_dir'))
    scalers_dir = os.path.join(output_dir, config.get('OUTPUT', 'scalers_dir'))
    results_dir = os.path.join(output_dir, config.get('OUTPUT', 'results_dir'))
    
    for d in [models_dir, scalers_dir, results_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Model path
    model_filename = f"{model_name}_{dataset_name}_{task}.keras"
    model_path = os.path.join(models_dir, model_filename)
    
    # Train model
    print("\nTraining...")
    history = train_model(model, data['X_train'], data['y_train'], 
                          config, model_path, data['class_weights'])
    
    # Evaluate model
    print("\nEvaluating...")
    # Use MAXIMUM batch size for prediction if GPU available (target: 95% utilization)
    predict_batch_size = 1024 if GPU_AVAILABLE else 32
    
    if task == 'classification':
        y_pred_probs = model.predict(data['X_test'], batch_size=predict_batch_size, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        metrics = evaluate_classification(data['y_test'], y_pred)
    else:
        y_pred = model.predict(data['X_test'], batch_size=predict_batch_size, verbose=0).flatten()
        metrics = evaluate_regression(data['y_test'], y_pred, data['target_scaler'])
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Save model
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save scalers and metadata
    scaler_path = os.path.join(scalers_dir, f"scaler_{dataset_name}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': data['feature_scaler'],
            'target_scaler': data['target_scaler'],
            'feature_names': data['feature_names'],
            'sequence_length': data['sequence_length'],
            'delta_used': data.get('delta_used')
        }, f)
    print(f"Scalers saved to: {scaler_path}")
    
    # Save training history
    if config.getboolean('OUTPUT', 'save_history'):
        history_path = os.path.join(results_dir, 
                                    f"history_{model_name}_{dataset_name}_{task}.json")
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {k: [float(v) for v in vals] 
                                   for k, vals in history.items()}
            json.dump(history_serializable, f, indent=2)
    
    # Generate plots
    if config.getboolean('OUTPUT', 'generate_plots'):
        # Training history plot
        history_plot_path = os.path.join(results_dir, 
                                         f"training_{model_name}_{dataset_name}_{task}.png")
        plot_training_history(history, history_plot_path)
        
        if task == 'classification':
            # Confusion matrix
            cm_plot_path = os.path.join(results_dir,
                                        f"confusion_{model_name}_{dataset_name}.png")
            plot_confusion_matrix(data['y_test'], y_pred, save_path=cm_plot_path)
        else:
            # Predictions plot
            pred_plot_path = os.path.join(results_dir,
                                          f"predictions_{model_name}_{dataset_name}.png")
            
            # Inverse transform for plotting
            y_true_inv = data['target_scaler'].inverse_transform(
                data['y_test'].reshape(-1, 1)).flatten()
            y_pred_inv = data['target_scaler'].inverse_transform(
                y_pred.reshape(-1, 1)).flatten()
            
            plot_predictions(y_true_inv, y_pred_inv,
                           title=f"{model_name.upper()} - {dataset_name}",
                           save_path=pred_plot_path)
        
        # Feature importance plot
        feature_plot_path = os.path.join(results_dir,
                                         f"features_{model_name}_{dataset_name}.png")
        plot_feature_importance(model, data['feature_names'], 
                               data['X_test'], feature_plot_path)
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'task': task,
        'metrics': metrics,
        'history': history,
        'model': model,
        'data': data
    }


def train_all_models(datasets_dir: str = 'datasets', 
                     config: ConfigParser = None,
                     task: str = 'regression',
                     models: List[str] = None,
                     specific_dataset: str = None,
                     use_ensemble: bool = False) -> List[Dict]:
    """Train all models on all datasets."""
    if config is None:
        config = load_config()
    
    if models is None:
        models = ['lstm', 'gru', 'bilstm', 'dlstm']
    
    # Find all CSV files in datasets directory
    datasets_path = Path(datasets_dir)
    csv_files = list(datasets_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {datasets_dir}")
        return []
    
    # Filter by specific dataset if provided
    if specific_dataset:
        csv_files = [f for f in csv_files if specific_dataset in f.stem]
        if not csv_files:
            print(f"Dataset '{specific_dataset}' not found")
            return []
    
    print(f"\nFound {len(csv_files)} dataset(s):")
    for f in csv_files:
        print(f"  - {f.name}")
    
    print(f"\nModels to train: {', '.join(models)}")
    print(f"Task: {task}")
    print(f"Ensemble: {use_ensemble}")
    
    all_results = []
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem
        print(f"\n\n{'#'*60}")
        print(f"# Dataset: {dataset_name}")
        print('#'*60)
        
        # Load dataset
        df = load_dataset(str(csv_file))
        print(f"Loaded {len(df)} records")
        
        dataset_results = []
        trained_models = []
        
        for model_name in models:
            try:
                result = train_single_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    df=df,
                    config=config,
                    task=task,
                    output_dir=str(datasets_path.parent)
                )
                dataset_results.append(result)
                all_results.append(result)
                trained_models.append(result['model'])
            except Exception as e:
                print(f"\nError training {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create and evaluate ensemble if enabled
        if use_ensemble and len(trained_models) > 1:
            print(f"\n{'='*60}")
            print(f"Creating ENSEMBLE for {dataset_name}")
            print('='*60)
            
            try:
                # Use data from last result
                data = dataset_results[-1]['data']
                
                # Ensemble predictions
                y_pred = ensemble_predict(trained_models, data['X_test'], task)
                
                if task == 'classification':
                    metrics = evaluate_classification(data['y_test'], y_pred)
                else:
                    metrics = evaluate_regression(data['y_test'], y_pred, 
                                                 data['target_scaler'])
                
                print("\nEnsemble Metrics:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.6f}")
                
                ensemble_result = {
                    'model_name': 'ensemble',
                    'dataset': dataset_name,
                    'task': task,
                    'metrics': metrics,
                    'history': {}
                }
                all_results.append(ensemble_result)
                
            except Exception as e:
                print(f"Error creating ensemble: {e}")
    
    return all_results


def generate_comparison_report(results: List[Dict], output_path: str = None):
    """Generate comparison report of all trained models."""
    if not results:
        print("No results to compare")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON REPORT")
    print("="*80)
    
    # Separate by task
    regression_results = [r for r in results if r['task'] == 'regression']
    classification_results = [r for r in results if r['task'] == 'classification']
    
    if regression_results:
        print("\n--- REGRESSION RESULTS ---")
        df_reg = pd.DataFrame([
            {
                'Model': r['model_name'].upper(),
                'Dataset': r['dataset'],
                **{k: v for k, v in r['metrics'].items() if not k.startswith('Precision_')}
            }
            for r in regression_results
        ])
        print(df_reg.to_string(index=False))
        
        # Find best model by MAPE
        best_idx = df_reg['MAPE'].idxmin()
        print(f"\nBest Model (lowest MAPE): {df_reg.loc[best_idx, 'Model']} on {df_reg.loc[best_idx, 'Dataset']}")
    
    if classification_results:
        print("\n--- CLASSIFICATION RESULTS ---")
        df_cls = pd.DataFrame([
            {
                'Model': r['model_name'].upper(),
                'Dataset': r['dataset'],
                **{k: v for k, v in r['metrics'].items() if not k.startswith('Precision_')}
            }
            for r in classification_results
        ])
        print(df_cls.to_string(index=False))
        
        # Find best model by Accuracy
        best_idx = df_cls['Accuracy'].idxmax()
        print(f"\nBest Model (highest Accuracy): {df_cls.loc[best_idx, 'Model']} on {df_cls.loc[best_idx, 'Dataset']}")
        
        # Find best model by F1-Score
        best_f1_idx = df_cls['F1-Score'].idxmax()
        print(f"Best Model (highest F1): {df_cls.loc[best_f1_idx, 'Model']} on {df_cls.loc[best_f1_idx, 'Dataset']}")
    
    # Save comparison to CSV
    if output_path:
        all_results = pd.DataFrame([
            {
                'Model': r['model_name'].upper(),
                'Dataset': r['dataset'],
                'Task': r['task'],
                **{k: v for k, v in r['metrics'].items() if not k.startswith('Precision_')}
            }
            for r in results
        ])
        all_results.to_csv(output_path, index=False)
        print(f"\nComparison saved to: {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train cryptocurrency price prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py                          # Train all models (regression)
  python train_models.py --task classification    # Train with classification
  python train_models.py --model dlstm            # Train specific model
  python train_models.py --dataset ETH-EUR        # Train on specific dataset
  python train_models.py --model gru --task classification --dataset ADA-EUR
  python train_models.py --ensemble               # Train all + ensemble
        """
    )
    
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type (default: regression)')
    
    parser.add_argument('--model', type=str, default=None,
                        choices=['lstm', 'gru', 'bilstm', 'dlstm'],
                        help='Specific model to train (default: all)')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to train on (default: all)')
    
    parser.add_argument('--config', type=str, default='training_config.txt',
                        help='Path to configuration file')
    
    parser.add_argument('--datasets-dir', type=str, default='datasets',
                        help='Directory containing datasets')
    
    parser.add_argument('--ensemble', action='store_true',
                        help='Use ensemble of all models')
    
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtesting after training (classification only)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("="*60)
    print("CRYPTOCURRENCY PRICE PREDICTION TRAINING (Enhanced)")
    print("="*60)
    
    # Print GPU status
    if GPU_AVAILABLE:
        print(f"\n✅ GPU Acceleration: ENABLED")
        print(f"   TensorFlow version: {tf.__version__}")
        print(f"   CUDA available: {tf.test.is_built_with_cuda()}")
        if tf.test.is_built_with_cuda():
            print(f"   GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
    else:
        print(f"\n⚠️  GPU Acceleration: DISABLED (using CPU)")
    
    print(f"\nTask: {args.task}")
    print(f"Config: {args.config}")
    print(f"Datasets directory: {args.datasets_dir}")
    print(f"Ensemble: {args.ensemble}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine models to train
    if args.model:
        models = [args.model]
    else:
        models = ['lstm', 'gru', 'bilstm', 'dlstm']
    
    # Train models
    results = train_all_models(
        datasets_dir=args.datasets_dir,
        config=config,
        task=args.task,
        models=models,
        specific_dataset=args.dataset,
        use_ensemble=args.ensemble
    )
    
    # Generate comparison report
    if results:
        results_dir = config.get('OUTPUT', 'results_dir')
        os.makedirs(results_dir, exist_ok=True)
        comparison_path = os.path.join(results_dir, 'model_comparison.csv')
        generate_comparison_report(results, comparison_path)
    
    # Run backtesting if requested (classification only)
    if args.backtest and args.task == 'classification':
        print("\n" + "="*60)
        print("RUNNING BACKTESTING")
        print("="*60)
        try:
            from backtest_trading import backtest_model
            
            for result in results:
                if result.get('task') == 'classification':
                    model_name = result['model_name']
                    dataset_name = result['dataset']
                    try:
                        backtest_model(
                            model_name=model_name,
                            dataset_name=dataset_name,
                            config=config,
                            transaction_cost=0.0025  # Bitvavo 0.25% fee
                        )
                    except Exception as e:
                        print(f"Error backtesting {model_name} on {dataset_name}: {e}")
        except ImportError:
            print("Warning: Could not import backtest_trading module")
        except Exception as e:
            print(f"Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
