#!/usr/bin/env python3
"""
PyTorch Model Backtesting Script
=================================
Simulates trading with trained PyTorch classification models to verify profitability.

Based on Imperial College paper methodology:
- Enter long position when prediction = 2 (Rise)
- Exit long position when prediction = 0 (Fall)
- Enter short position when prediction = 0 (Fall)
- Exit short position when prediction = 2 (Rise)
- Hold on prediction = 1 (Stationary)
- Apply transaction costs: 0.25% (Bitvavo maker/taker fee)

Usage:
    python backtest_models.py --model dlstm --dataset ADA-EUR_1H_20240101-20251231
    python backtest_models.py --model all --dataset ADA-EUR_1H_20240101-20251231
"""

import os
import sys
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from configparser import ConfigParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore')

# Import functions from train_models.py
sys.path.insert(0, str(Path(__file__).parent))
from train_models import (
    load_dataset, prepare_data, load_config,
    create_sequences, add_technical_indicators
)
from pytorch_models import get_model as get_pytorch_model
from pytorch_train import get_device

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class TradingSimulator:
    """Simulate trading with model predictions."""
    
    def __init__(self, transaction_cost: float = 0.0025):
        """
        Initialize trading simulator.
        
        Args:
            transaction_cost: Fee per trade (0.0025 = 0.25% for Bitvavo)
        """
        self.transaction_cost = transaction_cost
        self.trades = []
        self.equity_curve = []
        
    def simulate(self, predictions: np.ndarray, prices: np.ndarray,
                 timestamps: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate trading based on predictions.
        
        Based on Imperial College paper methodology:
        - Long on Rise (2), exit on Fall (0)
        - Short on Fall (0), exit on Rise (2)
        - Hold on Stationary (1)
        
        Args:
            predictions: Model predictions (0=Fall, 1=Stationary, 2=Rise)
            prices: Actual prices for calculating returns
            timestamps: Optional timestamps for trade log
            
        Returns:
            Dictionary with trading metrics
        """
        self.trades = []
        self.equity_curve = []
        
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        entry_time = 0
        equity = 1.0  # Starting equity (normalized to 1)
        self.equity_curve.append(equity)
        
        for i in range(len(predictions)):
            pred = int(predictions[i])
            current_price = prices[i]
            current_time = timestamps[i] if timestamps is not None else i
            
            # Trading logic based on Imperial College paper
            if position == 0:  # No position
                if pred == 2:  # Rise -> Enter long
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                elif pred == 0:  # Fall -> Enter short
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                # Stationary (1) -> Hold (no position)
            elif position == 1:  # Long position
                if pred == 0:  # Fall -> Exit long
                    return_pct = (current_price - entry_price) / entry_price
                    net_return = return_pct - (2 * self.transaction_cost)  # Entry + exit costs
                    equity *= (1 + net_return)
                    self.trades.append({
                        'type': 'long',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct * 100,
                        'net_return_pct': net_return * 100,
                        'equity': equity
                    })
                    position = 0
                # Rise (2) or Stationary (1) -> Hold position
            elif position == -1:  # Short position
                if pred == 2:  # Rise -> Exit short
                    return_pct = (entry_price - current_price) / entry_price
                    net_return = return_pct - (2 * self.transaction_cost)  # Entry + exit costs
                    equity *= (1 + net_return)
                    self.trades.append({
                        'type': 'short',
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'return_pct': return_pct * 100,
                        'net_return_pct': net_return * 100,
                        'equity': equity
                    })
                    position = 0
                # Fall (0) or Stationary (1) -> Hold position
            
            self.equity_curve.append(equity)
        
        # Close any open position at the end
        if position != 0:
            final_price = prices[-1]
            if position == 1:
                return_pct = (final_price - entry_price) / entry_price
            else:
                return_pct = (entry_price - final_price) / entry_price
            
            net_return = return_pct - (2 * self.transaction_cost)
            equity *= (1 + net_return)
            self.trades.append({
                'type': 'long' if position == 1 else 'short',
                'entry_time': entry_time,
                'exit_time': timestamps[-1] if timestamps is not None else len(predictions) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'return_pct': return_pct * 100,
                'net_return_pct': net_return * 100,
                'equity': equity
            })
            self.equity_curve[-1] = equity
        
        return self._calculate_metrics(prices)
    
    def _calculate_metrics(self, prices: np.ndarray) -> Dict:
        """Calculate trading performance metrics."""
        if not self.trades:
            return {
                'total_return_pct': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'avg_return_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'cpr': 0.0,  # Cumulative Price Return
                'trades': pd.DataFrame(),
                'equity_curve': np.array(self.equity_curve)
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Total return
        total_return_pct = (self.equity_curve[-1] - 1.0) * 100
        
        # Number of trades
        num_trades = len(self.trades)
        
        # Win rate
        winning_trades = trades_df[trades_df['net_return_pct'] > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        
        # Average return per trade
        avg_return_per_trade = trades_df['net_return_pct'].mean()
        
        # Sharpe Ratio (annualized, assuming hourly data)
        returns = trades_df['net_return_pct'].values / 100.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100
        
        # Cumulative Price Return (CPR) - as per Imperial College paper
        # This is the sum of all trade returns
        cpr = trades_df['net_return_pct'].sum()
        
        return {
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cpr': cpr,
            'trades': trades_df,
            'equity_curve': np.array(self.equity_curve)
        }


def load_pytorch_model_and_scaler(model_path: str, scaler_path: str, config: ConfigParser):
    """Load trained PyTorch model and scaler."""
    device = get_device()
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    # Handle both dict format and direct scaler
    if isinstance(scaler_data, dict):
        feature_scaler = scaler_data.get('feature_scaler')
        sequence_length = scaler_data.get('sequence_length', config.getint('DATA', 'sequence_length'))
        feature_names = scaler_data.get('feature_names', [])
    else:
        feature_scaler = scaler_data
        sequence_length = config.getint('DATA', 'sequence_length')
        feature_names = []
    
    # Load model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model architecture info from checkpoint or config
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Checkpoint format with metadata
        model_state = checkpoint['model_state_dict']
        input_shape = checkpoint.get('input_shape', (sequence_length, len(feature_names) if feature_names else 10))
        output_units = checkpoint.get('output_units', 3)  # Classification: 3 classes
        units = checkpoint.get('units', config.getint('MODEL', 'units'))
        dropout = checkpoint.get('dropout', config.getfloat('MODEL', 'dropout'))
        model_name = checkpoint.get('model_name', 'dlstm')
    else:
        # Direct state dict
        model_state = checkpoint
        input_shape = (sequence_length, len(feature_names) if feature_names else 10)
        output_units = 3
        units = config.getint('MODEL', 'units')
        dropout = config.getfloat('MODEL', 'dropout')
        model_name = 'dlstm'  # Default, will try to infer from filename
    
    # Infer model name from filename if not in checkpoint
    if model_name == 'dlstm' and 'bilstm' in str(model_path).lower():
        model_name = 'bilstm'
    elif model_name == 'dlstm' and 'lstm' in str(model_path).lower() and 'bilstm' not in str(model_path).lower():
        model_name = 'lstm'
    elif model_name == 'dlstm' and 'gru' in str(model_path).lower():
        model_name = 'gru'
    
    # Create model
    ma_window = config.getint('DLSTM', 'moving_average_window', fallback=10) if model_name == 'dlstm' else 10
    model = get_pytorch_model(
        model_name=model_name,
        input_shape=input_shape,
        output_units=output_units,
        units=units,
        dropout=dropout,
        task='classification',
        ma_window=ma_window
    )
    
    # Load state
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model, {
        'feature_scaler': feature_scaler,
        'target_scaler': feature_scaler,  # Same scaler for features
        'feature_names': feature_names,
        'sequence_length': sequence_length
    }


def backtest_model(model_name: str, dataset_name: str, 
                   config: ConfigParser, transaction_cost: float = 0.0025) -> Dict:
    """
    Backtest a trained PyTorch model.
    
    Args:
        model_name: Name of the model (lstm, gru, bilstm, dlstm)
        dataset_name: Name of the dataset
        config: Configuration object
        transaction_cost: Transaction cost per trade (default 0.25%)
        
    Returns:
        Dictionary with backtesting results
    """
    print(f"\n{'='*60}")
    print(f"Backtesting {model_name.upper()} on {dataset_name}")
    print(f"{'='*60}")
    
    # Paths
    base_dir = Path(__file__).parent
    models_dir = base_dir / config.get('OUTPUT', 'models_dir')
    scalers_dir = base_dir / config.get('OUTPUT', 'scalers_dir')
    datasets_dir = base_dir / 'datasets'
    results_dir = base_dir / config.get('OUTPUT', 'results_dir')
    
    # Find model and scaler files
    model_pattern = f"{model_name}_{dataset_name}_classification.pth"
    scaler_pattern = f"scaler_{dataset_name}.pkl"
    
    model_file = None
    scaler_file = None
    
    for f in models_dir.glob('*.pth'):
        if model_pattern in f.name or f"{model_name}_{dataset_name}" in f.name:
            model_file = f
            break
    
    for f in scalers_dir.glob('*.pkl'):
        if scaler_pattern in f.name:
            scaler_file = f
            break
    
    if model_file is None:
        print(f"  ✗ Model file not found: {model_pattern}")
        return None
    
    if scaler_file is None:
        print(f"  ✗ Scaler file not found: {scaler_pattern}")
        return None
    
    print(f"  ✓ Model: {model_file.name}")
    print(f"  ✓ Scaler: {scaler_file.name}")
    
    # Load model and scaler
    try:
        model, scaler_data = load_pytorch_model_and_scaler(
            str(model_file), str(scaler_file), config
        )
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load dataset
    dataset_path = datasets_dir / f"{dataset_name}.csv"
    if not dataset_path.exists():
        print(f"  ✗ Dataset not found: {dataset_path}")
        return None
    
    print(f"  ✓ Loading dataset: {dataset_path.name}")
    df = load_dataset(str(dataset_path))
    
    # Prepare data (same as training)
    print(f"  ✓ Preparing data...")
    data_dict = prepare_data(df, config, task='classification')
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    feature_scaler = data_dict['feature_scaler']
    target_scaler = data_dict['target_scaler']
    n_features = data_dict['n_features']
    sequence_length = config.getint('DATA', 'sequence_length')
    
    # Get prices for test set (need to account for sequence_length offset)
    train_test_split = config.getfloat('DATA', 'train_test_split')
    k = config.getint('CLASSIFICATION', 'smoothing_k')
    # Adjust for sequence_length and smoothing_k offsets
    total_offset = sequence_length + k
    split_idx = int(len(df) * train_test_split)
    prices_test = df['close'].values[split_idx + total_offset:split_idx + total_offset + len(X_test)]
    
    print(f"  ✓ Test set: {len(X_test)} samples")
    
    # Make predictions
    print(f"  ✓ Making predictions...")
    device = get_device()
    model.eval()
    
    predictions = []
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        batch_size = 1024
        
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            probs = model(batch).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            predictions.extend(preds)
    
    predictions = np.array(predictions)
    
    # Simulate trading
    print(f"  ✓ Simulating trading with {transaction_cost*100:.2f}% transaction costs...")
    simulator = TradingSimulator(transaction_cost=transaction_cost)
    results = simulator.simulate(predictions, prices_test)
    
    # Print results
    print(f"\n  Results:")
    print(f"    Total Return: {results['total_return_pct']:.2f}%")
    print(f"    Number of Trades: {results['num_trades']}")
    print(f"    Win Rate: {results['win_rate']*100:.2f}%")
    print(f"    Avg Return per Trade: {results['avg_return_per_trade']:.2f}%")
    print(f"    Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"    CPR (Cumulative Price Return): {results['cpr']:.2f}%")
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"backtest_{model_name}_{dataset_name}.json"
    
    import json
    results_dict = {
        'model': model_name,
        'dataset': dataset_name,
        'transaction_cost': transaction_cost,
        'total_return_pct': results['total_return_pct'],
        'num_trades': results['num_trades'],
        'win_rate': results['win_rate'],
        'avg_return_per_trade': results['avg_return_per_trade'],
        'sharpe_ratio': results['sharpe_ratio'],
        'max_drawdown': results['max_drawdown'],
        'cpr': results['cpr']
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"  ✓ Results saved to: {results_file}")
    
    # Plot equity curve
    if len(results['equity_curve']) > 0:
        plot_file = results_dir / f"backtest_equity_{model_name}_{dataset_name}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f'{model_name.upper()} Backtest Equity Curve\n'
                 f'Total Return: {results["total_return_pct"]:.2f}% | '
                 f'Sharpe: {results["sharpe_ratio"]:.2f} | '
                 f'Max DD: {results["max_drawdown"]:.2f}%')
        plt.xlabel('Time Step')
        plt.ylabel('Equity (normalized)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Equity curve saved to: {plot_file}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Backtest PyTorch trading models')
    parser.add_argument('--model', type=str, default='dlstm',
                       choices=['lstm', 'gru', 'bilstm', 'dlstm', 'all'],
                       help='Model to backtest')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ADA-EUR_1H_20240101-20251231)')
    parser.add_argument('--config', type=str, default='training_config.txt',
                       help='Path to training config file')
    parser.add_argument('--transaction_cost', type=float, default=0.0025,
                       help='Transaction cost per trade (default: 0.0025 = 0.25%%)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Backtest model(s)
    if args.model == 'all':
        models = ['lstm', 'gru', 'bilstm', 'dlstm']
        all_results = {}
        for model_name in models:
            results = backtest_model(model_name, args.dataset, config, args.transaction_cost)
            if results:
                all_results[model_name] = results
        
        # Print comparison
        if all_results:
            print(f"\n{'='*60}")
            print("COMPARISON OF ALL MODELS")
            print(f"{'='*60}")
            print(f"{'Model':<10} {'Return %':<12} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Max DD %':<10}")
            print("-" * 60)
            for model_name, results in all_results.items():
                print(f"{model_name.upper():<10} "
                      f"{results['total_return_pct']:>10.2f}% "
                      f"{results['num_trades']:>7} "
                      f"{results['win_rate']*100:>9.2f}% "
                      f"{results['sharpe_ratio']:>7.2f} "
                      f"{results['max_drawdown']:>9.2f}%")
    else:
        backtest_model(args.model, args.dataset, config, args.transaction_cost)


if __name__ == '__main__':
    main()

