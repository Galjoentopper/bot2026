#!/usr/bin/env python3
"""
Trading Backtesting Script
==========================
Simulates trading with trained classification models to verify profitability.

Based on Imperial College paper methodology:
- Enter long position when prediction = 2 (Rise)
- Exit long position when prediction = 0 (Fall)
- Enter short position when prediction = 0 (Fall)
- Exit short position when prediction = 2 (Rise)
- Hold on prediction = 1 (Stationary)
- Apply transaction costs: 0.25% (Bitvavo maker/taker fee)

Usage:
    python backtest_trading.py --model dlstm --dataset ETH-EUR
    python backtest_trading.py --model all --dataset ETH-EUR
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

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# Import functions from train_models.py
sys.path.insert(0, str(Path(__file__).parent))
from train_models import (
    load_dataset, prepare_data, load_config,
    create_sequences, add_technical_indicators
)

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
                if pred == 2:  # Rise prediction -> Enter long
                    position = 1
                    entry_price = current_price
                    entry_time = current_time
                elif pred == 0:  # Fall prediction -> Enter short
                    position = -1
                    entry_price = current_price
                    entry_time = current_time
                # Stationary (1) -> Do nothing
            
            elif position == 1:  # Long position
                if pred == 0:  # Fall prediction -> Exit long
                    # Calculate return
                    return_pct = (current_price - entry_price) / entry_price
                    # Apply transaction costs (entry + exit)
                    net_return = return_pct - (2 * self.transaction_cost)
                    
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
                if pred == 2:  # Rise prediction -> Exit short
                    # Calculate return (inverted for short)
                    return_pct = (entry_price - current_price) / entry_price
                    # Apply transaction costs (entry + exit)
                    net_return = return_pct - (2 * self.transaction_cost)
                    
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
                'cpr': 0.0  # Cumulative Price Return
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


def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load trained model and scaler."""
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    with open(scaler_path, 'rb') as f:
        scaler_data = pickle.load(f)
    
    # Handle both dict format (from train_models.py) and direct scaler
    if isinstance(scaler_data, dict):
        return model, scaler_data
    else:
        # Legacy format - wrap in dict
        return model, {
            'feature_scaler': scaler_data,
            'target_scaler': scaler_data,
            'feature_names': [],
            'sequence_length': 60
        }


def backtest_model(model_name: str, dataset_name: str, 
                   config: ConfigParser, transaction_cost: float = 0.0025) -> Dict:
    """
    Backtest a trained model.
    
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
    model_pattern = f"{model_name}_{dataset_name}_classification.keras"
    scaler_pattern = f"scaler_{dataset_name}.pkl"
    
    model_file = None
    scaler_file = None
    
    for f in models_dir.glob('*.keras'):
        if model_pattern in f.name:
            model_file = f
            break
    
    for f in scalers_dir.glob('*.pkl'):
        if scaler_pattern in f.name:
            scaler_file = f
            break
    
    if not model_file or not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_pattern}")
    if not scaler_file or not scaler_file.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_pattern}")
    
    print(f"Loading model: {model_file.name}")
    print(f"Loading scaler: {scaler_file.name}")
    
    # Load model and scaler
    model, scaler_data = load_model_and_scaler(str(model_file), str(scaler_file))
    
    # Load dataset
    dataset_file = datasets_dir / f"{dataset_name}.csv"
    if not dataset_file.exists():
        # Try with full name
        csv_files = list(datasets_dir.glob('*.csv'))
        matching = [f for f in csv_files if dataset_name in f.stem]
        if matching:
            dataset_file = matching[0]
        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    
    print(f"Loading dataset: {dataset_file.name}")
    df = load_dataset(str(dataset_file))
    
    # Prepare data (same as training)
    data = prepare_data(df, config, task='classification')
    
    # Get test data
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Generate predictions
    print("Generating predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Get actual prices for test set
    # Need to reconstruct prices from test indices
    target_col = config.get('DATA', 'target')
    train_test_split = config.getfloat('DATA', 'train_test_split')
    k = config.getint('CLASSIFICATION', 'smoothing_k')
    
    # Get prices (skip first k and last k due to smoothing)
    prices = df[target_col].values
    split_idx = int(len(prices) * train_test_split)
    test_prices = prices[split_idx + k:-k] if len(prices) > split_idx + 2*k else prices[split_idx:]
    
    # Align prices with predictions (they should match)
    if len(test_prices) != len(y_pred):
        # Take the last len(y_pred) prices
        test_prices = test_prices[-len(y_pred):]
    
    # Simulate trading
    print("Simulating trading...")
    simulator = TradingSimulator(transaction_cost=transaction_cost)
    metrics = simulator.simulate(y_pred, test_prices)
    
    # Print results
    print("\n" + "-"*60)
    print("BACKTESTING RESULTS")
    print("-"*60)
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Cumulative Price Return (CPR): {metrics['cpr']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    print(f"Average Return per Trade: {metrics['avg_return_per_trade']:.2f}%")
    print(f"Sharpe Ratio (annualized): {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Save results
    results_file = results_dir / f"backtest_{model_name}_{dataset_name}.csv"
    metrics_to_save = {k: v for k, v in metrics.items() if k not in ['trades', 'equity_curve']}
    pd.DataFrame([metrics_to_save]).to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save trade log
    if len(metrics['trades']) > 0:
        trades_file = results_dir / f"trades_{model_name}_{dataset_name}.csv"
        metrics['trades'].to_csv(trades_file, index=False)
        print(f"Trade log saved to: {trades_file}")
    
    # Plot equity curve
    plot_file = results_dir / f"equity_curve_{model_name}_{dataset_name}.png"
    plot_equity_curve(metrics['equity_curve'], str(plot_file))
    print(f"Equity curve saved to: {plot_file}")
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'metrics': metrics_to_save,
        'trades': metrics['trades'] if len(metrics['trades']) > 0 else pd.DataFrame()
    }


def plot_equity_curve(equity_curve: np.ndarray, save_path: str):
    """Plot equity curve."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(equity_curve, linewidth=2, label='Equity Curve')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break Even')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Equity (Normalized)')
    ax.set_title('Trading Equity Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final return annotation
    final_return = (equity_curve[-1] - 1.0) * 100
    ax.text(0.02, 0.98, f'Total Return: {final_return:.2f}%',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Backtest trading models')
    parser.add_argument('--model', type=str, default='dlstm',
                       choices=['lstm', 'gru', 'bilstm', 'dlstm', 'all'],
                       help='Model to backtest')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ETH-EUR_1H_20240101-20251231)')
    parser.add_argument('--config', type=str, default='training_config.txt',
                       help='Configuration file path')
    parser.add_argument('--transaction-cost', type=float, default=0.0025,
                       help='Transaction cost per trade (default: 0.0025 = 0.25%%)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine models to backtest
    if args.model == 'all':
        models = ['lstm', 'gru', 'bilstm', 'dlstm']
    else:
        models = [args.model]
    
    # Run backtests
    results = []
    for model_name in models:
        try:
            result = backtest_model(
                model_name=model_name,
                dataset_name=args.dataset,
                config=config,
                transaction_cost=args.transaction_cost
            )
            results.append(result)
        except Exception as e:
            print(f"\nError backtesting {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    if len(results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame([
            {
                'Model': r['model_name'].upper(),
                'Total Return %': r['metrics']['total_return_pct'],
                'CPR %': r['metrics']['cpr'],
                'Sharpe Ratio': r['metrics']['sharpe_ratio'],
                'Max Drawdown %': r['metrics']['max_drawdown'],
                'Num Trades': r['metrics']['num_trades'],
                'Win Rate %': r['metrics']['win_rate'] * 100
            }
            for r in results
        ])
        
        print(comparison.to_string(index=False))
        
        # Save comparison
        base_dir = Path(__file__).parent
        results_dir = base_dir / config.get('OUTPUT', 'results_dir')
        comparison_file = results_dir / f"backtest_comparison_{args.dataset}.csv"
        comparison.to_csv(comparison_file, index=False)
        print(f"\nComparison saved to: {comparison_file}")


if __name__ == '__main__':
    main()

