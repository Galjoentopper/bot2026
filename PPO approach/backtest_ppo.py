#!/usr/bin/env python3
"""
PPO Agent Backtesting Script
==============================
Comprehensive backtesting for trained PPO trading agents.

Usage:
    python backtest_ppo.py --model ppo_ensemble_ADA-EUR_1H_20240101-20251231.zip
    python backtest_ppo.py --model ppo_ensemble_ADA-EUR_1H_20240101-20251231.zip --episodes 20
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from colab_utils import (
    get_project_path,
    get_ppo_path,
    get_ppo_models_path,
    get_results_path,
    get_datasets_path,
    setup_environment
)
from utils import calculate_sharpe, calculate_max_drawdown

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


def calculate_trading_metrics(equity_curve: List[float], trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive trading metrics.
    
    Args:
        equity_curve: List of equity values over time
        trades: List of trade dictionaries with 'pnl', 'entry_time', 'exit_time'
        
    Returns:
        Dictionary of trading metrics
    """
    if not equity_curve:
        return {}
    
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    # Basic metrics
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    sharpe_ratio = calculate_sharpe(returns) if len(returns) > 1 else 0.0
    max_dd = calculate_max_drawdown(equity)
    
    # Trade metrics
    if trades:
        pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        win_rate = len(winning_trades) / len(pnls) * 100 if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        
        # Holding periods
        holding_periods = []
        for trade in trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                holding_periods.append(trade['exit_time'] - trade['entry_time'])
        avg_holding = np.mean(holding_periods) if holding_periods else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_holding = 0
        pnls = []
    
    return {
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_dd,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_holding_period': avg_holding,
        'final_equity': equity[-1],
        'initial_equity': equity[0],
    }


def backtest_ppo_agent(
    model_path: str,
    dataset_name: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    save_results: bool = True
) -> Dict:
    """
    Backtest a trained PPO agent.
    
    Args:
        model_path: Path to saved model (.zip)
        dataset_name: Dataset to backtest on
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        save_results: Save results to file
        
    Returns:
        Dictionary of backtest results
    """
    print("=" * 80)
    print("PPO AGENT BACKTESTING")
    print("=" * 80)
    
    # Import dependencies
    try:
        from stable_baselines3 import PPO
        from prediction_wrapper import load_ensemble
        from trading_env import TradingEnv
    except ImportError as e:
        print(f"Error importing: {e}")
        print("Install dependencies: pip install stable-baselines3 gymnasium")
        return {}
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        # Try in models directory
        model_path = get_ppo_models_path() / model_path.name
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return {}
    
    print(f"\nLoading model: {model_path}")
    model = PPO.load(str(model_path))
    
    # Load prediction models
    print(f"\nLoading prediction models for: {dataset_name}")
    prediction_models = load_ensemble(dataset_name)
    if not prediction_models.loaded:
        print("Failed to load prediction models")
        return {}
    
    # Find dataset
    datasets_path = get_datasets_path()
    dataset_files = list(datasets_path.glob(f"*{dataset_name}*.csv"))
    
    if not dataset_files:
        print(f"Dataset not found: {dataset_name}")
        return {}
    
    dataset_path = dataset_files[0]
    print(f"Using dataset: {dataset_path.name}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {len(df)} rows")
    
    # Create environment (use test split)
    from train_ppo_agent import load_config
    ppo_path = get_ppo_path()
    config_path = ppo_path / 'ppo_config.txt'
    config = load_config(str(config_path) if config_path.exists() else None)
    
    env = TradingEnv(
        dataset_path=str(dataset_path),
        prediction_models=prediction_models,
        transaction_cost=config['environment']['transaction_cost'],
        initial_capital=config['environment']['initial_capital'],
        sequence_length=config['environment']['sequence_length'],
        train_mode=False,  # Use test data
        validation_mode=False,  # Use test data (20%, models never saw this)
        train_split=config['training']['train_test_split'],
        validation_split=config['training'].get('validation_split', 0.2),
        reward_config=config['reward'],
        max_episode_steps=config['training']['max_episode_steps'],
        prediction_horizons=config['models'].get('prediction_horizons', [1, 2, 3]),
    )
    
    print(f"\nRunning {n_episodes} backtest episodes...")
    print("-" * 80)
    
    all_episodes = []
    all_equity_curves = []
    all_trades = []
    all_rewards = []
    all_actions = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_rewards = []
        episode_actions = []
        episode_equity = [env.portfolio.state.total_equity]
        episode_trades = []
        
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            episode_actions.append(int(action))
            episode_equity.append(env.portfolio.state.total_equity)
            
            # Track trades if available in info
            if 'trade' in info and info['trade']:
                episode_trades.append(info['trade'])
            
            step += 1
        
        # Calculate episode metrics
        episode_metrics = calculate_trading_metrics(episode_equity, episode_trades)
        episode_metrics['episode'] = episode + 1
        episode_metrics['episode_length'] = step
        episode_metrics['total_reward'] = sum(episode_rewards)
        episode_metrics['mean_reward'] = np.mean(episode_rewards) if episode_rewards else 0
        
        all_episodes.append(episode_metrics)
        all_equity_curves.append(episode_equity)
        all_trades.extend(episode_trades)
        all_rewards.extend(episode_rewards)
        all_actions.extend(episode_actions)
        
        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Return: {episode_metrics['total_return_pct']:.2f}%, "
              f"Trades: {episode_metrics['num_trades']}, "
              f"Reward: {episode_metrics['mean_reward']:.2f}")
    
    env.close()
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        'num_episodes': n_episodes,
        'mean_return': np.mean([e['total_return_pct'] for e in all_episodes]),
        'std_return': np.std([e['total_return_pct'] for e in all_episodes]),
        'mean_sharpe': np.mean([e['sharpe_ratio'] for e in all_episodes]),
        'mean_max_dd': np.mean([e['max_drawdown'] for e in all_episodes]),
        'total_trades': len(all_trades),
        'mean_win_rate': np.mean([e['win_rate'] for e in all_episodes]),
        'mean_profit_factor': np.mean([e['profit_factor'] for e in all_episodes if e['profit_factor'] > 0]),
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
    }
    
    # Action distribution
    action_counts = {}
    for action in all_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    aggregate_metrics['action_distribution'] = action_counts
    
    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nAggregate Metrics ({n_episodes} episodes):")
    print(f"  Mean Return: {aggregate_metrics['mean_return']:.2f}% ± {aggregate_metrics['std_return']:.2f}%")
    print(f"  Mean Sharpe Ratio: {aggregate_metrics['mean_sharpe']:.2f}")
    print(f"  Mean Max Drawdown: {aggregate_metrics['mean_max_dd']:.2f}")
    print(f"  Total Trades: {aggregate_metrics['total_trades']}")
    print(f"  Mean Win Rate: {aggregate_metrics['mean_win_rate']:.2f}%")
    print(f"  Mean Profit Factor: {aggregate_metrics['mean_profit_factor']:.2f}")
    print(f"  Mean Reward: {aggregate_metrics['mean_reward']:.2f} ± {aggregate_metrics['std_reward']:.2f}")
    
    print(f"\nAction Distribution:")
    for action, count in sorted(action_counts.items()):
        pct = count / len(all_actions) * 100
        print(f"  Action {action}: {count} ({pct:.1f}%)")
    
    # Generate visualizations
    if save_results:
        results_dir = get_results_path()
        results_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = model_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot equity curves
        fig, ax = plt.subplots(figsize=(14, 8))
        for i, equity_curve in enumerate(all_equity_curves):
            ax.plot(equity_curve, alpha=0.3, label=f'Episode {i+1}' if i < 5 else None)
        
        # Average equity curve
        if all_equity_curves:
            max_len = max(len(eq) for eq in all_equity_curves)
            avg_equity = []
            for step in range(max_len):
                step_values = [eq[step] for eq in all_equity_curves if step < len(eq)]
                if step_values:
                    avg_equity.append(np.mean(step_values))
            ax.plot(avg_equity, 'k-', linewidth=2, label='Average')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Equity')
        ax.set_title(f'Equity Curves - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = results_dir / f'backtest_equity_{model_name}_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved equity curve plot: {plot_path}")
        
        # Save results JSON
        results_data = {
            'model': str(model_path),
            'dataset': dataset_name,
            'timestamp': timestamp,
            'aggregate_metrics': aggregate_metrics,
            'episodes': all_episodes,
        }
        
        json_path = results_dir / f'backtest_{model_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved results JSON: {json_path}")
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'episodes': all_episodes,
        'equity_curves': all_equity_curves,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backtest PPO trading agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PPO model (.zip file)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (auto-detected from model if not specified)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic actions')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    # Auto-detect dataset from model name if not provided
    dataset_name = args.dataset
    if not dataset_name:
        model_name = Path(args.model).stem
        # Extract dataset from model name (e.g., ppo_ensemble_ADA-EUR_1H_20240101-20251231)
        parts = model_name.split('_')
        if len(parts) >= 3:
            dataset_name = '_'.join(parts[2:])
        else:
            print("Error: Could not auto-detect dataset name. Please specify --dataset")
            return 1
    
    results = backtest_ppo_agent(
        model_path=args.model,
        dataset_name=dataset_name,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        save_results=not args.no_save
    )
    
    if results:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

