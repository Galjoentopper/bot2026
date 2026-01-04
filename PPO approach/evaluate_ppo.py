#!/usr/bin/env python3
"""
PPO Agent Evaluation Script
============================
Evaluate trained PPO agent and compare with baselines.

Usage:
    python evaluate_ppo.py --model models/ppo_final.zip --dataset ETH-EUR_1H_20240101-20251231
    python evaluate_ppo.py --model models/ppo_dlstm.zip --compare-baselines
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
from utils import format_metrics, save_metrics, calculate_sharpe, calculate_max_drawdown


def evaluate_ppo_agent(
    model_path: str,
    dataset_name: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    save_results: bool = True
) -> Dict:
    """
    Evaluate a trained PPO agent.
    
    Args:
        model_path: Path to saved model (.zip)
        dataset_name: Dataset to evaluate on
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        render: Render environment
        save_results: Save results to file
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 60)
    print("PPO AGENT EVALUATION")
    print("=" * 60)
    
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
    
    # Find dataset
    datasets_path = get_datasets_path()
    dataset_files = list(datasets_path.glob(f"*{dataset_name}*"))
    
    if not dataset_files:
        print(f"Dataset not found: {dataset_name}")
        return {}
    
    dataset_path = dataset_files[0]
    print(f"Using dataset: {dataset_path.name}")
    
    # Create evaluation environment (test data)
    env = TradingEnv(
        dataset_path=dataset_path,
        prediction_models=prediction_models if prediction_models.loaded else None,
        train_mode=False,  # Use test data
        validation_mode=False,  # Use test data (20%, models never saw this)
        train_split=0.6,
        validation_split=0.2,
    )
    
    print(f"\nRunning {n_episodes} evaluation episodes...")
    
    # Collect episode results
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    episode_trades = []
    episode_sharpe = []
    episode_drawdown = []
    all_actions = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            actions.append(int(action))
            
            if render:
                env.render()
        
        # Get episode metrics
        metrics = env.get_episode_metrics()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_returns.append(metrics.get('total_return_pct', 0))
        episode_trades.append(metrics.get('num_trades', 0))
        episode_sharpe.append(metrics.get('sharpe_ratio', 0))
        episode_drawdown.append(metrics.get('max_drawdown', 0))
        all_actions.extend(actions)
        
        print(f"  Episode {ep + 1}: Reward={episode_reward:.2f}, Return={metrics.get('total_return_pct', 0):.2f}%, Trades={metrics.get('num_trades', 0)}")
    
    env.close()
    
    # Calculate aggregate metrics
    results = {
        'model_path': str(model_path),
        'dataset': dataset_name,
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_return_pct': float(np.mean(episode_returns)),
        'std_return_pct': float(np.std(episode_returns)),
        'mean_trades': float(np.mean(episode_trades)),
        'mean_sharpe': float(np.mean(episode_sharpe)),
        'mean_max_drawdown': float(np.mean(episode_drawdown)),
        'mean_episode_length': float(np.mean(episode_lengths)),
    }
    
    # Action distribution
    action_counts = {}
    action_names = ['Hold', 'Buy Small', 'Buy Medium', 'Buy Large', 
                   'Sell Small', 'Sell Medium', 'Sell Large', 
                   'Close Position', 'Reverse Position']
    for a in range(9):
        count = all_actions.count(a)
        results[f'action_{action_names[a]}_pct'] = count / len(all_actions) * 100 if all_actions else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nPerformance Metrics:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Return: {results['mean_return_pct']:.2f}% ± {results['std_return_pct']:.2f}%")
    print(f"  Mean Trades: {results['mean_trades']:.1f}")
    print(f"  Mean Sharpe Ratio: {results['mean_sharpe']:.3f}")
    print(f"  Mean Max Drawdown: {results['mean_max_drawdown']:.2f}%")
    
    print(f"\nAction Distribution:")
    for i, name in enumerate(action_names):
        pct = results.get(f'action_{name}_pct', 0)
        if pct > 0:
            print(f"  {name}: {pct:.1f}%")
    
    # Save results
    if save_results:
        results_path = get_results_path()
        model_name = model_path.stem
        results_file = results_path / f"eval_{model_name}_{dataset_name}.json"
        save_metrics(results, str(results_file))
        print(f"\nResults saved to: {results_file}")
    
    return results


def compare_with_baselines(
    model_path: str,
    dataset_name: str,
    n_episodes: int = 5
) -> pd.DataFrame:
    """
    Compare PPO agent with baseline strategies.
    
    Baselines:
    - Buy and Hold
    - Rule-based (prediction model signals)
    - Random
    
    Args:
        model_path: Path to PPO model
        dataset_name: Dataset name
        n_episodes: Number of episodes
        
    Returns:
        DataFrame with comparison results
    """
    print("\n" + "=" * 60)
    print("COMPARING WITH BASELINES")
    print("=" * 60)
    
    results = []
    
    # 1. Evaluate PPO agent
    print("\n1. PPO Agent")
    ppo_results = evaluate_ppo_agent(
        model_path=model_path,
        dataset_name=dataset_name,
        n_episodes=n_episodes,
        save_results=False
    )
    if ppo_results:
        results.append({
            'Strategy': 'PPO Agent',
            'Mean Return %': ppo_results['mean_return_pct'],
            'Std Return %': ppo_results['std_return_pct'],
            'Mean Trades': ppo_results['mean_trades'],
            'Mean Sharpe': ppo_results['mean_sharpe'],
            'Mean Max DD %': ppo_results['mean_max_drawdown'],
        })
    
    # 2. Buy and Hold baseline
    print("\n2. Buy and Hold Baseline")
    bh_results = evaluate_buy_and_hold(dataset_name)
    if bh_results:
        results.append({
            'Strategy': 'Buy & Hold',
            'Mean Return %': bh_results['return_pct'],
            'Std Return %': 0,
            'Mean Trades': 1,
            'Mean Sharpe': bh_results['sharpe_ratio'],
            'Mean Max DD %': bh_results['max_drawdown'],
        })
    
    # 3. Random baseline
    print("\n3. Random Baseline")
    random_results = evaluate_random_agent(dataset_name, n_episodes)
    if random_results:
        results.append({
            'Strategy': 'Random',
            'Mean Return %': random_results['mean_return_pct'],
            'Std Return %': random_results['std_return_pct'],
            'Mean Trades': random_results['mean_trades'],
            'Mean Sharpe': random_results['mean_sharpe'],
            'Mean Max DD %': random_results['mean_max_drawdown'],
        })
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # Save comparison
    results_path = get_results_path()
    comparison_file = results_path / f"comparison_{dataset_name}.csv"
    df.to_csv(comparison_file, index=False)
    print(f"\nComparison saved to: {comparison_file}")
    
    return df


def evaluate_buy_and_hold(dataset_name: str) -> Dict:
    """Evaluate buy and hold strategy."""
    try:
        datasets_path = get_datasets_path()
        dataset_files = list(datasets_path.glob(f"*{dataset_name}*"))
        
        if not dataset_files:
            return {}
        
        df = pd.read_csv(dataset_files[0])
        prices = df['close'].values
        
        # Use test portion (last 20%)
        split_idx = int(len(prices) * 0.8)
        test_prices = prices[split_idx:]
        
        # Calculate return
        return_pct = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
        
        # Calculate daily returns for Sharpe
        returns = np.diff(test_prices) / test_prices[:-1]
        sharpe = calculate_sharpe(list(returns))
        
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(test_prices)
        
        return {
            'return_pct': return_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
        }
    except Exception as e:
        print(f"Error in buy and hold: {e}")
        return {}


def evaluate_random_agent(dataset_name: str, n_episodes: int = 5) -> Dict:
    """Evaluate random action agent."""
    try:
        from trading_env import TradingEnv
        
        datasets_path = get_datasets_path()
        dataset_files = list(datasets_path.glob(f"*{dataset_name}*"))
        
        if not dataset_files:
            return {}
        
        env = TradingEnv(
            dataset_path=dataset_files[0],
            prediction_models=None,
            train_mode=False,
            validation_mode=False,  # Use test data
            train_split=0.6,
            validation_split=0.2,
            max_episode_steps=1000,
        )
        
        episode_returns = []
        episode_trades = []
        episode_sharpe = []
        episode_drawdown = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            
            metrics = env.get_episode_metrics()
            episode_returns.append(metrics.get('total_return_pct', 0))
            episode_trades.append(metrics.get('num_trades', 0))
            episode_sharpe.append(metrics.get('sharpe_ratio', 0))
            episode_drawdown.append(metrics.get('max_drawdown', 0))
        
        env.close()
        
        return {
            'mean_return_pct': np.mean(episode_returns),
            'std_return_pct': np.std(episode_returns),
            'mean_trades': np.mean(episode_trades),
            'mean_sharpe': np.mean(episode_sharpe),
            'mean_max_drawdown': np.mean(episode_drawdown),
        }
    except Exception as e:
        print(f"Error in random agent: {e}")
        return {}


def evaluate_agent(model, env, n_episodes: int = 10, deterministic: bool = True) -> Dict:
    """
    Simple evaluation function (for use in notebooks).
    
    Args:
        model: Trained PPO model
        env: Environment
        n_episodes: Number of episodes
        deterministic: Use deterministic actions
        
    Returns:
        Results dictionary
    """
    episode_rewards = []
    episode_returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            episode_returns.append(metrics.get('total_return_pct', 0))
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_return_pct': np.mean(episode_returns) if episode_returns else 0,
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO trading agent')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved PPO model (.zip)')
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    parser.add_argument('--compare-baselines', action='store_true',
                       help='Compare with baseline strategies')
    
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment(verbose=False)
    
    if args.compare_baselines:
        compare_with_baselines(
            model_path=args.model,
            dataset_name=args.dataset,
            n_episodes=args.episodes
        )
    else:
        evaluate_ppo_agent(
            model_path=args.model,
            dataset_name=args.dataset,
            n_episodes=args.episodes,
            render=args.render
        )


if __name__ == '__main__':
    main()


