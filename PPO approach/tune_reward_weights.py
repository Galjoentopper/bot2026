#!/usr/bin/env python3
"""
Reward Function Weight Tuning Script
=====================================
Automated tuning of reward function weights for PPO agent.

Tests different reward weight combinations and selects the best configuration.

Usage:
    python tune_reward_weights.py --dataset ADA-EUR_1H_20240101-20251231 --timesteps 50000
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from colab_utils import (
    get_project_path,
    get_ppo_path,
    get_results_path,
    get_datasets_path,
    setup_environment
)

# Import dependencies
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from prediction_wrapper import load_ensemble
    from trading_env import TradingEnv
    from train_ppo_agent import load_config, create_ppo_agent
except ImportError as e:
    print(f"Error importing: {e}")
    print("Install dependencies: pip install stable-baselines3 gymnasium")
    sys.exit(1)


def test_reward_config(
    reward_config: Dict,
    dataset_name: str,
    timesteps: int = 50000,
    n_envs: int = 4,
    eval_freq: int = 10000,
) -> Dict:
    """
    Test a specific reward configuration.
    
    Args:
        reward_config: Dictionary with reward parameters
        dataset_name: Dataset to train on
        timesteps: Number of training timesteps
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
        
    Returns:
        Dictionary with test results
    """
    print(f"\nTesting config: profit_scale={reward_config['profit_scale']}, "
          f"sharpe_bonus={reward_config['sharpe_bonus']}, "
          f"drawdown_penalty={reward_config['drawdown_penalty']}, "
          f"hold_penalty={reward_config.get('enable_hold_penalty', False)}")
    
    try:
        # Load base config
        ppo_path = get_ppo_path()
        config_path = ppo_path / 'ppo_config.txt'
        config = load_config(str(config_path) if config_path.exists() else None)
        
        # Update reward config
        config['reward'].update(reward_config)
        
        # Load prediction models
        prediction_models = load_ensemble(dataset_name)
        if not prediction_models.loaded:
            print("  Failed to load prediction models")
            return {'error': 'Failed to load models'}
        
        # Find dataset
        datasets_path = get_datasets_path()
        dataset_files = list(datasets_path.glob(f"*{dataset_name}*.csv"))
        if not dataset_files:
            print(f"  Dataset not found: {dataset_name}")
            return {'error': 'Dataset not found'}
        
        dataset_path = dataset_files[0]
        
        # Create environments
        def make_env(train_mode=True):
            def _init():
                return TradingEnv(
                    dataset_path=str(dataset_path),
                    prediction_models=prediction_models,
                    transaction_cost=config['environment']['transaction_cost'],
                    initial_capital=config['environment']['initial_capital'],
                    sequence_length=config['environment']['sequence_length'],
                    train_mode=train_mode,
                    train_split=config['training']['train_test_split'],
                    reward_config=config['reward'],
                    max_episode_steps=config['training']['max_episode_steps'],
                    prediction_horizons=config['models'].get('prediction_horizons', [1, 2, 3]),
                )
            return _init
        
        train_env = DummyVecEnv([make_env(train_mode=True) for _ in range(n_envs)])
        eval_env = DummyVecEnv([make_env(train_mode=False) for _ in range(1)])
        
        # Create PPO agent
        ppo_config = config['ppo']
        model = create_ppo_agent(
            env=train_env,
            config=ppo_config,
            tensorboard_log=None,  # Disable for tuning
            device='auto'
        )
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,  # Don't save during tuning
            log_path=None,
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )
        
        # Train
        model.learn(
            total_timesteps=timesteps,
            callback=eval_callback,
            progress_bar=False,
        )
        
        # Get evaluation results
        mean_reward = eval_callback.best_mean_reward if hasattr(eval_callback, 'best_mean_reward') else 0
        
        # Run final evaluation
        obs = eval_env.reset()
        episode_rewards = []
        for _ in range(5):  # 5 evaluation episodes
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_rewards.append(episode_reward)
            obs = eval_env.reset()
        
        final_mean_reward = np.mean(episode_rewards)
        final_std_reward = np.std(episode_rewards)
        
        # Cleanup
        train_env.close()
        eval_env.close()
        
        results = {
            'reward_config': reward_config,
            'mean_reward': float(final_mean_reward),
            'std_reward': float(final_std_reward),
            'best_mean_reward': float(mean_reward),
            'timesteps': timesteps,
        }
        
        print(f"  Results: Mean reward = {final_mean_reward:.2f} ± {final_std_reward:.2f}")
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def tune_reward_weights(
    dataset_name: str,
    timesteps: int = 50000,
    n_envs: int = 4,
) -> Dict:
    """
    Tune reward function weights by testing different combinations.
    
    Args:
        dataset_name: Dataset to train on
        timesteps: Number of training timesteps per configuration
        n_envs: Number of parallel environments
        
    Returns:
        Dictionary with best configuration and all results
    """
    print("=" * 80)
    print("REWARD FUNCTION WEIGHT TUNING")
    print("=" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"Timesteps per config: {timesteps}")
    print(f"Parallel environments: {n_envs}")
    
    # Define search space
    profit_scales = [25, 50, 75, 100]
    sharpe_bonuses = [0.5, 1.0, 1.5, 2.0]
    drawdown_penalties = [0.01, 0.05, 0.1, 0.2]
    enable_hold_penalties = [True, False]
    
    # Generate all combinations
    configs = []
    for ps, sb, dp, hp in product(profit_scales, sharpe_bonuses, drawdown_penalties, enable_hold_penalties):
        configs.append({
            'profit_scale': ps,
            'sharpe_bonus': sb,
            'drawdown_penalty': dp,
            'enable_hold_penalty': hp,
            'hold_penalty': 0.005 if hp else 0.0,
        })
    
    print(f"\nTesting {len(configs)} configurations...")
    print("=" * 80)
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] ", end='')
        result = test_reward_config(
            reward_config=config,
            dataset_name=dataset_name,
            timesteps=timesteps,
            n_envs=n_envs,
        )
        
        if 'error' not in result:
            results.append(result)
    
    if not results:
        print("\nNo successful configurations!")
        return {}
    
    # Find best configuration
    best_result = max(results, key=lambda x: x['mean_reward'])
    best_config = best_result['reward_config']
    
    # Print summary
    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    print(f"\nBest Configuration:")
    print(f"  profit_scale: {best_config['profit_scale']}")
    print(f"  sharpe_bonus: {best_config['sharpe_bonus']}")
    print(f"  drawdown_penalty: {best_config['drawdown_penalty']}")
    print(f"  enable_hold_penalty: {best_config['enable_hold_penalty']}")
    print(f"\nPerformance:")
    print(f"  Mean Reward: {best_result['mean_reward']:.2f} ± {best_result['std_reward']:.2f}")
    
    # Top 5 configurations
    sorted_results = sorted(results, key=lambda x: x['mean_reward'], reverse=True)
    print(f"\nTop 5 Configurations:")
    for i, result in enumerate(sorted_results[:5], 1):
        cfg = result['reward_config']
        print(f"  {i}. Reward: {result['mean_reward']:.2f} | "
              f"profit={cfg['profit_scale']}, sharpe={cfg['sharpe_bonus']}, "
              f"dd={cfg['drawdown_penalty']}, hold={cfg['enable_hold_penalty']}")
    
    # Save results
    results_dir = get_results_path()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f'reward_tuning_{dataset_name}_{timestamp}.json'
    
    output = {
        'dataset': dataset_name,
        'timestamp': timestamp,
        'best_config': best_config,
        'best_result': best_result,
        'all_results': results,
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Tune reward function weights')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Training timesteps per configuration')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    
    args = parser.parse_args()
    
    results = tune_reward_weights(
        dataset_name=args.dataset,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
    )
    
    if results:
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())

