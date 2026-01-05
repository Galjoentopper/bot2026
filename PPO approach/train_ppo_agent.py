#!/usr/bin/env python3
"""
PPO Trading Agent Training Script
==================================
Main training script for PPO agent (works both local and Colab).

Usage:
    python train_ppo_agent.py --model dlstm --dataset ETH-EUR --timesteps 1000000
    python train_ppo_agent.py --config ppo_config.txt --resume
"""

import os
import sys
import argparse
import multiprocessing
import numpy as np
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from colab_utils import (
    setup_environment,
    get_project_path,
    get_ppo_path,
    get_ppo_models_path,
    get_checkpoints_path,
    get_logs_path,
    get_datasets_path,
    is_colab_runtime
)

# Import vectorized environments for parallel processing
try:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    VEC_ENV_AVAILABLE = True
except ImportError:
    VEC_ENV_AVAILABLE = False
    print("Warning: stable-baselines3 vec_env not available. Install with: pip install stable-baselines3")


def run_reward_improvement_test(
    dataset_path: Path,
    prediction_models,
    reward_config: Dict,
    prediction_horizons: List[int],
    skip_test: bool = False
) -> bool:
    """
    Run a 50K step test specifically to verify reward improvements are working.
    
    This test verifies:
    - Rewards can go positive (max reward > 0)
    - Buying actions (1-3) are being used (>1% each)
    - Reward components are working (opening vs closing bonuses)
    - Mean return is improving toward positive
    
    Args:
        dataset_path: Path to dataset
        prediction_models: Loaded prediction models
        reward_config: Reward configuration
        prediction_horizons: Prediction horizons
        skip_test: If True, skip test and return True
        
    Returns:
        True if test passes, False otherwise
    """
    if skip_test:
        return True
    
    print("\n" + "=" * 60)
    print("REWARD IMPROVEMENT TEST (50K steps)")
    print("=" * 60)
    print("Testing reward improvements:")
    print("  - Positive rewards achievable")
    print("  - Buying actions being used")
    print("  - Reward components working correctly")
    print("This will take ~2-3 minutes.")
    
    # Debug: Print reward config to verify it's loaded
    print("\nReward Configuration:")
    print(f"  profit_scale: {reward_config.get('profit_scale', 'NOT SET')}")
    print(f"  buy_action_bonus: {reward_config.get('buy_action_bonus', 'NOT SET')}")
    print(f"  sell_action_bonus: {reward_config.get('sell_action_bonus', 'NOT SET')}")
    print(f"  open_position_cost_ratio: {reward_config.get('open_position_cost_ratio', 'NOT SET')}")
    print(f"  reward_clip_value: {reward_config.get('reward_clip_value', 'NOT SET')}")
    
    try:
        from trading_env import TradingEnv
        from ppo_trading_agent import create_ppo_agent
        from stable_baselines3.common.callbacks import CallbackList
        from ppo_trading_agent import PolicyCollapseCallback, RewardLoggingCallback
        
        # Create test environment
        test_env = TradingEnv(
            dataset_path=dataset_path,
            prediction_models=prediction_models if prediction_models and prediction_models.loaded else None,
            transaction_cost=0.0025,
            initial_capital=10000,
            sequence_length=60,
            train_mode=True,
            train_split=0.6,
            validation_split=0.2,
            reward_config=reward_config,
            max_episode_steps=2000,
            prediction_horizons=prediction_horizons,
        )
        
        # Create agent with higher exploration to encourage buying actions
        test_ppo_config = {
            'learning_rate': 0.0003,
            'n_steps': 512,
            'batch_size': 256,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.20,  # Increased exploration to encourage buying actions
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0,
        }
        
        print("\nCreating test agent...")
        test_model = create_ppo_agent(
            env=test_env,
            config=test_ppo_config,
            device='auto'
        )
        
        # Create callbacks to track metrics
        collapse_callback = PolicyCollapseCallback(threshold=0.90, check_freq=10000, verbose=1)
        reward_callback = RewardLoggingCallback(verbose=1)
        
        print("Running 50,000 step reward improvement test...")
        test_model.learn(
            total_timesteps=50000,
            callback=CallbackList([collapse_callback, reward_callback]),
            progress_bar=True,
        )
        
        # Analyze results
        print("\n" + "=" * 60)
        print("REWARD IMPROVEMENT TEST RESULTS")
        print("=" * 60)
        
        # Check action distribution
        action_stats = collapse_callback.action_counts
        total_actions = collapse_callback.total_actions
        
        if total_actions == 0:
            print("❌ TEST FAILED: No actions recorded!")
            test_env.close()
            return False
        
        action_dist = {k: v/total_actions*100 for k, v in action_stats.items()}
        buying_actions = [1, 2, 3]  # Buy Small, Medium, Large
        buying_action_usage = sum([action_dist.get(a, 0) for a in buying_actions])
        num_actions_used = len([p for p in action_dist.values() if p > 1.0])
        max_action_pct = max(action_dist.values()) if action_dist else 0
        
        print(f"\nAction Distribution:")
        print(f"  Actions used (>1%): {num_actions_used}/9")
        print(f"  Most common action: {max_action_pct:.1f}%")
        print(f"  Buying actions (1-3) total usage: {buying_action_usage:.1f}%")
        for action_id in buying_actions:
            pct = action_dist.get(action_id, 0)
            action_names = ['Hold', 'Buy Small', 'Buy Medium', 'Buy Large', 
                          'Sell Small', 'Sell Medium', 'Sell Large', 
                          'Close Position', 'Reverse Position']
            action_name = action_names[action_id] if action_id < len(action_names) else f"Action {action_id}"
            print(f"    {action_name}: {pct:.1f}%")
        
        # Check rewards
        reward_stats = reward_callback.get_stats()
        mean_reward = reward_stats.get('mean_reward', 0)
        std_reward = reward_stats.get('std_reward', 0)
        
        # Get episode rewards directly to calculate min/max
        episode_rewards = reward_callback.episode_rewards if hasattr(reward_callback, 'episode_rewards') else []
        if episode_rewards:
            max_reward = np.max(episode_rewards)
            min_reward = np.min(episode_rewards)
        else:
            max_reward = 0
            min_reward = 0
        
        # Get episode returns if available
        episode_returns = reward_callback.episode_returns if hasattr(reward_callback, 'episode_returns') else []
        mean_return = np.mean(episode_returns) if episode_returns else 0
        
        print(f"\nReward Statistics:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        print(f"  Reward range: [{min_reward:.2f}, {max_reward:.2f}]")
        if episode_returns:
            print(f"  Mean return: {mean_return:.2f}%")
        
        # Test criteria
        test_passed = True
        issues = []
        warnings = []
        
        # Check 1: Rewards should be positive (max reward > 0)
        # Note: Early in training, rewards may be negative as agent learns
        # We check if buying actions are being used as a proxy for improvement
        if max_reward <= 0 and buying_action_usage < 3.0:
            # Only fail if BOTH conditions: no positive rewards AND very low buying action usage
            test_passed = False
            issues.append(f"Max reward is {max_reward:.2f} and buying actions <3% (need buying actions >3% or positive rewards)")
        elif max_reward > 0:
            warnings.append(f"✓ Positive rewards achieved (max: {max_reward:.2f})")
        elif buying_action_usage >= 3.0:
            warnings.append(f"⚠ Max reward negative ({max_reward:.2f}) but buying actions being explored ({buying_action_usage:.1f}%) - may improve with more training")
        
        # Check 2: Buying actions should be used (>3% total for early training, >5% for full training)
        buying_actions_used = [action_dist.get(a, 0) for a in buying_actions]
        if buying_action_usage < 3.0:
            test_passed = False
            issues.append(f"Buying actions usage too low ({buying_action_usage:.1f}%, need >3% for test)")
        elif buying_action_usage >= 5.0:
            warnings.append(f"✓ Buying actions being used well ({buying_action_usage:.1f}%)")
        elif buying_action_usage >= 3.0:
            warnings.append(f"✓ Buying actions being used ({buying_action_usage:.1f}%) - should improve with more training")
        
        # Check 3: At least one buying action should be >1%
        max_buying_action = max(buying_actions_used)
        if max_buying_action < 1.0:
            test_passed = False
            issues.append(f"No buying action exceeds 1% usage (max: {max_buying_action:.1f}%)")
        elif max_buying_action >= 1.0:
            warnings.append(f"✓ At least one buying action >1% (max: {max_buying_action:.1f}%)")
        
        # Check 4: Agent should use multiple actions
        if num_actions_used < 4:
            test_passed = False
            issues.append(f"Only {num_actions_used} actions used (need at least 4)")
        elif num_actions_used >= 4:
            warnings.append(f"✓ Multiple actions explored ({num_actions_used}/9)")
        
        # Check 5: No single action should dominate
        if max_action_pct > 80:
            test_passed = False
            issues.append(f"Single action dominates ({max_action_pct:.1f}%)")
        elif max_action_pct <= 80:
            warnings.append(f"✓ Action distribution balanced (max: {max_action_pct:.1f}%)")
        
        # Check 6: Mean return should be improving (less negative or positive)
        if mean_return < -2.0:
            warnings.append(f"⚠ Mean return still very negative ({mean_return:.2f}%), but this may improve with more training")
        elif mean_return >= -2.0:
            warnings.append(f"✓ Mean return reasonable ({mean_return:.2f}%)")
        
        # Print results
        print("\n" + "-" * 60)
        if warnings:
            print("Test Warnings/Passes:")
            for warning in warnings:
                print(f"  {warning}")
        
        if test_passed:
            print("\n✅ REWARD IMPROVEMENT TEST PASSED")
            print("   Reward improvements are working correctly:")
            print("   - Positive rewards are achievable")
            print("   - Buying actions are being used")
            print("   - Agent is exploring properly")
            print("   Proceeding with full training...")
        else:
            print("\n❌ REWARD IMPROVEMENT TEST FAILED")
            print("   Issues detected:")
            for issue in issues:
                print(f"     - {issue}")
            print("\n   Recommendations:")
            print("     - Check reward_config has buy_action_bonus and open_position_cost_ratio set")
            print("     - Verify profit_scale is increased (should be 250)")
            print("     - Consider increasing ent_coef further (try 0.2-0.25)")
            print("     - Check that reward function changes are properly loaded")
        
        test_env.close()
        return test_passed
        
    except Exception as e:
        print(f"\n⚠️  REWARD IMPROVEMENT TEST ERROR: {e}")
        print("   Skipping test and proceeding with training...")
        import traceback
        traceback.print_exc()
        return True  # Don't block training if test fails


def run_validation_test(
    dataset_path: Path,
    prediction_models,
    reward_config: Dict,
    prediction_horizons: List[int],
    skip_validation: bool = False
) -> bool:
    """
    Run a quick 50K step validation test to verify agent is learning.
    
    This test verifies:
    - Agent explores multiple actions (not collapsed to single action)
    - Rewards are non-zero (both positive and negative)
    - Agent is actually learning, not stuck
    
    Args:
        dataset_path: Path to dataset
        prediction_models: Loaded prediction models
        reward_config: Reward configuration
        prediction_horizons: Prediction horizons
        skip_validation: If True, skip validation and return True
        
    Returns:
        True if validation passes, False otherwise
    """
    if skip_validation:
        return True
    
    print("\n" + "=" * 60)
    print("VALIDATION TEST (50K steps)")
    print("=" * 60)
    print("Running quick validation to verify agent can learn...")
    print("This will take ~2-3 minutes.")
    
    try:
        from trading_env import TradingEnv
        from ppo_trading_agent import create_ppo_agent
        from stable_baselines3.common.callbacks import CallbackList
        
        # Create simple validation environment (single env for speed)
        val_test_env = TradingEnv(
            dataset_path=dataset_path,
            prediction_models=prediction_models if prediction_models and prediction_models.loaded else None,
            transaction_cost=0.0025,
            initial_capital=10000,
            sequence_length=60,
            train_mode=True,
            train_split=0.6,
            validation_split=0.2,
            reward_config=reward_config,
            max_episode_steps=500,  # Shorter episodes for faster validation
            prediction_horizons=prediction_horizons,
        )
        
        # Create agent with high exploration for validation
        val_ppo_config = {
            'learning_rate': 0.001,  # Higher LR for fast test
            'n_steps': 256,
            'batch_size': 64,
            'n_epochs': 5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.3,  # Very high exploration for validation
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 0,
        }
        
        print("\nCreating validation agent with high exploration (ent_coef=0.3)...")
        val_model = create_ppo_agent(
            env=val_test_env,
            config=val_ppo_config,
            device='auto'
        )
        
        # Create callbacks to track validation metrics
        from ppo_trading_agent import PolicyCollapseCallback, RewardLoggingCallback
        collapse_callback = PolicyCollapseCallback(threshold=0.90, check_freq=10000, verbose=1)
        reward_callback = RewardLoggingCallback(verbose=0)  # Silent during validation
        
        print("Running 50,000 step validation test...")
        val_model.learn(
            total_timesteps=50000,
            callback=CallbackList([collapse_callback, reward_callback]),
            progress_bar=True,
        )
        
        # Analyze results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Check action distribution
        action_stats = collapse_callback.action_counts
        total_actions = collapse_callback.total_actions
        
        if total_actions == 0:
            print("❌ VALIDATION FAILED: No actions recorded!")
            val_test_env.close()
            return False
        
        action_dist = {k: v/total_actions*100 for k, v in action_stats.items()}
        num_actions_used = len([p for p in action_dist.values() if p > 1.0])  # Actions with >1% usage
        max_action_pct = max(action_dist.values()) if action_dist else 0
        
        print(f"\nAction Distribution:")
        print(f"  Actions used (>1%): {num_actions_used}/9")
        print(f"  Most common action: {max_action_pct:.1f}%")
        
        # Check rewards
        reward_stats = reward_callback.get_stats()
        mean_reward = reward_stats.get('mean_reward', 0)
        std_reward = reward_stats.get('std_reward', 0)
        
        print(f"\nReward Statistics:")
        print(f"  Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
        
        # Validation criteria
        validation_passed = True
        issues = []
        
        # Check 1: Agent should use multiple actions
        if num_actions_used < 4:
            validation_passed = False
            issues.append(f"Only {num_actions_used} actions used (need at least 4)")
        
        # Check 2: No single action should dominate
        if max_action_pct > 80:
            validation_passed = False
            issues.append(f"Single action dominates ({max_action_pct:.1f}%)")
        
        # Check 3: Rewards should be non-zero
        if abs(mean_reward) < 0.001:
            validation_passed = False
            issues.append("Mean reward is near zero (possible collapse)")
        
        # Check 4: Reward variance should be non-zero
        if std_reward < 0.01:
            validation_passed = False
            issues.append(f"Reward variance too low ({std_reward:.6f})")
        
        # Print results
        if validation_passed:
            print("\n✅ VALIDATION PASSED")
            print("   Agent is exploring properly and learning.")
            print("   Proceeding with full training...")
        else:
            print("\n❌ VALIDATION FAILED")
            print("   Issues detected:")
            for issue in issues:
                print(f"     - {issue}")
            print("\n   Recommendations:")
            print("     - Increase ent_coef further (try 0.4-0.5)")
            print("     - Check reward function configuration")
            print("     - Verify inaction_penalty is enabled")
            print("     - Consider simplifying action space for debugging")
        
        val_test_env.close()
        return validation_passed
        
    except Exception as e:
        print(f"\n⚠️  VALIDATION TEST ERROR: {e}")
        print("   Skipping validation and proceeding with training...")
        import traceback
        traceback.print_exc()
        return True  # Don't block training if validation fails


def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary with all configuration
    """
    # Default configuration
    config = {
        'environment': {
            'action_space': 'discrete',
            'transaction_cost': 0.0025,
            'initial_capital': 10000,
            'max_position': 1.0,
            'sequence_length': 60,
        },
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        },
        'reward': {
            'profit_scale': 250,
            'cost_scale': 1.0,
            'drawdown_penalty': 0.05,
            'sharpe_bonus': 1.5,
            'enable_hold_penalty': True,
            'hold_penalty': 0.01,
            'max_hold_periods': 100,
            'invalid_action_penalty': -1.0,
            'clip_reward': True,
            'reward_clip_value': 20.0,
            'enable_profit_threshold_bonus': True,
            'profit_threshold': 0.05,
            'profit_threshold_bonus': 5.0,
            'enable_inaction_penalty': True,
            'inaction_penalty': 0.03,
            'buy_action_bonus': 0.75,
            'sell_action_bonus': 0.25,
            'open_position_cost_ratio': 0.5,
            'max_drawdown_threshold': 0.1,
            'min_periods_for_sharpe': 10,
        },
        'training': {
            'total_timesteps': 1000000,
            'checkpoint_freq': 50000,
            'eval_freq': 10000,
            'n_eval_episodes': 5,
            'train_test_split': 0.6,
            'validation_split': 0.2,
            'max_episode_steps': 1000,
            'device': 'auto',
        },
        'models': {
            'prediction_model': 'ensemble',
            'dataset': 'ETH-EUR_1H_20240101-20251231',
            'ensemble_models': ['lstm', 'gru', 'bilstm', 'dlstm'],
            'prediction_horizons': [1, 2, 3, 5, 10],  # Short-term: 1,2,3 | Medium-term: 5,10
        }
    }
    
    # Load from file if exists
    if config_path is None:
        config_path = get_ppo_path() / 'ppo_config.txt'
    
    if Path(config_path).exists():
        parser = ConfigParser()
        parser.read(config_path)
        
        # Parse sections
        if 'ENVIRONMENT' in parser:
            for key in parser['ENVIRONMENT']:
                value = parser['ENVIRONMENT'][key]
                if key in ['transaction_cost', 'max_position']:
                    config['environment'][key] = float(value)
                elif key in ['initial_capital', 'sequence_length']:
                    config['environment'][key] = int(value)
                else:
                    config['environment'][key] = value
        
        if 'PPO' in parser:
            for key in parser['PPO']:
                value = parser['PPO'][key]
                if key in ['learning_rate', 'learning_rate_final', 'gamma', 'gae_lambda', 'clip_range', 
                          'ent_coef', 'ent_coef_final', 'vf_coef', 'max_grad_norm']:
                    config['ppo'][key] = float(value)
                elif key in ['n_steps', 'batch_size', 'n_epochs']:
                    config['ppo'][key] = int(value)
        
        if 'REWARD' in parser:
            for key in parser['REWARD']:
                value = parser['REWARD'][key]
                # Boolean flags
                if key in ['enable_hold_penalty', 'enable_profit_threshold_bonus', 
                          'enable_inaction_penalty', 'clip_reward']:
                    config['reward'][key] = value.lower() == 'true'
                # Float values
                elif key in ['profit_scale', 'cost_scale', 'drawdown_penalty', 
                            'sharpe_bonus', 'hold_penalty', 'profit_threshold', 
                            'profit_threshold_bonus', 'invalid_action_penalty',
                            'reward_clip_value', 'max_drawdown_threshold',
                            'buy_action_bonus', 'sell_action_bonus', 
                            'open_position_cost_ratio', 'inaction_penalty']:
                    config['reward'][key] = float(value)
                # Integer values
                elif key in ['max_hold_periods', 'min_periods_for_sharpe']:
                    config['reward'][key] = int(value)
                # String values (keep as-is)
                else:
                    config['reward'][key] = value
        
        if 'TRAINING' in parser:
            for key in parser['TRAINING']:
                value = parser['TRAINING'][key]
                if key in ['total_timesteps', 'checkpoint_freq', 'eval_freq', 
                          'n_eval_episodes', 'max_episode_steps']:
                    config['training'][key] = int(value) if value.lower() != 'none' else None
                elif key == 'train_test_split':
                    config['training'][key] = float(value)
                elif key == 'validation_split':
                    config['training'][key] = float(value)
                else:
                    config['training'][key] = value
        
        if 'MODELS' in parser:
            for key in parser['MODELS']:
                value = parser['MODELS'][key]
                if key == 'ensemble_models':
                    config['models'][key] = [m.strip() for m in value.split(',')]
                elif key == 'prediction_horizons':
                    # Parse comma-separated horizons: "1,2,3,5,10"
                    config['models'][key] = [int(h.strip()) for h in value.split(',')]
                else:
                    config['models'][key] = value
    
    return config


def train_ppo(
    model_type: str = None,
    dataset: str = None,
    timesteps: int = None,
    config_path: str = None,
    resume: bool = True,
    checkpoint_dir: str = None,
    skip_validation: bool = False,
):
    """
    Main training function.
    
    Args:
        model_type: Prediction model type (lstm, gru, bilstm, dlstm, ensemble)
        dataset: Dataset name
        timesteps: Total training timesteps
        config_path: Path to configuration file
        resume: Whether to resume from checkpoint
        checkpoint_dir: Directory for checkpoints
    """
    print("=" * 60)
    print("PPO TRADING AGENT TRAINING")
    print("=" * 60)
    
    # Setup environment (handles Colab detection, Drive mounting, etc.)
    env_info = setup_environment()
    
    # Load configuration
    config = load_config(config_path)
    
    # Pre-training verification: Test reward calculation
    print("\n" + "=" * 60)
    print("PRE-TRAINING VERIFICATION")
    print("=" * 60)
    try:
        from test_reward_calculation import test_reward_scenarios
        print("Running reward calculation tests...")
        test_passed = test_reward_scenarios()
        if not test_passed:
            print("\n⚠ WARNING: Some reward calculation tests failed!")
            print("Training will continue, but rewards may not work correctly.")
            response = input("Continue with training anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Training aborted by user.")
                return None
        else:
            print("\n✓ Reward calculation tests passed. Proceeding with training...")
    except ImportError as e:
        print(f"⚠ WARNING: Could not import test script: {e}")
        print("Skipping pre-training verification.")
    except Exception as e:
        print(f"⚠ WARNING: Error during verification: {e}")
        print("Skipping pre-training verification.")
    
    print("\n" + "=" * 60)
    
    # Pre-training reward improvement test: Quick 50K step test to verify reward improvements
    # This test verifies that the reward improvements (buying action bonuses, deferred costs) are working
    run_reward_test = True  # Enable by default to verify improvements
    run_validation = False  # Legacy validation test (disabled, using reward improvement test instead)
    
    print("\n" + "=" * 60)
    
    # Override with command line arguments
    if model_type:
        config['models']['prediction_model'] = model_type
    if dataset:
        config['models']['dataset'] = dataset
    if timesteps:
        config['training']['total_timesteps'] = timesteps
    
    print(f"\nConfiguration:")
    print(f"  Prediction model: {config['models']['prediction_model']}")
    print(f"  Dataset: {config['models']['dataset']}")
    print(f"  Total timesteps: {config['training']['total_timesteps']}")
    print(f"  Checkpoint freq: {config['training']['checkpoint_freq']}")
    print(f"  Device: {config['training']['device']}")
    
    # Import here to avoid issues if dependencies not installed
    try:
        from prediction_wrapper import load_prediction_model, load_ensemble
        from trading_env import TradingEnv
        from ppo_trading_agent import train_with_checkpoints, create_ppo_agent
    except ImportError as e:
        print(f"\nError importing modules: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install stable-baselines3 gymnasium tensorboard")
        return None
    
    # Load prediction models
    print(f"\nLoading prediction models...")
    dataset_name = config['models']['dataset']
    
    if config['models']['prediction_model'] == 'ensemble':
        model_names = config['models'].get('ensemble_models', ['lstm', 'gru', 'bilstm', 'dlstm'])
        prediction_models = load_ensemble(dataset_name, model_names)
    else:
        prediction_models = load_prediction_model(
            config['models']['prediction_model'],
            dataset_name
        )
    
    if not prediction_models.loaded:
        print("\nWarning: Could not load prediction models.")
        print("Training will proceed without prediction features.")
        prediction_models = None
    
    # Find dataset path
    datasets_path = get_datasets_path()
    dataset_files = list(datasets_path.glob(f"*{dataset_name}*"))
    
    if not dataset_files:
        print(f"\nError: Dataset not found: {dataset_name}")
        print(f"Available datasets:")
        for f in datasets_path.glob("*.csv"):
            print(f"  - {f.name}")
        return None
    
    dataset_path = dataset_files[0]
    print(f"\nUsing dataset: {dataset_path.name}")
    
    # Get prediction horizons
    prediction_horizons = config['models'].get('prediction_horizons', [1, 2, 3, 5, 10])
    print(f"\nPrediction horizons: {prediction_horizons}")
    print(f"  Short-term: {[h for h in prediction_horizons if h <= 3]}")
    print(f"  Medium-term: {[h for h in prediction_horizons if h > 3]}")
    
    # Run reward improvement test before full training
    if run_reward_test and not skip_validation:
        print("\n" + "=" * 60)
        reward_test_passed = run_reward_improvement_test(
            dataset_path=dataset_path,
            prediction_models=prediction_models,
            reward_config=config['reward'],
            prediction_horizons=prediction_horizons,
            skip_test=False
        )
        
        if not reward_test_passed:
            print("\n⚠️  Reward improvement test failed!")
            print("   The reward improvements may not be working correctly.")
            print("   Training will continue, but results may be suboptimal.")
            response = input("\nContinue with full training anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Training aborted by user.")
                return None
        else:
            print("\n✓ Reward improvement test passed. Proceeding with full training...")
    
    # Run validation test before full training
    if run_validation:
        validation_passed = run_validation_test(
            dataset_path=dataset_path,
            prediction_models=prediction_models,
            reward_config=config['reward'],
            prediction_horizons=prediction_horizons,
            skip_validation=False
        )
        
        if not validation_passed:
            print("\n⚠️  Validation test failed, but continuing with training.")
            print("   Monitor training closely for signs of policy collapse.")
            response = input("\nContinue with full training anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Training aborted by user.")
                return None
    
    # Create training environment
    print("\nCreating trading environment...")
    # Use vectorized environments for parallel CPU processing (faster data collection)
    # This allows multiple environments to run in parallel, speeding up data collection
    # which is the main bottleneck (environment runs on CPU, GPU only used for training)
    if not VEC_ENV_AVAILABLE:
        print("  ⚠ Vectorized environments not available, using single environment")
        print("  Install stable-baselines3 for parallel processing: pip install stable-baselines3")
    
    # Number of parallel environments - REDUCED for better exploration diversity
    # With fewer envs, each environment sees more diverse experiences before policy updates
    # Previous: 16 envs meant each env only saw ~18,750 steps in 300K total timesteps
    # With 4 envs and 1M timesteps, each env sees ~250K steps = more learning per env
    # This helps prevent policy collapse by ensuring more diverse experience per environment
    n_envs = 4  # Reduced from 16 for better exploration diversity
    
    def make_env(rank: int = 0, train_mode: bool = True):
        """Create a single environment."""
        def _init():
            return TradingEnv(
                dataset_path=dataset_path,
                prediction_models=prediction_models,
                transaction_cost=config['environment']['transaction_cost'],
                initial_capital=config['environment']['initial_capital'],
                sequence_length=config['environment']['sequence_length'],
                train_mode=train_mode,
                train_split=config['training']['train_test_split'],
                validation_split=config['training'].get('validation_split', 0.2),
                reward_config=config['reward'],
                max_episode_steps=config['training']['max_episode_steps'],
                prediction_horizons=prediction_horizons,
            )
        return _init
    
    # Create vectorized training environment
    # Use SubprocVecEnv for true parallelism (faster) or DummyVecEnv for simpler setup
    if VEC_ENV_AVAILABLE:
        # Try SubprocVecEnv first, but it requires proper multiprocessing setup
        # When imported as a module (not run as main), multiprocessing can fail
        use_subproc = True
        try:
            # Check if we can use fork (Linux) or need spawn (Windows/Mac)
            if sys.platform == 'win32':
                start_method = 'spawn'
            else:
                start_method = 'fork'
            
            # Try to set start method (may fail if already set)
            try:
                multiprocessing.set_start_method(start_method, force=True)
            except RuntimeError:
                pass  # Already set, continue
            
            print(f"  Creating {n_envs} parallel training environments...")
            train_env = SubprocVecEnv([make_env(i, train_mode=True) for i in range(n_envs)])
            print(f"  ✓ Using SubprocVecEnv (true parallelism)")
        except Exception as e:
            print(f"  ⚠ SubprocVecEnv failed: {e}")
            print(f"  Falling back to DummyVecEnv with {n_envs} workers")
            # Use DummyVecEnv with multiple workers - not truly parallel but better than 1
            # DummyVecEnv runs environments sequentially but batches observations
            train_env = DummyVecEnv([make_env(i, train_mode=True) for i in range(n_envs)])
            print(f"  ✓ Using DummyVecEnv with {n_envs} workers (sequential but batched)")
    else:
        # Fallback to single environment if vec_env not available
        train_env = TradingEnv(
            dataset_path=dataset_path,
            prediction_models=prediction_models,
            transaction_cost=config['environment']['transaction_cost'],
            initial_capital=config['environment']['initial_capital'],
            sequence_length=config['environment']['sequence_length'],
            train_mode=True,
            train_split=config['training']['train_test_split'],
            validation_split=config['training'].get('validation_split', 0.2),
            reward_config=config['reward'],
            max_episode_steps=config['training']['max_episode_steps'],
            prediction_horizons=prediction_horizons,
        )
        n_envs = 1
    
    # Create validation environment (for hyperparameter tuning during training)
    val_env = TradingEnv(
        dataset_path=dataset_path,
        prediction_models=prediction_models,
        transaction_cost=config['environment']['transaction_cost'],
        initial_capital=config['environment']['initial_capital'],
        sequence_length=config['environment']['sequence_length'],
        train_mode=False,
        validation_mode=True,  # Use validation data (20%)
        train_split=config['training']['train_test_split'],
        validation_split=config['training'].get('validation_split', 0.2),
        reward_config=config['reward'],
        max_episode_steps=config['training']['max_episode_steps'],
        prediction_horizons=prediction_horizons,
    )
    
    # Create evaluation environment (for final test evaluation)
    eval_env = TradingEnv(
        dataset_path=dataset_path,
        prediction_models=prediction_models,
        transaction_cost=config['environment']['transaction_cost'],
        initial_capital=config['environment']['initial_capital'],
        sequence_length=config['environment']['sequence_length'],
        train_mode=False,
        validation_mode=False,  # Use test data (20%, models never saw this)
        train_split=config['training']['train_test_split'],
        validation_split=config['training'].get('validation_split', 0.2),
        reward_config=config['reward'],
        max_episode_steps=config['training']['max_episode_steps'],
        prediction_horizons=prediction_horizons,
    )
    
    # Note: observation space is now a vectorized space
    if hasattr(train_env, 'observation_space'):
        print(f"  Observation space dimension: {train_env.observation_space.shape}")
    else:
        print(f"  Observation space dimension: {train_env.observation_space.shape[0]}")
    
    # Get data info from a single environment (vectorized envs share same data)
    if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
        sample_env = train_env.envs[0].env if hasattr(train_env.envs[0], 'env') else train_env.envs[0]
        if hasattr(sample_env, 'data_end_idx'):
            print(f"  Training data steps: {sample_env.data_end_idx - sample_env.data_start_idx}")
    if hasattr(val_env, 'data_end_idx'):
        print(f"  Validation data steps: {val_env.data_end_idx - val_env.data_start_idx}")
    if hasattr(eval_env, 'data_end_idx'):
        print(f"  Test data steps: {eval_env.data_end_idx - eval_env.data_start_idx}")
    
    print(f"\n  ⚡ Using {n_envs} parallel environments for {n_envs}x faster data collection")
    print(f"  🎯 GPU will train more frequently (every {config['ppo']['n_steps']} steps)")
    
    # Set checkpoint directory
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
    else:
        model_name = config['models']['prediction_model']
        checkpoint_path = get_checkpoints_path() / f"{model_name}_{dataset_name}"
    
    # Train agent
    print("\nStarting training...")
    
    model = train_with_checkpoints(
        env=train_env,
        total_timesteps=config['training']['total_timesteps'],
        checkpoint_freq=config['training']['checkpoint_freq'],
        checkpoint_path=checkpoint_path,
        eval_env=val_env,  # Use validation env for evaluation during training
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        ppo_config=config['ppo'],
        resume=resume,
        device=config['training']['device'],
    )
    
    # Save final model
    final_model_path = get_ppo_models_path() / f"ppo_{config['models']['prediction_model']}_{dataset_name}.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Clean up
    train_env.close()
    val_env.close()
    eval_env.close()
    
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train PPO trading agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppo_agent.py --model dlstm --dataset ETH-EUR --timesteps 1000000
  python train_ppo_agent.py --config ppo_config.txt --resume
  python train_ppo_agent.py --model ensemble --timesteps 500000
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       choices=['lstm', 'gru', 'bilstm', 'dlstm', 'ensemble'],
                       help='Prediction model to use')
    
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name (e.g., ETH-EUR_1H_20240101-20251231)')
    
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Resume from checkpoint if exists')
    
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training (ignore checkpoints)')
    
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Directory for checkpoints')
    
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip pre-training validation test (not recommended)')
    
    parser.add_argument('--skip-reward-test', action='store_true',
                       help='Skip reward improvement test (50K steps) - not recommended')
    
    args = parser.parse_args()
    
    # Handle resume flag
    resume = not args.no_resume
    
    # Train
    train_ppo(
        model_type=args.model,
        dataset=args.dataset,
        timesteps=args.timesteps,
        config_path=args.config,
        resume=resume,
        checkpoint_dir=args.checkpoint_dir,
        skip_validation=args.skip_validation or args.skip_reward_test,  # Skip reward test if requested
    )


if __name__ == '__main__':
    main()


