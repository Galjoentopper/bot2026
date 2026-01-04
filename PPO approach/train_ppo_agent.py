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
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Optional

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
            'profit_scale': 100,
            'cost_scale': 1.0,
            'drawdown_penalty': 0.1,
            'sharpe_bonus': 0.5,
            'enable_hold_penalty': False,
            'hold_penalty': 0.01,
            'max_hold_periods': 100,
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
                if key in ['learning_rate', 'gamma', 'gae_lambda', 'clip_range', 
                          'ent_coef', 'vf_coef', 'max_grad_norm']:
                    config['ppo'][key] = float(value)
                elif key in ['n_steps', 'batch_size', 'n_epochs']:
                    config['ppo'][key] = int(value)
        
        if 'REWARD' in parser:
            for key in parser['REWARD']:
                value = parser['REWARD'][key]
                if key == 'enable_hold_penalty':
                    config['reward'][key] = value.lower() == 'true'
                elif key in ['profit_scale', 'cost_scale', 'drawdown_penalty', 
                            'sharpe_bonus', 'hold_penalty']:
                    config['reward'][key] = float(value)
                elif key == 'max_hold_periods':
                    config['reward'][key] = int(value)
        
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
    
    # Create training environment
    print("\nCreating trading environment...")
    # Use vectorized environments for parallel CPU processing (faster data collection)
    # This allows multiple environments to run in parallel, speeding up data collection
    # which is the main bottleneck (environment runs on CPU, GPU only used for training)
    if not VEC_ENV_AVAILABLE:
        print("  ⚠ Vectorized environments not available, using single environment")
        print("  Install stable-baselines3 for parallel processing: pip install stable-baselines3")
    
    # Number of parallel environments - OPTIMIZED for RTX 4090 with 194GB RAM
    # With 194GB RAM available, we can run many more parallel environments
    # Each env uses ~1.5GB RAM (includes prediction models), so 16 envs = ~24GB RAM
    # This leaves 170GB+ headroom for GPU operations and prevents RAM bottleneck
    # More parallel envs = faster data collection = better GPU utilization
    n_envs = 16  # OPTIMIZED for high-RAM systems (194GB available)
    
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
    )


if __name__ == '__main__':
    main()


