"""
PPO Trading Agent
=================
Main PPO agent implementation using stable-baselines3.

Features:
- PPO policy network (MLP)
- Hyperparameters configuration
- Training loop with checkpointing
- Model saving/loading with resume support
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from datetime import datetime

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback, 
        CheckpointCallback, 
        EvalCallback,
        CallbackList
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")

from colab_utils import (
    get_ppo_models_path, 
    get_checkpoints_path, 
    get_logs_path,
    is_colab_runtime
)


# Default PPO hyperparameters (as per plan)
DEFAULT_PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'verbose': 1,
}


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    
    Saves progress to JSON file for checkpoint resume.
    """
    
    def __init__(
        self, 
        checkpoint_path: Path,
        save_freq: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.progress_file = self.checkpoint_path / 'progress.json'
        self.start_time = None
        self.best_mean_reward = -np.inf
    
    def _on_training_start(self):
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self._save_progress()
        return True
    
    def _on_training_end(self):
        self._save_progress()
    
    def _save_progress(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        progress = {
            'timesteps': self.num_timesteps,
            'best_reward': float(self.best_mean_reward),
            'elapsed_time': elapsed,
            'elapsed_time_str': str(time.strftime('%H:%M:%S', time.gmtime(elapsed))),
            'last_checkpoint': datetime.now().isoformat(),
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        if self.verbose > 0:
            print(f"Progress saved: {self.num_timesteps} timesteps, {progress['elapsed_time_str']}")


class RewardLoggingCallback(BaseCallback):
    """Callback to log rewards and episode info with action distribution."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = {}
        self.total_steps = 0
    
    def _on_step(self) -> bool:
        # Track actions
        if 'action' in self.locals.get('infos', [{}])[0]:
            action = self.locals['infos'][0].get('action', 0)
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.total_steps += 1
        
        # Check if episode ended
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]['episode']
            self.episode_rewards.append(info['r'])
            self.episode_lengths.append(info['l'])
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                # Print action distribution
                action_dist = {k: v/self.total_steps*100 for k, v in self.action_counts.items()}
                action_dist_str = ', '.join([f"Action {k}: {v:.1f}%" for k, v in sorted(action_dist.items())])
                print(f"Episode {len(self.episode_rewards)}: Mean reward = {mean_reward:.2f} | {action_dist_str}")
        
        return True
    
    def get_stats(self) -> Dict:
        action_dist = {k: v/self.total_steps*100 if self.total_steps > 0 else 0 
                      for k, v in self.action_counts.items()}
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'num_episodes': len(self.episode_rewards),
            'action_distribution': action_dist,
        }


def load_ppo_config(config_path: str = None) -> Dict:
    """
    Load PPO configuration from file.
    
    Args:
        config_path: Path to config file (ppo_config.txt)
        
    Returns:
        Dictionary of PPO hyperparameters
    """
    config = DEFAULT_PPO_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read(config_path)
        
        if 'PPO' in parser:
            for key in parser['PPO']:
                value = parser['PPO'][key]
                # Convert to appropriate type
                if key in ['learning_rate', 'gamma', 'gae_lambda', 'clip_range', 
                          'ent_coef', 'vf_coef', 'max_grad_norm']:
                    config[key] = float(value)
                elif key in ['n_steps', 'batch_size', 'n_epochs', 'verbose']:
                    config[key] = int(value)
    
    return config


def create_ppo_agent(
    env,
    config: Dict = None,
    tensorboard_log: str = None,
    device: str = 'auto'
) -> 'PPO':
    """
    Create a new PPO agent.
    
    Args:
        env: Gymnasium environment
        config: PPO hyperparameters
        tensorboard_log: Path for TensorBoard logs
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
    
    config = config or DEFAULT_PPO_CONFIG.copy()
    
    # Set tensorboard log path
    if tensorboard_log is None:
        tensorboard_log = str(get_logs_path())
    
    # Create agent with OPTIMIZED network architecture for RTX 4090 (24GB VRAM)
    # Larger network uses more GPU memory and can learn more complex patterns
    # [1280, 640] = 2 hidden layers: 1280 units (shared) + 640 units (value/policy split)
    # This will use ~16-18GB GPU memory for ~85-90% utilization on RTX 4090
    # Reduced from [1536, 768] to stay safely under 24GB VRAM limit
    policy_kwargs = {
        'net_arch': [1280, 640]  # OPTIMIZED for RTX 4090 (24GB VRAM) - safe memory usage
    }
    
    # Create agent
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        tensorboard_log=tensorboard_log,
        verbose=config.get('verbose', 1),
        device=device,
        policy_kwargs=policy_kwargs,  # Larger network architecture
    )
    
    return model


def load_ppo_agent(
    model_path: str,
    env = None,
    device: str = 'auto'
) -> 'PPO':
    """
    Load a saved PPO agent.
    
    Args:
        model_path: Path to saved model (.zip file)
        env: Environment to attach (optional)
        device: Device to use
        
    Returns:
        Loaded PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")
    
    model = PPO.load(model_path, env=env, device=device)
    return model


def save_ppo_agent(model: 'PPO', save_path: str, include_replay_buffer: bool = False):
    """
    Save PPO agent to disk.
    
    Args:
        model: PPO agent to save
        save_path: Path to save model
        include_replay_buffer: Not used for PPO (on-policy)
    """
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def train_with_checkpoints(
    env,
    total_timesteps: int,
    checkpoint_freq: int = 50000,
    checkpoint_path: Path = None,
    eval_env = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    ppo_config: Dict = None,
    tensorboard_log: str = None,
    resume: bool = True,
    device: str = 'auto',
) -> 'PPO':
    """
    Train PPO agent with automatic checkpointing for Colab session recovery.
    
    Args:
        env: Training environment
        total_timesteps: Total timesteps to train
        checkpoint_freq: Save checkpoint every N timesteps
        checkpoint_path: Path to save checkpoints
        eval_env: Evaluation environment (optional)
        eval_freq: Evaluate every N timesteps
        n_eval_episodes: Number of evaluation episodes
        ppo_config: PPO hyperparameters
        tensorboard_log: TensorBoard log path
        resume: Whether to resume from checkpoint if exists
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        Trained PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")
    
    # Set default paths
    if checkpoint_path is None:
        checkpoint_path = get_checkpoints_path()
    else:
        checkpoint_path = Path(checkpoint_path)
    
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    if tensorboard_log is None:
        tensorboard_log = str(get_logs_path())
    
    # Check for existing checkpoint
    latest_checkpoint = checkpoint_path / 'latest.zip'
    progress_file = checkpoint_path / 'progress.json'
    
    start_timesteps = 0
    model = None
    
    if resume and latest_checkpoint.exists():
        print("=" * 50)
        print("Resuming from checkpoint...")
        print("=" * 50)
        
        try:
            model = PPO.load(str(latest_checkpoint), env=env, device=device)
            
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                start_timesteps = progress.get('timesteps', 0)
                print(f"Resuming from timestep {start_timesteps}")
            
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Starting fresh training...")
            model = None
    
    if model is None:
        print("=" * 50)
        print("Starting new training...")
        print("=" * 50)
        model = create_ppo_agent(
            env=env,
            config=ppo_config,
            tensorboard_log=tensorboard_log,
            device=device
        )
    
    # Create callbacks
    callbacks = []
    
    # Progress tracking callback
    progress_callback = TrainingProgressCallback(
        checkpoint_path=checkpoint_path,
        save_freq=checkpoint_freq,
        verbose=1
    )
    callbacks.append(progress_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_path / 'best'),
            log_path=str(checkpoint_path / 'eval_logs'),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_callback)
    
    # Reward logging callback
    reward_callback = RewardLoggingCallback(verbose=1)
    callbacks.append(reward_callback)
    
    # Calculate remaining timesteps
    remaining_timesteps = total_timesteps - start_timesteps
    
    if remaining_timesteps <= 0:
        print(f"Training already completed ({start_timesteps} >= {total_timesteps})")
        return model
    
    print(f"\nTraining for {remaining_timesteps} timesteps...")
    print(f"Checkpoints saved to: {checkpoint_path}")
    print(f"TensorBoard logs: {tensorboard_log}")
    print()
    
    # Train
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
    finally:
        # Save final checkpoint
        model.save(str(latest_checkpoint))
        progress_callback._save_progress()
        print(f"\nCheckpoint saved to: {latest_checkpoint}")
    
    # Print final stats
    stats = reward_callback.get_stats()
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Total episodes: {stats['num_episodes']}")
    print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean episode length: {stats['mean_length']:.1f}")
    
    # Print action distribution
    if 'action_distribution' in stats:
        print("\nAction Distribution:")
        for action, pct in sorted(stats['action_distribution'].items()):
            action_names = ['Hold', 'Buy Small', 'Buy Medium', 'Buy Large', 
                          'Sell Small', 'Sell Medium', 'Sell Large', 
                          'Close Position', 'Reverse Position']
            action_name = action_names[int(action)] if int(action) < len(action_names) else f"Action {action}"
            print(f"  {action_name} (Action {action}): {pct:.1f}%")
        
        # Warn if agent is only holding
        if stats['action_distribution'].get(0, 0) > 95:
            print("\n⚠ WARNING: Agent is taking action 0 (Hold) >95% of the time!")
            print("  This suggests the reward function may need further tuning.")
            print("  Consider:")
            print("    - Increasing profit_scale")
            print("    - Increasing hold_penalty")
            print("    - Increasing ent_coef for more exploration")
    
    return model


def quick_train(
    env,
    timesteps: int = 100000,
    device: str = 'auto'
) -> 'PPO':
    """
    Quick training without checkpointing (for testing).
    
    Args:
        env: Training environment
        timesteps: Number of timesteps
        device: Device to use
        
    Returns:
        Trained PPO agent
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required")
    
    model = create_ppo_agent(env, device=device)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    return model


def evaluate_agent(
    model: 'PPO',
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
) -> Dict:
    """
    Evaluate a trained PPO agent.
    
    Args:
        model: Trained PPO agent
        env: Evaluation environment
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        render: Render environment
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_metrics = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Get episode-specific metrics if available
        if hasattr(env, 'get_episode_metrics'):
            metrics = env.get_episode_metrics()
            episode_metrics.append(metrics)
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'n_episodes': n_episodes,
    }
    
    # Aggregate episode metrics
    if episode_metrics:
        for key in episode_metrics[0].keys():
            values = [m[key] for m in episode_metrics if key in m]
            if values and isinstance(values[0], (int, float)):
                results[f'mean_{key}'] = np.mean(values)
    
    return results


if __name__ == '__main__':
    print("PPO Trading Agent Module")
    print("=" * 50)
    
    if not SB3_AVAILABLE:
        print("\nstable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
    else:
        print("\nstable-baselines3 is available!")
        print(f"\nDefault PPO config:")
        for k, v in DEFAULT_PPO_CONFIG.items():
            print(f"  {k}: {v}")


