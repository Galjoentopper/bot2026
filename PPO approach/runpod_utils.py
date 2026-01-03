"""
RunPod Utilities
================
Environment detection and path management for RunPod environment.

Similar to colab_utils.py but optimized for RunPod with standardized path detection.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def is_runpod_runtime() -> bool:
    """
    Detect if code is running on RunPod environment.
    
    Returns:
        True if running on RunPod, False otherwise
    """
    # Check for /workspace mount (RunPod's persistent storage)
    if os.path.exists('/workspace'):
        # Additional check: look for RunPod-specific environment variables
        if 'RUNPOD_POD_ID' in os.environ or 'RUNPOD_API_KEY' in os.environ:
            return True
        # If /workspace exists and we're not on Colab, likely RunPod
        try:
            import google.colab
            return False  # Colab, not RunPod
        except ImportError:
            # No Colab, and /workspace exists - likely RunPod
            return True
    
    # Check for other RunPod mount points
    if os.path.exists('/runpod-volume') or os.path.exists('/data'):
        return True
    
    return False


def get_project_path() -> Path:
    """
    Return project root based on environment.
    Unified path resolution for RunPod, Colab, and local environments.
    
    Returns:
        Path to project root:
        - RunPod: Searches in /workspace, /runpod-volume, /data
        - Colab: Uses colab_utils.get_project_path()
        - Local: Parent of 'PPO approach' folder
    """
    # Import colab_utils to check for Colab
    try:
        from colab_utils import is_colab_runtime
        if is_colab_runtime():
            # Use colab_utils for Colab environment
            from colab_utils import get_project_path as colab_get_project_path
            return colab_get_project_path()
    except ImportError:
        pass
    
    if is_runpod_runtime():
        # RunPod: Search in common mount points
        possible_paths = [
            Path('/workspace/bot2026'),           # Standard RunPod clone location
            Path('/workspace/bot-2026'),          # Alternative naming
            Path('/workspace/Bot 2026'),          # Colab-style naming
            Path('/runpod-volume/bot2026'),
            Path('/data/bot2026'),
            Path.cwd(),                           # Current directory
        ]
        
        for path in possible_paths:
            if path.exists():
                # Check if it's the project root (has 'PPO approach' folder)
                if (path / 'PPO approach').exists():
                    return path
                # Also check if we're already in the project root
                if path.name in ['bot2026', 'bot-2026', 'Bot 2026'] and (path / 'PPO approach').exists():
                    return path
        
        # If not found, try to find it dynamically
        workspace_root = Path('/workspace')
        if workspace_root.exists():
            # Search for project folder
            for bot_folder in workspace_root.glob('**/PPO approach'):
                if bot_folder.parent.is_dir():
                    return bot_folder.parent
        
        # Fallback to current directory
        current = Path.cwd()
        if (current / 'PPO approach').exists():
            return current
        
        # Last resort: assume /workspace/bot2026
        return Path('/workspace/bot2026')
    else:
        # Local: This file is in PPO approach/, parent is project root
        return Path(__file__).parent.parent


def get_ppo_path() -> Path:
    """Return path to PPO approach folder."""
    return get_project_path() / 'PPO approach'


def get_models_path() -> Path:
    """Return path to prediction models folder."""
    return get_project_path() / 'models'


def get_scalers_path() -> Path:
    """Return path to scalers folder."""
    return get_project_path() / 'scalers'


def get_datasets_path() -> Path:
    """Return path to datasets folder."""
    return get_project_path() / 'datasets'


def get_ppo_models_path() -> Path:
    """Return path to PPO models folder (for saving trained PPO agents)."""
    path = get_ppo_path() / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoints_path() -> Path:
    """Return path to checkpoints folder."""
    path = get_ppo_path() / 'checkpoints'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logs_path() -> Path:
    """Return path to logs folder."""
    path = get_project_path() / 'logs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path() -> Path:
    """Return path to results folder."""
    path = get_ppo_path() / 'results'
    path.mkdir(parents=True, exist_ok=True)
    return path


def verify_project_structure() -> bool:
    """
    Verify that all required project files and folders exist.
    
    Returns:
        True if all required directories exist, False otherwise
    """
    project_path = get_project_path()
    
    required_dirs = [
        'PPO approach',
        'models',
        'scalers',
        'datasets',
    ]
    
    all_exist = True
    
    print("Verifying project structure...")
    
    for dir_name in required_dirs:
        dir_path = project_path / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (MISSING)")
            all_exist = False
    
    # Check for write permissions
    for dir_name in ['models', 'PPO approach/models', 'PPO approach/checkpoints']:
        dir_path = project_path / dir_name
        if dir_path.exists():
            if os.access(dir_path, os.W_OK):
                print(f"  ✓ {dir_name}/ (writable)")
            else:
                print(f"  ⚠ {dir_name}/ (NOT WRITABLE)")
                all_exist = False
    
    return all_exist


def setup_environment(verbose: bool = True) -> dict:
    """
    One-call setup for RunPod environment.
    
    This function:
    1. Detects if running on RunPod
    2. Sets up project paths
    3. Adds project paths to sys.path
    4. Returns environment info
    
    Args:
        verbose: Print status messages
        
    Returns:
        Dictionary with environment information
    """
    env_info = {
        'is_runpod': is_runpod_runtime(),
        'is_colab': False,
        'project_path': None,
        'gpu_available': False,
    }
    
    if verbose:
        print("=" * 50)
        print("RunPod Training - Environment Setup")
        print("=" * 50)
    
    # Check for Colab (import colab_utils if available)
    try:
        from colab_utils import is_colab_runtime
        env_info['is_colab'] = is_colab_runtime()
        if env_info['is_colab']:
            if verbose:
                print("✓ Running on Google Colab (using colab_utils)")
            # Use colab_utils for Colab
            from colab_utils import setup_environment as colab_setup
            return colab_setup(verbose)
    except ImportError:
        pass
    
    # Detect environment
    if env_info['is_runpod']:
        if verbose:
            print("✓ Running on RunPod")
    else:
        if verbose:
            print("✓ Running locally")
    
    # Set project path
    env_info['project_path'] = get_project_path()
    
    if verbose:
        print(f"✓ Project path: {env_info['project_path']}")
    
    # Add paths to sys.path for imports
    project_path = str(env_info['project_path'])
    ppo_path = str(get_ppo_path())
    
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    if ppo_path not in sys.path:
        sys.path.insert(0, ppo_path)
    
    if verbose:
        print("✓ Added project paths to sys.path")
    
    # Check GPU availability
    try:
        import torch
        env_info['gpu_available'] = torch.cuda.is_available()
        if env_info['gpu_available']:
            gpu_name = torch.cuda.get_device_name(0)
            if verbose:
                print(f"✓ GPU available: {gpu_name}")
        else:
            if verbose:
                print("⚠ No GPU available - training will be slower")
    except ImportError:
        if verbose:
            print("⚠ PyTorch not installed - GPU check skipped")
    
    # Verify project structure
    if verbose:
        verify_project_structure()
        print("\n" + "=" * 50)
        print("Setup complete!")
        print("=" * 50)
    
    return env_info

