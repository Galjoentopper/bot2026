"""
Colab Utilities
===============
Environment detection and Google Drive integration for Colab VS Code Extension.

When using the Colab VS Code Extension:
1. Your notebook file stays on your local PC (in Cursor)
2. Code executes on Google's Colab servers (with GPU)
3. Local files are NOT accessible from Colab runtime
4. You must sync your project to Google Drive for Colab to access it

Also supports RunPod environment detection for unified path resolution.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def is_colab_runtime() -> bool:
    """
    Detect if code is running on Google Colab runtime.
    
    Returns:
        True if running on Colab, False otherwise
    """
    # Check for Colab-specific environment variables
    if 'COLAB_GPU' in os.environ:
        return True
    
    # Check for google.colab module
    try:
        import google.colab
        return True
    except ImportError:
        pass
    
    # Check IPython environment
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            if 'google.colab' in str(type(ipython)):
                return True
            # Check if running in Colab kernel
            if hasattr(ipython, 'kernel'):
                kernel_name = str(type(ipython.kernel))
                if 'colab' in kernel_name.lower():
                    return True
    except:
        pass
    
    # Check if /content directory exists (Colab's working directory)
    if os.path.exists('/content'):
        return True
    
    return False


def mount_drive(force: bool = False) -> bool:
    """
    Mount Google Drive (only works on Colab runtime).
    
    Args:
        force: Force remount even if already mounted
        
    Returns:
        True if mounted successfully, False otherwise
    """
    if not is_colab_runtime():
        print("Not running on Colab - Drive mount not needed")
        return False
    
    drive_path = Path('/content/drive')
    
    # Check if already mounted
    if drive_path.exists() and not force:
        if (drive_path / 'MyDrive').exists():
            print("Google Drive already mounted at /content/drive")
            return True
    
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=force)
        print("Google Drive mounted successfully!")
        return True
    except Exception as e:
        print(f"Failed to mount Google Drive: {e}")
        return False


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
        if not is_colab_runtime():
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
        - Colab: Searches for Bot 2026 folder in Drive
        - Local: Parent of 'PPO approach' folder
    """
    # Check for RunPod first
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
    
    if is_colab_runtime():
        # Try multiple possible paths (user may have different Drive setups)
        # Note: Colab uses English paths internally, but we check variants
        possible_paths = [
            Path('/content/drive/MyDrive/Bot 2026'),           # Standard English
            Path('/content/drive/Mijn Drive/Bot 2026'),        # Dutch locale
            Path('/content/drive/My Drive/Bot 2026'),          # With space
            Path('/content/drive/Othercomputers/Mijn laptop/Bot 2026'),
            Path('/content/drive/Othercomputers/My Laptop/Bot 2026'),
            Path('/content/drive/Shareddrives/Bot 2026'),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # If none found, try to find it dynamically
        drive_root = Path('/content/drive')
        if drive_root.exists():
            # Search in MyDrive
            for bot_folder in drive_root.glob('**/Bot 2026'):
                if bot_folder.is_dir():
                    return bot_folder
        
        # Fallback to default (will show error later)
        return Path('/content/drive/MyDrive/Bot 2026')
    else:
        # Local: This file is in PPO approach/, parent is Bot 2026/
        return Path(__file__).parent.parent


def get_ppo_path() -> Path:
    """
    Return path to PPO approach folder.
    
    Returns:
        Path to PPO approach folder
    """
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
    """Return path to TensorBoard logs folder."""
    path = get_ppo_path() / 'logs'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path() -> Path:
    """Return path to results folder."""
    path = get_ppo_path() / 'results'
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_environment(verbose: bool = True) -> dict:
    """
    One-call setup for notebook environment.
    
    This function:
    1. Detects if running on Colab, RunPod, or local
    2. Mounts Google Drive if on Colab
    3. Adds project paths to sys.path
    4. Returns environment info
    
    Args:
        verbose: Print status messages
        
    Returns:
        Dictionary with environment information
    """
    # Skip verbose output if we're in a subprocess (SubprocVecEnv creates subprocesses)
    import multiprocessing
    try:
        is_subprocess = multiprocessing.current_process().name != 'MainProcess'
        if is_subprocess:
            verbose = False  # Never print in subprocesses
    except:
        pass  # If multiprocessing check fails, continue normally
    
    env_info = {
        'is_colab': is_colab_runtime(),
        'is_runpod': is_runpod_runtime(),
        'project_path': None,
        'drive_mounted': False,
        'gpu_available': False,
    }
    
    if verbose:
        print("=" * 50)
        print("PPO Trading Agent - Environment Setup")
        print("=" * 50)
    
    # Detect environment
    if env_info['is_runpod']:
        if verbose:
            print("✓ Running on RunPod")
    elif env_info['is_colab']:
        if verbose:
            print("✓ Running on Google Colab")
        
        # Mount Drive
        env_info['drive_mounted'] = mount_drive()
        
        if not env_info['drive_mounted']:
            print("⚠ Warning: Drive not mounted. Files may not be accessible.")
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
            print(f"✓ GPU available: {gpu_name}")
        else:
            print("⚠ No GPU available - training will be slower")
    except ImportError:
        print("⚠ PyTorch not installed - GPU check skipped")
    
    # Verify key directories exist
    if verbose:
        print("\nVerifying directories...")
        
        dirs_to_check = [
            ('Models', get_models_path()),
            ('Scalers', get_scalers_path()),
            ('Datasets', get_datasets_path()),
        ]
        
        for name, path in dirs_to_check:
            if path.exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: {path} (NOT FOUND)")
    
    if verbose:
        print("\n" + "=" * 50)
        print("Setup complete!")
        print("=" * 50)
    
    return env_info


def verify_project_structure() -> bool:
    """
    Verify that all required project files and folders exist.
    
    Returns:
        True if all required files exist, False otherwise
    """
    project_path = get_project_path()
    
    required_dirs = [
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
    
    # Check for at least one model file
    models_path = get_models_path()
    if models_path.exists():
        keras_files = list(models_path.glob('*.keras'))
        if keras_files:
            print(f"  ✓ Found {len(keras_files)} model file(s)")
        else:
            print("  ⚠ No .keras model files found")
    
    # Check for at least one dataset
    datasets_path = get_datasets_path()
    if datasets_path.exists():
        csv_files = list(datasets_path.glob('*.csv'))
        if csv_files:
            print(f"  ✓ Found {len(csv_files)} dataset file(s)")
        else:
            print("  ⚠ No .csv dataset files found")
    
    return all_exist


if __name__ == '__main__':
    # Test the utilities
    print("Testing colab_utils.py...")
    print()
    
    env_info = setup_environment()
    print()
    
    print("Environment Info:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")
    print()
    
    verify_project_structure()

