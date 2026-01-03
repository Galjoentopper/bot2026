#!/usr/bin/env python3
"""
RunPod Training Script - Complete Training Pipeline
Adapted from colab_training_notebook.ipynb for RunPod
"""

import os
import sys
from pathlib import Path
import subprocess

# Set LD_LIBRARY_PATH for TensorFlow GPU support
# TensorFlow needs to find CUDA/cuDNN libraries at runtime
# Comprehensive library paths for RunPod CUDA 12.4 installation
cuda_paths = [
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib',  # Main CUDA libraries (libcudart, libcublas, etc.)
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs',  # Development stubs for linking
    '/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib',  # cuDNN libraries
    '/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib',  # CUDA runtime
    '/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib',  # cuBLAS
    '/usr/lib/x86_64-linux-gnu',  # System CUDA driver (libcuda.so)
    '/usr/local/nvidia/lib',  # RunPod default NVIDIA libraries
    '/usr/local/nvidia/lib64',  # RunPod default NVIDIA libraries (64-bit)
    '/usr/local/cuda-12.4/lib64',  # Alternative CUDA path
    '/usr/local/cuda/lib64',  # Generic CUDA path (if symlinked)
]

current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
current_paths = current_ld_path.split(':') if current_ld_path else []
ld_paths_to_add = []

for cuda_path in cuda_paths:
    if os.path.exists(cuda_path) and cuda_path not in current_paths:
        ld_paths_to_add.append(cuda_path)

if ld_paths_to_add:
    new_ld_path = ':'.join(ld_paths_to_add)
    if current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
    print(f"✓ Set LD_LIBRARY_PATH for CUDA libraries")
    print(f"  Added {len(ld_paths_to_add)} path(s) to LD_LIBRARY_PATH")

# Set TensorFlow-specific environment variables for GPU support
# These help with GPU initialization and library loading
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # Allow GPU memory growth
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # Ensure GPU 0 is visible
# Keep TF_CPP_MIN_LOG_LEVEL at 1 (warnings only) unless debugging
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TensorFlow log noise

# ============================================
# STEP 1: Setup Paths
# ============================================
print("=" * 60)
print("STEP 1: Setting up paths...")
print("=" * 60)

# CONFIGURATION: Set your dataset name here
# Can be overridden via environment variable DATASET_NAME
DATASET_NAME = os.environ.get('DATASET_NAME', "ADA-EUR_1H_20240101-20251231")

# Use runpod_utils for path detection
try:
    # Add PPO approach to path first
    current_dir = Path(__file__).parent
    ppo_path = current_dir / 'PPO approach'
    if ppo_path.exists():
        sys.path.insert(0, str(ppo_path))
    else:
        # Try if we're already in PPO approach
        if current_dir.name == 'PPO approach' or (current_dir / 'PPO approach').exists():
            sys.path.insert(0, str(current_dir))
    
    from runpod_utils import get_project_path, setup_environment, get_datasets_path, get_ppo_path
    from colab_utils import get_ppo_models_path, get_checkpoints_path
    
    # Setup environment
    env_info = setup_environment(verbose=True)
    PROJECT_PATH = env_info['project_path']
    
except ImportError:
    # Fallback to manual path detection
    print("⚠ runpod_utils not found, using fallback path detection")
    possible_paths = [
        Path('/workspace/bot2026'),
        Path('/workspace/bot-2026'),
        Path('/workspace/Bot 2026'),
        Path('/workspace'),
        Path('/runpod-volume/bot2026'),
        Path('/data/bot2026'),
        Path.cwd(),
    ]
    
    PROJECT_PATH = None
    for path in possible_paths:
        if path.exists() and (path / 'PPO approach').exists():
            PROJECT_PATH = path
            break
    
    if PROJECT_PATH is None:
        PROJECT_PATH = Path.cwd()
        print(f"⚠ Using current directory: {PROJECT_PATH}")
    
    # Add paths to sys.path
    if str(PROJECT_PATH) not in sys.path:
        sys.path.insert(0, str(PROJECT_PATH))
    if str(PROJECT_PATH / 'PPO approach') not in sys.path:
        sys.path.insert(0, str(PROJECT_PATH / 'PPO approach'))
    
    # Import path functions
    try:
        from colab_utils import get_datasets_path, get_ppo_path, get_ppo_models_path, get_checkpoints_path
    except ImportError:
        # Define fallback functions
        def get_datasets_path():
            return PROJECT_PATH / 'datasets'
        def get_ppo_path():
            return PROJECT_PATH / 'PPO approach'
        def get_ppo_models_path():
            path = get_ppo_path() / 'models'
            path.mkdir(parents=True, exist_ok=True)
            return path
        def get_checkpoints_path():
            path = get_ppo_path() / 'checkpoints'
            path.mkdir(parents=True, exist_ok=True)
            return path

# Change to project directory
os.chdir(PROJECT_PATH)

print(f"✓ Project path: {PROJECT_PATH}")
print(f"✓ Dataset name: {DATASET_NAME}")
print(f"✓ Current directory: {os.getcwd()}")

# ============================================
# STEP 2: Verify/Install Dependencies
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Verifying dependencies...")
print("=" * 60)

# Check if requirements.txt exists and install from it
requirements_file = PROJECT_PATH / 'requirements.txt'
if requirements_file.exists():
    print(f"✓ Found requirements.txt")
    
    # Check if packages are already installed
    print("Checking installed packages...")
    missing_packages = []
    
    # Read requirements.txt
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    # Check key packages
    key_packages = {
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'stable_baselines3': 'stable-baselines3',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'gymnasium': 'gymnasium',
    }
    
    for module_name, package_name in key_packages.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package_name}: {version}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ✗ {package_name}: not installed")
    
    # Install from requirements.txt if packages are missing
    if missing_packages:
        print(f"\n⚠ Missing {len(missing_packages)} package(s). Installing from requirements.txt...")
        print("  This may take a while...")
        
        # Use pip cache if available
        pip_cache_dir = os.environ.get('PIP_CACHE_DIR', '/workspace/.pip_cache' if os.path.exists('/workspace') else None)
        pip_args = [sys.executable, "-m", "pip", "install", "-q"]
        if pip_cache_dir:
            pip_args.extend(["--cache-dir", pip_cache_dir])
        pip_args.append("-r")
        pip_args.append(str(requirements_file))
        
        subprocess.check_call(pip_args)
        print("✓ Dependencies installed from requirements.txt")
    else:
        print("✓ All required packages are installed")
else:
    print("⚠ requirements.txt not found - skipping dependency installation")
    print("  Please ensure all dependencies are installed manually")

# Verify key packages are available
try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
    
    # Check TensorFlow GPU support
    print(f"  CUDA built-in: {tf.test.is_built_with_cuda()}")
    print(f"  GPU support: {tf.test.is_built_with_gpu_support()}")
    
    # Try to list GPU devices
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✓ GPU devices detected: {len(gpus)}")
        else:
            print("  ⚠ No GPU devices found by TensorFlow")
            print("  💡 This may be a library loading issue")
    except Exception as e:
        print(f"  ⚠ Error checking GPU: {e}")
        
except ImportError:
    print("⚠ TensorFlow not available")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError:
    print("⚠ PyTorch not available")

try:
    import stable_baselines3
    print(f"✓ stable-baselines3: {stable_baselines3.__version__}")
except ImportError:
    print("⚠ stable-baselines3 not available")

# ============================================
# STEP 3: Verify GPU
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Verifying GPU...")
print("=" * 60)

import torch
import tensorflow as tf

# PyTorch GPU
if torch.cuda.is_available():
    print(f"✓ PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ No PyTorch GPU detected!")
    print("  Training will be VERY slow on CPU")

# TensorFlow GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ TensorFlow GPU: {len(gpus)} GPU(s) available")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
else:
    print("⚠ No TensorFlow GPU detected!")

if not torch.cuda.is_available() and not gpus:
    print("\n" + "⚠" * 30)
    print("⚠ WARNING: No GPU detected!")
    print("   Training will be VERY slow on CPU.")
    print("⚠" * 30)
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# ============================================
# STEP 4: Verify Dataset
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Verifying dataset...")
print("=" * 60)

import pandas as pd

datasets_path = get_datasets_path()
if not datasets_path.exists():
    print(f"⚠ Datasets directory not found: {datasets_path}")
    print("  Creating directory...")
    datasets_path.mkdir(parents=True, exist_ok=True)

dataset_file = datasets_path / f"{DATASET_NAME}.csv"

if not dataset_file.exists():
    # Try to find with partial match
    matches = list(datasets_path.glob(f"*{DATASET_NAME}*.csv"))
    if matches:
        dataset_file = matches[0]
        print(f"⚠ Found similar file: {dataset_file.name}")
    else:
        print(f"❌ Dataset not found: {DATASET_NAME}.csv")
        print(f"\nAvailable datasets in {datasets_path}:")
        csv_files = list(datasets_path.glob("*.csv"))
        if csv_files:
            for f in sorted(csv_files):
                print(f"  - {f.name}")
        else:
            print("  (No CSV files found)")
        print(f"\nPlease upload your dataset to: {datasets_path}")
        sys.exit(1)

print(f"✓ Dataset found: {dataset_file.name}")

# Load and display dataset info
try:
    df = pd.read_csv(dataset_file)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Check required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n⚠ WARNING: Missing required columns: {missing_cols}")
        print("   Dataset may not work correctly for training")
    else:
        print(f"  ✓ All required columns present")
    
    # Display date range if timestamp exists
    if 'timestamp' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"\nDate Range:")
            print(f"  Start: {df['datetime'].min()}")
            print(f"  End: {df['datetime'].max()}")
            print(f"  Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
        except:
            print("  (Could not parse timestamps)")
    
except Exception as e:
    print(f"\n❌ ERROR loading dataset: {e}")
    raise

# ============================================
# STEP 5: Train Prediction Models
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Training prediction models...")
print("=" * 60)

from train_models import train_all_models, load_config

# Load configuration
config_path = PROJECT_PATH / 'training_config.txt'
if config_path.exists():
    print(f"Loading config from: {config_path}")
    config = load_config(str(config_path))
else:
    # Try config.txt, but skip if it's JSON format
    config_path_alt = PROJECT_PATH / 'config.txt'
    if config_path_alt.exists():
        # Check if it's JSON (starts with {)
        with open(config_path_alt, 'r') as f:
            first_char = f.read(1).strip()
        if first_char == '{':
            print("⚠ config.txt is JSON format (for crypto downloader), skipping")
            print("  Using default training configuration")
            config = None
        else:
            config_path = config_path_alt
            print(f"Loading config from: {config_path}")
            config = load_config(str(config_path))
    else:
        print("⚠ Config file not found, using defaults")
        config = None

# Train all models
print(f"\nTraining models on dataset: {DATASET_NAME}")
print("Models: LSTM, GRU, BiLSTM, DLSTM")
print("Task: Classification (Fall, Stationary, Rise)")
print("-" * 60)

try:
    results = train_all_models(
        datasets_dir=str(get_datasets_path()),
        config=config,
        task='classification',
        models=['lstm', 'gru', 'bilstm', 'dlstm'],
        specific_dataset=DATASET_NAME,
        use_ensemble=False
    )
    
    if results:
        print("\n" + "=" * 60)
        print("✓ All prediction models trained successfully!")
        print("=" * 60)
        
        # Print summary
        print("\nTraining Summary:")
        for result in results:
            model_name = result['model_name']
            metrics = result['metrics']
            print(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.6f}")
                else:
                    print(f"  {metric}: {value}")
    else:
        print("\n⚠ No models were trained. Check dataset name and paths.")
        
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================
# STEP 6: Validate Models Before PPO Training
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Validating models before PPO training...")
print("=" * 60)

import numpy as np

# Import prediction wrapper
from prediction_wrapper import PredictionModel, EnsemblePredictionModel

# Test loading each individual model
print("\n1. Testing Individual Model Loading:")
print("-" * 60)
models_to_test = ['lstm', 'gru', 'bilstm', 'dlstm']
loaded_models = {}

for model_name in models_to_test:
    try:
        model = PredictionModel(model_name, DATASET_NAME)
        if model.load():
            print(f"  ✓ {model_name.upper()} loaded successfully")
            loaded_models[model_name] = model
        else:
            print(f"  ✗ {model_name.upper()} failed to load")
    except Exception as e:
        print(f"  ✗ {model_name.upper()} error: {e}")

# Test ensemble loading
print("\n2. Testing Ensemble Loading:")
print("-" * 60)
try:
    ensemble = EnsemblePredictionModel(DATASET_NAME, list(loaded_models.keys()))
    if ensemble.load():
        print(f"  ✓ Ensemble loaded with {len(ensemble.models)} model(s)")
    else:
        print(f"  ✗ Ensemble failed to load")
        raise RuntimeError("Ensemble loading failed")
except Exception as e:
    print(f"  ✗ Ensemble error: {e}")
    raise

# Verify model inference works
print("\n3. Testing Model Inference:")
print("-" * 60)
try:
    # Create dummy sequence for testing
    seq_len = ensemble.sequence_length
    # Get number of features
    if ensemble.models and hasattr(ensemble.models[0], 'feature_names') and ensemble.models[0].feature_names:
        n_features = len(ensemble.models[0].feature_names)
    elif ensemble.feature_scaler:
        if hasattr(ensemble.feature_scaler, 'n_features_in_'):
            n_features = ensemble.feature_scaler.n_features_in_
        elif hasattr(ensemble.feature_scaler, 'feature_names_in_'):
            n_features = len(ensemble.feature_scaler.feature_names_in_)
        else:
            n_features = 27  # Default
    else:
        n_features = 27  # Default fallback
    
    dummy_seq = np.random.randn(seq_len, n_features)
    print(f"  Test sequence shape: {dummy_seq.shape}")
    
    # Test single-step prediction
    pred_class, confidence, probs = ensemble.predict(dummy_seq)
    print(f"  ✓ Single-step prediction:")
    print(f"    Class: {pred_class} ({['Fall', 'Stationary', 'Rise'][pred_class]})")
    print(f"    Confidence: {confidence:.4f}")
    print(f"    Probabilities: Fall={probs[0]:.3f}, Stationary={probs[1]:.3f}, Rise={probs[2]:.3f}")
    
except Exception as e:
    print(f"  ✗ Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n✓ Model validation completed - ready for PPO training")

# ============================================
# STEP 7: Train PPO Agent
# ============================================
print("\n" + "=" * 60)
print("STEP 7: Training PPO agent...")
print("=" * 60)

# Change to PPO approach directory
ppo_path = get_ppo_path()
os.chdir(ppo_path)
sys.path.insert(0, str(ppo_path))

from train_ppo_agent import train_ppo, load_config

# Load PPO configuration
ppo_config_path = ppo_path / 'ppo_config.txt'
print(f"\nLoading PPO config from: {ppo_config_path}")

if not ppo_config_path.exists():
    print("⚠ PPO config file not found, using defaults")
    ppo_config = load_config(None)
else:
    ppo_config = load_config(str(ppo_config_path))

# Update dataset name in config if needed
if ppo_config['models']['dataset'] != DATASET_NAME:
    print(f"\n⚠ Updating dataset name in config: {ppo_config['models']['dataset']} -> {DATASET_NAME}")
    ppo_config['models']['dataset'] = DATASET_NAME

# Verify PPO config matches prediction model settings
print(f"\nConfiguration Check:")
print(f"  Dataset: {ppo_config['models']['dataset']}")
print(f"  Sequence length: {ppo_config['environment']['sequence_length']}")
print(f"  Prediction model: {ppo_config['models']['prediction_model']}")

# Configure prediction horizons
prediction_horizons = ppo_config['models'].get('prediction_horizons', [1, 2, 3, 5, 10])
print(f"\nPrediction Horizons Configuration:")
print(f"  Horizons: {prediction_horizons}")

# Train PPO agent
print(f"\n{'='*60}")
print("Starting PPO Training...")
print("="*60)

try:
    model = train_ppo(
        model_type=ppo_config['models']['prediction_model'],
        dataset=ppo_config['models']['dataset'],
        timesteps=ppo_config['training']['total_timesteps'],
        config_path=str(ppo_config_path) if ppo_config_path.exists() else None,
        resume=True,  # Resume from checkpoint if exists
    )
    
    if model is not None:
        print("\n" + "="*60)
        print("✓ PPO training completed successfully!")
        print("="*60)
        
        # Display training summary
        final_model_path = get_ppo_models_path() / f"ppo_{ppo_config['models']['prediction_model']}_{DATASET_NAME}.zip"
        if final_model_path.exists():
            size_mb = final_model_path.stat().st_size / (1024 * 1024)
            print(f"\nFinal model saved: {final_model_path.name} ({size_mb:.2f} MB)")
        
        checkpoint_path = get_checkpoints_path() / f"{ppo_config['models']['prediction_model']}_{DATASET_NAME}"
        if checkpoint_path.exists():
            checkpoints = list(checkpoint_path.glob("*.zip"))
            print(f"Checkpoints available: {len(checkpoints)}")
    else:
        print("\n⚠ PPO training returned None - check for errors above")
        
except Exception as e:
    print(f"\n❌ Error during PPO training: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================
# DONE!
# ============================================
print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nAll models saved to: {PROJECT_PATH}")
print(f"  Prediction models: {PROJECT_PATH / 'models'}")
print(f"  PPO models: {get_ppo_models_path()}")
print(f"  Checkpoints: {get_checkpoints_path()}")
print(f"  Results: {PROJECT_PATH / 'results'}")
print("\n" + "=" * 60)

# ============================================
# STEP 8: Push Models to GitHub (Optional)
# ============================================
push_to_github = os.environ.get('RUNPOD_PUSH_TO_GITHUB', 'false').lower() == 'true'
if push_to_github:
    print("\n" + "=" * 60)
    print("STEP 8: Pushing models to GitHub...")
    print("=" * 60)
    
    try:
        from runpod_github import push_models_to_github
        
        # Collect model files, datasets, scalers, and results
        files_to_push = []
        
        # Prediction models
        models_dir = PROJECT_PATH / 'models'
        if models_dir.exists():
            files_to_push.extend(models_dir.glob('*.keras'))
            files_to_push.extend(models_dir.glob('*.h5'))
            files_to_push.extend(models_dir.glob('*.hdf5'))
        
        # PPO models
        ppo_models_dir = get_ppo_models_path()
        if ppo_models_dir.exists():
            files_to_push.extend(ppo_models_dir.glob('*.zip'))
        
        # Checkpoints
        checkpoints_dir = get_checkpoints_path()
        if checkpoints_dir.exists():
            files_to_push.extend(checkpoints_dir.glob('**/*.zip'))
        
        # Scalers
        scalers_dir = PROJECT_PATH / 'scalers'
        if scalers_dir.exists():
            files_to_push.extend(scalers_dir.glob('*.pkl'))
            files_to_push.extend(scalers_dir.glob('**/*.pkl'))
        
        # Datasets (CSV files)
        datasets_dir = get_datasets_path()
        if datasets_dir.exists():
            files_to_push.extend(datasets_dir.glob('*.csv'))
        
        # Results (plots, JSON files)
        results_dir = PROJECT_PATH / 'results'
        if results_dir.exists():
            files_to_push.extend(results_dir.glob('*.png'))
            files_to_push.extend(results_dir.glob('*.jpg'))
            files_to_push.extend(results_dir.glob('*.json'))
            files_to_push.extend(results_dir.glob('**/*.png'))
            files_to_push.extend(results_dir.glob('**/*.jpg'))
            files_to_push.extend(results_dir.glob('**/*.json'))
        
        # PPO results (in PPO approach/results/)
        ppo_results_dir = PROJECT_PATH / 'PPO approach' / 'results'
        if ppo_results_dir.exists():
            files_to_push.extend(ppo_results_dir.glob('*.png'))
            files_to_push.extend(ppo_results_dir.glob('*.jpg'))
            files_to_push.extend(ppo_results_dir.glob('*.json'))
            files_to_push.extend(ppo_results_dir.glob('**/*.png'))
            files_to_push.extend(ppo_results_dir.glob('**/*.jpg'))
            files_to_push.extend(ppo_results_dir.glob('**/*.json'))
        
        # Log files
        logs_dir = PROJECT_PATH / 'logs'
        if logs_dir.exists():
            files_to_push.extend(logs_dir.glob('*.log'))
            files_to_push.extend(logs_dir.glob('**/*.log'))
            files_to_push.extend(logs_dir.glob('**/*.log.*'))
            # TensorBoard event files
            files_to_push.extend(logs_dir.glob('**/events.out.tfevents.*'))
        
        # PPO logs (in PPO approach/logs/)
        ppo_logs_dir = PROJECT_PATH / 'PPO approach' / 'logs'
        if ppo_logs_dir.exists():
            files_to_push.extend(ppo_logs_dir.glob('*.log'))
            files_to_push.extend(ppo_logs_dir.glob('**/*.log'))
            files_to_push.extend(ppo_logs_dir.glob('**/*.log.*'))
            # TensorBoard event files
            files_to_push.extend(ppo_logs_dir.glob('**/events.out.tfevents.*'))
        
        # Other log files in root
        for log_file in PROJECT_PATH.glob('*.log'):
            files_to_push.append(log_file)
        for log_file in PROJECT_PATH.glob('*.log.*'):
            files_to_push.append(log_file)
        for out_file in PROJECT_PATH.glob('*.out'):
            files_to_push.append(out_file)
        if (PROJECT_PATH / 'nohup.out').exists():
            files_to_push.append(PROJECT_PATH / 'nohup.out')
        
        # Runs directory (if exists)
        runs_dir = PROJECT_PATH / 'runs'
        if runs_dir.exists():
            files_to_push.extend(runs_dir.glob('**/*'))
        
        if files_to_push:
            print(f"Found {len(files_to_push)} file(s) to push:")
            print(f"  - Models: {len([f for f in files_to_push if f.suffix in ['.keras', '.h5', '.hdf5', '.zip']])}")
            print(f"  - Datasets: {len([f for f in files_to_push if f.suffix == '.csv'])}")
            print(f"  - Scalers: {len([f for f in files_to_push if f.suffix == '.pkl'])}")
            print(f"  - Results: {len([f for f in files_to_push if f.suffix in ['.png', '.jpg', '.json']])}")
            print(f"  - Logs: {len([f for f in files_to_push if '.log' in f.name or '.out' in f.name or 'events.out.tfevents' in f.name])}")
            # Push all files without size restrictions
            if push_models_to_github(PROJECT_PATH, files_to_push, skip_large_files=False):
                print("✓ All files pushed to GitHub successfully")
            else:
                print("⚠ Failed to push files to GitHub")
        else:
            print("No files found to push")
    except ImportError:
        print("⚠ runpod_github.py not found - skipping GitHub push")
    except Exception as e:
        print(f"⚠ Error pushing to GitHub: {e}")
else:
    print("\n⚠ GitHub push disabled (set RUNPOD_PUSH_TO_GITHUB=true to enable)")

