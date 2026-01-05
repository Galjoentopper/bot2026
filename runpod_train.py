#!/usr/bin/env python3
"""
RunPod Training Script - Complete Training Pipeline
Adapted from colab_training_notebook.ipynb for RunPod
"""

import os
import sys
import json
from pathlib import Path
import subprocess
import multiprocessing

# PyTorch handles GPU detection automatically - no manual setup needed!
# CUDA_VISIBLE_DEVICES can still be set if needed
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure GPU 0 is visible

# Guard to prevent duplicate execution when SubprocVecEnv creates subprocesses
# Check if we're in a multiprocessing subprocess
try:
    _IS_SUBPROCESS = (multiprocessing.current_process().name != 'MainProcess')
except:
    _IS_SUBPROCESS = False

# Only execute setup steps if this is the main script (not imported, not in subprocess)
# When SubprocVecEnv forks subprocesses, they inherit parent's state but shouldn't re-run setup
_SKIP_SETUP = (__name__ != '__main__' or _IS_SUBPROCESS)

# Helper function to conditionally print STEP messages
def _print_step(step_num, step_name):
    """Print step header only if not in subprocess."""
    if not _SKIP_SETUP:
        print("=" * 60)
        print(f"STEP {step_num}: {step_name}")
        print("=" * 60)

# CONFIGURATION: Set your dataset name here
# Can be overridden via environment variable DATASET_NAME
DATASET_NAME = os.environ.get('DATASET_NAME', "ADA-EUR_1H_20240101-20251231")

# Only execute setup steps if not skipped
if not _SKIP_SETUP:
    _print_step(1, "Setting up paths...")

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
_print_step(2, "Verifying dependencies...")

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
        'torch': 'torch',
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
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  ⚠ No GPU devices detected")
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
_print_step(3, "Verifying GPU...")

import torch

# PyTorch GPU
if torch.cuda.is_available():
    print(f"✓ PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
else:
    print("⚠ No PyTorch GPU detected!")
    print("  Training will be slower on CPU")
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# ============================================
# STEP 4: Verify Dataset
# ============================================
_print_step(4, "Verifying dataset...")

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
_print_step(5, "Training prediction models...")

from train_models import train_all_models, load_config

def select_best_ensemble(dataset_name: str, results_dir: Path, n_models: int = 2) -> list:
    """
    Automatically select best ensemble models based on performance metrics.
    
    Prioritizes:
    1. High validation accuracy
    2. Low overfitting (< 10% gap preferred)
    3. Good convergence status
    
    Args:
        dataset_name: Dataset identifier
        results_dir: Path to results directory containing history files
        n_models: Number of models to select (default: 2)
        
    Returns:
        List of selected model names (e.g., ['dlstm', 'bilstm'])
    """
    try:
        # Find all history files for this dataset
        history_files = list(results_dir.glob(f'history_*_{dataset_name}_classification.json'))
        
        if not history_files:
            print(f"  No history files found for {dataset_name}")
            return None
        
        model_metrics = []
        
        for history_file in history_files:
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Extract model name from filename
                model_name = history_file.stem.replace('history_', '').replace('_classification', '').split('_')[0].lower()
                
                # Calculate metrics
                final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
                best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
                final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
                overfitting = final_train_acc - final_val_acc
                
                # Calculate score: validation accuracy with overfitting penalty
                # Penalize overfitting more if gap > 10%
                if overfitting > 10:
                    overfitting_penalty = 0.5  # Heavy penalty
                elif overfitting > 5:
                    overfitting_penalty = 0.2  # Moderate penalty
                else:
                    overfitting_penalty = 0.1  # Light penalty
                
                score = best_val_acc - (overfitting_penalty * overfitting)
                
                model_metrics.append({
                    'model_name': model_name,
                    'best_val_accuracy': best_val_acc,
                    'final_val_accuracy': final_val_acc,
                    'overfitting': overfitting,
                    'score': score,
                    'convergence': 'Good' if abs(final_train_acc - max(history['accuracy'])) < 2 else 'Needs more training'
                })
                
            except Exception as e:
                print(f"  Warning: Could not analyze {history_file.name}: {e}")
                continue
        
        if not model_metrics:
            return None
        
        # Sort by score (best first)
        model_metrics.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top n_models
        selected = [m['model_name'] for m in model_metrics[:n_models]]
        
        # Print selection reasoning
        print(f"\n  Ensemble Selection Analysis:")
        print(f"  {'Model':<10} {'Val Acc':<12} {'Overfit':<12} {'Score':<12}")
        print(f"  {'-'*50}")
        for m in model_metrics:
            marker = "✓" if m['model_name'] in selected else " "
            print(f"  {marker} {m['model_name']:<8} {m['best_val_accuracy']:>6.2f}%     {m['overfitting']:>6.2f}%     {m['score']:>6.2f}")
        
        return selected
        
    except Exception as e:
        print(f"  Error in ensemble selection: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_and_retrain_bilstm(dataset_name: str, project_path: Path, config) -> bool:
    """
    Check if BiLSTM needs retraining and retrain if necessary.
    
    Detection criteria:
    - Final validation accuracy < 50%
    - Training collapsed (accuracy dropped significantly)
    - Best validation accuracy achieved early but not maintained
    
    Args:
        dataset_name: Dataset identifier
        project_path: Project root path
        config: Training configuration
        
    Returns:
        True if retraining was performed, False otherwise
    """
    try:
        results_dir = project_path / 'results'
        history_file = results_dir / f'history_bilstm_{dataset_name}_classification.json'
        
        if not history_file.exists():
            print("  BiLSTM history file not found, skipping retrain check")
            return False
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
        best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
        final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
        epochs = len(history['accuracy'])
        
        # Check if retraining is needed
        needs_retrain = False
        reason = ""
        
        if final_val_acc < 50:
            needs_retrain = True
            reason = f"Low validation accuracy ({final_val_acc:.2f}% < 50%)"
        elif epochs < 30 and best_val_acc < 60:
            needs_retrain = True
            reason = f"Under-trained ({epochs} epochs, best: {best_val_acc:.2f}%)"
        elif final_train_acc < 30:  # Training collapsed
            needs_retrain = True
            reason = f"Training collapsed (final train acc: {final_train_acc:.2f}%)"
        
        if not needs_retrain:
            print(f"  BiLSTM performance acceptable (val acc: {final_val_acc:.2f}%, epochs: {epochs})")
            return False
        
        print(f"\n  ⚠ BiLSTM needs retraining: {reason}")
        print(f"  Current: {epochs} epochs, Val Acc: {final_val_acc:.2f}%, Best: {best_val_acc:.2f}%")
        print(f"  Retraining with 40-50 epochs and increased patience...")
        
        # Load dataset
        from train_models import load_dataset, train_single_model
        datasets_path = project_path / 'datasets'
        dataset_file = datasets_path / f"{dataset_name}.csv"
        
        if not dataset_file.exists():
            matches = list(datasets_path.glob(f"*{dataset_name}*.csv"))
            if matches:
                dataset_file = matches[0]
            else:
                print(f"  Error: Dataset file not found for {dataset_name}")
                return False
        
        df = load_dataset(str(dataset_file))
        
        # Temporarily modify config for retraining
        import copy
        from configparser import ConfigParser
        retrain_config = copy.deepcopy(config) if config else load_config(str(project_path / 'training_config.txt'))
        
        if isinstance(retrain_config, ConfigParser):
            # Increase epochs and patience
            retrain_config.set('TRAINING', 'epochs', '50')
            retrain_config.set('TRAINING', 'early_stopping_patience', '20')
        
        # Retrain BiLSTM
        print(f"  Starting BiLSTM retraining...")
        result = train_single_model(
            model_name='bilstm',
            dataset_name=dataset_name,
            df=df,
            config=retrain_config,
            task='classification',
            output_dir=str(project_path)
        )
        
        if result and result.get('metrics'):
            new_val_acc = result['metrics'].get('val_accuracy', 0)
            print(f"  ✓ Retraining complete! New validation accuracy: {new_val_acc:.2f}%")
            return True
        else:
            print(f"  ⚠ Retraining may have failed, check logs")
            return False
            
    except Exception as e:
        print(f"  Error in BiLSTM retrain check: {e}")
        import traceback
        traceback.print_exc()
        return False

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

# Load PPO configuration early to determine which models to train
ppo_path = get_ppo_path()
ppo_config_path = ppo_path / 'ppo_config.txt'

if not ppo_config_path.exists():
    print("⚠ PPO config file not found, using defaults")
    from train_ppo_agent import load_config
    ppo_config = load_config(None)
else:
    from train_ppo_agent import load_config
    ppo_config = load_config(str(ppo_config_path))

# Determine which models to train based on PPO config
# If using ensemble, only train ensemble models
if ppo_config['models'].get('prediction_model') == 'ensemble':
    ensemble_models = ppo_config['models'].get('ensemble_models', ['dlstm', 'bilstm'])
    # Handle both list and string formats
    if isinstance(ensemble_models, str):
        required_models = [m.strip() for m in ensemble_models.split(',')]
    elif isinstance(ensemble_models, list):
        required_models = [m.strip() if isinstance(m, str) else str(m) for m in ensemble_models]
    else:
        # Fallback to default
        required_models = ['dlstm', 'bilstm']
    print(f"\n📦 Using ensemble mode - will only train: {', '.join(required_models).upper()}")
else:
    # Single model mode - train all 4 models (for flexibility)
    required_models = ['lstm', 'gru', 'bilstm', 'dlstm']
    print(f"\n📦 Using single model mode - will train all models for flexibility")

# Check if all required models already exist
models_dir = PROJECT_PATH / 'models'
model_files = {}
all_models_exist = True

# Try to use version manager for model checking
try:
    from model_versioning import get_version_manager
    version_manager = get_version_manager(models_dir / 'manifest.json')
    # Initialize manifest from existing models if needed
    if not (models_dir / 'manifest.json').exists():
        version_manager.initialize_from_existing_models()
    use_versioning = True
except ImportError:
    use_versioning = False
    version_manager = None

for model_name in required_models:
    model_path = None
    
    # Try version manager first
    if use_versioning and version_manager:
        try:
            model_path = version_manager.get_model_file(
                model_name,
                DATASET_NAME,
                version=None,  # Latest version
                task='classification'
            )
        except Exception as e:
            pass  # Fall back to old method
    
    # Fallback to old pattern matching
    if model_path is None or not model_path.exists():
        # Try versioned pattern first
        matching = list(models_dir.glob(f"{model_name}_{DATASET_NAME}_v*_classification.pth"))
        if matching:
            model_path = matching[0]
        else:
            # Try old pattern (no version)
            model_pattern = f"{model_name}_{DATASET_NAME}_classification.pth"
            model_path = models_dir / model_pattern
            
            if not model_path.exists():
                # Try partial match
                matching = list(models_dir.glob(f"{model_name}*{DATASET_NAME}*.pth"))
                if matching:
                    model_path = matching[0]
                else:
                    all_models_exist = False
                    break
    
    model_files[model_name] = model_path

if all_models_exist:
    print(f"✓ All prediction models already exist for dataset: {DATASET_NAME}")
    print("  Skipping prediction model training...")
    for model_name, model_path in model_files.items():
        print(f"  ✓ {model_name.upper()}: {model_path.name}")
    results = []  # Empty results, models already exist
    
    # Still run ensemble selection even if models already exist
    print("\n" + "=" * 60)
    print("STEP 5.5: Automatic Ensemble Selection")
    print("=" * 60)
    try:
        selected_ensemble = select_best_ensemble(DATASET_NAME, PROJECT_PATH / 'results', n_models=2)
        if selected_ensemble:
            print(f"\n✓ Selected ensemble: {', '.join([m.upper() for m in selected_ensemble])}")
            # Update required_models for PPO training
            required_models = selected_ensemble
            # Update PPO config if using ensemble mode
            if ppo_config['models'].get('prediction_model') == 'ensemble':
                ppo_config['models']['ensemble_models'] = selected_ensemble
                print(f"  Updated PPO config ensemble_models to: {', '.join(selected_ensemble)}")
        else:
            print("⚠ Could not select ensemble automatically, using configured models")
    except Exception as e:
        print(f"⚠ Error in ensemble selection: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠ Some models missing. Training prediction models...")
    print("=" * 60)
    
    # Train only required models (based on ensemble config)
    models_str = ', '.join([m.upper() for m in required_models])
    print(f"\nTraining models on dataset: {DATASET_NAME}")
    print(f"Models: {models_str}")
    print("Task: Classification (Fall, Stationary, Rise)")
    print("-" * 60)

    try:
        results = train_all_models(
            datasets_dir=str(get_datasets_path()),
            config=config,
            task='classification',
            models=required_models,  # Use required_models instead of hardcoded list
            specific_dataset=DATASET_NAME,
            use_ensemble=False
        )
        
        if results:
            print("\n" + "=" * 60)
            print("✓ All prediction models trained successfully!")
            print("=" * 60)
            
            # Automatic ensemble selection based on model performance
            print("\n" + "=" * 60)
            print("STEP 5.5: Automatic Ensemble Selection")
            print("=" * 60)
            try:
                selected_ensemble = select_best_ensemble(DATASET_NAME, PROJECT_PATH / 'results', n_models=2)
                if selected_ensemble:
                    print(f"\n✓ Selected ensemble: {', '.join([m.upper() for m in selected_ensemble])}")
                    # Update required_models for PPO training
                    required_models = selected_ensemble
                    # Update PPO config if using ensemble mode
                    if ppo_config['models'].get('prediction_model') == 'ensemble':
                        ppo_config['models']['ensemble_models'] = selected_ensemble
                        print(f"  Updated PPO config ensemble_models to: {', '.join(selected_ensemble)}")
                else:
                    print("⚠ Could not select ensemble automatically, using configured models")
            except Exception as e:
                print(f"⚠ Error in ensemble selection: {e}")
                import traceback
                traceback.print_exc()
            
            # Check if BiLSTM needs retraining
            print("\n" + "=" * 60)
            print("STEP 5.6: Checking for Underperforming Models")
            print("=" * 60)
            try:
                bilstm_needs_retrain = check_and_retrain_bilstm(DATASET_NAME, PROJECT_PATH, config)
                if bilstm_needs_retrain:
                    print("✓ BiLSTM retraining completed")
            except Exception as e:
                print(f"⚠ Error checking BiLSTM: {e}")
                import traceback
                traceback.print_exc()
            
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
_print_step(6, "Validating models before PPO training...")

import numpy as np

# Import prediction wrapper
from prediction_wrapper import PredictionModel, EnsemblePredictionModel

# Test loading each individual model (only test required models)
print("\n1. Testing Individual Model Loading:")
print("-" * 60)
models_to_test = required_models  # Only test models that were trained/required
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
_print_step(7, "Training PPO agent...")

# Change to PPO approach directory
ppo_path = get_ppo_path()
os.chdir(ppo_path)
sys.path.insert(0, str(ppo_path))

from train_ppo_agent import train_ppo

# Update dataset name in PPO config if needed (ppo_config already loaded above)
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
        skip_setup=True,  # Skip verbose setup - already done in steps 1-6
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
        
        # Run backtesting on the trained model
        print("\n" + "=" * 60)
        print("STEP 7.5: Running Backtesting on Trained PPO Agent")
        print("=" * 60)
        try:
            # Add PPO approach to path for import
            import sys
            ppo_approach_path = get_ppo_path()
            if str(ppo_approach_path) not in sys.path:
                sys.path.insert(0, str(ppo_approach_path))
            from backtest_ppo import backtest_ppo_agent
            final_model_path = get_ppo_models_path() / f"ppo_{ppo_config['models']['prediction_model']}_{DATASET_NAME}.zip"
            if final_model_path.exists():
                print(f"\nBacktesting model: {final_model_path.name}")
                backtest_results = backtest_ppo_agent(
                    model_path=str(final_model_path),
                    dataset_name=DATASET_NAME,
                    n_episodes=10,
                    deterministic=True,
                    save_results=True
                )
                if backtest_results:
                    print("\n✓ Backtesting completed successfully!")
                else:
                    print("\n⚠ Backtesting completed with warnings")
            else:
                print(f"⚠ Model file not found for backtesting: {final_model_path}")
        except ImportError:
            print("⚠ backtest_ppo.py not found - skipping backtesting")
        except Exception as e:
            print(f"⚠ Error during backtesting: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the pipeline if backtesting fails
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
# STEP 9: Push Models to GitHub (Optional)
# ============================================
push_to_github = os.environ.get('RUNPOD_PUSH_TO_GITHUB', 'false').lower() == 'true'
if push_to_github:
    print("\n" + "=" * 60)
    print("STEP 9: Pushing models to GitHub...")
    print("=" * 60)
    
    try:
        from runpod_github import push_models_to_github
        
        # Collect model files, datasets, scalers, and results
        files_to_push = []
        
        # Prediction models
        models_dir = PROJECT_PATH / 'models'
        if models_dir.exists():
            files_to_push.extend(models_dir.glob('*.pth'))
            files_to_push.extend(models_dir.glob('*.pt'))  # Alternative PyTorch extension
        
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
            print(f"  - Models: {len([f for f in files_to_push if f.suffix in ['.pth', '.pt', '.zip']])}")
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

# ============================================
# Pipeline completed successfully - explicit exit
# ============================================
if not _SKIP_SETUP:
    print("\n" + "=" * 60)
    print("Pipeline completed successfully. Exiting...")
    print("=" * 60)
sys.exit(0)

