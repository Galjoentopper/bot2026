#!/bin/bash
# RunPod Setup Script - Run this first to verify everything is ready

set -e  # Exit on error

echo "=========================================="
echo "RunPod Setup & Verification"
echo "=========================================="

# Configuration
REPO_URL="${REPO_URL:-https://github.com/Galjoentopper/bot2026}"
REPO_NAME="bot2026"
WORKSPACE_PATH="/workspace"

# Check current directory
echo ""
echo "1. Current directory:"
pwd

# Check workspace
echo ""
echo "2. Workspace contents:"
if [ -d "$WORKSPACE_PATH" ]; then
    cd "$WORKSPACE_PATH"
    ls -la | head -20
else
    echo "⚠ $WORKSPACE_PATH not found, using current directory"
    WORKSPACE_PATH="$(pwd)"
fi

# Clone repository if not already present
echo ""
echo "3. Checking repository..."
if [ -d "$WORKSPACE_PATH/$REPO_NAME" ]; then
    echo "✓ Repository already exists at $WORKSPACE_PATH/$REPO_NAME"
    cd "$WORKSPACE_PATH/$REPO_NAME"
else
    echo "Cloning repository from $REPO_URL..."
    cd "$WORKSPACE_PATH"
    git clone "$REPO_URL" "$REPO_NAME" || {
        echo "⚠ Git clone failed. Continuing with existing directory if available..."
    }
    if [ -d "$REPO_NAME" ]; then
        cd "$REPO_NAME"
        echo "✓ Repository cloned successfully"
    else
        echo "⚠ Could not clone repository. Please clone manually or ensure you're in the project directory."
        exit 1
    fi
fi

PROJECT_DIR="$(pwd)"
echo "✓ Project directory: $PROJECT_DIR"

# Check GPU
echo ""
echo "4. GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "✓ GPU detected"
else
    echo "⚠ nvidia-smi not found - GPU may not be available"
fi

# Check Python
echo ""
echo "5. Python version:"
python3 --version
pip3 --version

# Check if project exists
echo ""
echo "6. Checking project structure:"
if [ -f "runpod_train.py" ]; then
    echo "✓ Found runpod_train.py"
else
    echo "⚠ runpod_train.py not found in current directory"
fi

if [ -d "PPO approach" ]; then
    echo "✓ Found PPO approach directory"
else
    echo "⚠ PPO approach directory not found"
fi

if [ -d "datasets" ]; then
    echo "✓ Found datasets directory"
    DATASET_COUNT=$(find datasets -name "*.csv" 2>/dev/null | wc -l)
    echo "  CSV files: $DATASET_COUNT"
else
    echo "⚠ datasets directory not found"
    mkdir -p datasets
    echo "  Created datasets directory"
fi

# Install dependencies from requirements.txt
echo ""
echo "7. Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "✓ Found requirements.txt"
    
    # Check if pip cache directory exists (for optimization)
    PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_PATH/.pip_cache}"
    export PIP_CACHE_DIR
    
    echo "Installing packages (this may take a while)..."
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Set LD_LIBRARY_PATH for TensorFlow GPU support (if CUDA libraries exist)
    if [ -d "/usr/local/cuda/lib64" ]; then
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        echo "✓ CUDA libraries found, setting LD_LIBRARY_PATH"
    fi
    
    python3 -m pip install -r requirements.txt
    
    echo "✓ Dependencies installed"
else
    echo "⚠ requirements.txt not found"
fi

# Check PyTorch
echo ""
echo "8. Checking PyTorch:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "⚠ PyTorch not installed"

# Check TensorFlow
echo ""
echo "9. Checking TensorFlow:"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'GPU available: {len(gpus) > 0}'); [print(f'  {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "⚠ TensorFlow not installed"

# Check stable-baselines3
echo ""
echo "10. Checking stable-baselines3:"
python3 -c "import stable_baselines3; print(f'stable-baselines3: {stable_baselines3.__version__}')" 2>/dev/null || echo "⚠ stable-baselines3 not installed"

# Check other dependencies
echo ""
echo "11. Checking other dependencies:"
python3 -c "import pandas, numpy, sklearn, gymnasium; print('✓ Core dependencies available')" 2>/dev/null || echo "⚠ Some dependencies missing"

# Verify Git setup
echo ""
echo "12. Checking Git configuration:"
if command -v git &> /dev/null; then
    echo "✓ Git is installed"
    if [ -d ".git" ]; then
        echo "✓ Git repository detected"
        git remote -v 2>/dev/null || echo "⚠ No remote configured"
    else
        echo "⚠ Not a git repository"
    fi
else
    echo "⚠ Git not installed"
fi

echo ""
echo "=========================================="
echo "Setup check complete!"
echo "=========================================="
