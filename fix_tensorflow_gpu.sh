#!/bin/bash
# Fix TensorFlow GPU Detection on RunPod
# This script tries multiple approaches to get TensorFlow GPU working

set -e

echo "=========================================="
echo "TensorFlow GPU Fix Script"
echo "=========================================="
echo ""

# Check current TensorFlow version
echo "1. Checking current TensorFlow installation..."
python3 -c "import tensorflow as tf; print(f'Current TensorFlow: {tf.__version__}')" 2>/dev/null || echo "TensorFlow not installed"

# Check CUDA and CuDNN versions
echo ""
echo "2. Checking CUDA and CuDNN versions..."
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release" || echo "CUDA version check failed"
else
    echo "nvcc not found, checking CUDA via library..."
    find /usr/local/cuda* -name "libcudart.so*" 2>/dev/null | head -1
fi

# Check CuDNN version
if [ -f "/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/include/cudnn_version.h" ]; then
    echo "CuDNN version:"
    grep CUDNN_MAJOR /usr/local/lib/python3.11/dist-packages/nvidia/cudnn/include/cudnn_version.h -A 2
elif [ -f "/usr/local/cuda/include/cudnn_version.h" ]; then
    echo "CuDNN version:"
    grep CUDNN_MAJOR /usr/local/cuda/include/cudnn_version.h -A 2
else
    echo "CuDNN version file not found in standard locations"
fi

echo ""
echo "3. Setting up LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:/usr/local/cuda-12.4/lib64:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0

echo "✓ Environment variables set"
echo ""

echo "4. Testing current TensorFlow GPU detection..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices found: {len(gpus)}')
if gpus:
    for gpu in gpus:
        print(f'  {gpu.name}')
else:
    print('  No GPU detected')
" 2>&1 | grep -E "(GPU|devices|detected|No GPU)" || echo "Test failed"

echo ""
echo "5. Attempting fixes..."
echo ""

# Option 1: Try TensorFlow 2.15.0.post1 (hotfix version)
echo "Option 1: Installing TensorFlow 2.15.0.post1 (hotfix)..."
pip uninstall -y tensorflow tensorflow-cpu 2>/dev/null || true
pip install tensorflow==2.15.0.post1 --no-cache-dir

echo "Testing TensorFlow 2.15.0.post1..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpus)}')
" 2>&1 | tail -1

if python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
    echo "✓ SUCCESS! TensorFlow 2.15.0.post1 detects GPU"
    echo ""
    echo "Update requirements.txt to use: tensorflow==2.15.0.post1"
    exit 0
fi

echo ""
echo "Option 2: Trying TensorFlow 2.16.0 (may support newer CuDNN)..."
pip uninstall -y tensorflow tensorflow-cpu 2>/dev/null || true
pip install tensorflow==2.16.1 --no-cache-dir 2>/dev/null || pip install tensorflow==2.16.0 --no-cache-dir

echo "Testing TensorFlow 2.16.x..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpus)}')
" 2>&1 | tail -1

if python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
    echo "✓ SUCCESS! TensorFlow 2.16.x detects GPU"
    echo ""
    echo "Update requirements.txt to use: tensorflow==2.16.1 (or 2.16.0)"
    exit 0
fi

echo ""
echo "Option 3: Trying TensorFlow nightly build..."
pip uninstall -y tensorflow tensorflow-cpu 2>/dev/null || true
pip install tf-nightly --no-cache-dir 2>/dev/null || echo "Nightly build not available"

echo "Testing TensorFlow nightly..."
python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU devices: {len(gpus)}')
" 2>&1 | tail -1

if python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); exit(0 if len(gpus) > 0 else 1)" 2>/dev/null; then
    echo "✓ SUCCESS! TensorFlow nightly detects GPU"
    echo ""
    echo "Update requirements.txt to use: tf-nightly"
    exit 0
fi

echo ""
echo "=========================================="
echo "All options tried. GPU detection still failing."
echo "=========================================="
echo ""
echo "This may be a fundamental version incompatibility."
echo "Recommendations:"
echo "1. Use CPU training (slower but works)"
echo "2. Contact RunPod support about CuDNN 9.3.0 installation"
echo "3. Use PyTorch for GPU training (it works correctly)"
echo ""





