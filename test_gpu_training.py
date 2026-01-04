#!/usr/bin/env python3
"""
Quick test to verify GPU is actually used during TensorFlow training
"""

import os
import sys

# Set LD_LIBRARY_PATH before importing TensorFlow
cuda_paths = [
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib',
    '/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs',
    '/usr/local/cuda-12.4/lib64',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib',
    '/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib',
    '/usr/lib/x86_64-linux-gnu',
]

current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
current_paths = current_ld_path.split(':') if current_ld_path else []
ld_paths_to_add = [p for p in cuda_paths if os.path.exists(p) and p not in current_paths]

if ld_paths_to_add:
    new_ld_path = ':'.join(ld_paths_to_add)
    os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}" if current_ld_path else new_ld_path

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

print("=" * 60)
print("GPU Training Test")
print("=" * 60)
print()

# Check GPU detection
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU devices detected: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"    Device: {details.get('device_name', 'Unknown')}")
        except:
            pass
else:
    print("  ⚠ No GPU detected - test will use CPU")
    sys.exit(1)

print()

# Configure GPU
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ GPU memory growth enabled")
except:
    pass

# Create a simple model and test training
print()
print("Creating test model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Generate dummy data
print("Generating test data...")
X_train = np.random.random((1000, 10)).astype(np.float32)
y_train = np.random.random((1000, 1)).astype(np.float32)

print()
print("Training test model (5 epochs)...")
print("Watch for GPU utilization in another terminal: nvidia-smi")
print()

# Train and check which device is used
with tf.device('/GPU:0'):
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        verbose=1
    )

print()
print("=" * 60)
print("✓ Test completed successfully!")
print("=" * 60)
print()
print("If you saw GPU utilization in nvidia-smi, GPU training is working!")
print("You can now run your actual training with:")
print("  python runpod_main.py --dataset ADA-EUR_1H_20240101-20251231 --verbose")
print()




