# ============================================================
# PPO TRADING AGENT - SIMPLE COLAB TRAINING SCRIPT
# ============================================================
# Copy this entire code into a Google Colab notebook cell
# Make sure your Bot 2026 folder is in Google Drive
# ============================================================

# CELL 1: Mount Drive and Setup
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

# Set paths
PROJECT_PATH = '/content/drive/MyDrive/Bot 2026'
PPO_PATH = f'{PROJECT_PATH}/PPO approach'

# Verify folder exists
if not os.path.exists(PROJECT_PATH):
    raise FileNotFoundError(f"Folder not found: {PROJECT_PATH}\nMake sure 'Bot 2026' is in your Google Drive!")

print(f"✓ Found project at: {PROJECT_PATH}")

# Add to Python path
sys.path.insert(0, PROJECT_PATH)
sys.path.insert(0, PPO_PATH)
os.chdir(PPO_PATH)

print(f"✓ Working directory: {os.getcwd()}")

# ============================================================
# CELL 2: Install Dependencies
# ============================================================
# !pip install stable-baselines3 gymnasium tensorboard shimmy -q

# ============================================================
# CELL 3: Verify GPU
# ============================================================
# import torch
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CELL 4: Import and Train
# ============================================================
# from train_ppo_agent import train_ppo
# 
# train_ppo(
#     model_type='ensemble',
#     dataset='ETH-EUR_1H_20240101-20251231',  # or ADA-EUR
#     timesteps=500000,
#     resume=True
# )

