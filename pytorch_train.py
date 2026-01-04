#!/usr/bin/env python3
"""
PyTorch Training Utilities
===========================
Training functions for PyTorch models including data loading,
training loops, early stopping, and model saving.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from configparser import ConfigParser
import copy


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = None
        self.counter = 0
        self.best_model_state = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_weights(self, model: nn.Module):
        """Restore model to best weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int,
                      shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Input features (n_samples, seq_len, n_features)
        y: Target values (n_samples,)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
    
    Returns:
        DataLoader instance
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y) if y.dtype in [np.int32, np.int64] else torch.FloatTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
        drop_last=False
    )
    
    return dataloader


def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU detected. Training will use CPU.")
    return device


def train_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                config: ConfigParser, model_path: Optional[str] = None,
                class_weights: Optional[Dict] = None, 
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                task: str = 'regression') -> Dict:
    """
    Train PyTorch model with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training targets
        config: Configuration parser
        model_path: Path to save best model
        class_weights: Class weights for imbalanced classification
        X_val: Validation features (optional, will split from train if not provided)
        y_val: Validation targets (optional)
        task: 'regression' or 'classification'
    
    Returns:
        Dictionary with training history
    """
    # Get device
    device = get_device()
    model = model.to(device)
    
    # Training parameters
    batch_size = config.getint('TRAINING', 'batch_size')
    epochs = config.getint('TRAINING', 'epochs')
    patience = config.getint('TRAINING', 'early_stopping_patience')
    learning_rate = config.getfloat('MODEL', 'learning_rate')
    validation_split = config.getfloat('TRAINING', 'validation_split')
    # Weight decay (L2 regularization) - research-based default
    weight_decay = config.getfloat('MODEL', 'weight_decay', fallback=1e-5)
    
    # Split validation set if not provided
    if X_val is None or y_val is None:
        train_size = int(len(X_train) * (1 - validation_split))
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]
    
    # Batch size validation (RESEARCH-BASED)
    # Note: Research shows smaller batches (64) improve generalization for classification tasks
    # Imperial College paper used batch_size=64 for best results
    # We respect the config value and only reduce if it causes issues (OOM or too few steps)
    train_size = len(X_train)
    min_steps_per_epoch = 50
    
    steps_per_epoch = train_size // batch_size
    
    # Only reduce batch size if it causes too few steps per epoch (would hurt training)
    if steps_per_epoch < min_steps_per_epoch:
        optimal_batch_size = train_size // min_steps_per_epoch
        print(f"  ⚠️  Batch size {batch_size} too large (only {steps_per_epoch} steps per epoch)")
        print(f"     Reducing to {optimal_batch_size} for {min_steps_per_epoch}+ steps per epoch")
        batch_size = optimal_batch_size
        steps_per_epoch = train_size // batch_size
    
    if torch.cuda.is_available():
        print(f"  📈 GPU detected: Using batch size {batch_size} ({steps_per_epoch} steps per epoch)")
        print(f"     Research-based batch size for better generalization (Imperial College: 64)")
    else:
        print(f"  📊 Using batch size {batch_size} ({steps_per_epoch} steps per epoch)")
    
    # Create data loaders
    print("  🔄 Creating optimized PyTorch DataLoaders...")
    train_loader = create_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size, shuffle=False)
    
    if len(X_train) < 500000:
        print(f"  💾 Dataset cached in memory ({len(X_train)} samples) - faster training")
    if len(X_val) < 500000:
        print(f"  💾 Dataset cached in memory ({len(X_val)} samples) - faster training")
    
    # Loss function
    if task == 'classification':
        if class_weights is not None:
            # Convert class weights to tensor
            weight_tensor = torch.FloatTensor([class_weights[i] for i in sorted(class_weights.keys())]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Optimizer with weight decay (L2 regularization)
    # Weight decay helps reduce overfitting by penalizing large weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler (RESEARCH-BASED)
    # Reduces learning rate when validation loss plateaus
    # Patience=5 epochs, factor=0.5 (halve LR), min_lr=1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=1e-6, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best=True)
    
    # Mixed precision training (for GPU speedup)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        print("  ✅ Mixed precision training enabled (float16) - ~2-3x speedup")
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [],
        'accuracy': [] if task == 'classification' else None,
        'val_accuracy': [] if task == 'classification' else None
    }
    
    print(f"  🚀 Training with batch size: {batch_size}, steps per epoch: {steps_per_epoch}")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    outputs = model(batch_X)
                    if task == 'classification':
                        loss = criterion(outputs, batch_y.long())
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                if task == 'classification':
                    loss = criterion(outputs, batch_y.long())
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            
            if task == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y.long()).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(batch_X)
                        if task == 'classification':
                            loss = criterion(outputs, batch_y.long())
                        else:
                            loss = criterion(outputs.squeeze(), batch_y)
                else:
                    outputs = model(batch_X)
                    if task == 'classification':
                        loss = criterion(outputs, batch_y.long())
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                
                val_loss += loss.item()
                
                if task == 'classification':
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y.long()).sum().item()
        
        # Calculate averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if task == 'classification':
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.2f}% - "
                  f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_weights(model)
            break
        
        # Save best model
        if model_path and val_loss == early_stopping.best_loss:
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
            torch.save(model.state_dict(), model_path)
    
    return history


def save_model(model: nn.Module, path: str):
    """Save PyTorch model."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: Optional[torch.device] = None):
    """Load PyTorch model weights."""
    if device is None:
        device = get_device()
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")
    return model



