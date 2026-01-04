"""
Prediction Wrapper
==================
Load trained prediction models (LSTM/GRU/BiLSTM/DLSTM) and generate features for PPO.

This module wraps PyTorch models to use them as feature extractors for the PPO trading agent.
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')

from colab_utils import get_project_path, get_models_path, get_scalers_path, get_datasets_path

# Import PyTorch models
sys.path.insert(0, str(Path(__file__).parent.parent))
from pytorch_models import get_model as get_pytorch_model


class PredictionModel:
    """Wrapper for a single prediction model."""
    
    def __init__(self, model_name: str, dataset_name: str, version: Optional[str] = None):
        """
        Initialize prediction model wrapper.
        
        Args:
            model_name: Model type ('lstm', 'gru', 'bilstm', 'dlstm')
            dataset_name: Dataset identifier (e.g., 'ETH-EUR_1H_20240101-20251231')
            version: Model version (e.g., 'v1.0.0'). If None, loads latest version.
        """
        self.model_name = model_name.lower()
        self.dataset_name = dataset_name
        self.version = version
        self.model = None
        self.scaler_data = None
        self.feature_scaler = None
        self.sequence_length = 60
        self.feature_names = []
        self.loaded = False
    
    def load(self) -> bool:
        """
        Load model and scaler from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Get device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  No GPU available, using CPU")
            
            models_path = get_models_path()
            scalers_path = get_scalers_path()
            
            # Try to use version manager if available
            model_path = None
            try:
                import sys
                from pathlib import Path
                # Add parent directory to path to import model_versioning
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                
                from model_versioning import get_version_manager
                version_manager = get_version_manager(models_path / 'manifest.json')
                
                # Get model file from version manager
                model_path = version_manager.get_model_file(
                    self.model_name,
                    self.dataset_name,
                    version=self.version,
                    task='classification'
                )
                
                if model_path:
                    print(f"  Loading version: {self.version or 'latest'}")
            except ImportError:
                # Versioning not available, fall back to old method
                pass
            except Exception as e:
                print(f"  Warning: Version manager error: {e}, using fallback")
            
            # Fallback to old naming if version manager didn't find it
            if model_path is None or not model_path.exists():
                # Find model file (PyTorch format) - try versioned first, then old format
                if self.version:
                    model_pattern = f"{self.model_name}_{self.dataset_name}_{self.version}_classification.pth"
                    model_path = models_path / model_pattern
                
                if model_path is None or not model_path.exists():
                    # Try old format (no version)
                    model_pattern = f"{self.model_name}_{self.dataset_name}_classification.pth"
                    model_path = models_path / model_pattern
                
                if not model_path.exists():
                    # Try to find with partial match
                    matching = list(models_path.glob(f"{self.model_name}*{self.dataset_name}*.pth"))
                    if matching:
                        model_path = matching[0]
                    else:
                        print(f"Model not found: {model_pattern}")
                        return False
            
            # Load scaler first to get model architecture info
            scaler_pattern = f"scaler_{self.dataset_name}.pkl"
            scaler_path = scalers_path / scaler_pattern
            
            if not scaler_path.exists():
                # Try partial match
                matching = list(scalers_path.glob(f"scaler*{self.dataset_name}*.pkl"))
                if matching:
                    scaler_path = matching[0]
                else:
                    # Try any scaler with dataset in name
                    dataset_base = self.dataset_name.split('_')[0]  # e.g., 'ETH-EUR'
                    matching = list(scalers_path.glob(f"*{dataset_base}*.pkl"))
                    if matching:
                        scaler_path = matching[0]
                    else:
                        print(f"Scaler not found for dataset: {self.dataset_name}")
                        return False
            
            print(f"Loading scaler: {scaler_path.name}")
            with open(scaler_path, 'rb') as f:
                self.scaler_data = pickle.load(f)
            
            # Extract scaler components
            if isinstance(self.scaler_data, dict):
                self.feature_scaler = self.scaler_data.get('feature_scaler')
                self.feature_names = self.scaler_data.get('feature_names', [])
                self.sequence_length = self.scaler_data.get('sequence_length', 60)
            else:
                self.feature_scaler = self.scaler_data
            
            # Build model architecture (need to match training architecture)
            input_shape = (self.sequence_length, len(self.feature_names) if self.feature_names else 26)
            output_units = 3  # Classification: Fall, Stationary, Rise
            
            # Create model with same architecture as training
            self.model = get_pytorch_model(
                model_name=self.model_name,
                input_shape=input_shape,
                output_units=output_units,
                units=256,  # Default from training
                dropout=0.2,  # Default from training
                task='classification'
            )
            
            # Load model weights
            print(f"Loading PyTorch model: {model_path.name}")
            self.model.load_state_dict(torch.load(str(model_path), map_location=device))
            self.model.to(device)
            self.model.eval()
            
            print(f"  Model configured for {'GPU' if torch.cuda.is_available() else 'CPU'} inference")
            
            self.loaded = True
            print(f"✓ Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Make prediction on a sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, n_features)
                     or (batch, sequence_length, n_features)
        
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
            - predicted_class: 0=Fall, 1=Stationary, 2=Rise
            - confidence: Max probability
            - probabilities: Full softmax output [p_fall, p_stationary, p_rise]
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, ...]  # Add batch dimension
        
        # Get probabilities - USE CPU for environment steps to free GPU for PPO training
        # This is critical: prediction models on CPU during env steps = more GPU for PPO
        # Environment steps are CPU-bound anyway, so CPU inference is fine
        # PPO training will use GPU exclusively, maximizing GPU utilization
        device = torch.device('cpu')  # Force CPU for env steps - free GPU for PPO
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).to(device)
        
        # Move model to CPU temporarily for inference
        original_device = next(self.model.parameters()).device
        self.model.to(device)
        
        with torch.no_grad():
            probs = self.model(sequence_tensor).cpu().numpy()
        
        # Move model back to original device
        self.model.to(original_device)
        
        if len(probs.shape) > 1:
            probs = probs[0]  # Get first batch item
        
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        return predicted_class, confidence, probs
    
    def predict_multi_horizon(self, sequence: np.ndarray, horizons: List[int] = [1, 2, 3, 5, 10]) -> Dict[int, Tuple[int, float, np.ndarray]]:
        """
        Predict multiple future steps ahead for direction prediction.
        
        For direction classification, we predict each horizon independently using
        the same input sequence. This is reasonable because:
        1. Direction prediction models learn patterns that generalize across horizons
        2. True autoregression is complex for classification (would need to predict features)
        3. Research shows models can learn multi-horizon patterns from single input
        
        Args:
            sequence: Input sequence of shape (sequence_length, n_features)
            horizons: List of steps ahead to predict (e.g., [1, 2, 3, 5, 10])
        
        Returns:
            Dictionary mapping horizon -> (predicted_class, confidence, probabilities)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence[np.newaxis, ...]  # Add batch dimension
        
        results = {}
        
        import tensorflow as tf
        
        # For direction prediction, we use the same sequence for all horizons
        # The model has learned to predict movement patterns that apply to different timeframes
        # This is a common approach in financial prediction (see research papers)
        # Use CPU for environment steps to free GPU for PPO training
        with tf.device('/CPU:0'):
            # Predict once - the model outputs direction which applies to multiple horizons
            probs = self.model.predict(sequence, verbose=0, batch_size=1)
            
            if len(probs.shape) > 1:
                probs = probs[0]  # Get first batch item
            
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            
            # For now, use same prediction for all horizons
            # In future, could train separate models for each horizon or use seq2seq
            for horizon in horizons:
                results[horizon] = (predicted_class, confidence, probs.copy())
        
        return results
    
    def get_features(self, sequence: np.ndarray, horizons: List[int] = None) -> np.ndarray:
        """
        Extract features from sequence for PPO observation.
        
        Args:
            sequence: Input sequence
            horizons: Optional list of horizons for multi-step prediction
        
        Returns:
            Feature vector including prediction info
            - If horizons=None: 5 features (class, confidence, 3 probs)
            - If horizons provided: 5 + (len(horizons) * 4) features
              (each horizon adds: class_norm, confidence, prob_fall, prob_rise)
        """
        if horizons is None or len(horizons) == 0:
            # Single-step prediction (backward compatible)
            pred_class, confidence, probs = self.predict(sequence)
            
            features = np.array([
                pred_class / 2.0,  # Normalize class to 0-1
                confidence,
                probs[0],  # Fall probability
                probs[1],  # Stationary probability
                probs[2],  # Rise probability
            ])
        else:
            # Multi-horizon prediction
            multi_preds = self.predict_multi_horizon(sequence, horizons=horizons)
            
            # Start with immediate prediction (t+1)
            if 1 in multi_preds:
                pred_class, confidence, probs = multi_preds[1]
                features = [
                    pred_class / 2.0,
                    confidence,
                    probs[0],  # Fall
                    probs[1],  # Stationary
                    probs[2],  # Rise
                ]
            else:
                # Fallback if t+1 not in horizons
                pred_class, confidence, probs = self.predict(sequence)
                features = [
                    pred_class / 2.0,
                    confidence,
                    probs[0],
                    probs[1],
                    probs[2],
                ]
            
            # Add features for each additional horizon
            for horizon in sorted(horizons):
                if horizon == 1:
                    continue  # Already added
                
                if horizon in multi_preds:
                    h_class, h_conf, h_probs = multi_preds[horizon]
                    # Add: normalized class, confidence, fall prob, rise prob
                    # (skip stationary to reduce dimensionality)
                    features.extend([
                        h_class / 2.0,
                        h_conf,
                        h_probs[0],  # Fall
                        h_probs[2],  # Rise
                    ])
                else:
                    # Pad with zeros if prediction failed
                    features.extend([0.5, 0.33, 0.33, 0.34])
            
            features = np.array(features)
        
        return features


class EnsemblePredictionModel:
    """Ensemble of multiple prediction models."""
    
    def __init__(self, dataset_name: str, model_names: List[str] = None):
        """
        Initialize ensemble.
        
        Args:
            dataset_name: Dataset identifier
            model_names: List of model names to include (default: all)
        """
        self.dataset_name = dataset_name
        self.model_names = model_names or ['lstm', 'gru', 'bilstm', 'dlstm']
        self.models: List[PredictionModel] = []
        self.loaded = False
    
    def load(self) -> bool:
        """
        Load all models in the ensemble.
        
        Returns:
            True if at least one model loaded successfully
        """
        print(f"\nLoading ensemble for dataset: {self.dataset_name}")
        print("-" * 50)
        
        for model_name in self.model_names:
            model = PredictionModel(model_name, self.dataset_name)
            if model.load():
                self.models.append(model)
        
        self.loaded = len(self.models) > 0
        
        if self.loaded:
            print(f"\n✓ Ensemble loaded with {len(self.models)} model(s)")
        else:
            print("\n✗ No models loaded")
        
        return self.loaded
    
    def predict(self, sequence: np.ndarray, voting: str = 'soft') -> Tuple[int, float, np.ndarray]:
        """
        Make ensemble prediction.
        
        Args:
            sequence: Input sequence
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
            
        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        if not self.loaded:
            raise RuntimeError("Ensemble not loaded. Call load() first.")
        
        all_probs = []
        for model in self.models:
            _, _, probs = model.predict(sequence)
            all_probs.append(probs)
        
        if voting == 'soft':
            # Average probabilities
            avg_probs = np.mean(all_probs, axis=0)
            predicted_class = int(np.argmax(avg_probs))
            confidence = float(np.max(avg_probs))
            return predicted_class, confidence, avg_probs
        else:
            # Hard voting (majority)
            predictions = [np.argmax(p) for p in all_probs]
            from collections import Counter
            votes = Counter(predictions)
            predicted_class = votes.most_common(1)[0][0]
            confidence = votes[predicted_class] / len(predictions)
            # Return average probs for consistency
            avg_probs = np.mean(all_probs, axis=0)
            return predicted_class, confidence, avg_probs
    
    def predict_multi_horizon(self, sequence: np.ndarray, horizons: List[int] = [1, 2, 3, 5, 10],
                              voting: str = 'soft') -> Dict[int, Tuple[int, float, np.ndarray]]:
        """
        Make multi-horizon ensemble predictions.
        
        Args:
            sequence: Input sequence
            horizons: List of steps ahead to predict
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        
        Returns:
            Dictionary mapping horizon -> (predicted_class, confidence, probabilities)
        """
        if not self.loaded:
            raise RuntimeError("Ensemble not loaded. Call load() first.")
        
        # Get predictions from all models for all horizons
        all_horizon_preds = {}
        for model in self.models:
            model_preds = model.predict_multi_horizon(sequence, horizons=horizons)
            for horizon, (h_class, h_conf, h_probs) in model_preds.items():
                if horizon not in all_horizon_preds:
                    all_horizon_preds[horizon] = []
                all_horizon_preds[horizon].append((h_class, h_conf, h_probs))
        
        # Combine predictions for each horizon
        results = {}
        for horizon in horizons:
            if horizon not in all_horizon_preds:
                continue
            
            horizon_preds = all_horizon_preds[horizon]
            
            if voting == 'soft':
                # Average probabilities across models
                avg_probs = np.mean([p for _, _, p in horizon_preds], axis=0)
                predicted_class = int(np.argmax(avg_probs))
                confidence = float(np.max(avg_probs))
                results[horizon] = (predicted_class, confidence, avg_probs)
            else:
                # Hard voting (majority)
                predictions = [c for c, _, _ in horizon_preds]
                from collections import Counter
                votes = Counter(predictions)
                predicted_class = votes.most_common(1)[0][0]
                confidence = votes[predicted_class] / len(predictions)
                avg_probs = np.mean([p for _, _, p in horizon_preds], axis=0)
                results[horizon] = (predicted_class, confidence, avg_probs)
        
        return results
    
    def get_features(self, sequence: np.ndarray, horizons: List[int] = None) -> np.ndarray:
        """
        Extract ensemble features with optional multi-horizon predictions.
        
        Args:
            sequence: Input sequence
            horizons: Optional list of horizons for multi-step prediction
        
        Returns:
            Feature vector (same format as PredictionModel.get_features)
        """
        if horizons is None or len(horizons) == 0:
            # Single-step prediction (backward compatible)
            pred_class, confidence, probs = self.predict(sequence)
            
            features = np.array([
                pred_class / 2.0,
                confidence,
                probs[0],
                probs[1],
                probs[2],
            ])
        else:
            # Multi-horizon prediction
            multi_preds = self.predict_multi_horizon(sequence, horizons=horizons)
            
            # Start with immediate prediction (t+1)
            if 1 in multi_preds:
                pred_class, confidence, probs = multi_preds[1]
                features = [
                    pred_class / 2.0,
                    confidence,
                    probs[0],  # Fall
                    probs[1],  # Stationary
                    probs[2],  # Rise
                ]
            else:
                # Fallback if t+1 not in horizons
                pred_class, confidence, probs = self.predict(sequence)
                features = [
                    pred_class / 2.0,
                    confidence,
                    probs[0],
                    probs[1],
                    probs[2],
                ]
            
            # Add features for each additional horizon
            for horizon in sorted(horizons):
                if horizon == 1:
                    continue  # Already added
                
                if horizon in multi_preds:
                    h_class, h_conf, h_probs = multi_preds[horizon]
                    # Add: normalized class, confidence, fall prob, rise prob
                    features.extend([
                        h_class / 2.0,
                        h_conf,
                        h_probs[0],  # Fall
                        h_probs[2],  # Rise
                    ])
                else:
                    # Pad with zeros if prediction failed
                    features.extend([0.5, 0.33, 0.33, 0.34])
            
            features = np.array(features)
        
        return features
    
    @property
    def sequence_length(self) -> int:
        """Get sequence length from first model."""
        if self.models:
            return self.models[0].sequence_length
        return 60
    
    @property
    def feature_scaler(self):
        """Get feature scaler from first model."""
        if self.models:
            return self.models[0].feature_scaler
        return None


def load_prediction_model(model_name: str, dataset_name: str) -> PredictionModel:
    """
    Load a single prediction model.
    
    Args:
        model_name: Model type ('lstm', 'gru', 'bilstm', 'dlstm')
        dataset_name: Dataset identifier
        
    Returns:
        Loaded PredictionModel
    """
    model = PredictionModel(model_name, dataset_name)
    model.load()
    return model


def load_ensemble(dataset_name: str, model_names: List[str] = None) -> EnsemblePredictionModel:
    """
    Load an ensemble of prediction models.
    
    Args:
        dataset_name: Dataset identifier
        model_names: List of models to include (default: all)
        
    Returns:
        Loaded EnsemblePredictionModel
    """
    ensemble = EnsemblePredictionModel(dataset_name, model_names)
    ensemble.load()
    return ensemble


def get_available_models(dataset_name: str = None) -> List[str]:
    """
    Get list of available trained models.
    
    Args:
        dataset_name: Optional filter by dataset
        
    Returns:
        List of model file names
    """
    models_path = get_models_path()
    
    if not models_path.exists():
        return []
    
    pattern = f"*{dataset_name}*.pth" if dataset_name else "*.pth"
    return [f.name for f in models_path.glob(pattern)]


def get_available_datasets() -> List[str]:
    """
    Get list of available datasets.
    
    Returns:
        List of dataset names
    """
    datasets_path = get_datasets_path()
    
    if not datasets_path.exists():
        return []
    
    return [f.stem for f in datasets_path.glob("*.csv")]


class DataPreprocessor:
    """Preprocess data for prediction models."""
    
    def __init__(self, feature_scaler, sequence_length: int = 60):
        """
        Initialize preprocessor.
        
        Args:
            feature_scaler: Fitted sklearn scaler
            sequence_length: Sequence length for model input
        """
        self.feature_scaler = feature_scaler
        self.sequence_length = sequence_length
        self.buffer = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.
        
        Replicates the indicators from train_models.py
        """
        result = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # Bollinger Bands
        sma = df['close'].rolling(window=20, min_periods=1).mean()
        std = df['close'].rolling(window=20, min_periods=1).std()
        result['bb_upper'] = sma + (std * 2)
        result['bb_lower'] = sma - (std * 2)
        result['bb_percent'] = (df['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'] + 1e-10)
        
        # Volume indicators
        result['volume_ma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        result['volume_ratio'] = df['volume'] / (result['volume_ma'] + 1e-10)
        
        # Price features
        result['price_change'] = df['close'].pct_change().fillna(0)
        result['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        result['close_open_range'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            result[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            result[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        return result.fillna(0)
    
    def prepare_sequence(self, data: np.ndarray) -> np.ndarray:
        """
        Prepare a sequence for model input.
        
        Args:
            data: Raw feature data of shape (n_samples, n_features)
            
        Returns:
            Scaled sequence ready for prediction
        """
        if self.feature_scaler is not None:
            scaled = self.feature_scaler.transform(data)
        else:
            scaled = data
        
        return scaled[-self.sequence_length:]
    
    def update_buffer(self, new_data: np.ndarray):
        """
        Update rolling buffer with new data.
        
        Args:
            new_data: New data point(s) to add
        """
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
        
        for row in new_data:
            self.buffer.append(row)
        
        # Keep only needed history
        if len(self.buffer) > self.sequence_length * 2:
            self.buffer = self.buffer[-self.sequence_length * 2:]
    
    def get_current_sequence(self) -> Optional[np.ndarray]:
        """
        Get current sequence from buffer.
        
        Returns:
            Prepared sequence or None if buffer too small
        """
        if len(self.buffer) < self.sequence_length:
            return None
        
        data = np.array(self.buffer[-self.sequence_length:])
        return self.prepare_sequence(data)


if __name__ == '__main__':
    # Test the prediction wrapper
    print("Testing prediction_wrapper.py...")
    print()
    
    # List available models and datasets
    print("Available datasets:")
    for ds in get_available_datasets():
        print(f"  - {ds}")
    
    print("\nAvailable models:")
    for model in get_available_models():
        print(f"  - {model}")
    
    # Try to load a model
    datasets = get_available_datasets()
    if datasets:
        dataset_name = datasets[0]
        print(f"\nTrying to load ensemble for: {dataset_name}")
        
        ensemble = load_ensemble(dataset_name)
        
        if ensemble.loaded:
            # Create dummy sequence for testing
            seq_len = ensemble.sequence_length
            n_features = 30  # Approximate feature count
            dummy_seq = np.random.randn(seq_len, n_features)
            
            print(f"\nTest prediction with dummy sequence ({seq_len} x {n_features}):")
            pred_class, confidence, probs = ensemble.predict(dummy_seq)
            
            class_names = ['Fall', 'Stationary', 'Rise']
            print(f"  Predicted: {class_names[pred_class]}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Probabilities: {probs}")


