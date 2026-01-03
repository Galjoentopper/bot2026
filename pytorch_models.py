#!/usr/bin/env python3
"""
PyTorch Model Implementations
==============================
PyTorch implementations of LSTM, GRU, BiLSTM, and DLSTM models
matching the TensorFlow/Keras architectures.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMModel(nn.Module):
    """LSTM model with two LSTM layers and dropout."""
    
    def __init__(self, input_shape: Tuple, output_units: int = 1,
                 units: int = 256, dropout: float = 0.2,
                 task: str = 'regression'):
        """
        Args:
            input_shape: (sequence_length, num_features)
            output_units: Number of output units
            units: Number of LSTM units (hidden size)
            dropout: Dropout rate
            task: 'regression' or 'classification'
        """
        super(LSTMModel, self).__init__()
        self.task = task
        seq_len, num_features = input_shape
        
        # First LSTM layer (returns sequences)
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=units,
            batch_first=True,
            dropout=0  # Dropout handled manually
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer (returns only last output)
        self.lstm2 = nn.LSTM(
            input_size=units,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        if task == 'classification':
            self.fc = nn.Linear(units, output_units)
        else:
            self.fc = nn.Linear(units, output_units)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Take only the last output (return_sequences=False equivalent)
        lstm2_out = lstm2_out[:, -1, :]  # (batch, units)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Output layer
        output = self.fc(lstm2_out)
        
        if self.task == 'classification':
            output = torch.softmax(output, dim=1)
        
        return output


class GRUModel(nn.Module):
    """GRU model with two GRU layers and dropout."""
    
    def __init__(self, input_shape: Tuple, output_units: int = 1,
                 units: int = 256, dropout: float = 0.2,
                 task: str = 'regression'):
        """
        Args:
            input_shape: (sequence_length, num_features)
            output_units: Number of output units
            units: Number of GRU units (hidden size)
            dropout: Dropout rate
            task: 'regression' or 'classification'
        """
        super(GRUModel, self).__init__()
        self.task = task
        seq_len, num_features = input_shape
        
        # First GRU layer
        self.gru1 = nn.GRU(
            input_size=num_features,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second GRU layer
        self.gru2 = nn.GRU(
            input_size=units,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        if task == 'classification':
            self.fc = nn.Linear(units, output_units)
        else:
            self.fc = nn.Linear(units, output_units)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # First GRU layer
        gru1_out, _ = self.gru1(x)
        gru1_out = self.dropout1(gru1_out)
        
        # Second GRU layer
        gru2_out, _ = self.gru2(gru1_out)
        # Take only the last output
        gru2_out = gru2_out[:, -1, :]  # (batch, units)
        gru2_out = self.dropout2(gru2_out)
        
        # Output layer
        output = self.fc(gru2_out)
        
        if self.task == 'classification':
            output = torch.softmax(output, dim=1)
        
        return output


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model with two bidirectional LSTM layers."""
    
    def __init__(self, input_shape: Tuple, output_units: int = 1,
                 units: int = 256, dropout: float = 0.2,
                 task: str = 'regression'):
        """
        Args:
            input_shape: (sequence_length, num_features)
            output_units: Number of output units
            units: Number of LSTM units per direction (hidden size)
            dropout: Dropout rate
            task: 'regression' or 'classification'
        """
        super(BiLSTMModel, self).__init__()
        self.task = task
        seq_len, num_features = input_shape
        
        # First bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=units,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=units * 2,  # Bidirectional doubles the size
            hidden_size=units,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer (bidirectional doubles hidden size)
        if task == 'classification':
            self.fc = nn.Linear(units * 2, output_units)
        else:
            self.fc = nn.Linear(units * 2, output_units)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # First bidirectional LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Second bidirectional LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Take only the last output
        lstm2_out = lstm2_out[:, -1, :]  # (batch, units * 2)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Output layer
        output = self.fc(lstm2_out)
        
        if self.task == 'classification':
            output = torch.softmax(output, dim=1)
        
        return output


class DLSTMModel(nn.Module):
    """
    Decomposition LSTM model.
    
    Based on Imperial College paper - best trading profitability.
    Uses proper time series decomposition: Trend (AvgPool) + Remainder
    """
    
    def __init__(self, input_shape: Tuple, output_units: int = 1,
                 units: int = 256, dropout: float = 0.2,
                 task: str = 'regression', ma_window: int = 10):
        """
        Args:
            input_shape: (sequence_length, num_features)
            output_units: Number of output units
            units: Number of LSTM units (hidden size)
            dropout: Dropout rate
            task: 'regression' or 'classification'
            ma_window: Moving average window for trend extraction
        """
        super(DLSTMModel, self).__init__()
        self.task = task
        seq_len, num_features = input_shape
        
        # Trend extraction using Average Pooling
        # padding='same' equivalent: padding = (kernel_size - 1) // 2
        # But we need to ensure output size matches input size exactly
        # For odd kernel_size: padding = (kernel_size - 1) // 2
        # For even kernel_size: we need to handle it differently
        if ma_window % 2 == 1:
            padding = (ma_window - 1) // 2
        else:
            padding = (ma_window - 2) // 2  # Adjust for even kernel size
        
        self.avg_pool = nn.AvgPool1d(
            kernel_size=ma_window,
            stride=1,
            padding=padding
        )
        
        # LSTM on Trend branch
        self.trend_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.trend_dropout1 = nn.Dropout(dropout)
        self.trend_lstm2 = nn.LSTM(
            input_size=units,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.trend_dropout2 = nn.Dropout(dropout)
        
        # LSTM on Remainder branch
        self.remainder_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.remainder_dropout1 = nn.Dropout(dropout)
        self.remainder_lstm2 = nn.LSTM(
            input_size=units,
            hidden_size=units,
            batch_first=True,
            dropout=0
        )
        self.remainder_dropout2 = nn.Dropout(dropout)
        
        # Output layer (merged branches)
        if task == 'classification':
            self.fc = nn.Linear(units, output_units)
        else:
            self.fc = nn.Linear(units, output_units)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Convert to (batch, features, seq_len) for AvgPool1d
        x_transposed = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Trend extraction using Average Pooling
        trend = self.avg_pool(x_transposed)
        # Convert back to (batch, seq_len, features)
        trend = trend.transpose(1, 2)
        
        # Ensure trend has same sequence length as input (handle padding edge cases)
        seq_len = x.size(1)
        if trend.size(1) != seq_len:
            # Trim or pad to match
            if trend.size(1) > seq_len:
                trend = trend[:, :seq_len, :]
            else:
                # Pad with last value
                padding = seq_len - trend.size(1)
                last_val = trend[:, -1:, :].expand(-1, padding, -1)
                trend = torch.cat([trend, last_val], dim=1)
        
        # Remainder: Original - Trend
        remainder = x - trend
        
        # LSTM on Trend branch
        trend_out, _ = self.trend_lstm1(trend)
        trend_out = self.trend_dropout1(trend_out)
        trend_out, _ = self.trend_lstm2(trend_out)
        trend_out = trend_out[:, -1, :]  # Take last output
        trend_out = self.trend_dropout2(trend_out)
        
        # LSTM on Remainder branch
        remainder_out, _ = self.remainder_lstm1(remainder)
        remainder_out = self.remainder_dropout1(remainder_out)
        remainder_out, _ = self.remainder_lstm2(remainder_out)
        remainder_out = remainder_out[:, -1, :]  # Take last output
        remainder_out = self.remainder_dropout2(remainder_out)
        
        # Merge hidden states (simple additive fusion)
        merged = trend_out + remainder_out
        
        # Output layer
        output = self.fc(merged)
        
        if self.task == 'classification':
            output = torch.softmax(output, dim=1)
        
        return output


def get_model(model_name: str, input_shape: Tuple, output_units: int,
              units: int = 256, dropout: float = 0.2,
              task: str = 'regression', ma_window: int = 10) -> nn.Module:
    """
    Factory function to get model by name.
    
    Args:
        model_name: 'lstm', 'gru', 'bilstm', or 'dlstm'
        input_shape: (sequence_length, num_features)
        output_units: Number of output units
        units: Number of hidden units
        dropout: Dropout rate
        task: 'regression' or 'classification'
        ma_window: Moving average window for DLSTM
    
    Returns:
        PyTorch model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'lstm':
        return LSTMModel(
            input_shape=input_shape,
            output_units=output_units,
            units=units,
            dropout=dropout,
            task=task
        )
    elif model_name == 'gru':
        return GRUModel(
            input_shape=input_shape,
            output_units=output_units,
            units=units,
            dropout=dropout,
            task=task
        )
    elif model_name == 'bilstm':
        return BiLSTMModel(
            input_shape=input_shape,
            output_units=output_units,
            units=units,
            dropout=dropout,
            task=task
        )
    elif model_name == 'dlstm':
        return DLSTMModel(
            input_shape=input_shape,
            output_units=output_units,
            units=units,
            dropout=dropout,
            task=task,
            ma_window=ma_window
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: lstm, gru, bilstm, dlstm")

