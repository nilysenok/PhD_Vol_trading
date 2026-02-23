"""GRU model for volatility forecasting."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
import warnings
from .base import BaseVolatilityModel

# Conditional imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. GRU models will not be available.")


if TORCH_AVAILABLE:
    class GRUNetwork(nn.Module):
        """GRU neural network for sequence modeling."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2,
            bidirectional: bool = False
        ):
            """Initialize GRU network.

            Args:
                input_size: Number of input features.
                hidden_size: Number of hidden units.
                num_layers: Number of GRU layers.
                dropout: Dropout rate.
                bidirectional: Use bidirectional GRU.
            """
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )

            self.fc = nn.Linear(hidden_size * self.num_directions, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor of shape (batch, seq_len, features).

            Returns:
                Output tensor of shape (batch, 1).
            """
            # GRU forward
            out, _ = self.gru(x)

            # Take last time step
            out = out[:, -1, :]

            # Fully connected layer
            out = self.fc(out)

            return out


class GRUModel(BaseVolatilityModel):
    """GRU model for volatility forecasting.

    Gated Recurrent Units are effective for capturing temporal
    dependencies in time series data.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 22,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize GRU model.

        Args:
            hidden_size: Number of hidden units.
            num_layers: Number of GRU layers.
            dropout: Dropout rate.
            bidirectional: Use bidirectional GRU.
            sequence_length: Length of input sequences.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            device: Device to use (None for auto-detect).
            **kwargs: Additional parameters.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__(name="GRU", **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._training_history = {"train_loss": [], "val_loss": []}

    def _create_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences for GRU input.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples,).

        Returns:
            Tuple of (X_seq, y_seq) with shapes (n_seq, seq_len, n_features)
            and (n_seq,).
        """
        n_samples, n_features = X.shape
        n_sequences = n_samples - self.sequence_length + 1

        X_seq = np.zeros((n_sequences, self.sequence_length, n_features))

        for i in range(n_sequences):
            X_seq[i] = X[i:i + self.sequence_length]

        if y is not None:
            # Target is the value at the end of each sequence
            y_seq = y[self.sequence_length - 1:]
            return X_seq, y_seq

        return X_seq, None

    def _normalize(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Normalize data.

        Args:
            X: Features.
            y: Target.
            fit: Whether to fit scalers.

        Returns:
            Normalized (X, y).
        """
        if fit:
            self._mean_X = X.mean(axis=0)
            self._std_X = X.std(axis=0) + 1e-8
            if y is not None:
                self._mean_y = y.mean()
                self._std_y = y.std() + 1e-8

        X_norm = (X - self._mean_X) / self._std_X

        if y is not None:
            y_norm = (y - self._mean_y) / self._std_y
            return X_norm, y_norm

        return X_norm, None

    def _denormalize_y(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        return y * self._std_y + self._mean_y

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> "GRUModel":
        """Fit GRU model.

        Args:
            X: Training features.
            y: Training target.
            X_val: Validation features.
            y_val: Validation target.
            **kwargs: Additional parameters.

        Returns:
            Self.
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # Normalize
        X, y = self._normalize(X, y, fit=True)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)

        # Create model
        n_features = X.shape[1]
        self._model = GRUNetwork(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)

        # Create data loader
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val, y_val = self._normalize(X_val, y_val, fit=False)
            X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self._model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self._model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self._training_history["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = self._model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self._training_history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self._best_state = self._model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        # Restore best model
                        self._model.load_state_dict(self._best_state)
                        break

        self.is_fitted = True
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Normalize
        X, _ = self._normalize(X, None, fit=False)

        # Create sequences
        X_seq, _ = self._create_sequences(X)

        # Predict
        self._model.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            y_pred = self._model(X_tensor).cpu().numpy().flatten()

        # Denormalize
        y_pred = self._denormalize_y(y_pred)

        # Ensure non-negative (RV should be positive)
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def get_training_history(self) -> dict:
        """Get training history.

        Returns:
            Dictionary with training and validation losses.
        """
        return self._training_history
