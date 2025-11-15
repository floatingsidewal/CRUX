"""
GNN Training Module

Handles training loop, evaluation, and early stopping for GNN models.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import logging
import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from torch_geometric.data import Data

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PyGData = None
    logger.warning("PyTorch Geometric not available. GNN training will be limited.")

from .graph_models import BaseGNN
from .evaluation import ModelEvaluator


class GNNTrainer:
    """
    Trainer for Graph Neural Network models.

    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: BaseGNN,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        device: Optional[str] = None,
    ):
        """
        Initialize GNN trainer.

        Args:
            model: GNN model to train
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GNN training")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_macro': [],
        }

    def train_epoch(self, train_data: List[Data]) -> float:
        """
        Train for one epoch.

        Args:
            train_data: List of training graphs

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for data in train_data:
            # Move data to device
            data = data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)

            # Compute loss
            loss = self.criterion(out, data.y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self, data_list: List[Data]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate model on validation/test set.

        Args:
            data_list: List of graphs to evaluate

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)

                # Forward pass
                out = self.model(data.x, data.edge_index)

                # Compute loss
                loss = self.criterion(out, data.y)
                total_loss += loss.item()

                # Convert to predictions
                probs = torch.sigmoid(out).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels = data.y.cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(data_list) if data_list else 0

        # Concatenate all predictions and labels
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)

        # Compute metrics using ModelEvaluator
        evaluator = ModelEvaluator(label_names=self.model.label_names)
        metrics = evaluator.evaluate(y_true, y_pred)

        return avg_loss, metrics

    def train(
        self,
        train_data: List[Data],
        val_data: List[Data],
        num_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping.

        Args:
            train_data: Training graphs
            val_data: Validation graphs
            num_epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
            min_delta: Minimum improvement to reset patience
            verbose: Print progress

        Returns:
            Training history dictionary
        """
        best_val_f1 = 0
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        logger.info(f"Training {self.model.name} on {self.device}")
        logger.info(f"Train: {len(train_data)} graphs, Val: {len(val_data)} graphs")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_data)

            # Validate
            val_loss, val_metrics = self.evaluate(val_data)
            val_f1_macro = val_metrics['f1_macro']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1_macro'].append(val_f1_macro)

            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val F1 Macro: {val_f1_macro:.4f}"
                )

            # Early stopping
            if val_f1_macro > best_val_f1 + min_delta:
                best_val_f1 = val_f1_macro
                best_epoch = epoch
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} "
                       f"(Val F1 Macro: {best_val_f1:.4f})")

        return self.history

    def predict(self, data_list: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.

        Args:
            data_list: List of graphs

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)

                # Forward pass
                out = self.model(data.x, data.edge_index)

                # Convert to predictions
                probs = torch.sigmoid(out).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.append(preds)
                all_probs.append(probs)

        y_pred = np.vstack(all_preds)
        y_proba = np.vstack(all_probs)

        return y_pred, y_proba

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint.

        Args:
            path: Output file path
            metadata: Optional metadata to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model.get_config(),
            'feature_names': self.model.feature_names,
            'label_names': self.model.label_names,
            'history': self.history,
            'metadata': metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            path: Input file path

        Returns:
            Metadata dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.feature_names = checkpoint.get('feature_names', [])
        self.model.label_names = checkpoint.get('label_names', [])
        self.history = checkpoint.get('history', {})

        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('metadata', {})
