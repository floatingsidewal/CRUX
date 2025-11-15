"""
ML Module

Machine learning components for training and evaluating models on CRUX datasets.
"""

from .dataset import DatasetLoader
from .features import FeatureExtractor
from .models import BaselineModel, XGBoostModel, RandomForestModel
from .evaluation import ModelEvaluator

# GNN components (optional, require PyTorch Geometric)
try:
    from .graph_loader import GraphDatasetLoader
    from .graph_features import GraphFeatureExtractor
    from .graph_models import BaseGNN, GCNModel, GATModel, GraphSAGEModel
    from .graph_trainer import GNNTrainer
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

__all__ = [
    "DatasetLoader",
    "FeatureExtractor",
    "BaselineModel",
    "XGBoostModel",
    "RandomForestModel",
    "ModelEvaluator",
]

if GNN_AVAILABLE:
    __all__.extend([
        "GraphDatasetLoader",
        "GraphFeatureExtractor",
        "BaseGNN",
        "GCNModel",
        "GATModel",
        "GraphSAGEModel",
        "GNNTrainer",
    ])
