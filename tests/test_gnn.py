"""
Tests for GNN components

Tests graph feature extraction, graph loading, and GNN models.
"""

import pytest
import numpy as np
import networkx as nx
from pathlib import Path

# Try to import PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from crux.ml.graph_features import GraphFeatureExtractor

# Mark all tests in this module as requiring torch
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch Geometric not available"
)


class TestGraphFeatureExtractor:
    """Test graph feature extraction."""

    def test_init(self):
        """Test feature extractor initialization."""
        extractor = GraphFeatureExtractor(max_features=20)
        assert extractor.max_features == 20
        assert not extractor._fitted

    def test_fit_single_graph(self):
        """Test fitting on a single graph."""
        # Create simple graph
        graph = nx.DiGraph()
        graph.add_node("resource1", type="Microsoft.Storage/storageAccounts", properties={"tier": "Standard"})
        graph.add_node("resource2", type="Microsoft.Network/virtualNetworks", properties={"tier": "Premium"})

        extractor = GraphFeatureExtractor(max_features=10)
        extractor.fit([graph])

        assert extractor._fitted
        assert len(extractor.type_encoder.classes_) == 2
        assert "Microsoft.Storage/storageAccounts" in extractor.type_encoder.classes_
        assert "Microsoft.Network/virtualNetworks" in extractor.type_encoder.classes_

    def test_transform_graph(self):
        """Test transforming a graph to feature matrix."""
        # Create graph
        graph = nx.DiGraph()
        graph.add_node("res1", type="Microsoft.Storage/storageAccounts", properties={"enabled": True})
        graph.add_node("res2", type="Microsoft.Storage/storageAccounts", properties={"enabled": False})

        extractor = GraphFeatureExtractor(max_features=10)
        extractor.fit([graph])

        X, node_to_idx = extractor.transform(graph)

        assert X.shape[0] == 2  # 2 nodes
        assert X.shape[1] == extractor.get_num_features()
        assert len(node_to_idx) == 2
        assert "res1" in node_to_idx
        assert "res2" in node_to_idx

    def test_fit_transform(self):
        """Test fit_transform convenience method."""
        graph1 = nx.DiGraph()
        graph1.add_node("res1", type="Microsoft.Storage/storageAccounts", properties={})

        graph2 = nx.DiGraph()
        graph2.add_node("res2", type="Microsoft.Network/virtualNetworks", properties={})

        extractor = GraphFeatureExtractor(max_features=10)
        results = extractor.fit_transform([graph1, graph2])

        assert len(results) == 2
        assert extractor._fitted

        X1, mapping1 = results[0]
        assert X1.shape[0] == 1
        assert len(mapping1) == 1

    def test_feature_consistency(self):
        """Test that same graph produces same features."""
        graph = nx.DiGraph()
        graph.add_node("res1", type="Microsoft.Storage/storageAccounts", properties={"tier": "Standard"})

        extractor = GraphFeatureExtractor(max_features=10)
        extractor.fit([graph])

        X1, _ = extractor.transform(graph)
        X2, _ = extractor.transform(graph)

        np.testing.assert_array_equal(X1, X2)

    def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = nx.DiGraph()

        extractor = GraphFeatureExtractor(max_features=10)
        extractor.fit([nx.DiGraph()])  # Fit on dummy graph with data

        # Add a node for fitting
        dummy = nx.DiGraph()
        dummy.add_node("dummy", type="Test", properties={})
        extractor.fit([dummy])

        X, mapping = extractor.transform(graph)

        assert X.shape[0] == 0
        assert len(mapping) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestGraphDataLoader:
    """Test graph dataset loader."""

    def test_graph_to_pyg_data(self):
        """Test converting NetworkX graph to PyTorch Geometric Data."""
        from crux.ml.graph_loader import GraphDatasetLoader
        from sklearn.preprocessing import MultiLabelBinarizer

        # Create graph
        graph = nx.DiGraph()
        graph.add_node("res1", type="Microsoft.Storage/storageAccounts", properties={"tier": "Standard"})
        graph.add_node("res2", type="Microsoft.Network/virtualNetworks", properties={})
        graph.add_edge("res1", "res2", relationship="dependsOn")

        # Labels
        labels_dict = {
            "res1": ["Storage_PublicAccess"],
            "res2": [],
        }

        # Feature extractor
        extractor = GraphFeatureExtractor(max_features=10)
        extractor.fit([graph])

        # MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit([["Storage_PublicAccess"]])

        # Convert
        loader = GraphDatasetLoader.__new__(GraphDatasetLoader)  # Create without __init__
        data = loader._graph_to_pyg_data(graph, labels_dict, extractor, mlb)

        assert data is not None
        assert data.x.shape[0] == 2  # 2 nodes
        assert data.edge_index.shape == (2, 1)  # 1 edge
        assert data.y.shape[0] == 2  # 2 nodes
        assert len(data.node_ids) == 2

    def test_split_graphs(self):
        """Test splitting graphs into train/val/test."""
        from crux.ml.graph_loader import GraphDatasetLoader

        # Create dummy data
        data_list = []
        for i in range(10):
            x = torch.randn(5, 10)  # 5 nodes, 10 features
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
            y = torch.zeros(5, 3)  # 5 nodes, 3 labels
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # Split
        loader = GraphDatasetLoader.__new__(GraphDatasetLoader)
        train, val, test = loader.split_graphs(data_list, test_size=0.2, val_size=0.1, random_state=42)

        assert len(train) + len(val) + len(test) == 10
        assert len(test) == 2  # 20% of 10
        assert len(val) == 1   # ~10% of 8 (remaining after test)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestGNNModels:
    """Test GNN model architectures."""

    def test_gcn_init(self):
        """Test GCN model initialization."""
        from crux.ml.graph_models import GCNModel

        model = GCNModel(
            in_channels=10,
            hidden_channels=32,
            out_channels=5,
            num_layers=2,
            dropout=0.5,
        )

        assert model.in_channels == 10
        assert model.hidden_channels == 32
        assert model.out_channels == 5
        assert model.num_layers == 2
        assert model.dropout == 0.5
        assert len(model.convs) == 2

    def test_gcn_forward(self):
        """Test GCN forward pass."""
        from crux.ml.graph_models import GCNModel

        model = GCNModel(
            in_channels=10,
            hidden_channels=16,
            out_channels=3,
            num_layers=2,
        )
        model.eval()

        # Create dummy input
        x = torch.randn(5, 10)  # 5 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        # Forward pass
        out = model(x, edge_index)

        assert out.shape == (5, 3)  # 5 nodes, 3 output labels

    def test_gat_init(self):
        """Test GAT model initialization."""
        from crux.ml.graph_models import GATModel

        model = GATModel(
            in_channels=10,
            hidden_channels=16,
            out_channels=5,
            num_layers=2,
            heads=4,
            dropout=0.5,
        )

        assert model.heads == 4
        assert len(model.convs) == 2

    def test_gat_forward(self):
        """Test GAT forward pass."""
        from crux.ml.graph_models import GATModel

        model = GATModel(
            in_channels=10,
            hidden_channels=8,
            out_channels=3,
            num_layers=2,
            heads=2,
        )
        model.eval()

        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        out = model(x, edge_index)

        assert out.shape == (5, 3)

    def test_graphsage_init(self):
        """Test GraphSAGE model initialization."""
        from crux.ml.graph_models import GraphSAGEModel

        model = GraphSAGEModel(
            in_channels=10,
            hidden_channels=16,
            out_channels=5,
            num_layers=2,
            aggr="mean",
        )

        assert model.aggr == "mean"
        assert len(model.convs) == 2

    def test_graphsage_forward(self):
        """Test GraphSAGE forward pass."""
        from crux.ml.graph_models import GraphSAGEModel

        model = GraphSAGEModel(
            in_channels=10,
            hidden_channels=16,
            out_channels=3,
            num_layers=2,
        )
        model.eval()

        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        out = model(x, edge_index)

        assert out.shape == (5, 3)

    def test_model_save_load(self, tmp_path):
        """Test model save and load."""
        from crux.ml.graph_models import GCNModel

        # Create and save model
        model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        model.feature_names = ["feat1", "feat2"]
        model.label_names = ["label1", "label2", "label3"]

        model_path = tmp_path / "model.pt"
        model.save(str(model_path))

        # Load model
        loaded_model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        loaded_model.load(str(model_path))

        assert loaded_model.feature_names == ["feat1", "feat2"]
        assert loaded_model.label_names == ["label1", "label2", "label3"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
class TestGNNTrainer:
    """Test GNN trainer."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        from crux.ml.graph_models import GCNModel
        from crux.ml.graph_trainer import GNNTrainer

        model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        trainer = GNNTrainer(model, learning_rate=0.01)

        assert trainer.learning_rate == 0.01
        assert trainer.model == model
        assert trainer.device is not None

    def test_train_single_epoch(self):
        """Test training for one epoch."""
        from crux.ml.graph_models import GCNModel
        from crux.ml.graph_trainer import GNNTrainer

        # Create model
        model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        model.label_names = ["l1", "l2", "l3"]

        # Create dummy training data
        data_list = []
        for _ in range(5):
            x = torch.randn(3, 10)  # 3 nodes, 10 features
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
            y = torch.randint(0, 2, (3, 3)).float()  # 3 nodes, 3 labels (binary)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        trainer = GNNTrainer(model)
        loss = trainer.train_epoch(data_list)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate(self):
        """Test model evaluation."""
        from crux.ml.graph_models import GCNModel
        from crux.ml.graph_trainer import GNNTrainer

        # Create model
        model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        model.label_names = ["l1", "l2", "l3"]

        # Create dummy data
        data_list = []
        for _ in range(3):
            x = torch.randn(3, 10)
            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
            y = torch.randint(0, 2, (3, 3)).float()
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        trainer = GNNTrainer(model)
        loss, metrics = trainer.evaluate(data_list)

        assert isinstance(loss, float)
        assert loss >= 0
        assert "f1_macro" in metrics
        assert "precision_micro" in metrics

    def test_predict(self):
        """Test model prediction."""
        from crux.ml.graph_models import GCNModel
        from crux.ml.graph_trainer import GNNTrainer

        model = GCNModel(in_channels=10, hidden_channels=16, out_channels=3)
        model.label_names = ["l1", "l2", "l3"]

        # Create dummy data
        x = torch.randn(5, 10)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        y = torch.zeros(5, 3)
        data = Data(x=x, edge_index=edge_index, y=y)

        trainer = GNNTrainer(model)
        y_pred, y_proba = trainer.predict([data])

        assert y_pred.shape == (5, 3)
        assert y_proba.shape == (5, 3)
        assert np.all((y_pred == 0) | (y_pred == 1))  # Binary predictions
        assert np.all((y_proba >= 0) & (y_proba <= 1))  # Probabilities in [0, 1]
