"""
Graph Dataset Loader

Loads dependency graphs from CRUX datasets and prepares them for GNN training.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import json
import logging
from pathlib import Path
import networkx as nx
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

if TYPE_CHECKING:
    from torch_geometric.data import Data

try:
    import torch
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    PyGData = None

from .graph_features import GraphFeatureExtractor

logger = logging.getLogger(__name__)


class GraphDatasetLoader:
    """
    Load and prepare CRUX dependency graphs for GNN training.

    Handles loading graphs from JSON format, extracting node features,
    and creating PyTorch Geometric Data objects.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize the graph dataset loader.

        Args:
            dataset_path: Path to CRUX dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.graphs_dir = self.dataset_path / "graphs"
        self.labels_path = self.dataset_path / "labels.json"

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")

    def load_graphs(self) -> List[nx.DiGraph]:
        """
        Load all dependency graphs from the dataset.

        Returns:
            List of NetworkX directed graphs
        """
        if not self.graphs_dir.exists():
            logger.warning(f"No graphs directory found at {self.graphs_dir}")
            return []

        graphs = []
        for graph_file in self.graphs_dir.glob("*_graph.json"):
            try:
                graph = self._load_graph_from_json(graph_file)
                if graph.number_of_nodes() > 0:
                    graphs.append(graph)
            except Exception as e:
                logger.error(f"Error loading graph {graph_file}: {e}")

        logger.info(f"Loaded {len(graphs)} graphs from {self.graphs_dir}")
        return graphs

    def _load_graph_from_json(self, json_path: Path) -> nx.DiGraph:
        """Load a graph from node-link JSON format."""
        with open(json_path) as f:
            data = json.load(f)

        # Convert from node-link format
        graph = nx.node_link_graph(data, directed=True)
        return graph

    def load_labels(self) -> Dict[str, List[str]]:
        """
        Load labels for all resources.

        Returns:
            Dictionary mapping resource IDs to lists of label strings
        """
        if not self.labels_path.exists():
            logger.warning(f"No labels file found at {self.labels_path}")
            return {}

        with open(self.labels_path) as f:
            labels = json.load(f)

        return labels

    def prepare_graph_data(
        self,
        graphs: List[nx.DiGraph],
        labels_dict: Dict[str, List[str]],
        feature_extractor: Optional[GraphFeatureExtractor] = None,
    ) -> Tuple[List[Data], GraphFeatureExtractor, List[str]]:
        """
        Prepare graphs for GNN training.

        Args:
            graphs: List of NetworkX graphs
            labels_dict: Dictionary mapping resource IDs to label lists
            feature_extractor: Optional pre-fitted feature extractor

        Returns:
            Tuple of (pyg_data_list, feature_extractor, label_names)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN functionality. "
                            "Install with: pip install torch torch-geometric")

        # Fit feature extractor if not provided
        if feature_extractor is None:
            feature_extractor = GraphFeatureExtractor(max_features=50)
            feature_extractor.fit(graphs)

        # Encode labels
        mlb = MultiLabelBinarizer()
        all_labels = list(labels_dict.values())
        if all_labels:
            mlb.fit(all_labels)
        else:
            logger.warning("No labels found in dataset")
            mlb.fit([[]])

        label_names = list(mlb.classes_)

        # Convert each graph to PyTorch Geometric Data object
        pyg_data_list = []
        for graph in graphs:
            data = self._graph_to_pyg_data(graph, labels_dict, feature_extractor, mlb)
            if data is not None:
                pyg_data_list.append(data)

        logger.info(f"Prepared {len(pyg_data_list)} PyTorch Geometric graphs with "
                   f"{feature_extractor.get_num_features()} node features and "
                   f"{len(label_names)} labels")

        return pyg_data_list, feature_extractor, label_names

    def _graph_to_pyg_data(
        self,
        graph: nx.DiGraph,
        labels_dict: Dict[str, List[str]],
        feature_extractor: GraphFeatureExtractor,
        mlb: MultiLabelBinarizer,
    ) -> Optional[Data]:
        """
        Convert a NetworkX graph to PyTorch Geometric Data object.

        Args:
            graph: NetworkX directed graph
            labels_dict: Labels for resources
            feature_extractor: Fitted feature extractor
            mlb: Fitted MultiLabelBinarizer

        Returns:
            PyTorch Geometric Data object or None if graph is empty
        """
        if graph.number_of_nodes() == 0:
            return None

        # Extract node features
        X, node_to_idx = feature_extractor.transform(graph)

        # Convert to torch tensors
        x = torch.tensor(X, dtype=torch.float)

        # Build edge index (COO format: [2, num_edges])
        edge_list = []
        for source, target in graph.edges():
            if source in node_to_idx and target in node_to_idx:
                edge_list.append([node_to_idx[source], node_to_idx[target]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Graph with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Build label matrix
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        node_labels = []
        for idx in range(len(node_to_idx)):
            node_id = idx_to_node[idx]
            labels = labels_dict.get(node_id, [])
            node_labels.append(labels)

        y = mlb.transform(node_labels)
        y = torch.tensor(y, dtype=torch.float)

        # Create PyTorch Geometric Data object
        data = PyGData(x=x, edge_index=edge_index, y=y)

        # Store node IDs for reference
        data.node_ids = [idx_to_node[i] for i in range(len(node_to_idx))]

        return data

    def split_graphs(
        self,
        data_list: List[Data],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Split graphs into train/validation/test sets.

        Args:
            data_list: List of PyTorch Geometric Data objects
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
            random_state: Random seed

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        np.random.seed(random_state)

        # Shuffle indices
        indices = np.random.permutation(len(data_list))

        # Calculate split sizes
        n_test = int(len(data_list) * test_size)
        n_val = int((len(data_list) - n_test) * val_size)

        # Split indices
        test_indices = indices[:n_test]
        val_indices = indices[n_test:n_test + n_val]
        train_indices = indices[n_test + n_val:]

        # Split data
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]

        logger.info(f"Split graphs: {len(train_data)} train, {len(val_data)} val, "
                   f"{len(test_data)} test")

        return train_data, val_data, test_data

    def load_and_prepare(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        max_features: int = 50,
    ) -> Tuple[List[Data], List[Data], List[Data], GraphFeatureExtractor, List[str]]:
        """
        Convenience method to load and prepare graphs in one step.

        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
            random_state: Random seed
            max_features: Maximum number of node features

        Returns:
            Tuple of (train_data, val_data, test_data, feature_extractor, label_names)
        """
        # Load graphs and labels
        graphs = self.load_graphs()
        labels_dict = self.load_labels()

        if not graphs:
            raise ValueError("No graphs found in dataset")

        # Prepare data
        feature_extractor = GraphFeatureExtractor(max_features=max_features)
        data_list, feature_extractor, label_names = self.prepare_graph_data(
            graphs, labels_dict, feature_extractor
        )

        if not data_list:
            raise ValueError("No valid graph data prepared")

        # Split data
        train_data, val_data, test_data = self.split_graphs(
            data_list, test_size, val_size, random_state
        )

        return train_data, val_data, test_data, feature_extractor, label_names
