"""
Graph Feature Extractor

Extracts node features from NetworkX graphs for GNN models.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import hashlib
import logging

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extract node features from NetworkX dependency graphs.

    Converts resource properties into numerical feature vectors
    for use with Graph Neural Networks.
    """

    def __init__(self, max_features: int = 50):
        """
        Initialize the graph feature extractor.

        Args:
            max_features: Maximum number of features per node
        """
        self.max_features = max_features
        self.type_encoder = LabelEncoder()
        self.feature_names = []
        self._fitted = False

    def fit(self, graphs: List[nx.DiGraph]) -> "GraphFeatureExtractor":
        """
        Learn feature schema from training graphs.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            Self for chaining
        """
        # Collect all resource types
        all_types = set()
        all_properties = set()

        for graph in graphs:
            for node, attrs in graph.nodes(data=True):
                # Collect resource types
                if "type" in attrs:
                    all_types.add(attrs["type"])

                # Collect property keys from properties dict
                if "properties" in attrs and isinstance(attrs["properties"], dict):
                    all_properties.update(self._get_property_paths(attrs["properties"]))

        # Fit type encoder
        self.type_encoder.fit(list(all_types))

        # Select top features by frequency
        self.feature_names = sorted(list(all_properties))[:self.max_features - 1]
        # -1 because we add resource type as first feature

        self._fitted = True
        logger.info(f"Fitted GraphFeatureExtractor with {len(self.type_encoder.classes_)} "
                   f"resource types and {len(self.feature_names)} property features")

        return self

    def transform(self, graph: nx.DiGraph) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Transform a graph to node feature matrix.

        Args:
            graph: NetworkX graph

        Returns:
            Tuple of (feature_matrix, node_to_idx_mapping)
            - feature_matrix: (num_nodes, num_features) array
            - node_to_idx_mapping: Maps node IDs to matrix row indices
        """
        if not self._fitted:
            raise ValueError("GraphFeatureExtractor must be fitted before transform")

        nodes = list(graph.nodes())
        num_nodes = len(nodes)
        num_features = len(self.feature_names) + 1  # +1 for resource type

        # Initialize feature matrix
        X = np.zeros((num_nodes, num_features), dtype=np.float32)

        # Create node ID to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Extract features for each node
        for idx, node in enumerate(nodes):
            attrs = graph.nodes[node]
            features = self._extract_node_features(attrs)
            X[idx] = features

        return X, node_to_idx

    def fit_transform(self, graphs: List[nx.DiGraph]) -> List[Tuple[np.ndarray, Dict[str, int]]]:
        """
        Fit and transform in one step.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            List of (feature_matrix, node_to_idx_mapping) tuples
        """
        self.fit(graphs)
        return [self.transform(graph) for graph in graphs]

    def _extract_node_features(self, attrs: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from node attributes."""
        num_features = len(self.feature_names) + 1
        # Initialize with -1.0 to distinguish missing properties from False booleans
        features = np.full(num_features, -1.0, dtype=np.float32)

        # Feature 0: Resource type (encoded)
        if "type" in attrs:
            try:
                features[0] = self.type_encoder.transform([attrs["type"]])[0]
            except ValueError:
                # Unknown type, use -1
                features[0] = -1

        # Features 1+: Properties
        if "properties" in attrs and isinstance(attrs["properties"], dict):
            prop_dict = self._flatten_properties(attrs["properties"])

            for i, feature_name in enumerate(self.feature_names):
                if feature_name in prop_dict:
                    value = prop_dict[feature_name]
                    features[i + 1] = self._convert_to_float(value)

        return features

    def _get_property_paths(self, obj: Any, prefix: str = "") -> List[str]:
        """Recursively get all property paths from nested dict."""
        paths = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                paths.append(path)

                # Recurse for nested dicts
                if isinstance(value, dict):
                    paths.extend(self._get_property_paths(value, path))

        return paths

    def _flatten_properties(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested properties to dot notation."""
        result = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                result[path] = value

                # Recurse for nested dicts
                if isinstance(value, dict):
                    result.update(self._flatten_properties(value, path))

        return result

    def _convert_to_float(self, value: Any) -> float:
        """Convert a value to float for ML."""
        if value is None:
            return 0.0
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Hash strings to [0, 1] range
            return self._hash_string(value)
        elif isinstance(value, (list, dict)):
            # Use length for collections
            return float(len(value))
        else:
            return 0.0

    def _hash_string(self, s: str) -> float:
        """Hash a string to a float in [0, 1] range."""
        hash_int = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return (hash_int % 10000) / 10000.0

    def get_num_features(self) -> int:
        """Get the number of features."""
        if not self._fitted:
            raise ValueError("GraphFeatureExtractor must be fitted first")
        return len(self.feature_names) + 1
