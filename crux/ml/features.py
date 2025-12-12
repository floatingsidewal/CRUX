"""
Feature Extraction

Converts JSON resource representations into numerical features for ML models.
"""

import hashlib
import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts numerical features from Azure resource JSON representations.

    Handles:
    - Resource type encoding
    - Property extraction
    - Boolean flags
    - Nested property flattening
    - String hashing for categorical values
    """

    def __init__(self, max_features: int = 100):
        """
        Initialize the feature extractor.

        Args:
            max_features: Maximum number of features to extract per resource
        """
        self.max_features = max_features
        self.feature_names: List[str] = []
        self.resource_type_encoder = LabelEncoder()
        self.known_resource_types: Set[str] = set()

    def fit(self, resources: List[Dict[str, Any]]) -> "FeatureExtractor":
        """
        Fit the feature extractor on a set of resources.

        This learns the feature schema from the data.

        Args:
            resources: List of resource dictionaries

        Returns:
            self
        """
        logger.info(f"Fitting feature extractor on {len(resources)} resources")

        # Collect all resource types
        resource_types = [r.get("type", "unknown") for r in resources]
        self.known_resource_types = set(resource_types)
        self.resource_type_encoder.fit(resource_types)

        # Collect all property paths
        all_properties = set()
        for resource in resources:
            properties = self._extract_property_paths(resource.get("properties", {}))
            all_properties.update(properties.keys())

        # Create feature names (limit to max_features)
        self.feature_names = [
            "resource_type_encoded",
            "has_properties",
            "num_properties",
        ]

        # Add property features
        sorted_properties = sorted(all_properties)[:self.max_features - 3]
        for prop in sorted_properties:
            self.feature_names.append(f"prop_{prop}")

        logger.info(f"Extracted {len(self.feature_names)} features")

        return self

    def transform(self, resources: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Transform resources into feature vectors.

        Args:
            resources: List of resource dictionaries

        Returns:
            Tuple of (feature_matrix, resource_ids)
        """
        if not self.feature_names:
            raise ValueError("FeatureExtractor must be fit before transform")

        features = []
        resource_ids = []

        for resource in resources:
            feature_vector = self._extract_features(resource)
            features.append(feature_vector)

            # Generate resource ID (use id or name or hash)
            resource_id = resource.get("id", resource.get("name", self._hash_resource(resource)))
            resource_ids.append(resource_id)

        return np.array(features), resource_ids

    def fit_transform(self, resources: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and transform in one step.

        Args:
            resources: List of resource dictionaries

        Returns:
            Tuple of (feature_matrix, resource_ids)
        """
        self.fit(resources)
        return self.transform(resources)

    def _extract_features(self, resource: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from a single resource."""
        # Initialize with -1.0 to distinguish missing properties from False booleans
        features = np.full(len(self.feature_names), -1.0)

        # Resource type encoding
        resource_type = resource.get("type", "unknown")
        if resource_type in self.known_resource_types:
            features[0] = self.resource_type_encoder.transform([resource_type])[0]
        else:
            # Use hash for unknown types
            features[0] = hash(resource_type) % 1000

        # Has properties (0 or 1, not -1)
        properties = resource.get("properties", {})
        features[1] = 1.0 if properties else 0.0

        # Number of properties (count, starts at 0)
        features[2] = float(len(properties)) if properties else 0.0

        # Extract property values
        property_paths = self._extract_property_paths(properties)

        # Fill in property features
        for i, feature_name in enumerate(self.feature_names[3:], start=3):
            prop_name = feature_name.replace("prop_", "")
            if prop_name in property_paths:
                features[i] = property_paths[prop_name]

        return features

    def _extract_property_paths(
        self, obj: Any, prefix: str = "", max_depth: int = 5
    ) -> Dict[str, float]:
        """
        Recursively extract property paths and convert to numerical values.

        Args:
            obj: Object to extract properties from
            prefix: Current path prefix
            max_depth: Maximum recursion depth

        Returns:
            Dictionary mapping property paths to numerical values
        """
        if max_depth == 0:
            return {}

        result = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key

                # Convert value to number
                if isinstance(value, bool):
                    result[path] = 1.0 if value else 0.0
                elif isinstance(value, (int, float)):
                    result[path] = float(value)
                elif isinstance(value, str):
                    # Hash string to number
                    result[path] = self._hash_string(value)
                elif isinstance(value, dict):
                    # Recurse into nested dict
                    nested = self._extract_property_paths(value, path, max_depth - 1)
                    result.update(nested)
                elif isinstance(value, list):
                    # Store list length
                    result[f"{path}._length"] = float(len(value))
                    # If list of primitives, hash them
                    if value and isinstance(value[0], (str, int, float, bool)):
                        result[path] = self._hash_string(str(value))

        return result

    def _hash_string(self, s: str) -> float:
        """Hash a string to a float in [0, 1]."""
        # Use first 8 bytes of MD5 hash
        hash_bytes = hashlib.md5(s.encode()).digest()[:8]
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        # Normalize to [0, 1]
        return hash_int / (2**64 - 1)

    def _hash_resource(self, resource: Dict[str, Any]) -> str:
        """Generate a unique ID for a resource by hashing its content."""
        content = str(sorted(resource.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get_feature_names(self) -> List[str]:
        """Get the names of all features."""
        return self.feature_names.copy()
