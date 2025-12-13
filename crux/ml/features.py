"""
Feature Extraction

Converts JSON resource representations into numerical features for ML models.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

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


# Curated security-relevant properties for logistic regression
CURATED_PROPERTIES = [
    # Storage Account properties
    "properties.allowBlobPublicAccess",
    "properties.supportsHttpsTrafficOnly",
    "properties.minimumTlsVersion",
    "properties.allowSharedKeyAccess",
    "properties.publicNetworkAccess",
    "properties.encryption.services.blob.enabled",
    "properties.encryption.services.file.enabled",
    "properties.networkAcls.defaultAction",
    # Key Vault properties
    "properties.enablePurgeProtection",
    "properties.enableSoftDelete",
    "properties.enableRbacAuthorization",
    "properties.softDeleteRetentionInDays",
    # Network properties
    "properties.enableDdosProtection",
    "properties.securityRules",  # Will be converted to count
    # VM properties
    "properties.osProfile.linuxConfiguration.disablePasswordAuthentication",
    "properties.storageProfile.osDisk.encryptionSettings.enabled",
    "properties.diagnosticsProfile.bootDiagnostics.enabled",
    # App Service properties
    "properties.httpsOnly",
    "properties.siteConfig.minTlsVersion",
    "properties.siteConfig.http20Enabled",
    # General properties
    "location",
    "sku.name",
    "sku.tier",
    "kind",
]


class NamedPropertyExtractor:
    """
    Extracts named properties from Azure resources for interpretable ML models.

    Unlike FeatureExtractor which hashes property names, this class preserves
    the original property names for use in statistical analysis (e.g., logistic
    regression where coefficients need interpretation).

    Modes:
        - 'curated': Use predefined security-relevant properties (~25 columns)
        - 'all': Extract all properties found across resources (sparse)
    """

    def __init__(
        self,
        mode: str = "curated",
        property_list: Optional[List[str]] = None,
    ):
        """
        Initialize the named property extractor.

        Args:
            mode: Extraction mode ('curated' or 'all')
            property_list: Custom list of property paths (overrides mode)
        """
        self.mode = mode
        self.property_list = property_list or (
            CURATED_PROPERTIES if mode == "curated" else None
        )

    def extract(self, resources: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract named properties from resources into a DataFrame.

        Args:
            resources: List of resource dictionaries

        Returns:
            DataFrame with one column per property, rows aligned with resources
        """
        if self.mode == "all" and self.property_list is None:
            # Discover all properties across all resources
            self.property_list = self._discover_all_properties(resources)
            logger.info(f"Discovered {len(self.property_list)} properties across all resources")

        # Extract property values for each resource
        rows = []
        for resource in resources:
            row = {}
            for prop_path in self.property_list:
                value = self._get_nested_value(resource, prop_path)
                # Convert to appropriate type for DataFrame
                row[prop_path] = self._convert_value(value)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Clean column names for easier use in statistical software
        df.columns = [self._clean_column_name(c) for c in df.columns]

        return df

    def _get_nested_value(self, obj: Dict[str, Any], path: str) -> Any:
        """
        Get a value from a nested dictionary using dot notation.

        Args:
            obj: Dictionary to search
            path: Dot-separated path (e.g., 'properties.allowBlobPublicAccess')

        Returns:
            Value at path or None if not found
        """
        parts = path.split(".")
        current = obj

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _convert_value(self, value: Any) -> Any:
        """
        Convert a value to a type suitable for DataFrame/statistical analysis.

        Args:
            value: Raw value from resource

        Returns:
            Converted value (float, int, or NaN)
        """
        if value is None:
            return np.nan
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Convert common string values to numeric
            lower_val = value.lower()
            if lower_val in ("true", "enabled", "allow", "yes"):
                return 1.0
            elif lower_val in ("false", "disabled", "deny", "no"):
                return 0.0
            elif lower_val.startswith("tls"):
                # Convert TLS versions to numeric (TLS1_0=1.0, TLS1_1=1.1, TLS1_2=1.2)
                try:
                    version = lower_val.replace("tls", "").replace("_", ".")
                    return float(version)
                except ValueError:
                    return np.nan
            else:
                # For other strings, return a hash (or NaN if not useful)
                # Using hash % 1000 to keep values reasonable
                return hash(value) % 1000
        elif isinstance(value, list):
            # Return list length
            return float(len(value))
        elif isinstance(value, dict):
            # Return number of keys
            return float(len(value))
        else:
            return np.nan

    def _clean_column_name(self, name: str) -> str:
        """
        Clean a property path to a valid column name for statistical software.

        Args:
            name: Original property path

        Returns:
            Cleaned column name
        """
        # Remove 'properties.' prefix for brevity
        if name.startswith("properties."):
            name = name[len("properties."):]

        # Replace dots with underscores
        name = name.replace(".", "_")

        # Ensure starts with letter (for R compatibility)
        if name[0].isdigit():
            name = "prop_" + name

        return name

    def _discover_all_properties(
        self,
        resources: List[Dict[str, Any]],
        max_depth: int = 5,
    ) -> List[str]:
        """
        Discover all property paths across all resources.

        Args:
            resources: List of resource dictionaries
            max_depth: Maximum recursion depth

        Returns:
            Sorted list of unique property paths
        """
        all_paths = set()

        for resource in resources:
            paths = self._get_all_paths(resource, max_depth=max_depth)
            all_paths.update(paths)

        return sorted(all_paths)

    def _get_all_paths(
        self,
        obj: Any,
        prefix: str = "",
        max_depth: int = 5,
    ) -> Set[str]:
        """
        Recursively get all paths in a nested dictionary.

        Args:
            obj: Object to explore
            prefix: Current path prefix
            max_depth: Maximum recursion depth

        Returns:
            Set of all paths found
        """
        if max_depth == 0:
            return set()

        paths = set()

        if isinstance(obj, dict):
            for key, value in obj.items():
                # Skip internal metadata
                if key.startswith("_"):
                    continue

                path = f"{prefix}.{key}" if prefix else key

                # Add this path
                paths.add(path)

                # Recurse into nested dicts
                if isinstance(value, dict):
                    paths.update(self._get_all_paths(value, path, max_depth - 1))

        return paths

    def get_property_list(self) -> List[str]:
        """Get the list of properties being extracted."""
        return self.property_list.copy() if self.property_list else []
