"""
Dataset Loader

Loads and prepares CRUX datasets for ML training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads CRUX datasets and prepares them for ML training.

    Handles:
    - Loading baseline and mutated resources
    - Loading labels
    - Multi-label target encoding
    - Train/validation/test splits
    """

    def __init__(self, dataset_dir: str):
        """
        Initialize the dataset loader.

        Args:
            dataset_dir: Path to the dataset directory (e.g., dataset/exp-20240101-120000)
        """
        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")

        # Paths to dataset files
        self.baseline_file = self.dataset_dir / "baseline" / "resources.json"
        self.mutated_file = self.dataset_dir / "mutated" / "resources.json"
        self.labels_file = self.dataset_dir / "labels.json"
        self.metadata_file = self.dataset_dir / "metadata.json"

        # Check files exist
        for file in [self.mutated_file, self.labels_file]:
            if not file.exists():
                raise ValueError(f"Required file not found: {file}")

        self.label_encoder = MultiLabelBinarizer()
        self.all_labels: List[str] = []

        logger.info(f"Initialized DatasetLoader for {dataset_dir}")

    def load_resources(
        self,
        include_baseline: bool = False,
        include_mutated: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Load resources from the dataset.

        Args:
            include_baseline: Whether to include baseline (unmutated) resources
            include_mutated: Whether to include mutated resources

        Returns:
            List of resource dictionaries
        """
        resources = []

        if include_baseline and self.baseline_file.exists():
            with open(self.baseline_file) as f:
                baseline = json.load(f)
                logger.info(f"Loaded {len(baseline)} baseline resources")
                resources.extend(baseline)

        if include_mutated:
            with open(self.mutated_file) as f:
                mutated = json.load(f)
                logger.info(f"Loaded {len(mutated)} mutated resources")
                resources.extend(mutated)

        return resources

    def load_labels(self) -> Dict[str, List[str]]:
        """
        Load labels from the dataset.

        Returns:
            Dictionary mapping resource IDs to label lists
        """
        with open(self.labels_file) as f:
            labels = json.load(f)
            logger.info(f"Loaded labels for {len(labels)} resources")
            return labels

    def load_metadata(self) -> Dict[str, Any]:
        """
        Load dataset metadata.

        Returns:
            Metadata dictionary
        """
        with open(self.metadata_file) as f:
            return json.load(f)

    def prepare_training_data(
        self,
        resources: List[Dict[str, Any]],
        labels_dict: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
        """
        Prepare training data by aligning resources with their labels.

        Args:
            resources: List of resource dictionaries
            labels_dict: Dictionary mapping resource IDs to labels

        Returns:
            Tuple of (filtered_resources, label_matrix, label_names)
        """
        # Filter resources to only those with labels
        labeled_resources = []
        label_lists = []

        for resource in resources:
            # Try different ID formats
            resource_id = resource.get("id", resource.get("name", "unknown"))

            # For mutated resources, also try the mutation_key format
            mutation_id = resource.get("_metadata", {}).get("mutation_id")
            if mutation_id:
                mutation_key = f"{resource_id}#{mutation_id}"
                if mutation_key in labels_dict:
                    resource_id = mutation_key

            if resource_id in labels_dict:
                labeled_resources.append(resource)
                label_lists.append(labels_dict[resource_id])

        logger.info(f"Found {len(labeled_resources)} resources with labels")

        # Fit label encoder and transform
        self.label_encoder.fit(label_lists)
        label_matrix = self.label_encoder.transform(label_lists)
        self.all_labels = list(self.label_encoder.classes_)

        logger.info(f"Encoded {len(self.all_labels)} unique labels")
        logger.info(f"Label distribution: {np.sum(label_matrix, axis=0).tolist()}")

        return labeled_resources, label_matrix, self.all_labels

    def split_data(
        self,
        resources: List[Dict[str, Any]],
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Split data into train, validation, and test sets.

        Args:
            resources: List of resource dictionaries
            labels: Label matrix
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_resources, val_resources, test_resources,
                     train_labels, val_labels, test_labels)
        """
        # First split: train+val vs test
        resources_train_val, resources_test, labels_train_val, labels_test = train_test_split(
            resources,
            labels,
            test_size=test_size,
            random_state=random_state,
        )

        # Second split: train vs val
        resources_train, resources_val, labels_train, labels_val = train_test_split(
            resources_train_val,
            labels_train_val,
            test_size=val_size / (1 - test_size),  # Adjust for already-split data
            random_state=random_state,
        )

        logger.info(f"Split: train={len(resources_train)}, "
                   f"val={len(resources_val)}, "
                   f"test={len(resources_test)}")

        return (
            resources_train,
            resources_val,
            resources_test,
            labels_train,
            labels_val,
            labels_test,
        )

    def get_label_statistics(self, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics about label distribution.

        Args:
            labels: Label matrix

        Returns:
            Dictionary of statistics
        """
        label_counts = np.sum(labels, axis=0)

        stats = {
            "num_samples": labels.shape[0],
            "num_labels": labels.shape[1],
            "total_label_occurrences": int(np.sum(labels)),
            "avg_labels_per_sample": float(np.mean(np.sum(labels, axis=1))),
            "label_distribution": {
                label: int(count)
                for label, count in zip(self.all_labels, label_counts)
            },
        }

        return stats

    def get_binary_labels(
        self,
        labels: np.ndarray,
        mode: str = "any",
    ) -> np.ndarray:
        """
        Convert multi-label matrix to binary classification labels.

        Args:
            labels: Multi-label binary matrix of shape (n_samples, n_labels)
            mode: Binary label mode:
                - 'any': 1 if resource has ANY labels, 0 otherwise (default)
                - 'specific:<label>': 1 if resource has specific label (future)
                - 'severity:high': 1 if resource has high-severity labels (future)

        Returns:
            Binary array of shape (n_samples,) with values 0 or 1
        """
        if mode == "any":
            # 1 if resource has at least one label, 0 otherwise
            binary_labels = (np.sum(labels, axis=1) > 0).astype(int)
            logger.info(f"Binary labels (mode='{mode}'): "
                       f"{np.sum(binary_labels)} positive, "
                       f"{len(binary_labels) - np.sum(binary_labels)} negative")
            return binary_labels
        else:
            raise ValueError(f"Unsupported binary mode: {mode}. Currently only 'any' is supported.")

    def load_and_prepare(
        self,
        include_baseline: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str],
    ]:
        """
        Convenience method to load and prepare data in one step.

        Args:
            include_baseline: Whether to include baseline resources
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed

        Returns:
            Tuple of (train_resources, val_resources, test_resources,
                     train_labels, val_labels, test_labels, label_names)
        """
        # Load data
        resources = self.load_resources(
            include_baseline=include_baseline,
            include_mutated=True,
        )
        labels_dict = self.load_labels()

        # Prepare training data
        labeled_resources, label_matrix, label_names = self.prepare_training_data(
            resources, labels_dict
        )

        # Split data
        (
            train_res,
            val_res,
            test_res,
            train_labels,
            val_labels,
            test_labels,
        ) = self.split_data(
            labeled_resources,
            label_matrix,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

        return (
            train_res,
            val_res,
            test_res,
            train_labels,
            val_labels,
            test_labels,
            label_names,
        )

    def export_to_csv(
        self,
        output_path: str,
        include_features: bool = True,
        max_features: int = 100,
        binary_mode: str = "any",
        include_baseline: bool = False,
        named_properties: str = "none",
        property_list: Optional[List[str]] = None,
    ) -> None:
        """
        Export dataset to CSV format for use in external tools.

        Args:
            output_path: Path to output CSV file
            include_features: Whether to include numerical feature columns
            max_features: Maximum number of features to extract
            binary_mode: How to create binary labels ('any', etc.)
            include_baseline: Whether to include baseline (unmutated) resources
            named_properties: Named property extraction mode:
                - 'none': Use hashed features (default)
                - 'all': Extract all ARM properties as columns (sparse)
                - 'curated': Use predefined security-relevant properties
            property_list: Custom list of property paths to extract (overrides named_properties)
        """
        import pandas as pd
        from .features import FeatureExtractor, NamedPropertyExtractor

        # Load mutated resources and labels
        mutated_resources = self.load_resources(
            include_baseline=False,
            include_mutated=True,
        )
        labels_dict = self.load_labels()

        # Prepare mutated training data
        labeled_mutated, label_matrix, label_names = self.prepare_training_data(
            mutated_resources, labels_dict
        )

        # Get binary labels for mutated resources
        binary_labels = self.get_binary_labels(label_matrix, mode=binary_mode)

        # Build rows for mutated resources
        rows = []
        all_resources = []  # For feature extraction later

        for resource, binary_label, multi_labels in zip(
            labeled_mutated,
            binary_labels,
            label_matrix,
        ):
            # Extract metadata
            resource_id = resource.get("id", resource.get("name", "unknown"))
            mutation_id = resource.get("_metadata", {}).get("mutation_id", "")
            if mutation_id:
                resource_id = f"{resource_id}#{mutation_id}"

            row = {
                "resource_id": resource_id,
                "resource_type": resource.get("type", "unknown"),
                "is_mutated": 1,
                "mutation_id": mutation_id,
                "mutation_severity": resource.get("_metadata", {}).get("mutation_severity", ""),
                "has_misconfiguration": int(binary_label),
                "label_count": int(np.sum(multi_labels)),
                "labels": ",".join([
                    label_names[i]
                    for i, val in enumerate(multi_labels)
                    if val == 1
                ]),
                "source_template": resource.get("_metadata", {}).get("source_template", ""),
            }
            rows.append(row)
            all_resources.append(resource)

        # Add baseline resources as negative class
        if include_baseline and self.baseline_file.exists():
            with open(self.baseline_file) as f:
                baseline_resources = json.load(f)

            logger.info(f"Adding {len(baseline_resources)} baseline resources as negative class")

            for resource in baseline_resources:
                resource_id = resource.get("id", resource.get("name", "unknown"))

                row = {
                    "resource_id": resource_id,
                    "resource_type": resource.get("type", "unknown"),
                    "is_mutated": 0,
                    "mutation_id": "",
                    "mutation_severity": "",
                    "has_misconfiguration": 0,  # Baseline = no misconfiguration
                    "label_count": 0,
                    "labels": "",
                    "source_template": resource.get("_metadata", {}).get("source_template", ""),
                }
                rows.append(row)
                all_resources.append(resource)

        df = pd.DataFrame(rows)

        # Add feature columns based on extraction mode
        if include_features:
            if named_properties == "none" and property_list is None:
                # Use hashed features (original behavior)
                logger.info(f"Extracting hashed features (max_features={max_features})...")
                extractor = FeatureExtractor(max_features=max_features)
                feature_matrix, _ = extractor.fit_transform(all_resources)

                # Add feature columns
                for i in range(feature_matrix.shape[1]):
                    if i < len(extractor.feature_names):
                        col_name = f"feature_{extractor.feature_names[i]}"
                    else:
                        col_name = f"feature_{i}"
                    df[col_name] = feature_matrix[:, i]

                logger.info(f"Added {feature_matrix.shape[1]} hashed feature columns")
            else:
                # Use named property extraction
                logger.info(f"Extracting named properties (mode={named_properties})...")
                named_extractor = NamedPropertyExtractor(
                    mode=named_properties,
                    property_list=property_list,
                )
                property_df = named_extractor.extract(all_resources)

                # Merge property columns into main dataframe
                for col in property_df.columns:
                    df[col] = property_df[col].values

                logger.info(f"Added {len(property_df.columns)} named property columns")

        # Write to CSV
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        # Calculate class distribution
        positive_count = df['has_misconfiguration'].sum()
        negative_count = len(df) - positive_count

        logger.info(f"Exported {len(df)} rows to {output_path}")
        logger.info(f"Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
        logger.info(f"Class distribution: positive={positive_count}, negative={negative_count}")
        if negative_count > 0:
            logger.info(f"Class balance: {positive_count/len(df)*100:.1f}% positive, {negative_count/len(df)*100:.1f}% negative")
