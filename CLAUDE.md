# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRUX (Cloud Resource Configuration Analyzer) generates labeled datasets for training ML models to detect Azure resource misconfigurations. It uses **static template analysis** (Option D): analyzing Bicep/ARM templates locally without deploying to Azure, enabling zero-cost, large-scale dataset generation.

## Documentation Organization

**IMPORTANT**: All documentation files MUST be stored in the `docs/` folder, except for:
- `README.md` - Project overview (stays in root)
- `CLAUDE.md` - Development guidelines (stays in root)

### docs/ Structure

```
docs/
├── csv-export-example.md   # CSV export usage guide
├── prd.md                  # Product requirements document
└── changelog.md            # Project changelog
```

### Guidelines

- **New documentation**: Always create new `.md` files in `docs/`
- **Research papers**: Store in `docs/` with descriptive names
- **Milestone reports**: Use format `MILESTONE-X.Y-TOPIC.md` in `docs/`
- **Examples/tutorials**: Store in `docs/` with `-example` or `-guide` suffix
- **Never create**: Empty folders (delete if no longer needed)

## Scripts Organization

The `scripts/` directory contains utility and operational scripts that are **not part of the core package**:

```
scripts/
└── generate_full_dataset.py   # Full dataset generation script
```

### What Goes in scripts/

- **One-time data generation scripts**: Large-scale dataset generation
- **Verification and debugging tools**: Testing specific fixes or features
- **Maintenance utilities**: Database migrations, cleanup tasks
- **Operational scripts**: Batch processing, data validation

### What Does NOT Go in scripts/

- **Core package code**: Use `crux/` for reusable components
- **Unit tests**: Use `tests/` for automated tests
- **User-facing commands**: Add to CLI in `crux/cli.py`
- **Examples for users**: Put in `docs/` as documentation

### Running Scripts

Scripts are meant to be run directly:
```bash
python3 scripts/verify_zero_f1_fix.py
python3 scripts/generate_full_dataset.py
```

Make scripts executable if needed:
```bash
chmod +x scripts/*.py
```

## Core Commands

### Development Setup
```bash
# Install CRUX in editable mode with dev dependencies
pip install -e .[dev]

# Install with ML libraries for model training
pip install -e .[ml]

# Run tests
pytest

# Format code
black crux/

# Type checking
mypy crux/

# Linting
ruff check crux/
```

### Using the CLI

```bash
# Fetch Azure Quickstart Templates (first time)
crux fetch-templates --limit 100

# Generate a dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name pilot-001 \
  --limit 50

# List available mutations and rules
crux list-mutations
crux list-rules
```

## Architecture (Option D: Static Analysis)

### Pipeline Overview

**Fetch → Compile → Extract → Mutate → Label → Export**

1. **Fetch** (`crux/templates/fetcher.py`): Download Azure Quickstart Templates from GitHub
2. **Compile** (`crux/templates/compiler.py`): Convert Bicep → ARM JSON using `az bicep build`
3. **Extract** (`crux/templates/extractor.py`): Parse ARM JSON to extract resource properties
4. **Graph** (`crux/templates/graph.py`): Build dependency graphs using NetworkX
5. **Mutate** (`crux/mutations/*.py`): Apply Python-defined mutations
6. **Label** (`crux/rules/evaluator.py`): Evaluate YAML-defined security rules
7. **Export** (`crux/dataset/generator.py`): Generate labeled datasets

### Directory Structure

```
crux/                          # Python package
├── templates/                 # Template operations
│   ├── fetcher.py             # Download from GitHub
│   ├── compiler.py            # Bicep → ARM JSON
│   ├── extractor.py           # ARM → resources
│   └── graph.py               # Dependency graphs
├── mutations/                 # Python mutations
│   ├── base.py                # Mutation framework
│   └── storage.py             # Storage mutations
├── rules/                     # Rule engine
│   └── evaluator.py           # YAML rule evaluation
├── dataset/                   # Dataset generation
│   └── generator.py           # End-to-end pipeline
└── cli.py                     # Command-line interface

rules/                         # YAML rule definitions (data)
├── storage.yaml
├── keyvault.yaml              # (To be added)
└── network.yaml               # (To be added)

templates/                     # Downloaded templates (gitignored)
dataset/                       # Generated datasets (gitignored)
tests/                         # Unit tests
```

### Key Design Patterns

#### 1. Mutations are Python Code (Type-Safe)

Mutations are defined as Python functions for flexibility and testability:

```python
# crux/mutations/storage.py
from .base import Mutation

def mutate_public_blob_access(resource):
    if "properties" not in resource:
        resource["properties"] = {}
    resource["properties"]["allowBlobPublicAccess"] = True
    return resource

STORAGE_PUBLIC_BLOB_ACCESS = Mutation(
    id="storage_public_blob_access",
    target_type="Microsoft.Storage/storageAccounts",
    description="Enable public blob access",
    severity="high",
    labels=["Storage_PublicAccess", "CIS_3.7"],
    cis_references=["3.7"],
    mutate=mutate_public_blob_access,
)
```

**Why Python?**
- Type-safe (IDE autocomplete, mypy)
- Testable (unit tests for each mutation)
- Flexible (can handle complex logic)
- Versionable (git tracks changes)

#### 2. Rules are YAML (Accessible to Security Teams)

Security rules are YAML for easy review by non-programmers:

```yaml
# rules/storage.yaml
rules:
  - id: storage-public-blob-access
    resource_type: Microsoft.Storage/storageAccounts
    severity: high
    cis_reference: "3.7"
    condition:
      property: properties.allowBlobPublicAccess
      equals: true
    labels:
      - Storage_PublicAccess
      - CIS_3.7
```

#### 3. Streaming Pipeline (Memory-Efficient)

The dataset generator uses Python generators for memory efficiency:

```python
# Processes 1000+ templates without loading all into memory
for template in template_paths:
    arm = compile(template)
    resources = extract(arm)
    for mutation in mutations:
        mutated = mutation.apply(resource)
        labels = evaluate_rules(mutated)
        yield (mutated, labels)
```

#### 4. Graph-Aware (NetworkX)

Dependency graphs are built using NetworkX for cross-resource analysis:
- Nodes: Resources (with properties as attributes)
- Edges: Dependencies (`dependsOn`, `reference()`)
- Export: GraphML (Gephi, Cytoscape), JSON (DGL, PyTorch Geometric)

### Dataset Structure

```
dataset/exp-YYYYMMDD-HHMMSS/
├── metadata.json              # Experiment info
├── baseline/
│   └── resources.json         # Original (unmutated) resources
├── mutated/
│   └── resources.json         # Mutated resources
├── graphs/                    # Dependency graphs
│   └── *.graphml
└── labels.json                # resource_id → [labels]
```

Each resource in `labels.json` maps to a list of misconfiguration labels:
```json
{
  "/subscriptions/.../Microsoft.Storage/storageAccounts/myaccount": [
    "Storage_PublicAccess",
    "CIS_3.7",
    "Storage_WeakTLS"
  ]
}
```

## ML Pipeline (Milestone 3)

### Overview

The ML pipeline trains machine learning models to detect misconfigurations from the generated datasets. It includes feature extraction, model training, evaluation, and prediction capabilities.

### Components

#### 1. Feature Extractor (`crux/ml/features.py`)

Converts JSON resources to numerical feature vectors:
- Encodes resource types
- Extracts and flattens nested properties
- Converts boolean/string/numeric values to floats
- Handles missing properties gracefully

```python
from crux.ml.features import FeatureExtractor

extractor = FeatureExtractor(max_features=100)
X, resource_ids = extractor.fit_transform(resources)
# X is (n_samples, n_features) numpy array
```

#### 2. Dataset Loader (`crux/ml/dataset.py`)

Loads and prepares datasets for training:
- Reads baseline and mutated resources
- Aligns resources with labels
- Multi-label encoding
- Train/validation/test splits

```python
from crux.ml.dataset import DatasetLoader

loader = DatasetLoader("dataset/exp-20240101-120000")
train_res, val_res, test_res, train_labels, val_labels, test_labels, label_names = loader.load_and_prepare()
```

#### 3. Baseline Models (`crux/ml/models.py`)

Three baseline models for multi-label classification:
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosted trees
- **Logistic Regression**: Linear model with regularization (L1, L2, ElasticNet)

```python
from crux.ml.models import RandomForestModel, XGBoostModel, LogisticRegressionModel

# Random Forest
model = RandomForestModel(n_estimators=100, random_state=42)
model.fit(X_train, y_train, feature_names=features, label_names=labels)

# XGBoost
model = XGBoostModel(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train, feature_names=features, label_names=labels)

# Logistic Regression
model = LogisticRegressionModel(C=1.0, penalty='l2', max_iter=1000, random_state=42)
model.fit(X_train, y_train, feature_names=features, label_names=labels)

# Predict
y_pred = model.predict(X_test)

# Save/Load
model.save("models/my_model.pkl")
model.load("models/my_model.pkl")
```

#### 4. Model Evaluator (`crux/ml/evaluation.py`)

Comprehensive evaluation metrics for multi-label classification:
- Macro/Micro/Weighted averaging
- Per-label performance
- Confusion matrices
- Error analysis

```python
from crux.ml.evaluation import ModelEvaluator

evaluator = ModelEvaluator(label_names=labels)
metrics = evaluator.evaluate(y_true, y_pred, y_proba)

# Print formatted report
evaluator.print_report(metrics)

# Save to JSON
evaluator.save_report(metrics, "results/evaluation.json")
```

#### 5. Binary Classification Support

CRUX supports binary classification (has ANY misconfiguration vs clean) in addition to multi-label classification:

**Binary Label Transformation:**
```python
from crux.ml.dataset import DatasetLoader

loader = DatasetLoader("dataset/exp-20240101-120000")
# Load multi-label data
resources, labels_multi, label_names = loader.prepare_training_data(resources, labels_dict)

# Convert to binary labels (0 = clean, 1 = has any misconfiguration)
binary_labels = loader.get_binary_labels(labels_multi, mode='any')
```

**Binary Evaluator:**
```python
from crux.ml.evaluation import BinaryEvaluator

evaluator = BinaryEvaluator()
metrics = evaluator.evaluate(y_true, y_pred, y_proba)

# Includes: accuracy, precision, recall, F1, ROC AUC, PR AUC, confusion matrix
evaluator.print_report(metrics)
evaluator.save_report(metrics, "results/binary_eval.json")
```

#### 6. CSV Export for External Tools

Export datasets to CSV format for use in Python, R, Excel, or other tools:

```python
from crux.ml.dataset import DatasetLoader

loader = DatasetLoader("dataset/exp-20240101-120000")
loader.export_to_csv(
    output_path="data/crux_data.csv",
    include_features=True,
    max_features=100,
    binary_mode="any",  # Binary target: has_misconfiguration (0/1)
    include_baseline=False,
)
```

**CSV Columns:**
- `resource_id`, `resource_type`, `is_mutated`, `mutation_id`, `mutation_severity`
- `has_misconfiguration` - Binary target (0/1)
- `label_count`, `labels` - Multi-label information
- `source_template` - Template path
- `feature_*` - Numerical feature columns (100 by default)

See `docs/csv-export-example.md` for detailed examples using the CSV in pandas, sklearn, R, Excel, and more.

### CLI Commands

#### Train a Model

```bash
# Train Random Forest on a dataset (multi-label)
crux train-model \
  --dataset dataset/exp-20240101-120000 \
  --model random-forest \
  --output models \
  --name my-rf-model

# Train XGBoost (multi-label)
crux train-model \
  --dataset dataset/exp-20240101-120000 \
  --model xgboost \
  --output models \
  --name my-xgb-model \
  --max-features 150

# Train Logistic Regression with binary classification
crux train-model \
  --dataset dataset/exp-20240101-120000 \
  --model logistic-regression \
  --binary-mode any \
  --output models \
  --name my-lr-binary
```

Output:
- `models/my-rf-model.pkl` - Trained model
- `models/my-rf-model_features.pkl` - Feature extractor
- `models/my-rf-model_metrics.json` - Evaluation metrics

#### Evaluate a Model

```bash
crux evaluate-model \
  --model models/my-rf-model.pkl \
  --dataset dataset/exp-20240101-120000 \
  --output results/evaluation.json
```

#### Export to CSV

```bash
# Export dataset to CSV with features and binary labels
crux export-csv \
  --dataset dataset/exp-20240101-120000 \
  --output data/crux_data.csv \
  --max-features 100 \
  --binary-mode any

# Export metadata only (no features)
crux export-csv \
  --dataset dataset/exp-20240101-120000 \
  --output data/crux_metadata.csv \
  --no-features

# Include baseline resources for balanced dataset
crux export-csv \
  --dataset dataset/exp-20240101-120000 \
  --output data/crux_full.csv \
  --include-baseline \
  --binary-mode any
```

Use the exported CSV in Python, R, Excel, or any other data analysis tool. See `docs/csv-export-example.md` for complete examples.

### Workflow Example

```bash
# 1. Generate a dataset (Milestone 2)
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name exp-001 \
  --limit 100

# 2. Train multi-label models (Milestone 3)
crux train-model \
  --dataset dataset/exp-001 \
  --model random-forest \
  --output models \
  --name rf-baseline

crux train-model \
  --dataset dataset/exp-001 \
  --model xgboost \
  --output models \
  --name xgb-baseline

# 3. Train binary classification models
crux train-model \
  --dataset dataset/exp-001 \
  --model logistic-regression \
  --binary-mode any \
  --output models \
  --name lr-binary

# 4. Export to CSV for external analysis
crux export-csv \
  --dataset dataset/exp-001 \
  --output data/exp-001.csv \
  --max-features 100 \
  --binary-mode any

# 5. Evaluate models
crux evaluate-model \
  --model models/rf-baseline.pkl \
  --dataset dataset/exp-001 \
  --output results/rf-eval.json
```

### Model Performance Metrics

The evaluation framework provides:

**Overall Metrics:**
- Exact Match Ratio: Percentage of samples with all labels correct
- Sample Accuracy: Accuracy at the sample level
- Hamming Loss: Fraction of incorrect labels

**Aggregated Metrics (Macro/Micro/Weighted):**
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1 Score: Harmonic mean of Precision and Recall

**Per-Label Metrics:**
- Individual Precision, Recall, F1 for each misconfiguration label
- Support: Number of true occurrences of each label

## GNN Pipeline (Milestone 4)

### Overview

The GNN (Graph Neural Network) pipeline leverages dependency graphs between Azure resources to detect misconfigurations. Unlike baseline models that treat resources independently, GNNs can learn from cross-resource patterns and dependencies.

### Key Advantages

- **Graph-Aware**: Considers relationships between resources (e.g., VM → VNet → NSG)
- **Message Passing**: Nodes share information with neighbors through graph convolutions
- **Context-Rich**: Learns from both node properties and graph structure
- **Multi-Resource Patterns**: Can detect issues that span multiple resources

### Components

#### 1. Graph Feature Extractor (`crux/ml/graph_features.py`)

Converts NetworkX dependency graphs to numerical node feature matrices:
- Encodes resource types as categorical features
- Extracts and flattens nested properties from resource configs
- Converts boolean/string/numeric values to floats
- Handles variable-length property sets gracefully

```python
from crux.ml.graph_features import GraphFeatureExtractor

# Create feature extractor
extractor = GraphFeatureExtractor(max_features=50)

# Fit on training graphs
extractor.fit(train_graphs)

# Transform a graph to feature matrix
X, node_to_idx = extractor.transform(graph)
# X is (num_nodes, num_features) numpy array
```

#### 2. Graph Dataset Loader (`crux/ml/graph_loader.py`)

Loads dependency graphs from CRUX datasets and prepares PyTorch Geometric Data objects:
- Reads graphs from `dataset/*/graphs/*.json` files
- Aligns graph nodes with labels from `labels.json`
- Builds edge indices in COO format for PyTorch Geometric
- Splits graphs into train/validation/test sets

```python
from crux.ml.graph_loader import GraphDatasetLoader

loader = GraphDatasetLoader("dataset/exp-20240101-120000")
train_data, val_data, test_data, feature_extractor, label_names = loader.load_and_prepare()
```

#### 3. GNN Models (`crux/ml/graph_models.py`)

Three state-of-the-art GNN architectures for multi-label node classification:

**GCN (Graph Convolutional Network):**
- Simple and effective graph convolutions
- Aggregates neighbor features via mean pooling
- Fast training and inference

**GAT (Graph Attention Network):**
- Uses attention mechanisms to weight neighbor contributions
- Learns which neighbors are most important
- Multiple attention heads for richer representations

**GraphSAGE:**
- Sampling-based approach for scalable graph learning
- Supports mean, max, or add aggregation
- Works well on large graphs

```python
from crux.ml.graph_models import GCNModel, GATModel, GraphSAGEModel

# Create GCN model
model = GCNModel(
    in_channels=50,        # Number of node features
    hidden_channels=64,     # Hidden layer size
    out_channels=37,        # Number of labels
    num_layers=2,
    dropout=0.5,
)

# Create GAT model with attention
model = GATModel(
    in_channels=50,
    hidden_channels=32,
    out_channels=37,
    num_layers=2,
    heads=4,               # Attention heads
    dropout=0.5,
)

# Create GraphSAGE model
model = GraphSAGEModel(
    in_channels=50,
    hidden_channels=64,
    out_channels=37,
    num_layers=2,
    aggr="mean",          # Aggregation method
)
```

#### 4. GNN Trainer (`crux/ml/graph_trainer.py`)

Handles training loop with early stopping and evaluation:
- Automatic device selection (GPU if available, else CPU)
- Early stopping based on validation F1 score
- Binary cross-entropy loss for multi-label classification
- Training history tracking

```python
from crux.ml.graph_trainer import GNNTrainer

trainer = GNNTrainer(
    model,
    learning_rate=0.001,
    weight_decay=5e-4,
)

# Train with early stopping
history = trainer.train(
    train_data,
    val_data,
    num_epochs=100,
    patience=10,  # Stop if no improvement for 10 epochs
)

# Evaluate on test set
test_loss, test_metrics = trainer.evaluate(test_data)

# Make predictions
y_pred, y_proba = trainer.predict(test_data)
```

### CLI Commands

#### Train a GNN Model

```bash
# Train GCN on a dataset
crux train-gnn \
  --dataset dataset/exp-20240101-120000 \
  --model gcn \
  --output models \
  --name my-gcn \
  --hidden-channels 64 \
  --num-layers 2 \
  --epochs 100 \
  --patience 10

# Train GAT with attention
crux train-gnn \
  --dataset dataset/exp-20240101-120000 \
  --model gat \
  --output models \
  --name my-gat \
  --hidden-channels 32 \
  --heads 4 \
  --num-layers 2

# Train GraphSAGE
crux train-gnn \
  --dataset dataset/exp-20240101-120000 \
  --model graphsage \
  --output models \
  --name my-sage \
  --hidden-channels 64 \
  --num-layers 3
```

Output files:
- `models/my-gcn.pt` - Trained model
- `models/my-gcn_features.pkl` - Graph feature extractor
- `models/my-gcn_metrics.json` - Test set evaluation metrics
- `models/my-gcn_history.json` - Training history (loss, F1 per epoch)

### Installation Requirements

GNN functionality requires PyTorch and PyTorch Geometric:

```bash
# CPU version (for development/testing)
pip install torch torchvision
pip install torch-geometric

# GPU version (for faster training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

**Note**: The GNN modules are optional. CRUX will work without PyTorch installed, but GNN training will not be available. The baseline models (Random Forest, XGBoost) do not require PyTorch.

### Workflow Example

```bash
# 1. Generate a dataset with dependency graphs (Milestone 2)
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name exp-gnn \
  --limit 100

# 2. Train baseline models for comparison (Milestone 3)
crux train-model \
  --dataset dataset/exp-gnn \
  --model random-forest \
  --output models \
  --name rf-baseline

# 3. Train GNN models (Milestone 4)
crux train-gnn \
  --dataset dataset/exp-gnn \
  --model gcn \
  --output models \
  --name gcn-model

crux train-gnn \
  --dataset dataset/exp-gnn \
  --model gat \
  --output models \
  --name gat-model

# 4. Compare results
# Check models/rf-baseline_metrics.json
# Check models/gcn-model_metrics.json
# Check models/gat-model_metrics.json
```

### Expected Performance

GNNs typically outperform baseline models on:
- **Cross-resource dependencies**: VM without NSG, VNet without DDoS protection
- **Structural patterns**: Resources in same connected component
- **Cascading misconfigurations**: One misconfigured resource affecting others

Baseline models may perform better on:
- **Simple single-resource rules**: Storage TLS version, encryption settings
- **Independent properties**: Resource-level configurations without dependencies
- **Small datasets**: GNNs need more data to learn graph patterns

### Model Selection Guide

**Use GCN when:**
- You have clear dependency structures
- Training speed is important
- You want a simple, interpretable baseline GNN

**Use GAT when:**
- Some dependencies are more important than others
- You want the model to learn attention weights
- You have sufficient training data (GAT has more parameters)

**Use GraphSAGE when:**
- You have very large graphs
- You want scalable inference
- You need different aggregation strategies

**Use Random Forest/XGBoost when:**
- You don't have dependency graph data
- You want fast training and inference
- You need feature importance analysis
- Resources are mostly independent

### Hyperparameter Tuning

Key hyperparameters for GNNs:

- `--hidden-channels`: Size of hidden layers (32-128 typical)
- `--num-layers`: Number of GNN layers (2-3 typical, deeper = more hops)
- `--dropout`: Regularization (0.3-0.7, higher for more data)
- `--lr`: Learning rate (0.001-0.01)
- `--weight-decay`: L2 regularization (1e-5 to 1e-3)
- `--patience`: Early stopping patience (10-20 epochs)
- `--heads`: (GAT only) Number of attention heads (2-8)

### Troubleshooting

**"PyTorch Geometric not available"**
- Install PyTorch and PyTorch Geometric (see Installation Requirements above)
- GNN functionality is optional - baseline models will still work

**"No graphs found in dataset"**
- Ensure dataset was generated with graph export enabled
- Check that `dataset/*/graphs/` directory exists and contains `*_graph.json` files

**OOM (Out of Memory) errors**
- Reduce `--hidden-channels` (e.g., 32 instead of 64)
- Reduce `--num-layers`
- Use GraphSAGE instead of GAT (fewer parameters)
- Train on CPU with `--device cpu`

**Poor model performance**
- Increase dataset size (more templates)
- Increase `--epochs` and `--patience`
- Try different model architectures
- Check that graphs have meaningful structure (not all disconnected nodes)

## Development Workflow

### Adding New Mutations

1. Create mutation function in `crux/mutations/<resource_type>.py`
2. Define `Mutation` object with metadata
3. Add to `ALL_MUTATIONS` list
4. Write unit test in `tests/test_mutations.py`

Example:
```python
# crux/mutations/keyvault.py
def mutate_no_purge_protection(resource):
    resource["properties"]["enablePurgeProtection"] = False
    return resource

KV_NO_PURGE_PROTECTION = Mutation(
    id="kv_no_purge_protection",
    target_type="Microsoft.KeyVault/vaults",
    description="Disable purge protection",
    severity="high",
    labels=["KeyVault_NoPurgeProtection", "CIS_8.4"],
    cis_references=["8.4"],
    mutate=mutate_no_purge_protection,
)
```

### Adding New Rules

1. Create or edit YAML file in `rules/<resource_type>.yaml`
2. Define rule with condition and labels
3. Rule evaluator automatically loads all `.yaml` files from `rules/`

Example:
```yaml
# rules/keyvault.yaml
rules:
  - id: kv-no-purge-protection
    resource_type: Microsoft.KeyVault/vaults
    severity: high
    cis_reference: "8.4"
    condition:
      property: properties.enablePurgeProtection
      equals: false
    labels:
      - KeyVault_NoPurgeProtection
      - CIS_8.4
```

### Testing Strategy

- **Unit tests**: Test individual components (mutations, rules, extractors)
- **Integration tests**: Test end-to-end pipeline on sample templates
- **Property-based tests**: Use `hypothesis` to generate random inputs
- **Validation tests**: Ensure mutations actually change properties

## Important Constraints

1. **Python 3.10+**: Use modern Python features (match/case, type hints)
2. **Azure CLI required**: `az bicep build` is external dependency
3. **No Azure deployment**: All analysis is static (template-level)
4. **Mutations are pure functions**: No side effects, always return new resource
5. **Rules are declarative**: No Python code in YAML rules

## Common Development Tasks

### Run a quick test

```bash
# Generate a small dataset for testing
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --limit 10 \
  --name test-run
```

### Debug a specific mutation

```python
# tests/test_mutations.py
def test_storage_public_access():
    from crux.mutations.storage import STORAGE_PUBLIC_BLOB_ACCESS

    resource = {
        "type": "Microsoft.Storage/storageAccounts",
        "properties": {"allowBlobPublicAccess": False}
    }

    mutated = STORAGE_PUBLIC_BLOB_ACCESS.apply(resource)
    assert mutated["properties"]["allowBlobPublicAccess"] == True
```

### Add logging

```python
import logging
logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed info for debugging")
logger.info("High-level progress updates")
logger.warning("Recoverable issues")
logger.error("Errors that prevent completion")
```

## Current Status

- ✅ **Milestone 0**: Repository restructuring, devcontainer with Claude Code CLI
- ✅ **Milestone 1**: Core infrastructure (templates, mutations, rules, dataset generator, CLI)
- ✅ **Milestone 2**: Large-scale dataset generation with dependency graphs
- ✅ **Milestone 3**: ML validation (XGBoost, Random Forest baselines)
- ✅ **Milestone 4**: GNN models (GCN, GAT, GraphSAGE for graph-aware detection)

