# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRUX (Cloud Resource Configuration Analyzer) generates labeled datasets for training ML models to detect Azure resource misconfigurations. It uses **static template analysis** (Option D): analyzing Bicep/ARM templates locally without deploying to Azure, enabling zero-cost, large-scale dataset generation.

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

Two baseline models for multi-label classification:
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosted trees

```python
from crux.ml.models import RandomForestModel, XGBoostModel

# Random Forest
model = RandomForestModel(n_estimators=100, random_state=42)
model.fit(X_train, y_train, feature_names=features, label_names=labels)

# XGBoost
model = XGBoostModel(n_estimators=100, learning_rate=0.1, random_state=42)
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

### CLI Commands

#### Train a Model

```bash
# Train Random Forest on a dataset
crux train-model \
  --dataset dataset/exp-20240101-120000 \
  --model random-forest \
  --output models \
  --name my-rf-model

# Train XGBoost
crux train-model \
  --dataset dataset/exp-20240101-120000 \
  --model xgboost \
  --output models \
  --name my-xgb-model \
  --max-features 150
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

### Workflow Example

```bash
# 1. Generate a dataset (Milestone 2)
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name exp-001 \
  --limit 100

# 2. Train a Random Forest model (Milestone 3)
crux train-model \
  --dataset dataset/exp-001 \
  --model random-forest \
  --output models \
  --name rf-baseline

# 3. Train an XGBoost model for comparison
crux train-model \
  --dataset dataset/exp-001 \
  --model xgboost \
  --output models \
  --name xgb-baseline

# 4. Evaluate the Random Forest model
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
- 🔄 **Milestone 2**: Large-scale dataset generation (1000+ templates)
- 🔄 **Milestone 3**: ML validation (XGBoost, Random Forest baselines)
- 🔄 **Milestone 4**: Graph analysis (GNN models for cross-resource patterns)

## Archive: Option A

The original Option A (Azure deployment-based analysis) is archived in the `archive/option-a` branch. We pivoted to Option D (static analysis) for:
- Zero Azure costs
- Faster iteration
- Larger dataset capacity
- Better reproducibility
