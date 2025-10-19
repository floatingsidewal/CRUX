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
