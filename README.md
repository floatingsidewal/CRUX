# CRUX

**CRUX** (Cloud Resource Configuration Analyzer) is a system for generating labeled datasets to train ML models that detect misconfigurations in Azure resources. CRUX analyzes Bicep/ARM templates statically, applies controlled mutations (a "configuration fuzzer"), and generates ground-truth labels based on security rules and CIS benchmarks.

## Why CRUX
- **Zero Azure cost**: Analyzes templates locally without deploying to Azure
- **Deterministic labels**: Every mutation is intentional and recorded → clean ground truth
- **Large-scale dataset generation**: Process 1000+ templates in hours, not weeks
- **Schema-agnostic**: Supports heterogeneous resource types (Storage, Key Vault, NSG, Function App, etc.)
- **Graph-aware**: Exports dependency graphs for cross-resource issue detection

## Architecture (Option D: Static Template Analysis)

CRUX uses a pipeline approach: **Fetch → Compile → Extract → Mutate → Label → Export**

1. **Fetch**: Download Azure Quickstart Templates from GitHub (1000+ real-world examples)
2. **Compile**: Convert Bicep to ARM JSON using `az bicep build`
3. **Extract**: Parse ARM JSON to extract resource properties and dependencies
4. **Mutate**: Apply Python-defined mutations to inject misconfigurations
5. **Label**: Evaluate YAML-defined security rules to generate labels
6. **Export**: Save labeled datasets (baseline + mutated) for ML training

## Quick Start

### Installation

```bash
# Install CRUX in development mode
pip install -e .

# Or with development dependencies
pip install -e .[dev]

# Or with ML libraries
pip install -e .[ml]
```

### Basic Workflow

```bash
# 1. Fetch Azure Quickstart Templates (first time only)
crux fetch-templates --limit 100 --output templates/

# 2. Generate a labeled dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name pilot-001 \
  --limit 50

# 3. Explore the results
ls -la dataset/pilot-001/
cat dataset/pilot-001/metadata.json
cat dataset/pilot-001/labels.json

# 4. List available mutations and rules
crux list-mutations
crux list-rules
```

## Repository Layout

```
crux/
  crux/                       # Main Python package
    templates/                # Template operations (fetch, compile, extract, graph)
    mutations/                # Python mutation definitions
      storage.py              # Storage account mutations
    rules/                    # Rule evaluation engine
    dataset/                  # Dataset generation pipeline
    cli.py                    # Command-line interface
  rules/                      # YAML security rule definitions
    storage.yaml              # CIS rules for storage accounts
  templates/                  # Downloaded templates (gitignored)
  dataset/                    # Generated datasets (gitignored)
  tests/                      # Unit tests
  pyproject.toml              # Package configuration
  README.md
  CLAUDE.md
  prd.md
```

## Dataset Structure

```
dataset/exp-YYYYMMDD-HHMMSS/
  metadata.json               # Experiment metadata
  baseline/
    resources.json            # Original (unmutated) resources
  mutated/
    resources.json            # Mutated resources with labels
  graphs/                     # Dependency graphs (GraphML)
  labels.json                 # resource_id → [labels] mapping
```

## Adding New Mutations

Mutations are defined in Python for type safety and flexibility:

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

## Adding New Rules

Security rules are defined in YAML for easy review by security teams:

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

## Development

```bash
# Run tests
pytest

# Format code
black crux/

# Type checking
mypy crux/

# Linting
ruff check crux/
```

## Roadmap

See **prd.md** for the full requirements and milestones.

- ✅ Milestone 0: Repository restructuring and devcontainer setup
- ✅ Milestone 1: Core infrastructure (templates, mutations, rules, dataset generator)
- 🔄 Milestone 2: Large-scale dataset generation (1000+ templates)
- 🔄 Milestone 3: ML validation (baseline models)
- 🔄 Milestone 4: Graph analysis and cross-resource patterns

## Archive: Option A (Azure Deployment-Based Analysis)

The original Option A approach (deploying to Azure and harvesting runtime state) has been archived to the `archive/option-a` branch. Option D (static template analysis) is now the primary approach due to:
- Zero Azure costs
- Faster iteration (hours vs. weeks)
- Larger dataset generation capacity
- Reproducibility (no Azure API flakiness)
