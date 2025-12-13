# CRUX

**CRUX** (Cloud Resource mUtation eXaminer) is a system for generating labeled datasets to train ML models that detect misconfigurations in Azure resources. CRUX analyzes Bicep/ARM templates statically, applies controlled mutations (a "configuration fuzzer"), and generates ground-truth labels based on security rules and CIS benchmarks.

## Why CRUX
- **Zero Azure cost**: Analyzes templates locally without deploying to Azure
- **Deterministic labels**: Every mutation is intentional and recorded → clean ground truth
- **Large-scale dataset generation**: Process 1000+ templates in hours, not weeks
- **Schema-agnostic**: Supports heterogeneous resource types (Storage, Key Vault, NSG, Function App, etc.)
- **Graph-aware**: Exports dependency graphs for cross-resource issue detection

## Architecture (Static Template Analysis)

CRUX uses a pipeline approach: **Fetch → Compile → Extract → Mutate → Label → Export**

1. **Fetch**: Download Azure Quickstart Templates from GitHub (1000+ real-world examples)
2. **Compile**: Convert Bicep to ARM JSON using `az bicep build`
3. **Extract**: Parse ARM JSON to extract resource properties and dependencies
4. **Mutate**: Apply Python-defined mutations to inject misconfigurations
5. **Label**: Evaluate YAML-defined security rules to generate labels
6. **Export**: Save labeled datasets (baseline + mutated) for ML training

## Long Term Goals
- Build robust ML models to detect misconfigurations in Azure resources using "Signatures", which reduce the need for exhaustive rule definitions or staff looking directly at running configuration to detect issues.
- Enable graph-based analysis to capture cross-resource misconfigurations (e.g., VM without NSG, Storage without encryption) and isolate the potential issue to a small set of resources.
- Provide a flexible framework for adding new mutations and rules as Azure services evolve.

## Quick Start

### Prerequisites

Before using CRUX, ensure you have:
- **Python 3.11+**
- **Azure CLI** ([Install Guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli))
- **Bicep CLI** ([Install Guide](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/install))

### Development Setup

#### Option 1: Devcontainer (Recommended)

The easiest way to get started is using the preconfigured devcontainer:

1. Open this repository in VS Code
2. When prompted, click "Reopen in Container" (or run **Dev Containers: Reopen in Container** from the command palette)
3. The container automatically installs all dependencies via `pip install -e .[dev]`

The devcontainer includes:
- Python 3.11
- Azure CLI & Bicep CLI
- All required VS Code extensions
- Pre-configured Python environment

#### Option 2: Local Development

If you prefer to work locally:

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate  # On Windows

# 2. Install CRUX in development mode
pip install -e .

# Or with development dependencies
pip install -e .[dev]

# Or with ML libraries
pip install -e .[ml]

# 3. Verify Azure CLI and Bicep are installed
az --version
az bicep version  # Install with: az bicep install
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

## Advanced Usage Examples

### Example 1: Large-Scale Dataset Generation (All Resource Types)

Generate a comprehensive dataset across all available Azure Quickstart Templates:

```bash
# Fetch a large collection of templates (first time only)
crux fetch-templates --limit 500 --output templates/

# Generate large-scale dataset with all resource types
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name large-scale-001 \
  --limit 500

# This will process templates containing:
# - Storage accounts (blob, file, queue, table)
# - Virtual machines and VM scale sets
# - Virtual networks, subnets, NSGs
# - Key vaults and secrets
# - Function apps and app services
# - Databases (SQL, Cosmos DB, etc.)
# - Load balancers and application gateways
# - And many more...

# Expected results (approximate):
# - Templates processed: 500
# - Resources extracted: 2000-5000
# - Mutated resources: 10000-25000
# - Labels generated: 50000-150000
# - Processing time: 2-6 hours (depending on hardware)
```

### Example 2: Core Infrastructure Resources (VM, Storage, Network, KeyVault)

Focus on essential Azure infrastructure components:

```bash
# Generate dataset targeting core infrastructure resource types
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/{compute,storage,network,key-vault}/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name core-infra-001 \
  --limit 100

# Alternative: Use grep to filter after fetching
# This captures templates that contain VM, storage, network, or keyvault resources
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name core-infra-002 \
  --limit 200

# Resource types included:
# Compute:
#   - Microsoft.Compute/virtualMachines
#   - Microsoft.Compute/virtualMachineScaleSets
#   - Microsoft.Compute/disks
# Storage:
#   - Microsoft.Storage/storageAccounts
#   - Microsoft.Storage/storageAccounts/blobServices
#   - Microsoft.Storage/storageAccounts/fileServices
# Network:
#   - Microsoft.Network/virtualNetworks
#   - Microsoft.Network/networkSecurityGroups
#   - Microsoft.Network/publicIPAddresses
#   - Microsoft.Network/networkInterfaces
#   - Microsoft.Network/loadBalancers
# Key Vault:
#   - Microsoft.KeyVault/vaults
#   - Microsoft.KeyVault/vaults/secrets
#   - Microsoft.KeyVault/vaults/keys

# Expected results:
# - Templates processed: 100-200
# - Resources extracted: 500-1500
# - Mutated resources: 2500-7500
# - Common mutation patterns:
#   * VM without NSG attached
#   * Storage without encryption
#   * Public IP without DDoS protection
#   * KeyVault without purge protection
#   * VNet without network policies
```

### Example 3: Security-Focused Dataset (High-Severity Issues)

Generate a dataset emphasizing high-severity security misconfigurations:

```bash
# Generate dataset focusing on CIS benchmark violations
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name security-cis-001 \
  --limit 300

# After generation, you can filter by severity in your ML pipeline
# High-severity labels include:
# - Storage_PublicAccess (CIS 3.7)
# - Storage_WeakTLS (CIS 3.1)
# - Storage_NoEncryption (CIS 3.9)
# - KeyVault_NoPurgeProtection (CIS 8.4)
# - Network_NoNSG (Custom)
# - VM_NoEncryption (CIS 7.1)

# Use the dataset for training models specialized in critical issues
```

### Example 4: Incremental Dataset Updates

Build datasets incrementally as you add new templates or mutations:

```bash
# Initial dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/microsoft.storage/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name storage-v1 \
  --limit 50

# Later: Add more template categories
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/microsoft.{storage,keyvault}/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name storage-kv-v2 \
  --limit 100

# Later: Add compute resources
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/microsoft.{storage,keyvault,compute}/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name multi-resource-v3 \
  --limit 200
```

### Example 5: Full ML Pipeline Workflow

Complete end-to-end workflow from dataset generation to model training:

```bash
# 1. Fetch templates (one-time setup)
crux fetch-templates --limit 300 --output templates/

# 2. Generate comprehensive dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name ml-experiment-001 \
  --limit 250

# 3. Train baseline models (requires: pip install -e .[ml])
crux train-model \
  --dataset dataset/ml-experiment-001 \
  --model random-forest \
  --output models/ \
  --name rf-baseline

crux train-model \
  --dataset dataset/ml-experiment-001 \
  --model xgboost \
  --output models/ \
  --name xgb-baseline

# 4. Train GNN models for graph-aware detection (requires: pip install torch torch-geometric)
crux train-gnn \
  --dataset dataset/ml-experiment-001 \
  --model gcn \
  --output models/ \
  --name gcn-graph-aware

crux train-gnn \
  --dataset dataset/ml-experiment-001 \
  --model gat \
  --output models/ \
  --name gat-attention

# 5. Export to CSV for external analysis
# Option A: Hashed features (100 numeric columns)
crux export-csv \
  --dataset dataset/ml-experiment-001 \
  --output data/ml-experiment-001.csv \
  --max-features 100 \
  --binary-mode any

# Option B: Named properties for interpretable ML (recommended for research)
crux export-csv \
  --dataset dataset/ml-experiment-001 \
  --output data/ml-capstone.csv \
  --include-baseline \
  --named-properties curated \
  --binary-mode any

# 6. Evaluate models
crux evaluate-model \
  --model models/rf-baseline.pkl \
  --dataset dataset/ml-experiment-001 \
  --output results/rf-eval.json

crux evaluate-model \
  --model models/xgb-baseline.pkl \
  --dataset dataset/ml-experiment-001 \
  --output results/xgb-eval.json
```

### Tips for Large-Scale Generation

**Performance Optimization:**
- Use `--limit` to control dataset size (start small, scale up)
- Use `--pattern` to filter template types early
- Run on machines with SSD for faster I/O
- Monitor disk space (datasets can be 100MB-10GB depending on scale)

**Template Selection:**
- **Diverse**: Use `"**/*.bicep"` to get all resource types
- **Focused**: Use `"quickstarts/microsoft.compute/**/*.bicep"` for specific categories
- **Multi-category**: Use `"quickstarts/microsoft.{storage,compute,network}/**/*.bicep"`

**Dataset Quality:**
- Check `metadata.json` for statistics (templates processed, resources extracted, mutations applied)
- Review `labels.json` for label distribution (avoid extreme imbalance)
- Use multiple small datasets for experimentation, large datasets for production models

**Reproducibility:**
- Save `metadata.json` with each experiment
- Use descriptive names (e.g., `core-infra-2024-01-15` instead of `test-001`)
- Document the `--pattern` and `--limit` used in your experiment notes

### CSV Export Options

The `crux export-csv` command supports multiple export modes for different use cases:

**Feature Types:**
- `--max-features N`: Hashed numeric features (default: 100 columns named `feature_0`, `feature_1`, etc.)
- `--named-properties curated`: 24 interpretable security-relevant properties (e.g., `allowBlobPublicAccess`, `enablePurgeProtection`)
- `--named-properties all`: All extracted properties as columns (can be 500+ columns)
- `--property-list FILE`: Custom list of property paths to extract

**Class Balance:**
- By default, only mutated (positive) samples are exported
- `--include-baseline`: Adds baseline resources as negative class (`has_misconfiguration=0`)
- Recommended for logistic regression and balanced ML training

**Example for Research/Capstone Projects:**
```bash
crux export-csv \
  --dataset dataset/maximum-001 \
  --output capstone_data.csv \
  --include-baseline \
  --named-properties curated \
  --binary-mode any
```

This produces a CSV with:
- ~20,000 rows (positive + negative samples)
- 33 columns (9 metadata + 24 named properties)
- ~63% positive / ~37% negative class balance
- Interpretable column names for logistic regression coefficients

See `docs/csv-export-example.md` for detailed Python and R examples.

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
  docs/                       # Documentation
  pyproject.toml              # Package configuration
  README.md
  CLAUDE.md
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

### Running Tests and Code Quality Tools

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

### Environment Notes

- **Virtual Environment**: If using local development (Option 2), always activate your virtual environment before running commands
- **Devcontainer**: All tools and dependencies are pre-installed; no additional setup needed
- **Azure Authentication**: CRUX analyzes templates locally and doesn't require Azure authentication for basic operations

## Roadmap

See **docs/prd.md** for the full requirements and milestones.

- ✅ Milestone 0: Repository restructuring and devcontainer setup
- ✅ Milestone 1: Core infrastructure (templates, mutations, rules, dataset generator)
- ✅ Milestone 2: Large-scale dataset generation (1000+ templates)
- ✅ Milestone 3: ML validation (baseline models)
- ✅ Milestone 4: Graph analysis and cross-resource patterns (GNN models)
