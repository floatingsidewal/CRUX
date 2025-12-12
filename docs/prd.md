# CRUX Product Requirements Document

**Cloud Resource mUtation eXtractor**
**Current Version**: 2.0 (Option D - Static Template Analysis)
**Last Updated**: 2025-12-12
**Status**: ✅ Production Ready

---

## Executive Summary

CRUX generates high-quality labeled datasets for training machine learning models to detect Azure resource misconfigurations. Unlike traditional rule-based tools, CRUX enables ML-based detection that can identify subtle, cross-resource issues and learn from patterns in real-world infrastructure templates.

**Key Achievement**: Successfully pivoted from costly Azure deployment-based approach (Option A) to **zero-cost static template analysis** (Option D), enabling large-scale dataset generation from 1,000+ templates.

---

## 1. Problem Statement

### The Challenge
- **Misconfigurations** are a leading cause of cloud security incidents and outages
- **Rule-based tools** catch obvious violations but miss subtle, cross-resource issues
- **Lack of labeled data** prevents ML-based misconfiguration detection research
- **High cost** of generating realistic labeled datasets from actual deployments

### The Solution
CRUX analyzes Azure Bicep/ARM templates locally (no deployment required), applies controlled mutations to inject misconfigurations, and evaluates security rules to generate ground-truth labels. This produces large-scale, high-quality datasets for ML model training at zero Azure cost.

---

## 2. Current Architecture (Option D)

### 2.1 Static Template Analysis Pipeline

```
Fetch → Compile → Extract → Mutate → Label → Export
```

1. **Fetch**: Download Azure Quickstart Templates from GitHub (1,205+ templates)
2. **Compile**: Convert Bicep to ARM JSON using `az bicep build`
3. **Extract**: Parse ARM JSON to extract resource configurations
4. **Graph**: Build dependency graphs using NetworkX
5. **Mutate**: Apply Python-defined mutations to inject misconfigurations
6. **Label**: Evaluate YAML-defined security rules
7. **Export**: Generate labeled datasets (JSON, CSV, GraphML)

### 2.2 Key Design Decisions

**Mutations are Python** (Type-Safe, Testable)
```python
def mutate_public_blob_access(resource):
    resource["properties"]["allowBlobPublicAccess"] = True
    return resource
```

**Rules are YAML** (Accessible to Security Teams)
```yaml
rules:
  - id: storage-public-blob-access
    condition:
      property: properties.allowBlobPublicAccess
      equals: true
    labels: [Storage_PublicAccess, CIS_3.7]
```

**Streaming Pipeline** (Memory-Efficient)
- Processes 1,000+ templates without loading all into memory
- Python generators enable scalable dataset generation

**Graph-Aware** (NetworkX)
- Dependency graphs capture resource relationships
- Export to GraphML (Gephi, Cytoscape) and JSON (PyTorch Geometric, DGL)

---

## 3. Machine Learning Pipeline Example

### 3.1 Baseline Models

**Implemented Models** (Milestone 3 ✅):
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosted trees
- **Logistic Regression**: Linear model with regularization

**Binary Classification Support** (NEW):
- Convert multi-label to binary (has ANY misconfiguration)
- Specialized BinaryEvaluator with ROC AUC, PR AUC, optimal threshold finding

### 3.2 Graph Neural Networks

**Implemented Models** (Milestone 4 ✅):
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **GraphSAGE**: Scalable graph learning

**Advantages**:
- Learn from cross-resource dependencies
- Detect multi-resource patterns
- Context-rich feature representations

### 3.3 Dataset Export

**Formats Supported**:
- **JSON**: Resource configurations and labels
- **CSV**: Tabular data for pandas, R, Excel (NEW)
- **GraphML**: Dependency graphs for visualization
- **PyTorch Geometric**: Graph data for GNNs

**CSV Export Features** (NEW):
- Numerical features (100+ by default)
- Binary target: `has_misconfiguration` (0/1)
- Metadata: resource_type, mutation_id, severity
- Compatible with sklearn, statsmodels, R caret

---

## 4. Current Status & Milestones

### 4.1 Achieved Milestones

| Milestone | Status | Description |
|-----------|--------|-------------|
| **0** | ✅ Complete | Repository restructuring, devcontainer setup |
| **1** | ✅ Complete | Core infrastructure (templates, mutations, rules) |
| **2** | ✅ Complete | Large-scale dataset generation |
| **3** | ✅ Complete | ML validation (baseline models) |
| **4** | ✅ Complete | GNN models (graph-aware detection) |

### 4.2 Key Achievements

1. **Zero-Cost Operation**: No Azure deployment required
2. **Large-Scale Dataset Generation**: Process 1,000+ templates
3. **Production-Ready**: Comprehensive CLI, tests, documentation
4. **Extensible**: Easy to add new mutations and rules
5. **Portable**: CSV export for external ML workflows

---

## 5. Repository Structure

```
CRUX/
├── crux/                      # Python package
│   ├── templates/             # Template operations
│   ├── mutations/             # Mutation definitions (Python)
│   ├── rules/                 # Rule engine
│   ├── dataset/               # Dataset generation
│   ├── ml/                    # ML models and evaluation
│   └── cli.py                 # Command-line interface
├── rules/                     # YAML rule definitions
│   ├── storage.yaml
│   ├── keyvault.yaml
│   ├── network.yaml
│   └── vm.yaml
├── tests/                     # Unit tests
├── docs/                      # Documentation
│   ├── csv-export-example.md
│   ├── changelog.md
│   └── prd.md
├── scripts/                   # Utility scripts
│   └── generate_full_dataset.py
├── templates/                 # Downloaded templates (gitignored)
├── dataset/                   # Generated datasets (gitignored)
└── models/                    # Trained models (gitignored)
```

---

## 6. Usage Workflows

### 6.1 Dataset Generation

```bash
# Fetch Azure Quickstart Templates
crux fetch-templates --limit 100 --output templates

# Generate labeled dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name my-experiment \
  --limit 50
```

### 6.2 Model Training

```bash
# Multi-label classification
crux train-model \
  --dataset dataset/my-experiment \
  --model xgboost \
  --output models \
  --name xgb-baseline

# Binary classification
crux train-model \
  --dataset dataset/my-experiment \
  --model logistic-regression \
  --binary-mode any \
  --output models \
  --name lr-binary

# Graph Neural Network
crux train-gnn \
  --dataset dataset/my-experiment \
  --model gat \
  --output models \
  --name gat-model
```

### 6.3 CSV Export for External Tools

```bash
# Export to CSV
crux export-csv \
  --dataset dataset/my-experiment \
  --output data/crux_data.csv \
  --max-features 100 \
  --binary-mode any

# Use in Python
import pandas as pd
df = pd.read_csv('data/crux_data.csv')
X = df[[col for col in df.columns if col.startswith('feature_')]].values
y = df['has_misconfiguration'].values
```

---

## 7. Resource Types Supported

| Resource Type | Mutations | Rules | Status |
|--------------|-----------|-------|--------|
| Storage Accounts | 13 | 13 | ✅ Production |
| Key Vaults | 8 | 8 | ✅ Production |
| Virtual Machines | 7 | 7 | ✅ Production |
| Virtual Networks | 4 | 4 | ✅ Production |
| Network Security Groups | 4 | 4 | ✅ Production |

**Total**: 36 mutations, 36 rules across 5 resource types

---

## 8. Quality Assurance

### 8.1 Testing

- **Unit Tests**: Test cases covering all components
- **Integration Tests**: End-to-end dataset generation
- **Continuous Validation**: Automated checks for label quality

### 8.2 Code Quality

- **Type Hints**: Full type annotations with mypy support
- **Code Formatting**: Black, isort
- **Linting**: Ruff
- **Documentation**: Comprehensive docstrings and user guides

---

## 9. Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Templates processed | 100+ | ✅ |
| Labeled samples | 1,000+ | ✅ |
| Model F1 Score | >0.70 | ✅ |
| Zero Azure cost | $0 | ✅ |
| Documentation | Complete | ✅ |

---

## 10. Future Enhancements

### 10.1 Potential Features (Not Committed)

- **Multi-cloud support**: AWS CloudFormation, GCP Deployment Manager
- **Terraform support**: Analyze Terraform configurations
- **Real-time detection**: Integrate with CI/CD pipelines
- **AutoML**: Automatic model selection and tuning
- **Explainability**: SHAP values, LIME for model interpretability
- **Active learning**: Prioritize uncertain samples for labeling

### 10.2 Research Directions

- **Transfer learning**: Pre-trained models on CRUX datasets
- **Anomaly detection**: Unsupervised misconfiguration detection
- **Temporal analysis**: Track configuration drift over time
- **Cross-resource attention**: Enhanced GNN architectures

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Template quality | Medium | Use Azure Quickstart Templates (curated) | ✅ Mitigated |
| Label imbalance | High | Large-scale dataset (6K+ samples) | ✅ Mitigated |
| Model overfitting | Medium | Cross-validation, regularization | ✅ Mitigated |
| Azure CLI dependency | Low | Well-documented, stable API | ✅ Acceptable |

---

## 12. Non-Goals

- **Production deployment scanner**: CRUX is a research tool for ML dataset generation
- **Real-time Azure monitoring**: Focus is on template analysis, not runtime
- **Comprehensive coverage**: Not all Azure resources (36 mutation types is sufficient)
- **Policy enforcement**: Use Azure Policy for governance, not CRUX

---

## 13. References

- **Documentation**: See `CLAUDE.md` for development guide
- **Examples**: See `docs/csv-export-example.md` for usage examples

---

**Maintained by**: Brad (Principal Software Engineer / Project Owner)
**Repository**: https://github.com/floatingsidewal/CRUX
**License**: MIT
