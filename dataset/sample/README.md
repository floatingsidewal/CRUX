# Sample Dataset

This is a small sample dataset for CRUX (Cloud Resource Configuration Analyzer).

## Dataset Statistics

- **Templates processed**: 5 Azure Quickstart Templates
- **Baseline resources**: 20 (unmutated)
- **Mutated resources**: 10 (with security misconfigurations)
- **Labels generated**: 47 misconfiguration labels

## Dataset Structure

```
sample/
├── baseline/
│   └── resources.json          # Original (secure) Azure resources
├── mutated/
│   └── resources.json          # Mutated resources with misconfigurations
├── graphs/
│   └── *.graphml               # Resource dependency graphs
├── labels.json                 # Misconfiguration labels per resource
└── metadata.json               # Dataset metadata and statistics
```

## Usage

### Explore the Dataset

```bash
# View labels
cat dataset/sample/labels.json

# View metadata
cat dataset/sample/metadata.json

# Count resources
jq 'length' dataset/sample/baseline/resources.json
jq 'length' dataset/sample/mutated/resources.json
```

### Train a Model

```bash
# Install ML dependencies
pip install -e .[ml]

# Train a Random Forest model on the sample
crux train-model \
  --dataset dataset/sample \
  --model random-forest \
  --output models \
  --name sample-rf

# Evaluate the model
crux evaluate-model \
  --model models/sample-rf.pkl \
  --dataset dataset/sample
```

## Labels

The dataset includes labels for common Azure security misconfigurations:

- **Storage**: Public blob access, weak TLS, insecure transfer
- **VM**: Missing disk encryption, unapproved VM sizes, missing diagnostics
- **Network**: Public IPs, insecure NSG rules, missing DDoS protection

See `labels.json` for the complete label assignment per resource.

## Purpose

This sample dataset serves as:
- **Example** for understanding CRUX data format
- **Quick start** for testing ML models
- **Reference** for integrating CRUX into your workflow
- **Validation** that CRUX setup is working correctly

For larger-scale benchmarking, generate a full dataset with 100+ templates:

```bash
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name benchmark \
  --limit 100 \
  --pattern "quickstarts/**/azuredeploy.json"
```
