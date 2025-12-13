# Sample Dataset

This is a small sample dataset for CRUX (Cloud Resource mUtation eXtractor).

## Dataset Statistics

- **Templates processed**: 10 Azure Quickstart Templates (Storage-focused)
- **Baseline resources**: 24 (unmutated, negative class)
- **Mutated resources**: 110 (with security misconfigurations, positive class)
- **Labels generated**: 451 misconfiguration labels
- **CSV rows**: 134 (110 positive + 24 negative)
- **Class balance**: 82.1% positive / 17.9% negative

## Files

- `data.csv` - ML-ready dataset with named properties and binary labels
- `baseline/resources.json` - Original (secure) Azure resources
- `mutated/resources.json` - Mutated resources with misconfigurations
- `graphs/*.graphml` - Resource dependency graphs
- `labels.json` - Misconfiguration labels per resource
- `metadata.json` - Dataset metadata and statistics

## CSV Schema

| Column | Description |
|--------|-------------|
| `resource_id` | Unique Azure resource identifier |
| `resource_type` | Azure resource type (e.g., Microsoft.Storage/storageAccounts) |
| `is_mutated` | Whether this resource was mutated (1) or baseline (0) |
| `mutation_id` | ID of the mutation applied (empty for baseline) |
| `mutation_severity` | Severity level (high, medium, low) |
| `has_misconfiguration` | Binary target: 1 if misconfigured, 0 if clean |
| `label_count` | Number of misconfiguration labels |
| `labels` | Comma-separated list of misconfiguration labels |
| `source_template` | Path to source template |
| Named properties | 24 security-relevant property columns |

## Usage

### Python (pandas + sklearn)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('data.csv')

# Select named properties as features
feature_cols = ['allowBlobPublicAccess', 'supportsHttpsTrafficOnly',
                'minimumTlsVersion', 'enablePurgeProtection']
X = df[feature_cols].fillna(0).values
y = df['has_misconfiguration'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### R

```r
library(caret)

# Load dataset
df <- read.csv('data.csv')

# Train/test split
set.seed(42)
train_idx <- createDataPartition(df$has_misconfiguration, p=0.8, list=FALSE)
train <- df[train_idx, ]
test <- df[-train_idx, ]

# Train logistic regression
model <- glm(has_misconfiguration ~ allowBlobPublicAccess + supportsHttpsTrafficOnly,
             data=train, family=binomial)
summary(model)
```

## Reproduce This Dataset

```bash
# 1. Generate the raw dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --pattern "quickstarts/microsoft.storage/**/*.bicep" \
  --rules rules/ \
  --output dataset/ \
  --name sample \
  --limit 10

# 2. Export to CSV with named properties and baseline
crux export-csv \
  --dataset dataset/sample \
  --output dataset/sample/data.csv \
  --include-baseline \
  --named-properties curated \
  --binary-mode any
```

## Labels

The dataset includes labels for common Azure security misconfigurations:

- **Storage**: Public blob access, weak TLS, insecure transfer
- **VM**: Missing disk encryption, unapproved VM sizes, missing diagnostics
- **Network**: Public IPs, insecure NSG rules, missing DDoS protection

See `labels.json` for the complete label assignment per resource.
