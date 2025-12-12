# CSV Export Usage Guide

This guide demonstrates how to export CRUX datasets to CSV format and use them in external tools (Python, R, Excel, SPSS, etc.) for custom analysis and machine learning.

## Overview

The CSV export feature converts CRUX datasets into a tabular format suitable for:
- Custom machine learning workflows
- Statistical analysis in R, Python, or SPSS
- Data exploration in Excel, Tableau, or other BI tools
- Integration with external systems

## Quick Start

### 1. Generate a Dataset

First, generate a CRUX dataset using the standard workflow:

```bash
# Fetch templates (if you haven't already)
crux fetch-templates --limit 100 --output templates

# Generate dataset
crux generate-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name my-experiment \
  --limit 50
```

### 2. Export to CSV

Export the dataset to CSV format with binary labels and features:

```bash
crux export-csv \
  --dataset dataset/my-experiment \
  --output data/crux_data.csv \
  --max-features 100 \
  --binary-mode any
```

**Output:** `data/crux_data.csv` with the following columns:
- `resource_id` - Unique resource identifier
- `resource_type` - Azure resource type (e.g., `Microsoft.Storage/storageAccounts`)
- `is_mutated` - Whether resource was mutated (0 = baseline, 1 = mutated)
- `mutation_id` - Applied mutation identifier
- `mutation_severity` - Severity level (high, medium, low)
- `has_misconfiguration` - **Binary target variable** (0 = clean, 1 = has misconfiguration)
- `label_count` - Number of misconfiguration labels
- `labels` - Comma-separated list of specific misconfigurations
- `source_template` - Source Bicep template path
- `feature_<name>` - Numerical feature columns (100 columns by default)

## Using the CSV Data

### Python with pandas and scikit-learn

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the data
df = pd.read_csv('data/crux_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Positive samples: {df['has_misconfiguration'].sum()}")
print(f"Negative samples: {(df['has_misconfiguration'] == 0).sum()}")

# 2. Separate features and target
feature_cols = [col for col in df.columns if col.startswith('feature_')]
X = df[feature_cols].values
y = df['has_misconfiguration'].values

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# 3. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 5. Evaluate
y_pred = lr_model.predict(X_test)
y_proba = lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 6. Compare with Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))
print(f"ROC AUC: {roc_auc_score(y_test, rf_proba):.3f}")

# 7. Feature importance (for tree-based models)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

### Python with statsmodels (Statistical Analysis)

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit

# Load data
df = pd.read_csv('data/crux_data.csv')

# Select a subset of features for interpretability
feature_cols = [col for col in df.columns if col.startswith('feature_')][:10]
X = df[feature_cols]
y = df['has_misconfiguration']

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X_with_const)
result = model.fit()

# Print detailed statistical summary
print(result.summary())

# Get odds ratios
odds_ratios = pd.DataFrame({
    'feature': X_with_const.columns,
    'odds_ratio': np.exp(result.params),
    'p_value': result.pvalues
})
print("\nOdds Ratios:")
print(odds_ratios)
```

### R with caret and glmnet

```r
library(tidyverse)
library(caret)
library(glmnet)

# 1. Load the data
df <- read.csv("data/crux_data.csv")

cat("Dataset dimensions:", dim(df), "\n")
cat("Positive samples:", sum(df$has_misconfiguration == 1), "\n")
cat("Negative samples:", sum(df$has_misconfiguration == 0), "\n")

# 2. Extract features and target
feature_cols <- grep("^feature_", names(df), value = TRUE)
X <- as.matrix(df[, feature_cols])
y <- factor(df$has_misconfiguration, levels = c(0, 1),
            labels = c("clean", "misconfigured"))

# 3. Split into train/test
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# 4. Train Logistic Regression with cross-validation
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

lr_model <- train(
  x = X_train,
  y = y_train,
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "ROC"
)

# 5. Evaluate on test set
y_pred_proba <- predict(lr_model, X_test, type = "prob")[, "misconfigured"]
y_pred <- predict(lr_model, X_test)

confusionMatrix(y_pred, y_test)

# 6. ROC curve and AUC
library(pROC)
roc_obj <- roc(y_test, y_pred_proba)
cat("ROC AUC:", auc(roc_obj), "\n")
plot(roc_obj, main = "ROC Curve - Misconfiguration Detection")

# 7. Train Elastic Net for feature selection
elastic_net <- train(
  x = X_train,
  y = y_train,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(
    alpha = seq(0, 1, by = 0.1),  # 0 = ridge, 1 = lasso
    lambda = seq(0.001, 0.1, by = 0.01)
  ),
  metric = "ROC"
)

# Get coefficient estimates (non-zero features)
coef_matrix <- coef(elastic_net$finalModel, s = elastic_net$bestTune$lambda)
non_zero_features <- which(coef_matrix != 0)
cat("Number of selected features:", length(non_zero_features), "\n")
```

### Excel / Tableau / Power BI

The CSV file can be opened directly in Excel or imported into BI tools:

**Excel:**
1. Open Excel
2. File → Open → Select `crux_data.csv`
3. Use Data → Filter to explore specific resource types or severities
4. Create PivotTables to summarize misconfigurations by resource type
5. Use built-in charts to visualize label distributions

**Tableau:**
1. Connect to Data → Text File → Select `crux_data.csv`
2. Create calculated fields:
   - `Misconfiguration Rate = SUM([has_misconfiguration]) / COUNT([resource_id])`
3. Build dashboards showing:
   - Misconfiguration rates by resource type
   - Distribution of mutation severities
   - Feature importance visualization

**Power BI:**
1. Get Data → Text/CSV → Select `crux_data.csv`
2. Use Power Query to transform data if needed
3. Create measures:
   ```DAX
   Misconfiguration Rate =
       DIVIDE(
           SUM('crux_data'[has_misconfiguration]),
           COUNT('crux_data'[resource_id])
       )
   ```

## Advanced Usage

### Export Without Features (Metadata Only)

If you only need labels and metadata without features:

```bash
crux export-csv \
  --dataset dataset/my-experiment \
  --output data/crux_metadata.csv \
  --no-features \
  --binary-mode any
```

This creates a smaller file useful for:
- Quick label exploration
- Joining with other datasets
- Reporting and documentation

### Include Baseline Resources

To include both mutated AND baseline (clean) resources:

```bash
crux export-csv \
  --dataset dataset/my-experiment \
  --output data/crux_full.csv \
  --include-baseline \
  --binary-mode any
```

This provides more balanced data with clean examples.

### Custom Feature Engineering in Python

```python
import pandas as pd

# Load data
df = pd.read_csv('data/crux_data.csv')

# Add custom features
df['is_storage'] = (df['resource_type'].str.contains('Storage')).astype(int)
df['is_compute'] = (df['resource_type'].str.contains('VirtualMachines|Compute')).astype(int)
df['is_network'] = (df['resource_type'].str.contains('Network|VirtualNetworks')).astype(int)

# Combine with existing features
feature_cols = [col for col in df.columns if col.startswith('feature_')]
feature_cols += ['is_storage', 'is_compute', 'is_network']

X = df[feature_cols].values
y = df['has_misconfiguration'].values

# Continue with modeling...
```

## Integration with CRUX Models

You can combine CSV exports with CRUX-trained models:

```bash
# 1. Export CSV for external analysis
crux export-csv \
  --dataset dataset/my-experiment \
  --output data/analysis.csv

# 2. Train logistic regression in CRUX
crux train-model \
  --dataset dataset/my-experiment \
  --model logistic-regression \
  --binary-mode any \
  --output models \
  --name lr-binary

# 3. Compare CRUX model with external models
# Use the CSV to train models in Python/R and compare results
```

## Schema Reference

### CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `resource_id` | string | Unique identifier | `myStorage#storage_public_blob_access` |
| `resource_type` | string | Azure resource type | `Microsoft.Storage/storageAccounts` |
| `is_mutated` | int (0/1) | Was resource mutated? | `1` |
| `mutation_id` | string | Mutation identifier | `storage_public_blob_access` |
| `mutation_severity` | string | Severity level | `high`, `medium`, `low` |
| `has_misconfiguration` | int (0/1) | **Target variable** | `1` |
| `label_count` | int | Number of labels | `3` |
| `labels` | string | Comma-separated labels | `Storage_PublicAccess,CIS_3.7` |
| `source_template` | string | Template path | `templates/quickstarts/storage.bicep` |
| `feature_*` | float | Numerical features | Various numeric values |

## Troubleshooting

### Large CSV Files

If the CSV is too large for Excel (>1M rows):
- Use `--max-features` to reduce columns
- Use `--no-features` for metadata only
- Process in chunks with pandas:
  ```python
  chunk_iter = pd.read_csv('data/large.csv', chunksize=10000)
  for chunk in chunk_iter:
      # Process each chunk
      pass
  ```

### Missing Values

Feature columns may contain `-1.0` for missing properties:
```python
# Replace -1 with NaN
df.replace(-1.0, pd.NA, inplace=True)

# Or drop features with too many missing values
missing_pct = df.isnull().sum() / len(df)
keep_cols = missing_pct[missing_pct < 0.5].index
df_clean = df[keep_cols]
```

### Class Imbalance

If you have imbalanced classes:
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use in LogisticRegression
lr = LogisticRegression(
    class_weight={0: class_weights[0], 1: class_weights[1]}
)

# Or use SMOTE for oversampling
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

## Next Steps

- **Model Comparison**: Compare CRUX built-in models vs custom models
- **Feature Engineering**: Create domain-specific features for Azure resources
- **Ensemble Methods**: Combine multiple models for better performance
- **Deployment**: Export trained models for production use
- **Monitoring**: Track misconfiguration rates over time

## Support

For questions or issues:
- GitHub: https://github.com/anthropics/crux/issues
- Documentation: See `CLAUDE.md` in the repository root
