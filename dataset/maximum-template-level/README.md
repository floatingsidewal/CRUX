# Maximum Template-Level Dataset

**Experiment ID**: `maximum-template-level`
**Generated**: 2025-12-19
**Observations**: 14,000 (1,000 templates × 14 scenarios)
**Status**: ✅ Meets 7,000+ observation requirement
**Purpose**: Academic logistic regression analysis of Azure resource misconfigurations

## Overview

This dataset provides **14,000 template-level observations** meeting the specification requirement for academic research. Each observation represents an Azure Resource Manager (ARM) template in a specific mutation scenario, with aggregated features across all resources in that template.

### Specification Compliance

Per `docs/CRUX_Template_Level_Dataset_Specification.md`:

| Requirement | Target | Actual | Status |
|-------------|--------|--------|--------|
| **Minimum observations** | 7,000 | 14,000 | ✅ Exceeds (2×) |
| **Templates** | ~1,000 | 1,000 | ✅ Meets |
| **Scenarios per template** | 14 | 14 | ✅ Meets |
| **Mathematical justification** | 1,000 × 14 = 14,000 | 1,000 × 14 = 14,000 | ✅ Matches |

### Key Characteristics

- **Template-level aggregation**: Each observation represents an entire ARM template in a specific mutation scenario
- **Probabilistic IV-DV relationships**: Template features create statistical (not deterministic) relationships with misconfiguration labels
- **Comprehensive scenarios**: All 14 mutation scenarios from baseline to combined mutations
- **Diverse templates**: 1,000 templates from Azure Quickstart Templates repository
- **Suitable for logistic regression**: Avoids tautological resource-level relationships

## Dataset Structure

### Files

```
dataset/maximum-template-level/
├── template_level_data.csv    # 14,000 observations × 60 columns
└── README.md                   # This file
```

### Dimensions

| Dimension | Count | Description |
|-----------|-------|-------------|
| **Total observations** | 14,000 | 1,000 templates × 14 scenarios |
| **Baseline observations** | 1,000 | One per template (no mutations) |
| **Mutated observations** | 13,000 | 13 mutation scenarios per template |
| **Total columns** | 60 | 5 metadata + 48 features + 6 labels + 1 control |
| **Unique templates** | 1,000 | From Azure Quickstart Templates |
| **Scenarios** | 14 | Baseline + 13 mutation scenarios |
| **File size** | ~2.7 MB | CSV format |

## Column Reference

### Metadata Columns (5)

| Column | Type | Description |
|--------|------|-------------|
| `template_id` | string | Template path (e.g., "quickstarts/vm-simple-0001/azuredeploy") |
| `template_name` | string | Template filename (e.g., "azuredeploy") |
| `scenario_id` | string | Mutation scenario identifier |
| `scenario_category` | string | Category: control, security, operational, reliability, combined |
| `is_mutated` | binary | 0 = baseline (clean), 1 = mutated (has intentional misconfigurations) |

### Feature Columns (48)

Features are organized into 5 categories:

#### Composition Features (13)
Template structure and resource types:
- `num_resources`, `num_resource_types`
- `has_storage`, `has_vm`, `has_nsg`, `has_vnet`, `has_keyvault`, `has_sql`, `has_webapp`
- `count_storage`, `count_vm`, `count_nsg`, `count_vnet`

#### Security Features (14)
Security-related aggregates:
- `any_public_access`, `all_https_only`, `any_weak_tls`, `any_http_allowed`
- `pct_secure_boot`, `pct_vtpm_enabled`
- `any_no_encryption`, `any_password_auth`
- `any_open_inbound`, `any_open_ssh`, `any_open_rdp`, `any_ddos_disabled`
- `all_encryption_enabled`, `any_diagnostics_disabled`

#### Operational Features (7)
Operational metrics:
- `any_no_patching`, `all_auto_patch`
- `pct_managed_identity`, `any_no_identity`
- `any_versioning_disabled`, `any_soft_delete_disabled`

#### Reliability Features (5)
Reliability metrics:
- `any_no_availability_set`, `all_managed_disks`, `any_unmanaged_disk`
- `any_no_service_endpoints`, `any_broad_address_space`

#### Graph Features (6)
Dependency graph statistics:
- `num_dependencies`, `avg_resource_degree`, `max_resource_degree`
- `has_isolated_resources`, `max_dependency_depth`, `dependency_density`

### Label Columns (6)

Dependent variables for misconfiguration detection:

| Column | Type | Description |
|--------|------|-------------|
| `has_any_misconfiguration` | binary | **PRIMARY DV** - 1 if any misconfiguration present |
| `misconfiguration_count` | integer | Total number of misconfigurations |
| `unique_rule_count` | integer | Number of unique rules violated |
| `security_issue_count` | integer | Count of security misconfigurations |
| `operational_issue_count` | integer | Count of operational misconfigurations |
| `reliability_issue_count` | integer | Count of reliability misconfigurations |

## Mutation Scenarios

### Baseline
- **scenario_id**: `baseline`
- **category**: control
- **description**: No mutations applied - templates as authored
- **observations**: 1,000 (one per template)

### Single-Category Scenarios

#### Security (3 scenarios)
1. **security_high**: High-severity security mutations (7 mutations)
2. **security_medium**: Medium-severity security mutations (5 mutations)
3. **security_all**: All security mutations (12 mutations)

#### Operational (3 scenarios)
4. **operational_high**: High-severity operational mutations (4 mutations)
5. **operational_medium**: Medium-severity operational mutations (3 mutations)
6. **operational_all**: All operational mutations (7 mutations)

#### Reliability (3 scenarios)
7. **reliability_high**: High-severity reliability mutations (3 mutations)
8. **reliability_medium**: Medium-severity reliability mutations (2 mutations)
9. **reliability_all**: All reliability mutations (5 mutations)

### Combined Scenarios (4 scenarios)

10. **security_operational**: All security + operational mutations (19 mutations)
11. **security_reliability**: All security + reliability mutations (17 mutations)
12. **operational_reliability**: All operational + reliability mutations (12 mutations)
13. **all_mutations**: All available mutations (24 mutations)

## Usage Examples

### Load Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/maximum-template-level/template_level_data.csv')

print(f"Observations: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Baseline: {(df['is_mutated'] == 0).sum():,}")
print(f"Mutated: {(df['is_mutated'] == 1).sum():,}")
print(f"Templates: {df['template_id'].nunique():,}")
print(f"Scenarios: {df['scenario_id'].nunique()}")
```

Expected output:
```
Observations: 14,000
Columns: 60
Baseline: 1,000
Mutated: 13,000
Templates: 1,000
Scenarios: 14
```

### Explore Scenarios

```python
# Scenario distribution
scenario_counts = df['scenario_id'].value_counts()
print(scenario_counts)

# Average misconfigurations by scenario
scenario_stats = df.groupby('scenario_id')[
    ['misconfiguration_count', 'has_any_misconfiguration']
].agg(['mean', 'sum'])
print(scenario_stats)

# Category distribution
category_dist = df.groupby('scenario_category').agg({
    'template_id': 'count',
    'has_any_misconfiguration': 'mean'
})
print(category_dist)
```

### Compare Baseline vs Mutated

```python
# Compare baseline vs mutated observations
baseline = df[df['is_mutated'] == 0]
mutated = df[df['is_mutated'] == 1]

print("Baseline statistics:")
print(f"  Observations: {len(baseline):,}")
print(f"  Misconfiguration rate: {baseline['has_any_misconfiguration'].mean():.1%}")
print(f"  Avg issues: {baseline['misconfiguration_count'].mean():.2f}")

print("\nMutated statistics:")
print(f"  Observations: {len(mutated):,}")
print(f"  Misconfiguration rate: {mutated['has_any_misconfiguration'].mean():.1%}")
print(f"  Avg issues: {mutated['misconfiguration_count'].mean():.2f}")
```

### Feature Analysis

```python
# Identify high-variance features
feature_cols = [col for col in df.columns if col.startswith(
    ('num_', 'has_', 'count_', 'any_', 'all_', 'pct_', 'avg_', 'max_')
)]

feature_variance = df[feature_cols].var().sort_values(ascending=False)
print("Top 10 highest variance features:")
print(feature_variance.head(10))

# Feature correlation with DV
feature_correlation = df[feature_cols].corrwith(df['has_any_misconfiguration'])
print("\nTop 10 features correlated with misconfigurations:")
print(feature_correlation.abs().sort_values(ascending=False).head(10))
```

### Logistic Regression Example

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Prepare features and target
feature_cols = [col for col in df.columns if col.startswith(
    ('num_', 'has_', 'count_', 'any_', 'all_', 'pct_', 'avg_', 'max_')
)]
X = df[feature_cols]
y = df['has_any_misconfiguration']

# Split dataset (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} observations")
print(f"Test set: {len(X_test):,} observations")

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
print("\nTop 10 most important features:")
print(feature_importance.head(10))
```

### Multi-Category Prediction

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, accuracy_score

# Prepare multi-label targets (security, operational, reliability issues)
label_cols = ['has_any_misconfiguration', 'security_issue_count',
              'operational_issue_count', 'reliability_issue_count']
X = df[feature_cols]
y = df[label_cols]

# Convert counts to binary (has issue or not)
y_binary = (y > 0).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Train multi-output classifier
model = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print(f"Exact Match Ratio: {accuracy_score(y_test, y_pred):.3f}")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.3f}")

# Per-label accuracy
for i, col in enumerate(label_cols):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col}: {acc:.3f}")
```

### Scenario Effectiveness Analysis

```python
import matplotlib.pyplot as plt

# Compare misconfiguration rates by scenario
scenario_analysis = df.groupby('scenario_id').agg({
    'has_any_misconfiguration': 'mean',
    'misconfiguration_count': 'mean',
    'security_issue_count': 'mean',
    'operational_issue_count': 'mean',
    'reliability_issue_count': 'mean'
}).round(3)

print(scenario_analysis)

# Visualize (if matplotlib available)
scenario_analysis[['security_issue_count', 'operational_issue_count',
                    'reliability_issue_count']].plot(kind='bar', stacked=True,
                                                      figsize=(14, 6))
plt.title('Average Misconfiguration Counts by Scenario')
plt.xlabel('Scenario')
plt.ylabel('Average Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('scenario_analysis.png')
print("\nPlot saved to: scenario_analysis.png")
```

### Template Complexity Analysis

```python
# Analyze relationship between template complexity and misconfigurations
complexity_analysis = df.groupby('num_resources').agg({
    'template_id': 'count',
    'has_any_misconfiguration': 'mean',
    'misconfiguration_count': 'mean',
    'num_dependencies': 'mean'
}).rename(columns={'template_id': 'observation_count'})

print("Misconfiguration rate by template complexity:")
print(complexity_analysis)

# Correlation between complexity and issues
print("\nCorrelation with misconfiguration count:")
print(df[['num_resources', 'num_resource_types', 'num_dependencies',
          'dependency_density']].corrwith(df['misconfiguration_count']))
```

## Statistical Properties

### Observation Distribution

- **Balanced scenarios**: Each scenario has exactly 1,000 observations (one per template)
- **Controlled comparison**: Each template appears in all 14 scenarios
- **Stratified analysis**: Can stratify by `scenario_category` (control, security, operational, reliability, combined)

### Feature Properties

- **Binary features**: `has_*`, `any_*`, `all_*` (0 or 1)
- **Count features**: `num_*`, `count_*` (non-negative integers)
- **Percentage features**: `pct_*` (0.0 to 1.0)
- **Graph features**: `num_dependencies`, `avg_resource_degree`, etc. (real numbers)

### Label Properties

- **Binary labels**: `has_any_misconfiguration` (0 or 1)
- **Count labels**: `*_count` (non-negative integers)
- **Baseline is clean**: All baseline observations have `has_any_misconfiguration = 0`
- **Mutations introduce issues**: Mutated scenarios have varying misconfiguration rates

### Expected Statistics

```python
# Load and compute statistics
df = pd.read_csv('dataset/maximum-template-level/template_level_data.csv')

print("Dataset Statistics:")
print(f"  Total observations: {len(df):,}")
print(f"  Unique templates: {df['template_id'].nunique():,}")
print(f"  Scenarios: {df['scenario_id'].nunique()}")
print(f"  Baseline observations: {(df['is_mutated'] == 0).sum():,}")
print(f"  Mutated observations: {(df['is_mutated'] == 1).sum():,}")
print(f"  Overall misconfiguration rate: {df['has_any_misconfiguration'].mean():.1%}")
print(f"  Avg misconfigurations per observation: {df['misconfiguration_count'].mean():.2f}")
```

Expected output:
```
Dataset Statistics:
  Total observations: 14,000
  Unique templates: 1,000
  Scenarios: 14
  Baseline observations: 1,000
  Mutated observations: 13,000
  Overall misconfiguration rate: 60-70%
  Avg misconfigurations per observation: 1.5-2.5
```

## Analysis Recommendations

### For Academic Research

1. **Binary Logistic Regression**: Use `has_any_misconfiguration` as DV, composition + security features as IVs
2. **Multi-class Classification**: Predict `scenario_category` from features
3. **Count Regression**: Use Poisson/Negative Binomial for `misconfiguration_count`
4. **Survival Analysis**: Time-to-detection using template complexity as predictor

### For Machine Learning

1. **Binary Classification**: Clean vs misconfigured templates
2. **Multi-label Classification**: Predict all issue types simultaneously
3. **Feature Selection**: LASSO, Random Forest importance, or Boruta
4. **Ensemble Methods**: Random Forest, XGBoost, Gradient Boosting

### For Security Research

1. **Risk Scoring**: Develop composite risk scores from multiple labels
2. **Pattern Mining**: Identify common misconfiguration patterns
3. **Benchmark Comparison**: Compare against CIS benchmarks
4. **Impact Assessment**: Correlate complexity with security issues

## Data Quality

- **No missing values**: All 14,000 observations have complete data
- **No duplicate rows**: Each (template, scenario) combination appears exactly once
- **Consistent schema**: All 60 columns present in every row
- **Valid ranges**: All features within expected ranges (binary 0/1, percentages 0.0-1.0, etc.)

## Limitations

### Synthetic Mutations
- Labels derived from predefined mutation scenarios (known ground truth)
- May not capture all real-world misconfiguration patterns
- Useful for controlled experiments and baseline establishment

### Template Coverage
- 1,000 templates from Azure Quickstart Templates
- May not represent full diversity of production templates
- Focus on common Azure resource types

### Label Accuracy
- Labels based on YAML rule evaluation
- Rules may not cover all security best practices
- CIS compliance percentage is estimated

## Citation

If you use this dataset in academic research, please cite:

```
CRUX Maximum Template-Level Dataset (2025)
1,000 Azure Quickstart Templates × 14 Mutation Scenarios = 14,000 observations
Generated using CRUX v1.0.0 - Cloud Resource mUtation eXtractor
https://github.com/[your-repo]/CRUX
```

## License

This dataset is provided for academic research and educational purposes.

## Support

For questions or issues:
1. Review the specification: `docs/CRUX_Template_Level_Dataset_Specification.md`
2. Check generation guide: `docs/Generating_Maximum_Dataset.md`
3. Open an issue on GitHub

---

**Dataset Version**: 1.0.0
**Last Updated**: 2025-12-19
**Generator**: CRUX Template-Level Dataset Generator v1.0.0
