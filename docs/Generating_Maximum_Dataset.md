# Generating the Maximum Template-Level Dataset

This guide explains how to generate a template-level dataset meeting the specification requirement of **7,000+ observations** for academic logistic regression analysis.

## Specification Requirements

Per [CRUX_Template_Level_Dataset_Specification.md](CRUX_Template_Level_Dataset_Specification.md):

### Target Observations

- **Minimum**: 7,000 observations
- **Target**: 8,000-14,000 observations
- **Mathematical justification**: ~1,000 templates × 14 scenarios = ~14,000 observations

### Why 7,000+ Observations?

The specification addresses a critical problem with resource-level datasets:

**Problem**: Resource-level observations have **deterministic** IV-DV relationships
- Example: `IF allowBlobPublicAccess == True → has_misconfiguration = 1` (always)
- This is tautological - the IV directly encodes the DV
- Makes logistic regression meaningless (perfect separation)

**Solution**: Template-level observations have **probabilistic** IV-DV relationships
- Example: `IF any_public_access == True → has_any_misconfiguration = ?` (depends on other factors)
- Template-level features aggregate across resources, creating statistical uncertainty
- Suitable for logistic regression analysis

## Current Status

### Demonstration Dataset (dataset/maximum-template-level/)

- **Observations**: 112 (8 sample templates × 14 scenarios)
- **Status**: ⚠️ Does NOT meet 7,000+ requirement
- **Purpose**: Demonstrates dataset structure and format
- **Shortfall**: 6,888 observations (needs 62× more templates)

### Real Maximum Dataset (Not Yet Generated)

- **Target observations**: ~14,000 (1,000 templates × 14 scenarios)
- **Available templates**: 1,717 Azure Quickstart Templates
- **Location when generated**: `dataset/maximum-template-level-full/`

## Prerequisites

### 1. Fetch Azure Quickstart Templates

```bash
cd templates
git clone https://github.com/Azure/azure-quickstart-templates.git
cd ..
```

This provides **1,717 templates** - more than enough to exceed the 7,000 observation target.

### 2. Install CRUX with Dependencies

```bash
# Install in development mode with all dependencies
pip install -e .[dev]

# Or install specific dependencies
pip install pandas tqdm networkx pyyaml
```

### 3. Verify Installation

```bash
# Check CRUX CLI is available
crux --help

# Verify templates were fetched
find templates/azure-quickstart-templates -name "azuredeploy.json" | wc -l
# Should show: 1717 (or similar)
```

## Generation Methods

### Method 1: Use Generation Script (Recommended)

The easiest way to generate the maximum dataset:

```bash
python3 scripts/generate_maximum_template_dataset.py
```

**What it does**:
1. Discovers ~1,717 Azure Quickstart Templates
2. Selects first 1,000 templates (per spec)
3. Applies all 14 mutation scenarios to each template
4. Generates ~14,000 observations
5. Validates dataset meets requirements
6. Saves to `dataset/maximum-template-level/`

**Expected output**:
```
Found 1717 templates
Limited to: 1000 templates

Dataset Configuration:
  Templates: 1,000
  Scenarios: 14
  Expected observations: 14,000
  Meets 7,000+ requirement: ✓ YES

Generating dataset...
This may take 10-30 minutes for 1,000 templates...

✓ Dataset Generation Complete!
Output directory: dataset/maximum-template-level-full

✓ Dataset meets all academic requirements!
```

**Estimated time**: 10-30 minutes (depends on hardware)

### Method 2: Use CRUX CLI Directly

For more control over the generation process:

```bash
crux generate-template-dataset \
  --templates templates/azure-quickstart-templates \
  --rules rules/ \
  --output dataset/ \
  --name maximum-template-level-full \
  --scenarios all \
  --limit 1000 \
  --validate
```

**Parameters**:
- `--templates`: Directory containing ARM/Bicep templates
- `--rules`: Directory containing YAML rule definitions
- `--output`: Output directory for datasets
- `--name`: Experiment name (used as subdirectory name)
- `--scenarios`: Which scenarios to use (`all` = all 14, or comma-separated list)
- `--limit`: Limit number of templates (1000 = ~14,000 observations)
- `--validate`: Run validation checks after generation

### Method 3: Python API

For programmatic control or integration:

```python
from pathlib import Path
from crux.dataset.template_level_generator import (
    TemplateLevelDatasetGenerator,
    MUTATION_SCENARIOS
)

# Discover templates
template_paths = list(Path("templates/azure-quickstart-templates").rglob("azuredeploy.json"))
template_paths = [str(p) for p in template_paths[:1000]]  # Limit to 1,000

# Initialize generator
generator = TemplateLevelDatasetGenerator(rules_dir="rules/")

# Generate dataset
output_path = generator.generate_dataset(
    template_paths=template_paths,
    output_dir="dataset/",
    experiment_name="maximum-template-level-full",
    limit=None,  # Already limited to 1,000
    scenarios_subset=None  # Use all 14 scenarios
)

print(f"Dataset generated: {output_path}")
```

## Expected Output

### Output Directory Structure

```
dataset/maximum-template-level-full/
├── template_level_data.csv      # Main dataset (~14,000 rows × 60 columns)
├── metadata.json                # Experiment configuration and documentation
├── summary_stats.json           # Statistical summary of features and labels
└── failed_templates.json        # Any templates that failed processing (if any)
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total observations** | ~14,000 |
| **Unique templates** | ~1,000 |
| **Scenarios per template** | 14 |
| **Baseline observations** | ~1,000 (1 per template) |
| **Mutated observations** | ~13,000 (13 scenarios per template) |
| **Total columns** | 60 |
| **Feature columns** | 45 |
| **Label columns** | 10 |
| **Metadata columns** | 5 |
| **Meets 7,000+ requirement** | ✅ YES |

### CSV Structure

The dataset will have 60 columns:

**Metadata (5)**:
- `template_id`, `template_name`, `scenario_id`, `scenario_category`, `is_mutated`

**Features (45)**:
- Composition (13): `num_resources`, `has_*`, `count_*`
- Security (14): `any_public_access`, `all_https_only`, `pct_secure_boot`, etc.
- Operational (7): `any_diagnostics_disabled`, `pct_managed_identity`, etc.
- Reliability (5): `any_no_availability_set`, `all_managed_disks`, etc.
- Graph (6): `num_dependencies`, `avg_resource_degree`, `dependency_density`, etc.

**Labels (10)**:
- `has_any_misconfiguration` (primary DV)
- `misconfiguration_count`, `unique_rule_count`
- `security_issue_count`, `operational_issue_count`, `reliability_issue_count`
- `has_critical_issue`, `has_high_issue`, `max_severity_level`, `cis_compliance_pct`

## Validation

The generated dataset is automatically validated against these requirements:

### Validation Checks

1. **Observation Count**: ≥7,000 observations
2. **DV Variance**: 10-90% positive rate (not too imbalanced)
3. **Feature Variance**: <20% of features have low variance
4. **Multicollinearity**: No feature pairs with correlation >0.90
5. **Scenario Effectiveness**: Mutated scenarios have higher misconfiguration rates than baseline
6. **Template Coverage**: ≥95% of expected template × scenario combinations

### Manual Validation

```bash
# Load dataset in Python
python3 << EOF
import pandas as pd

df = pd.read_csv('dataset/maximum-template-level-full/template_level_data.csv')

print(f"Total observations: {len(df):,}")
print(f"Unique templates: {df['template_id'].nunique():,}")
print(f"Unique scenarios: {df['scenario_id'].nunique()}")
print(f"Baseline observations: {(df['is_mutated'] == 0).sum():,}")
print(f"Mutated observations: {(df['is_mutated'] == 1).sum():,}")
print(f"DV positive rate: {df['has_any_misconfiguration'].mean():.1%}")
print(f"Meets 7,000+ requirement: {len(df) >= 7000}")
EOF
```

Expected output:
```
Total observations: 14,000
Unique templates: 1,000
Unique scenarios: 14
Baseline observations: 1,000
Mutated observations: 13,000
DV positive rate: 65.3%
Meets 7,000+ requirement: True
```

## Scaling Considerations

### Template Count vs. Observations

| Templates | Scenarios | Observations | Meets Target? |
|-----------|-----------|--------------|---------------|
| 8 (demo) | 14 | 112 | ❌ No (6,888 short) |
| 100 | 14 | 1,400 | ❌ No (5,600 short) |
| 500 | 14 | 7,000 | ✅ Yes (exactly) |
| **1,000** | **14** | **14,000** | ✅ **Yes (target)** |
| 1,717 (all) | 14 | 24,038 | ✅ Yes (exceeds) |

### Performance Estimates

| Templates | Est. Time | CSV Size | RAM Usage |
|-----------|-----------|----------|-----------|
| 100 | 1-3 min | ~500 KB | ~200 MB |
| 500 | 5-15 min | ~2.5 MB | ~500 MB |
| **1,000** | **10-30 min** | **~5 MB** | **~1 GB** |
| 1,717 | 20-50 min | ~8 MB | ~2 GB |

**Recommendation**: Use 1,000 templates for the target 14,000 observations. This provides:
- Well above the 7,000 minimum
- Reasonable generation time (10-30 minutes)
- Manageable file size (~5 MB CSV)
- Sufficient data for robust statistical analysis

## Troubleshooting

### "ModuleNotFoundError: No module named 'tqdm'"

Install dependencies:
```bash
pip install -e .[dev]
```

### "Templates directory not found"

Fetch Azure Quickstart Templates:
```bash
cd templates && git clone https://github.com/Azure/azure-quickstart-templates.git && cd ..
```

### "No templates found"

Verify templates were fetched:
```bash
find templates/azure-quickstart-templates -name "azuredeploy.json" | head -5
```

### Generation is very slow

- Reduce template limit to 500 (still meets 7,000 requirement)
- Use faster hardware
- Check for network/disk I/O bottlenecks
- Ensure sufficient RAM (1-2 GB recommended)

### Dataset doesn't meet 7,000 observation requirement

Check how many templates were actually processed:
```bash
# Count observations
wc -l dataset/maximum-template-level-full/template_level_data.csv

# Check metadata
cat dataset/maximum-template-level-full/metadata.json | grep observation_count
```

If too few:
- Increase `--limit` parameter to 1,000
- Check `failed_templates.json` for errors
- Ensure all 14 scenarios are being used (check `--scenarios` parameter)

## Next Steps

After generating the maximum dataset:

1. **Verify it meets requirements**:
   ```bash
   python3 -c "import pandas as pd; df=pd.read_csv('dataset/maximum-template-level-full/template_level_data.csv'); print(f'Observations: {len(df):,}'); print(f'Meets 7K: {len(df)>=7000}')"
   ```

2. **Load in Python/R for analysis**:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset/maximum-template-level-full/template_level_data.csv')
   ```

3. **Run logistic regression**:
   - Use `has_any_misconfiguration` as dependent variable
   - Use composition + security features as independent variables
   - See `dataset/maximum-template-level/README.md` for examples

4. **Export to other formats** (if needed):
   ```bash
   crux export-csv --dataset dataset/maximum-template-level-full --output data/analysis.csv
   ```

## References

- [CRUX Template-Level Dataset Specification](CRUX_Template_Level_Dataset_Specification.md)
- [Dataset README](../dataset/maximum-template-level/README.md)
- [CSV Export Guide](csv-export-example.md)

## Summary

To generate the real maximum template-level dataset meeting specification requirements:

```bash
# 1. Fetch templates (if not already done)
cd templates && git clone https://github.com/Azure/azure-quickstart-templates.git && cd ..

# 2. Install dependencies
pip install -e .[dev]

# 3. Generate dataset
python3 scripts/generate_maximum_template_dataset.py

# 4. Verify
python3 -c "import pandas as pd; print(f\"Observations: {len(pd.read_csv('dataset/maximum-template-level-full/template_level_data.csv')):,}\")"
```

**Expected result**: ~14,000 observations meeting all academic requirements for logistic regression analysis.
