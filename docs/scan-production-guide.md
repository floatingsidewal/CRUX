# CRUX Scan: Production Use Cases Guide

This guide covers real-world scenarios for using CRUX's scan functionality to assess Azure resource misconfigurations in production environments.

## Overview

CRUX Scan analyzes Bicep/ARM templates using trained machine learning models to detect potential security misconfigurations before deployment. It combines:

- **Rule-based detection**: Deterministic checks against CIS Azure Benchmark rules
- **ML-powered inference**: Probabilistic risk scoring from trained models
- **Research-backed recommendations**: Security guidance based on statistical analysis

### How It Works

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Bicep/ARM      │────▶│    CRUX      │────▶│   Risk Report   │
│  Template       │     │    Scan      │     │  (JSON/Text)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌─────▼─────┐       ┌─────▼─────┐
              │  Trained  │       │   YAML    │
              │  ML Model │       │   Rules   │
              └───────────┘       └───────────┘
```

1. **Input**: Bicep (`.bicep`) or ARM JSON (`.json`) templates
2. **Processing**: Template compilation, resource extraction, feature generation
3. **Analysis**: ML model inference + rule evaluation
4. **Output**: Risk scores, findings, and actionable recommendations

---

## Prerequisites

### Required Components

```bash
# 1. Install CRUX
pip install -e .

# 2. Verify Azure CLI (needed for Bicep compilation)
az --version
az bicep version

# 3. Trained model (from Phase I dataset generation + training)
ls models/
# rf-baseline.pkl          <- Random Forest model
# xgb-baseline.pkl         <- XGBoost model
# lr-binary.pkl            <- Logistic Regression model
```

### Model Selection Guide

| Model | Best For | Speed | Interpretability |
|-------|----------|-------|------------------|
| **Random Forest** | General-purpose scanning | Fast | Medium |
| **XGBoost** | High accuracy requirements | Medium | Medium |
| **Logistic Regression** | Interpretable risk factors | Fast | High |

---

## Production Use Cases

### Use Case 1: Developer Pre-Commit Scanning

**Scenario**: A developer wants to check their infrastructure code before committing.

```bash
# Scan a single template during development
crux scan \
  --template infra/storage-account.bicep \
  --model models/rf-baseline.pkl \
  --rules rules/

# Example output:
# Overall Risk Score: 0.35 (MEDIUM)
# Resources at Risk: 1/3
#
# [MEDIUM] Microsoft.Storage/storageAccounts - devStorage
#   Recommendations:
#     - Enable soft delete for blob data recovery
#     - Consider enabling versioning for data protection
```

**Git Pre-Commit Hook** (`.git/hooks/pre-commit`):

```bash
#!/bin/bash
set -e

# Find all modified Bicep files
BICEP_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.bicep$' || true)

if [ -n "$BICEP_FILES" ]; then
    echo "Running CRUX security scan on Bicep templates..."
    for file in $BICEP_FILES; do
        crux scan --template "$file" --model models/rf-baseline.pkl --fail-threshold 0.8
        if [ $? -ne 0 ]; then
            echo "Security scan failed for $file"
            exit 1
        fi
    done
    echo "Security scan passed!"
fi
```

---

### Use Case 2: CI/CD Pipeline Integration

**Scenario**: Block deployments that exceed risk thresholds in automated pipelines.

#### GitHub Actions

```yaml
# .github/workflows/security-scan.yml
name: Infrastructure Security Scan

on:
  pull_request:
    paths:
      - 'infra/**/*.bicep'
      - 'infra/**/*.json'

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Azure CLI
        run: |
          curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
          az bicep install

      - name: Install CRUX
        run: pip install -e .

      - name: Download trained model
        run: |
          # Download from your artifact storage
          aws s3 cp s3://my-bucket/models/rf-baseline.pkl models/

      - name: Run security scan
        run: |
          crux scan \
            --template infra/main.bicep \
            --model models/rf-baseline.pkl \
            --rules rules/ \
            --output-format json \
            --output scan-results.json \
            --fail-threshold 0.7

      - name: Upload scan results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: security-scan-results
          path: scan-results.json

      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('scan-results.json', 'utf8'));
            const body = `## Security Scan Failed

            **Risk Score**: ${results.summary.overall_risk_score.toFixed(2)} (${results.summary.overall_risk_level})
            **Resources at Risk**: ${results.summary.resources_at_risk}/${results.summary.total_resources}

            Please review the findings and address high-risk issues before merging.`;

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });
```

#### Azure DevOps

```yaml
# azure-pipelines.yml
trigger:
  paths:
    include:
      - infra/*

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: |
      curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
      az bicep install
    displayName: 'Install Azure CLI'

  - script: pip install -e .
    displayName: 'Install CRUX'

  - task: DownloadSecureFile@1
    inputs:
      secureFile: 'rf-baseline.pkl'
    displayName: 'Download trained model'

  - script: |
      crux scan \
        --template infra/main.bicep \
        --model $(Agent.TempDirectory)/rf-baseline.pkl \
        --rules rules/ \
        --ci \
        --output $(Build.ArtifactStagingDirectory)/scan-results.json
    displayName: 'Run CRUX Security Scan'

  - task: PublishBuildArtifacts@1
    condition: always()
    inputs:
      pathToPublish: '$(Build.ArtifactStagingDirectory)/scan-results.json'
      artifactName: 'SecurityScanResults'
```

---

### Use Case 3: Batch Scanning for Security Audits

**Scenario**: Security team needs to scan all infrastructure templates in a repository.

```bash
#!/bin/bash
# scripts/batch-scan.sh

MODEL="models/rf-baseline.pkl"
RULES_DIR="rules/"
OUTPUT_DIR="scan-results/$(date +%Y%m%d-%H%M%S)"
SUMMARY_FILE="$OUTPUT_DIR/summary.json"

mkdir -p "$OUTPUT_DIR"

echo "Starting batch security scan..."
echo "Output directory: $OUTPUT_DIR"

# Initialize summary
echo '{"scans": [], "total_templates": 0, "high_risk_count": 0, "timestamp": "'$(date -Iseconds)'"}' > "$SUMMARY_FILE"

# Find all Bicep and ARM templates
TEMPLATES=$(find . -type f \( -name "*.bicep" -o -name "azuredeploy.json" -o -name "main.json" \) | grep -v node_modules)

for template in $TEMPLATES; do
    echo "Scanning: $template"

    # Generate output filename
    output_name=$(echo "$template" | sed 's/[\/\.]/_/g')
    output_file="$OUTPUT_DIR/${output_name}.json"

    # Run scan
    crux scan \
        --template "$template" \
        --model "$MODEL" \
        --rules "$RULES_DIR" \
        --output-format json \
        --output "$output_file" 2>/dev/null

    if [ -f "$output_file" ]; then
        # Extract risk score and update summary
        risk_score=$(jq -r '.summary.overall_risk_score' "$output_file")
        risk_level=$(jq -r '.summary.overall_risk_level' "$output_file")

        echo "  Risk: $risk_score ($risk_level)"

        # Add to summary
        jq --arg t "$template" --arg s "$risk_score" --arg l "$risk_level" \
           '.scans += [{"template": $t, "risk_score": ($s | tonumber), "risk_level": $l}] | .total_templates += 1' \
           "$SUMMARY_FILE" > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"

        if [ "$risk_level" = "HIGH" ] || [ "$risk_level" = "CRITICAL" ]; then
            jq '.high_risk_count += 1' "$SUMMARY_FILE" > "$SUMMARY_FILE.tmp" && mv "$SUMMARY_FILE.tmp" "$SUMMARY_FILE"
        fi
    else
        echo "  Failed to scan (compilation error?)"
    fi
done

echo ""
echo "Batch scan complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Print summary
echo "=== SUMMARY ==="
jq -r '"Total templates scanned: \(.total_templates)"' "$SUMMARY_FILE"
jq -r '"High/Critical risk templates: \(.high_risk_count)"' "$SUMMARY_FILE"
echo ""
echo "Top 5 highest risk templates:"
jq -r '.scans | sort_by(-.risk_score) | .[0:5][] | "  \(.risk_score | tostring | .[0:4]) (\(.risk_level)): \(.template)"' "$SUMMARY_FILE"
```

**Usage**:
```bash
chmod +x scripts/batch-scan.sh
./scripts/batch-scan.sh

# Output:
# Starting batch security scan...
# Output directory: scan-results/20250125-143022
# Scanning: ./infra/storage.bicep
#   Risk: 0.45 (MEDIUM)
# Scanning: ./infra/network.bicep
#   Risk: 0.72 (HIGH)
# ...
#
# === SUMMARY ===
# Total templates scanned: 15
# High/Critical risk templates: 3
#
# Top 5 highest risk templates:
#   0.85 (CRITICAL): ./infra/legacy/old-storage.bicep
#   0.72 (HIGH): ./infra/network.bicep
#   0.68 (HIGH): ./infra/keyvault.bicep
```

---

### Use Case 4: Exporting Live Azure Resources for Analysis

**Scenario**: Analyze existing deployed resources by exporting their configurations.

#### Step 1: Export Resources from Azure

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "Your-Subscription-ID"

# Option A: Export a specific resource group as ARM template
az group export \
    --name "my-resource-group" \
    --output-folder ./exported-templates \
    --include-parameter-default-value

# Option B: Export specific resource types
az resource list \
    --resource-group "my-resource-group" \
    --resource-type "Microsoft.Storage/storageAccounts" \
    --query "[].id" -o tsv | while read id; do
        name=$(basename "$id")
        az resource show --ids "$id" -o json > "./exported-templates/${name}.json"
    done

# Option C: Export all resources in subscription
az resource list --query "[].id" -o tsv | while read id; do
    name=$(echo "$id" | sed 's/[\/]/_/g')
    az resource show --ids "$id" -o json > "./exported-templates/${name}.json" 2>/dev/null
done
```

#### Step 2: Convert to Scannable Format

The exported ARM templates may need preprocessing:

```python
#!/usr/bin/env python3
# scripts/prepare_exported_resources.py

import json
import sys
from pathlib import Path

def prepare_arm_template(resource_json_path: str, output_path: str):
    """Convert exported resource JSON to ARM template format."""
    with open(resource_json_path) as f:
        resource = json.load(f)

    # Wrap single resource in ARM template structure
    arm_template = {
        "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
        "contentVersion": "1.0.0.0",
        "resources": [resource] if isinstance(resource, dict) else resource
    }

    with open(output_path, 'w') as f:
        json.dump(arm_template, f, indent=2)

    print(f"Prepared: {output_path}")

if __name__ == "__main__":
    input_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "./exported-templates")
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./prepared-templates")
    output_dir.mkdir(exist_ok=True)

    for json_file in input_dir.glob("*.json"):
        output_file = output_dir / f"{json_file.stem}_template.json"
        try:
            prepare_arm_template(str(json_file), str(output_file))
        except Exception as e:
            print(f"Failed to prepare {json_file}: {e}")
```

#### Step 3: Scan Exported Resources

```bash
# Prepare exported resources
python scripts/prepare_exported_resources.py ./exported-templates ./prepared-templates

# Scan all prepared templates
for template in ./prepared-templates/*.json; do
    echo "Scanning: $template"
    crux scan \
        --template "$template" \
        --model models/rf-baseline.pkl \
        --rules rules/ \
        --output-format text
    echo "---"
done
```

---

### Use Case 5: Compliance Dashboard Integration

**Scenario**: Feed scan results into a compliance dashboard or SIEM.

```python
#!/usr/bin/env python3
# scripts/compliance_export.py

"""
Export CRUX scan results to compliance dashboard formats.
Supports: Splunk, Azure Sentinel, Generic CSV
"""

import json
import csv
import sys
from datetime import datetime
from pathlib import Path

def export_to_splunk(scan_result: dict, output_file: str):
    """Export to Splunk-compatible JSON format (one event per line)."""
    base_event = {
        "timestamp": scan_result["metadata"]["scan_timestamp"],
        "source": "crux",
        "sourcetype": "crux:scan",
        "template": scan_result["metadata"]["template_path"],
        "overall_risk_score": scan_result["summary"]["overall_risk_score"],
        "overall_risk_level": scan_result["summary"]["overall_risk_level"],
    }

    with open(output_file, 'w') as f:
        # Overall scan event
        f.write(json.dumps(base_event) + '\n')

        # Individual finding events
        for finding in scan_result.get("findings", []):
            event = {
                **base_event,
                "event_type": "finding",
                "resource_id": finding["resource_id"],
                "resource_type": finding["resource_type"],
                "resource_name": finding["resource_name"],
                "risk_score": finding["risk_score"],
                "risk_level": finding["risk_level"],
                "rule_violations": finding.get("rule_violations", []),
                "model_prediction": finding.get("model_prediction", False),
            }
            f.write(json.dumps(event) + '\n')

def export_to_csv(scan_result: dict, output_file: str):
    """Export to CSV for generic dashboards."""
    fieldnames = [
        "timestamp", "template", "resource_id", "resource_type",
        "resource_name", "risk_score", "risk_level", "rule_violations",
        "model_confidence", "recommendations"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for finding in scan_result.get("findings", []):
            writer.writerow({
                "timestamp": scan_result["metadata"]["scan_timestamp"],
                "template": scan_result["metadata"]["template_path"],
                "resource_id": finding["resource_id"],
                "resource_type": finding["resource_type"],
                "resource_name": finding["resource_name"],
                "risk_score": finding["risk_score"],
                "risk_level": finding["risk_level"],
                "rule_violations": ";".join(finding.get("rule_violations", [])),
                "model_confidence": finding.get("model_confidence", ""),
                "recommendations": ";".join(finding.get("recommendations", [])),
            })

def export_to_azure_sentinel(scan_result: dict, output_file: str):
    """Export to Azure Sentinel custom log format."""
    events = []

    for finding in scan_result.get("findings", []):
        event = {
            "TimeGenerated": scan_result["metadata"]["scan_timestamp"],
            "SourceSystem": "CRUX",
            "Type": "CRUXSecurityScan_CL",
            "TemplateFile_s": scan_result["metadata"]["template_path"],
            "ResourceId_s": finding["resource_id"],
            "ResourceType_s": finding["resource_type"],
            "ResourceName_s": finding["resource_name"],
            "RiskScore_d": finding["risk_score"],
            "RiskLevel_s": finding["risk_level"],
            "RuleViolations_s": json.dumps(finding.get("rule_violations", [])),
            "ModelPrediction_b": finding.get("model_prediction", False),
            "ModelConfidence_d": finding.get("model_confidence", 0),
            "Recommendations_s": json.dumps(finding.get("recommendations", [])),
        }
        events.append(event)

    with open(output_file, 'w') as f:
        json.dump(events, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: compliance_export.py <scan_result.json> <output_file> <format>")
        print("Formats: splunk, csv, sentinel")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        scan_result = json.load(f)

    output_file = sys.argv[2]
    format_type = sys.argv[3].lower()

    if format_type == "splunk":
        export_to_splunk(scan_result, output_file)
    elif format_type == "csv":
        export_to_csv(scan_result, output_file)
    elif format_type == "sentinel":
        export_to_azure_sentinel(scan_result, output_file)
    else:
        print(f"Unknown format: {format_type}")
        sys.exit(1)

    print(f"Exported to {output_file} ({format_type} format)")
```

**Usage**:
```bash
# Run scan with JSON output
crux scan --template infra/main.bicep --model models/rf-baseline.pkl --output-format json --output scan.json

# Export to different formats
python scripts/compliance_export.py scan.json splunk-events.json splunk
python scripts/compliance_export.py scan.json findings.csv csv
python scripts/compliance_export.py scan.json sentinel-logs.json sentinel
```

---

## Understanding Scan Results

### Risk Score Interpretation

| Score Range | Level | Action Required |
|-------------|-------|-----------------|
| 0.0 - 0.39 | LOW | Monitor, no immediate action |
| 0.4 - 0.59 | MEDIUM | Review and plan remediation |
| 0.6 - 0.79 | HIGH | Prioritize fixes before deployment |
| 0.8 - 1.0 | CRITICAL | Block deployment, immediate remediation |

### Key Risk Factors (from Research)

Based on Phase I logistic regression analysis:

| Factor | Odds Ratio | Interpretation |
|--------|------------|----------------|
| **Encryption Disabled** | 0.07 | Strong protective factor when enabled |
| **Versioning Disabled** | 10.89 | High risk - 10x more likely to be misconfigured |
| **Network ACLs Missing** | 0.28 | Protective when enabled |
| **Soft Delete Disabled** | 3.2 | Elevated risk for data loss |
| **Public Access Enabled** | 5.4 | Significant exposure risk |

### Common Recommendations

The scanner provides context-aware recommendations:

**Storage Accounts**:
- Enable soft delete for blob data recovery
- Enforce HTTPS-only traffic
- Use TLS 1.2 or higher
- Disable public blob access
- Enable blob versioning

**Key Vaults**:
- Enable purge protection
- Enable soft delete
- Use RBAC for access control
- Enable audit logging

**Networks**:
- Enable DDoS protection for public-facing resources
- Configure Network Security Groups
- Use private endpoints where possible

---

## Threshold Configuration Guide

### Recommended Thresholds by Environment

| Environment | Threshold | Rationale |
|-------------|-----------|-----------|
| **Development** | 0.9 (or none) | Allow experimentation, warn only |
| **Staging** | 0.7 | Catch high-risk issues before production |
| **Production** | 0.5 | Strict security posture |
| **Regulated** | 0.4 | Maximum security for compliance |

### Custom Threshold Examples

```bash
# Development: warn but don't fail
crux scan --template dev.bicep --model models/rf-baseline.pkl --fail-threshold 0.95

# Staging: moderate strictness
crux scan --template staging.bicep --model models/rf-baseline.pkl --fail-threshold 0.7

# Production: strict
crux scan --template prod.bicep --model models/rf-baseline.pkl --fail-threshold 0.5

# Compliance audit: very strict
crux scan --template regulated.bicep --model models/rf-baseline.pkl --fail-threshold 0.4
```

---

## Troubleshooting

### Common Issues

**"Model file not found"**
```bash
# Verify model path
ls -la models/
# Re-download or retrain model
crux train-model --dataset dataset/exp-001 --model random-forest --output models --name rf-baseline
```

**"Bicep compilation failed"**
```bash
# Check Azure CLI installation
az bicep version
# Upgrade Bicep
az bicep upgrade
# Check template syntax
az bicep build --file template.bicep --stdout
```

**"No resources found in template"**
```bash
# Verify template has resources
cat template.bicep | grep -A5 "resource "
# For ARM JSON, check resources array
jq '.resources | length' template.json
```

**"Empty scan results"**
```bash
# Run with verbose output
crux scan --template template.bicep --model models/rf-baseline.pkl --output-format both
# Check if template compiles correctly
az bicep build --file template.bicep --stdout | jq '.resources | length'
```

---

## Best Practices

1. **Version your models**: Store trained models with version tags and track which model was used for each scan.

2. **Baseline scanning**: Run scans on your templates regularly, not just during CI/CD, to catch drift.

3. **Gradual rollout**: Start with high thresholds (0.9) and gradually lower as you remediate issues.

4. **Document exceptions**: If a finding is a false positive, document why and consider adding a rule exception.

5. **Update models periodically**: Retrain models as you add new mutation patterns or rules.

6. **Combine with other tools**: CRUX complements (doesn't replace) tools like Azure Policy, Defender for Cloud, and manual reviews.

---

## Next Steps

- **Train custom models**: Generate datasets from your own templates for organization-specific risk patterns
- **Add custom rules**: Extend `rules/` with your organization's security policies
- **Integrate with ticketing**: Auto-create Jira/ServiceNow tickets for high-risk findings
- **Build dashboards**: Use the compliance export scripts to feed Power BI, Grafana, or Splunk dashboards
