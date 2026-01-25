# Milestone 2.6: CI/CD Corpus Validation

## Overview

This milestone adds automated testing of the CRUX scanner against the test corpus as part of the GitHub CI/CD pipeline. Every PR and push to main will validate that the scanner correctly detects all known misconfigurations in the test corpus.

## Goals

1. **Regression Prevention**: Catch model/rule changes that break detection of known issues
2. **Continuous Validation**: Ensure scanner accuracy stays above threshold
3. **PR Gating**: Block merges if scanner fails to detect known issues
4. **Metrics Tracking**: Track precision/recall over time

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions Workflow                               │
└─────────────────────────────────────────────────────────────────────────────┘

  Push/PR
     │
     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Checkout   │────▶│   Install    │────▶│   Download   │
│   Code       │     │   CRUX       │     │   Model      │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Upload     │◀────│   Generate   │◀────│   Validate   │
│   Artifacts  │     │   Report     │     │   Corpus     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                                          Pass/Fail Gate
```

## Workflow Triggers

| Trigger | Action |
|---------|--------|
| Push to `main` | Full validation, update baseline metrics |
| Pull Request | Full validation, compare to baseline |
| Manual dispatch | Full validation with custom parameters |
| Scheduled (weekly) | Full validation, regression tracking |

## Implementation

### 1. GitHub Actions Workflow

```yaml
# .github/workflows/corpus-validation.yml
name: Test Corpus Validation

on:
  push:
    branches: [main]
    paths:
      - 'crux/**'
      - 'rules/**'
      - 'test-corpus/**'
  pull_request:
    paths:
      - 'crux/**'
      - 'rules/**'
      - 'test-corpus/**'
  workflow_dispatch:
    inputs:
      threshold:
        description: 'Minimum pass rate (0.0-1.0)'
        required: false
        default: '0.95'
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM UTC

jobs:
  validate-corpus:
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
        run: pip install -e .[dev]

      - name: Download baseline model
        run: |
          # Option A: From GitHub Releases
          gh release download latest --pattern 'rf-baseline.pkl' -D models/

          # Option B: From artifact storage
          # aws s3 cp s3://crux-models/rf-baseline.pkl models/

      - name: Run corpus validation
        id: validate
        run: |
          crux validate-corpus \
            --model models/rf-baseline.pkl \
            --rules rules/ \
            --corpus-dir test-corpus \
            --output validation-results.json

          # Extract metrics for summary
          PASS_RATE=$(jq -r '.summary.pass_rate' validation-results.json)
          echo "pass_rate=$PASS_RATE" >> $GITHUB_OUTPUT

      - name: Check pass rate threshold
        run: |
          THRESHOLD="${{ github.event.inputs.threshold || '0.95' }}"
          PASS_RATE="${{ steps.validate.outputs.pass_rate }}"

          if (( $(echo "$PASS_RATE < $THRESHOLD" | bc -l) )); then
            echo "::error::Corpus validation failed: pass rate $PASS_RATE < $THRESHOLD"
            exit 1
          fi

      - name: Upload validation results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: corpus-validation-results
          path: validation-results.json

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('validation-results.json', 'utf8'));

            const summary = results.summary;
            const status = summary.pass_rate >= 0.95 ? '✅' : '❌';

            const body = `## ${status} Corpus Validation Results

            | Metric | Value |
            |--------|-------|
            | Pass Rate | ${(summary.pass_rate * 100).toFixed(1)}% |
            | Total Cases | ${summary.total_cases} |
            | Passed | ${summary.passed_cases} |
            | Failed | ${summary.failed_cases} |
            | Precision | ${(results.metrics.precision * 100).toFixed(1)}% |
            | Recall | ${(results.metrics.recall * 100).toFixed(1)}% |
            | F1 Score | ${(results.metrics.f1_score * 100).toFixed(1)}% |

            <details>
            <summary>View Details</summary>

            \`\`\`json
            ${JSON.stringify(results.case_results, null, 2)}
            \`\`\`
            </details>`;

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body
            });
```

### 2. CLI Enhancements

Add `--min-pass-rate` flag to `validate-corpus`:

```bash
# Fail if pass rate drops below threshold
crux validate-corpus \
  --model models/rf-baseline.pkl \
  --corpus-dir test-corpus \
  --min-pass-rate 0.95 \
  --output results.json
```

### 3. Baseline Metrics Tracking

Store baseline metrics in repository:

```json
// test-corpus/baseline-metrics.json
{
  "last_updated": "2025-01-25",
  "model": "rf-baseline.pkl",
  "model_version": "1.0.0",
  "metrics": {
    "pass_rate": 0.98,
    "precision": 0.95,
    "recall": 0.92,
    "f1_score": 0.935
  },
  "total_cases": 15,
  "by_category": {
    "storage": {"pass_rate": 1.0, "cases": 5},
    "keyvault": {"pass_rate": 1.0, "cases": 3},
    "network": {"pass_rate": 0.9, "cases": 4},
    "compute": {"pass_rate": 1.0, "cases": 3}
  }
}
```

### 4. PR Validation Criteria

| Criteria | Threshold | Action |
|----------|-----------|--------|
| Pass rate | >= 95% | Block merge if below |
| Precision drop | <= 5% | Warn in PR comment |
| Recall drop | <= 5% | Warn in PR comment |
| New test case fails | Any | Block merge |

## Test Corpus Requirements

For CI validation to work, test cases must:

1. **Be self-contained**: Templates should compile without external dependencies
2. **Have clear expectations**: `expected.json` must specify exact labels expected
3. **Be deterministic**: Same template should always produce same results

### Minimal Test Case Structure

```
test-corpus/
├── storage/
│   └── public-blob-access/
│       ├── template.bicep      # Minimal, compilable template
│       └── expected.json       # Expected findings
├── manifest.json               # Corpus metadata
└── baseline-metrics.json       # Baseline for comparison
```

## Model Versioning

### Option A: GitHub Releases

```bash
# Upload model to GitHub Release
gh release create v1.0.0 models/rf-baseline.pkl --notes "Initial model release"

# Download in CI
gh release download v1.0.0 --pattern 'rf-baseline.pkl' -D models/
```

### Option B: Git LFS

```bash
# Track large model files with Git LFS
git lfs track "*.pkl"
git add .gitattributes
git add models/rf-baseline.pkl
git commit -m "Add trained model via LFS"
```

### Option C: Cloud Storage

```bash
# Store in S3/Azure Blob, download in CI
aws s3 cp models/rf-baseline.pkl s3://crux-models/rf-baseline.pkl
# In CI: aws s3 cp s3://crux-models/rf-baseline.pkl models/
```

## Success Criteria

1. **CI runs on every PR** touching crux/, rules/, or test-corpus/
2. **Pass rate >= 95%** required to merge
3. **PR comments** show validation summary
4. **Weekly scheduled runs** track regression over time
5. **Metrics artifacts** uploaded for historical analysis

## Implementation Plan

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Create GitHub Actions workflow | 1 day |
| 2 | Add `--min-pass-rate` to CLI | 0.5 day |
| 3 | Set up model storage (Releases/LFS) | 0.5 day |
| 4 | Add baseline metrics tracking | 0.5 day |
| 5 | Create lightweight test model for CI | 0.5 day |

**Total: ~3 days**

## Lightweight CI Model

For fast CI runs, create a minimal model trained on the test corpus:

```bash
# Generate small dataset from test corpus templates
crux generate-dataset \
  --templates test-corpus/ \
  --rules rules/ \
  --output dataset/ci-minimal \
  --name ci-model

# Train lightweight model
crux train-model \
  --dataset dataset/ci-minimal \
  --model random-forest \
  --output models \
  --name ci-rf-minimal \
  --max-features 50
```

This ensures:
- Fast CI runs (< 2 minutes)
- Model guaranteed to work with test corpus
- No external dependencies for model download
