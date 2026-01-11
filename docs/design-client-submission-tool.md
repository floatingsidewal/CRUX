# Client Submission Tool Design

## Overview

This document outlines the design for a client-facing tool that enables users to submit Azure ARM/Bicep templates for analysis against CRUX's trained ML models. The tool provides actionable insights about potential misconfigurations and security issues.

## Goals

1. **Easy Template Submission**: Support multiple input formats (Bicep, ARM JSON, URLs)
2. **ML-Powered Analysis**: Leverage trained models for misconfiguration detection
3. **Actionable Insights**: Return prioritized findings with remediation guidance
4. **CI/CD Integration**: Enable automated scanning in deployment pipelines
5. **Community Contributions**: Allow users to contribute templates for model improvement

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Interfaces                              │
├──────────────────┬──────────────────┬──────────────────┬────────────────┤
│    CLI Tool      │   REST API       │  Python SDK      │  CI/CD Plugins │
│  crux analyze    │  /api/v1/scan    │  CruxClient()    │  GitHub Action │
└────────┬─────────┴────────┬─────────┴────────┬─────────┴───────┬────────┘
         │                  │                  │                 │
         └──────────────────┼──────────────────┼─────────────────┘
                            │                  │
                            ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Analysis Engine                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Template  │  │  Resource   │  │   Feature    │  │     Model      │  │
│  │  Parser    │──▶  Extractor  │──▶  Extractor   │──▶   Predictor    │  │
│  └────────────┘  └─────────────┘  └──────────────┘  └────────────────┘  │
│         │                                                    │          │
│         ▼                                                    ▼          │
│  ┌────────────┐                                    ┌────────────────┐   │
│  │   Rule     │                                    │    Report      │   │
│  │ Evaluator  │───────────────────────────────────▶│   Generator    │   │
│  └────────────┘                                    └────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Model Registry                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ RandomForest │  │   XGBoost    │  │   GNN/GAT    │  ...              │
│  │    v1.0      │  │    v1.0      │  │    v1.0      │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Template Parser
- Accepts Bicep (`.bicep`) or ARM JSON (`.json`) files
- Compiles Bicep to ARM using existing `BicepCompiler`
- Validates template structure before processing
- Supports batch processing of multiple templates

#### 2. Resource Extractor
- Uses existing `ResourceExtractor` to parse ARM templates
- Extracts resource properties, dependencies, and metadata
- Generates simulated resource IDs for tracking

#### 3. Feature Extractor
- Transforms resources to numerical features using `FeatureExtractor`
- Supports both hashed features (for ML) and named properties (for reports)
- Maintains compatibility with trained model expectations

#### 4. Rule Evaluator
- Applies YAML-defined rules from `rules/` directory
- Provides deterministic baseline for known patterns
- Complements ML predictions with rule-based detection

#### 5. Model Predictor
- Loads trained models from registry
- Supports ensemble predictions from multiple models
- Returns probabilities and confidence scores

#### 6. Report Generator
- Aggregates findings from rules and ML predictions
- Prioritizes issues by severity and confidence
- Generates remediation recommendations

---

## CLI Design

### New Command: `crux analyze`

```bash
# Analyze a single template
crux analyze template.bicep

# Analyze multiple templates
crux analyze *.bicep

# Analyze with specific model
crux analyze template.bicep --model models/rf-v1.pkl

# Output formats
crux analyze template.bicep --format json
crux analyze template.bicep --format table
crux analyze template.bicep --format sarif  # For GitHub integration

# Severity filtering
crux analyze template.bicep --min-severity high

# CI mode (non-zero exit on findings)
crux analyze template.bicep --ci --fail-on high

# Verbose output with remediation
crux analyze template.bicep --verbose
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model PATH` | Path to trained model | Uses bundled default |
| `--format FORMAT` | Output format: `json`, `table`, `sarif`, `markdown` | `table` |
| `--min-severity LEVEL` | Filter by severity: `low`, `medium`, `high`, `critical` | `low` |
| `--ci` | CI/CD mode: structured output, deterministic | `false` |
| `--fail-on LEVEL` | Exit non-zero if findings at this level or above | None |
| `--verbose` | Include remediation guidance | `false` |
| `--output PATH` | Write output to file | stdout |
| `--rules-only` | Use only rule-based detection (no ML) | `false` |
| `--include-passing` | Include resources with no findings | `false` |

### Example Output

```
$ crux analyze storage-account.bicep

CRUX Template Analysis Report
═════════════════════════════════════════════════════════════════════

Template: storage-account.bicep
Resources Analyzed: 3
Issues Found: 2

┌────────────────────────────────────────────────────────────────────┐
│ CRITICAL: Storage Account - Public Blob Access Enabled            │
├────────────────────────────────────────────────────────────────────┤
│ Resource: storageAccount (Microsoft.Storage/storageAccounts)       │
│ Property: properties.allowBlobPublicAccess = true                  │
│                                                                    │
│ Detection:                                                         │
│   • Rule: storage-public-blob-access (CIS 3.7)                    │
│   • Model: RandomForest (confidence: 94.2%)                        │
│                                                                    │
│ Remediation:                                                       │
│   Set allowBlobPublicAccess to false to prevent anonymous access   │
│   to blob containers.                                              │
│                                                                    │
│   properties:                                                      │
│     allowBlobPublicAccess: false  # ← Recommended                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ HIGH: Storage Account - Weak TLS Version                           │
├────────────────────────────────────────────────────────────────────┤
│ Resource: storageAccount (Microsoft.Storage/storageAccounts)       │
│ Property: properties.minimumTlsVersion = TLS1_0                    │
│                                                                    │
│ Detection:                                                         │
│   • Model: RandomForest (confidence: 87.5%)                        │
│                                                                    │
│ Remediation:                                                       │
│   Set minimumTlsVersion to TLS1_2 for secure connections.          │
│                                                                    │
│   properties:                                                      │
│     minimumTlsVersion: 'TLS1_2'  # ← Recommended                   │
└────────────────────────────────────────────────────────────────────┘

Summary: 1 critical, 1 high, 0 medium, 0 low
```

---

## Python SDK Design

### Installation

```bash
pip install crux-analyzer
# or
pip install crux[client]
```

### Basic Usage

```python
from crux.client import CruxAnalyzer

# Initialize analyzer
analyzer = CruxAnalyzer()

# Analyze a template
results = analyzer.analyze("path/to/template.bicep")

# Access findings
for finding in results.findings:
    print(f"{finding.severity}: {finding.title}")
    print(f"  Resource: {finding.resource_id}")
    print(f"  Confidence: {finding.confidence:.1%}")
    print(f"  Remediation: {finding.remediation}")
```

### Advanced Usage

```python
from crux.client import CruxAnalyzer, AnalysisConfig

# Custom configuration
config = AnalysisConfig(
    models=["models/rf-v1.pkl", "models/xgb-v1.pkl"],
    min_severity="medium",
    ensemble_strategy="majority",  # or "any", "all", "weighted"
    include_rules=True,
)

analyzer = CruxAnalyzer(config)

# Batch analysis
templates = ["vm.bicep", "network.bicep", "storage.bicep"]
batch_results = analyzer.analyze_batch(templates)

# Export results
batch_results.to_json("findings.json")
batch_results.to_sarif("findings.sarif")
batch_results.to_csv("findings.csv")
```

### SDK Classes

```python
@dataclass
class AnalysisResult:
    """Result of analyzing a single template."""
    template_path: str
    timestamp: datetime
    resources: List[ResourceAnalysis]
    findings: List[Finding]
    summary: AnalysisSummary

    def to_json(self) -> str: ...
    def to_dict(self) -> dict: ...
    def has_findings(self, min_severity: str = "low") -> bool: ...

@dataclass
class Finding:
    """A single misconfiguration finding."""
    id: str
    title: str
    severity: str  # critical, high, medium, low
    resource_id: str
    resource_type: str
    property_path: str
    current_value: Any

    # Detection metadata
    detected_by: List[str]  # ["rule:storage-public-access", "model:rf-v1"]
    confidence: float  # 0.0 - 1.0
    labels: List[str]  # ["Storage_PublicAccess", "CIS_3.7"]
    cis_references: List[str]

    # Remediation
    remediation: str
    recommended_value: Any

@dataclass
class AnalysisSummary:
    """Summary statistics for analysis."""
    total_resources: int
    resources_with_findings: int
    findings_by_severity: Dict[str, int]
    models_used: List[str]
    rules_evaluated: int
```

---

## REST API Design

### Endpoints

#### POST /api/v1/analyze

Analyze a template and return findings.

**Request:**
```json
{
  "template": "<base64-encoded template content>",
  "filename": "main.bicep",
  "options": {
    "models": ["rf-v1", "xgb-v1"],
    "min_severity": "medium",
    "include_remediation": true
  }
}
```

**Response:**
```json
{
  "id": "analysis-abc123",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "template": {
    "filename": "main.bicep",
    "hash": "sha256:abc123..."
  },
  "summary": {
    "total_resources": 5,
    "resources_with_findings": 2,
    "findings": {
      "critical": 1,
      "high": 1,
      "medium": 0,
      "low": 0
    }
  },
  "resources": [
    {
      "id": "/subscriptions/.../storageAccounts/myaccount",
      "type": "Microsoft.Storage/storageAccounts",
      "name": "myaccount",
      "findings": [
        {
          "id": "finding-001",
          "title": "Public Blob Access Enabled",
          "severity": "critical",
          "property": "properties.allowBlobPublicAccess",
          "current_value": true,
          "detected_by": [
            {"type": "rule", "id": "storage-public-blob-access"},
            {"type": "model", "id": "rf-v1", "confidence": 0.942}
          ],
          "labels": ["Storage_PublicAccess", "CIS_3.7"],
          "remediation": {
            "description": "Disable public blob access",
            "recommended_value": false,
            "code_snippet": "properties:\n  allowBlobPublicAccess: false"
          }
        }
      ]
    }
  ],
  "models_used": [
    {"id": "rf-v1", "version": "1.0.0", "trained_on": "2024-01-01"}
  ]
}
```

#### POST /api/v1/analyze/batch

Analyze multiple templates in a single request.

#### GET /api/v1/models

List available models and their metadata.

#### POST /api/v1/contribute

Submit a template for community contribution (see Community section).

---

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/template-scan.yml
name: CRUX Template Scan

on:
  pull_request:
    paths:
      - '**/*.bicep'
      - '**/*.json'

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install CRUX
        run: pip install crux-analyzer

      - name: Scan templates
        run: |
          crux analyze **/*.bicep \
            --format sarif \
            --output results.sarif \
            --ci \
            --fail-on high

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### Azure DevOps Pipeline

```yaml
# azure-pipelines.yml
trigger:
  paths:
    include:
      - '**/*.bicep'

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'

  - script: pip install crux-analyzer
    displayName: 'Install CRUX'

  - script: |
      crux analyze **/*.bicep \
        --format json \
        --output $(Build.ArtifactStagingDirectory)/crux-results.json \
        --ci
    displayName: 'Scan Templates'
    continueOnError: true

  - task: PublishBuildArtifacts@1
    inputs:
      pathToPublish: $(Build.ArtifactStagingDirectory)/crux-results.json
      artifactName: 'crux-scan-results'
```

### GitLab CI

```yaml
# .gitlab-ci.yml
crux-scan:
  image: python:3.11
  stage: test
  script:
    - pip install crux-analyzer
    - crux analyze **/*.bicep --format json --output gl-code-quality-report.json --ci
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
  rules:
    - changes:
        - "**/*.bicep"
        - "**/*.json"
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: crux-scan
        name: CRUX Template Scan
        entry: crux analyze
        language: python
        types: [file]
        files: \.(bicep|json)$
        args: ['--fail-on', 'high', '--ci']
```

---

## SARIF Output Format

For integration with GitHub Code Scanning and other security tools:

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "CRUX",
          "version": "1.0.0",
          "informationUri": "https://github.com/your-org/crux",
          "rules": [
            {
              "id": "storage-public-blob-access",
              "name": "PublicBlobAccess",
              "shortDescription": {
                "text": "Storage account allows public blob access"
              },
              "fullDescription": {
                "text": "Azure Storage accounts should have public blob access disabled to prevent anonymous access to data."
              },
              "help": {
                "text": "Set allowBlobPublicAccess to false",
                "markdown": "Set `allowBlobPublicAccess` to `false` in the storage account properties."
              },
              "defaultConfiguration": {
                "level": "error"
              },
              "properties": {
                "tags": ["security", "azure", "storage", "CIS_3.7"],
                "precision": "high"
              }
            }
          ]
        }
      },
      "results": [
        {
          "ruleId": "storage-public-blob-access",
          "level": "error",
          "message": {
            "text": "Storage account 'myaccount' has public blob access enabled"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": {
                  "uri": "storage.bicep"
                },
                "region": {
                  "startLine": 15,
                  "startColumn": 5
                }
              }
            }
          ],
          "properties": {
            "confidence": 0.942,
            "detectedBy": ["rule", "model:rf-v1"]
          }
        }
      ]
    }
  ]
}
```

---

## Community Contribution Workflow

### Template Contribution Flow

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│    User      │     │  CRUX Server  │     │   Dataset    │
│  Submits     │────▶│  Validates    │────▶│   Queue      │
│  Template    │     │  & Anonymizes │     │              │
└──────────────┘     └───────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Updated    │◀────│   Retrain     │◀────│   Review     │
│    Model     │     │   Pipeline    │     │   Process    │
└──────────────┘     └───────────────┘     └──────────────┘
```

### Contribution CLI

```bash
# Opt-in to contribute analyzed templates
crux analyze template.bicep --contribute

# Contribute with explicit consent
crux contribute template.bicep \
  --consent \
  --anonymize  # Remove subscription IDs, names, etc.

# Check contribution status
crux contribution-status
```

### Privacy & Security

1. **Anonymization**:
   - Remove subscription IDs, resource group names
   - Obfuscate resource names (preserve structure)
   - Strip comments and tags containing PII
   - Preserve only structural and security-relevant properties

2. **Consent**:
   - Explicit opt-in required
   - Clear data usage policy
   - Option to withdraw contributions

3. **Review Process**:
   - Automated security scan before inclusion
   - Manual review for sensitive patterns
   - Community moderation for quality

### Contribution API

```json
POST /api/v1/contribute
{
  "template": "<anonymized-base64-content>",
  "metadata": {
    "contributor_id": "anonymous-hash-123",
    "consent_version": "1.0",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "labels": {
    "user_provided": ["Storage_PublicAccess"],
    "analysis_result": ["Storage_PublicAccess", "CIS_3.7"]
  }
}
```

---

## Implementation Modules

### New Module: `crux/client/`

```
crux/client/
├── __init__.py
├── analyzer.py       # Main CruxAnalyzer class
├── config.py         # AnalysisConfig, options
├── findings.py       # Finding, AnalysisResult classes
├── formatters/
│   ├── __init__.py
│   ├── table.py      # Table output formatter
│   ├── json.py       # JSON formatter
│   ├── sarif.py      # SARIF formatter
│   └── markdown.py   # Markdown formatter
├── remediation.py    # Remediation suggestions
└── contrib.py        # Community contribution logic
```

### Integration with Existing Code

```python
# crux/client/analyzer.py
from crux.templates.compiler import BicepCompiler
from crux.templates.extractor import ResourceExtractor
from crux.rules.evaluator import RuleEvaluator
from crux.ml.features import FeatureExtractor
from crux.ml.models import BaselineModel

class CruxAnalyzer:
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.compiler = BicepCompiler()
        self.extractor = ResourceExtractor()
        self.rule_evaluator = RuleEvaluator()
        self._load_models()

    def analyze(self, template_path: str) -> AnalysisResult:
        # 1. Compile if Bicep
        if template_path.endswith('.bicep'):
            arm_json = self.compiler.compile(template_path)
        else:
            arm_json = self._load_json(template_path)

        # 2. Extract resources
        resources = self.extractor.extract_resources(arm_json)

        # 3. Run rule-based detection
        rule_findings = self._evaluate_rules(resources)

        # 4. Run ML prediction
        ml_findings = self._predict(resources)

        # 5. Merge and deduplicate findings
        findings = self._merge_findings(rule_findings, ml_findings)

        # 6. Generate report
        return self._build_result(template_path, resources, findings)
```

---

## Configuration

### Model Registry Configuration

```yaml
# crux-config.yaml (or ~/.crux/config.yaml)
models:
  default: "rf-v1"
  available:
    - id: "rf-v1"
      path: "~/.crux/models/rf-v1.pkl"
      features: "~/.crux/models/rf-v1_features.pkl"
      type: "random-forest"
      version: "1.0.0"

    - id: "xgb-v1"
      path: "~/.crux/models/xgb-v1.pkl"
      features: "~/.crux/models/xgb-v1_features.pkl"
      type: "xgboost"
      version: "1.0.0"

rules:
  paths:
    - "~/.crux/rules/"
    - "./rules/"  # Local overrides

output:
  default_format: "table"
  colors: true

contribution:
  enabled: false
  anonymize: true
  endpoint: "https://api.crux-project.org/contribute"
```

---

## Phased Implementation Plan

### Phase 1: CLI Enhancement
- Add `crux analyze` command
- Implement table and JSON formatters
- Basic single-model prediction
- Rule-based detection integration

### Phase 2: Output Formats & CI/CD
- SARIF output format
- Markdown format for reports
- GitHub Actions example
- Azure DevOps pipeline example
- Exit code handling for CI

### Phase 3: Python SDK
- `CruxAnalyzer` class
- Batch processing
- Ensemble predictions
- Programmatic access

### Phase 4: REST API (Optional)
- FastAPI-based server
- Authentication/rate limiting
- Async processing for large batches

### Phase 5: Community Features
- Anonymization pipeline
- Contribution CLI
- Review dashboard (web UI)

---

## Success Metrics

1. **Adoption**:
   - Number of templates analyzed per month
   - CI/CD pipeline integrations
   - SDK downloads

2. **Quality**:
   - False positive rate (user feedback)
   - Detection coverage vs. known issues
   - Time to analyze (performance)

3. **Community**:
   - Templates contributed
   - Model accuracy improvement over time
   - Community engagement

---

## Security Considerations

1. **Template Content**: Never log or store sensitive template content
2. **Model Integrity**: Sign models, verify checksums before loading
3. **API Security**: Rate limiting, authentication for production API
4. **Contribution Privacy**: Strict anonymization, consent tracking
5. **Dependency Security**: Regular dependency audits

---

## Appendix: Example Finding Remediation Database

```yaml
# remediations/storage.yaml
remediations:
  Storage_PublicAccess:
    title: "Disable Public Blob Access"
    description: |
      Public blob access allows anonymous users to read blob data.
      This can lead to data exposure if containers are misconfigured.
    steps:
      - "Set allowBlobPublicAccess to false"
      - "Review container access policies"
      - "Use SAS tokens for legitimate anonymous access"
    code:
      bicep: |
        properties: {
          allowBlobPublicAccess: false
        }
      arm: |
        "properties": {
          "allowBlobPublicAccess": false
        }
    references:
      - "https://docs.microsoft.com/azure/storage/blobs/anonymous-read-access-configure"
      - "CIS Azure Benchmark 3.7"
```
