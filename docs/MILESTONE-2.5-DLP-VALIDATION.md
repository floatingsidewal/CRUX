# Milestone 2.5: Data Leak Prevention for Sample Submissions

## Overview

This milestone adds a Data Leak Prevention (DLP) validation layer to the sample submission system to prevent accidental exposure of PII, business-specific data, or sensitive information when engineers submit templates.

**Goal**: Ensure submissions contain only configuration/settings patterns, not business-specific data.

## Problem Statement

When engineers submit templates with known misconfigurations:
- Templates may contain **PII** (names, emails, phone numbers in comments/params)
- Templates may contain **business data** (internal project names, customer identifiers)
- Templates may contain **secrets** (even after sanitization, patterns may leak)
- Templates may contain **internal URLs** (corporate domains, internal APIs)

Current sanitization handles known patterns (subscription IDs, passwords) but may miss:
- PII in resource names (`storageaccount-john-smith`)
- PII in comments (`// Created by John Smith, john@company.com`)
- Business identifiers (`customer-acme-corp-storage`)
- Internal hostnames (`api.internal.company.com`)

## Microsoft Presidio Evaluation

### What is Presidio?

[Microsoft Presidio](https://github.com/microsoft/presidio) is an open-source framework for detecting and anonymizing PII. It uses:
- **Regex patterns** for structured data (SSN, credit cards, phone numbers)
- **NLP/NER** (spaCy) for unstructured data (names, locations, organizations)
- **Context-aware detection** for improved accuracy

### Supported Entity Types

Presidio detects 30+ entity types out of the box:

| Category | Entity Types |
|----------|--------------|
| **Personal** | PERSON, EMAIL_ADDRESS, PHONE_NUMBER, DATE_OF_BIRTH |
| **Financial** | CREDIT_CARD, IBAN_CODE, US_BANK_NUMBER, CRYPTO |
| **Government** | US_SSN, US_PASSPORT, UK_NHS, AU_ABN, etc. |
| **Location** | LOCATION, IP_ADDRESS, URL |
| **Medical** | MEDICAL_LICENSE, US_DRIVER_LICENSE |
| **Technical** | IP_ADDRESS, URL, DOMAIN_NAME |

### Fit Assessment

| Requirement | Presidio Support | Notes |
|-------------|------------------|-------|
| Detect names in text | ✅ Strong | NER-based PERSON detection |
| Detect emails | ✅ Strong | Regex + context |
| Detect phone numbers | ✅ Strong | Multi-format support |
| Detect internal URLs | ⚠️ Partial | URL detection, needs custom patterns for internal domains |
| Detect business identifiers | ⚠️ Partial | ORGANIZATION entity, but may miss custom patterns |
| Detect Azure-specific secrets | ❌ Needs extension | Custom recognizers needed |
| Lightweight/fast | ✅ Good | Sub-second analysis for template-sized text |
| Python native | ✅ Excellent | Pure Python, pip installable |

### Verdict: **Recommended with Extensions**

Presidio provides a solid foundation. We need to add:
1. **Custom recognizers** for Azure-specific patterns
2. **Domain blocklist** for internal corporate URLs
3. **Template-aware parsing** to focus on relevant sections

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Sample Submission Pipeline                        │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐
  │   Template   │────▶│   Existing   │────▶│     DLP      │────▶│  Submit  │
  │    Input     │     │  Sanitizer   │     │  Validator   │     │  or Warn │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────────┘
                              │                    │
                              │                    │
                       Removes known         Detects remaining
                       patterns:             sensitive data:
                       - Subscription IDs    - PII (names, emails)
                       - Passwords           - Business identifiers
                       - Keys                - Internal URLs
                       - SAS tokens          - Custom patterns
```

### Validation Levels

```python
class DLPValidationLevel(Enum):
    STRICT = "strict"      # Block submission if ANY PII detected
    MODERATE = "moderate"  # Warn but allow with confirmation
    PERMISSIVE = "permissive"  # Warn only, always allow
```

### Components

#### 1. Presidio Integration (`crux/corpus/dlp.py`)

```python
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine

class TemplateDLPValidator:
    """Validates templates for PII/sensitive data leakage."""

    def __init__(self, custom_patterns: List[str] = None):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Add custom recognizers for Azure patterns
        self._add_azure_recognizers()

        # Add organization-specific patterns
        if custom_patterns:
            self._add_custom_recognizers(custom_patterns)

    def validate(self, content: str) -> DLPValidationResult:
        """
        Validate template content for sensitive data.

        Returns:
            DLPValidationResult with findings and recommendations
        """
        ...

    def get_safe_content(self, content: str) -> str:
        """
        Return content with detected PII redacted.
        """
        ...
```

#### 2. Custom Recognizers

```python
# Azure-specific patterns to detect
AZURE_PATTERNS = {
    "AZURE_STORAGE_KEY": r"[A-Za-z0-9+/]{86}==",  # 86 chars + ==
    "AZURE_SAS_TOKEN": r"\?sv=\d{4}-\d{2}-\d{2}&[^'\"\s]+",
    "AZURE_CONNECTION_STRING": r"(DefaultEndpoints|AccountName|AccountKey)=[^;]+;",
    "AZURE_TENANT_ID": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
}

# Organization-specific patterns (configurable)
INTERNAL_PATTERNS = {
    "INTERNAL_DOMAIN": r"\b[\w-]+\.(internal|corp|local)\.([\w-]+\.)*[a-z]{2,}\b",
    "EMPLOYEE_ID": r"\b[A-Z]{2,3}\d{5,8}\b",  # Example: EMP12345
}
```

#### 3. DLP Validation Result

```python
@dataclass
class DLPFinding:
    entity_type: str       # e.g., "PERSON", "EMAIL_ADDRESS"
    text: str              # The detected text (redacted in logs)
    start: int             # Start position
    end: int               # End position
    score: float           # Confidence score (0-1)
    context: str           # Surrounding context (redacted)
    recommendation: str    # What to do about it

@dataclass
class DLPValidationResult:
    is_safe: bool                    # No high-confidence PII found
    findings: List[DLPFinding]       # All detected items
    risk_score: float                # 0-1 aggregate risk
    blocking_findings: List[DLPFinding]  # High-confidence findings
    warning_findings: List[DLPFinding]   # Lower-confidence findings
    safe_content: str                # Content with PII redacted
```

## Implementation Plan

### Phase 1: Core Integration (MVP)

1. **Add Presidio dependency**
   ```toml
   # pyproject.toml
   [project.optional-dependencies]
   dlp = ["presidio-analyzer>=2.2", "presidio-anonymizer>=2.2"]
   ```

2. **Create DLP validator module** (`crux/corpus/dlp.py`)
   - Presidio engine initialization
   - Standard entity detection
   - Result formatting

3. **Integrate with submission pipeline**
   - Add `--validate-dlp` flag to `submit-sample`
   - Add `--dlp-level` option (strict/moderate/permissive)

4. **Update CLI output**
   ```
   $ crux submit-sample --template storage.bicep --validate-dlp ...

   [DLP] Scanning template for sensitive data...
   [DLP] Found 2 potential issues:
     - PERSON detected at line 5: "j**n s***h" (score: 0.85)
     - EMAIL_ADDRESS detected at line 12: "j***@e*****.com" (score: 0.95)

   [WARN] Template contains potential PII. Options:
     1. Edit template to remove PII and resubmit
     2. Use --dlp-level permissive to submit anyway
     3. Use --force to skip DLP validation (not recommended)
   ```

### Phase 2: Azure-Specific Recognizers

1. **Azure credential patterns**
   - Storage account keys
   - SAS tokens
   - Connection strings
   - Managed identity IDs

2. **Azure resource name patterns**
   - Resource group naming conventions
   - Subscription-specific prefixes

### Phase 3: Organization Configuration

1. **Configuration file** (`dlp-config.yaml`)
   ```yaml
   # Custom patterns to detect
   custom_patterns:
     - name: "INTERNAL_PROJECT"
       pattern: "(project|prj)-(alpha|beta|gamma|delta)-\d+"
       score: 0.9

     - name: "EMPLOYEE_EMAIL"
       pattern: "[a-z]+\.[a-z]+@mycompany\.com"
       score: 0.95

   # Domains to flag
   blocked_domains:
     - "internal.mycompany.com"
     - "*.corp.mycompany.com"

   # Allowlist (don't flag these)
   allowlist:
     - "example.com"
     - "contoso.com"  # Microsoft example domain
     - "fabrikam.com"
   ```

2. **CLI support**
   ```bash
   crux submit-sample \
     --template storage.bicep \
     --dlp-config dlp-config.yaml \
     ...
   ```

### Phase 4: Batch Validation

1. **Validate existing corpus**
   ```bash
   crux validate-dlp --corpus-dir test-corpus --output dlp-report.json
   ```

2. **Pre-commit hook integration**
   ```bash
   # .git/hooks/pre-commit
   crux validate-dlp --path . --fail-on-pii
   ```

## What Should NOT Be Flagged

To avoid false positives, we should allowlist:

| Pattern | Reason |
|---------|--------|
| `example.com`, `contoso.com` | Microsoft documentation domains |
| `eastus`, `westus2`, etc. | Azure region names |
| `Standard_LRS`, `Premium_LRS` | Azure SKU names |
| `Microsoft.*` resource types | Azure resource type identifiers |
| `00000000-0000-...` | Already-sanitized GUIDs |
| Parameter/variable names | `storageAccountName`, `location` |

## What SHOULD Be Flagged

| Pattern | Risk Level | Example |
|---------|------------|---------|
| Real person names | HIGH | `// Created by John Smith` |
| Real email addresses | HIGH | `admin@realcompany.com` |
| Phone numbers | HIGH | `+1-555-123-4567` |
| Internal URLs | MEDIUM | `api.internal.corp.com` |
| Customer identifiers | MEDIUM | `customer-acme-corp` |
| Project codenames | LOW | `project-manhattan` |
| Employee IDs | MEDIUM | `EMP12345` |

## Success Criteria

1. **Detection Rate**: Catch >95% of PII in test templates
2. **False Positive Rate**: <5% false positives on clean templates
3. **Performance**: <2 seconds validation for typical template
4. **User Experience**: Clear, actionable warnings

## Dependencies

```toml
[project.optional-dependencies]
dlp = [
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
    "spacy>=3.4.0",  # For NER
]
```

Note: spaCy model download required:
```bash
python -m spacy download en_core_web_lg
```

## Timeline Estimate

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 1 | 2-3 days | Core Presidio integration |
| Phase 2 | 1-2 days | Azure-specific recognizers |
| Phase 3 | 1-2 days | Organization configuration |
| Phase 4 | 1 day | Batch validation |

**Total: ~5-8 days**

## References

- [Microsoft Presidio GitHub](https://github.com/microsoft/presidio)
- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [Presidio Analyzer PyPI](https://pypi.org/project/presidio-analyzer/)
- [Custom Recognizers Guide](https://microsoft.github.io/presidio/analyzer/adding_recognizers/)
