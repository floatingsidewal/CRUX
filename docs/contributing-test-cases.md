# Contributing Test Cases to CRUX

This guide explains how to submit templates with known misconfigurations to help improve CRUX's detection capabilities.

## Why Contribute?

Your real-world examples help us:
- **Validate detection accuracy** - Ensure the scanner catches real issues
- **Improve the model** - Train on patterns from production environments
- **Cover edge cases** - Identify issues synthetic mutations might miss
- **Build comprehensive benchmarks** - Measure improvement over time

## Quick Start

```bash
# Submit a template with known issues
crux submit-sample \
  --template path/to/problematic-storage.bicep \
  --issue-types "Storage_PublicAccess,Storage_NoHttps" \
  --severity high \
  --description "Storage account found with public access during Q4 security audit"

# List available issue types
crux list-issue-types

# Check your submissions
crux list-submissions --status pending
```

## Submission Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Engineer   │────▶│   Submit    │────▶│   Review    │────▶│   Merge     │
│  Finds      │     │   Sample    │     │   Queue     │     │   to        │
│  Issue      │     │             │     │             │     │   Corpus    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Real-world          Anonymized          Validated           Training
  template            & labeled           by team             data
```

## Step-by-Step Guide

### 1. Identify an Issue

Find a template with a known misconfiguration:
- **Security audit finding** - Issues discovered during compliance reviews
- **Incident post-mortem** - Templates that caused security incidents
- **Manual review** - Misconfigurations spotted during code review
- **Existing scan results** - Templates flagged by other tools

### 2. Prepare the Template

You can submit either:

**Option A: The actual template (will be anonymized)**
```bash
crux submit-sample \
  --template infra/storage.bicep \
  --issue-types "Storage_PublicAccess" \
  --severity high \
  --description "Production storage with public access" \
  --environment production \
  --discovery-method audit
```

**Option B: A minimal reproduction**
Create a focused template that reproduces just the issue:

```bicep
// minimal-public-storage.bicep
// Minimal reproduction of public storage account issue

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'exampleStorage'
  location: 'eastus'
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: true  // THE ISSUE
  }
}
```

Then submit:
```bash
crux submit-sample \
  --template minimal-public-storage.bicep \
  --issue-types "Storage_PublicAccess" \
  --severity high \
  --description "Minimal repro: storage with public blob access" \
  --no-anonymize  # Already sanitized
```

### 3. Choose Issue Types

Use `crux list-issue-types` to see available types:

**Storage Issues**
| Issue Type | Description |
|------------|-------------|
| `Storage_PublicAccess` | Public blob access enabled |
| `Storage_NoHttps` | HTTPS-only not enforced |
| `Storage_WeakTLS` | TLS version below 1.2 |
| `Storage_NoEncryption` | Encryption not enabled |
| `Storage_NoSoftDelete` | Soft delete not enabled |
| `Storage_NoVersioning` | Blob versioning not enabled |

**Key Vault Issues**
| Issue Type | Description |
|------------|-------------|
| `KeyVault_NoPurgeProtection` | Purge protection disabled |
| `KeyVault_NoSoftDelete` | Soft delete disabled |
| `KeyVault_NoRBAC` | Using access policies instead of RBAC |
| `KeyVault_PublicNetwork` | Public network access enabled |

**Network Issues**
| Issue Type | Description |
|------------|-------------|
| `Network_NoDDoS` | DDoS protection disabled |
| `Network_NoNSG` | No Network Security Group |
| `Network_OpenPorts` | Dangerous ports open to internet |
| `Network_NoPrivateEndpoint` | Using public endpoints |

**Compute Issues**
| Issue Type | Description |
|------------|-------------|
| `VM_PasswordAuth` | Password auth instead of SSH keys |
| `VM_NoEncryption` | Disk encryption disabled |
| `VM_PublicIP` | Public IP directly attached |

**Database Issues**
| Issue Type | Description |
|------------|-------------|
| `SQL_PublicAccess` | Public network access enabled |
| `SQL_NoAuditing` | Auditing not enabled |
| `SQL_NoTDE` | Transparent Data Encryption disabled |

**Custom Issues**
For issues not in the list:
```bash
crux submit-sample \
  --issue-types "Custom_MyNewIssue" \
  ...
```

### 4. Add Context

The more context you provide, the more valuable the submission:

```bash
crux submit-sample \
  --template problematic.bicep \
  --issue-types "Storage_PublicAccess,Storage_NoVersioning" \
  --severity high \
  --description "Found during incident response - public bucket exposed customer PII for 2 weeks" \
  --environment production \
  --discovery-method incident \
  --cis-references "3.7,3.10" \
  --submitted-by "security-team@company.com"
```

### 5. Verify Submission

```bash
# Check your submission was recorded
crux list-submissions --status pending

# View submission details
cat test-corpus/submissions/SUB-XXXXXX/submission.json
```

## Anonymization

By default, CRUX automatically sanitizes templates:

**What gets anonymized:**
- Subscription IDs → `00000000-0000-0000-0000-000000000000`
- Tenant IDs → `00000000-0000-0000-0000-000000000000`
- IP addresses → `10.0.0.1`
- Email addresses → `user@example.com`
- Connection strings → `Server=sanitized;...`
- Passwords/secrets → `REDACTED`
- Storage keys → `REDACTED_KEY`
- SAS tokens → `?sv=REDACTED`

**What's preserved:**
- Resource types (e.g., `Microsoft.Storage/storageAccounts`)
- Property names and structure
- Boolean/numeric configuration values
- Security-relevant settings

**Skip anonymization** (if already sanitized):
```bash
crux submit-sample --no-anonymize ...
```

## Best Practices

### DO:
- **Submit real issues** - Actual misconfigurations found in production
- **Provide context** - Explain how it was discovered and its impact
- **Include CIS references** - Link to relevant benchmarks
- **Use specific issue types** - Be precise about what's wrong
- **Submit multi-issue templates** - Complex cases are valuable

### DON'T:
- **Submit secrets** - Even though we anonymize, avoid submitting actual credentials
- **Submit proprietary code** - Only submit templates you're authorized to share
- **Fabricate issues** - Only submit real misconfigurations you've verified
- **Skip review** - All submissions go through security review before merging

## Review Process

After submission:

1. **Pending** - Your submission is in the queue
2. **Under Review** - Security team is validating the issue
3. **Approved** - Issue confirmed, ready to merge
4. **Merged** - Added to test corpus and training data
5. **Rejected** - Not suitable (duplicate, false positive, etc.)

Check status:
```bash
crux list-submissions --status pending
crux list-submissions --status merged
```

## Validating Against the Corpus

After your submission is merged, you can validate the scanner detects it:

```bash
# Run validation against full corpus
crux validate-corpus \
  --model models/rf-baseline.pkl \
  --corpus-dir test-corpus \
  --output validation-results.json
```

Expected output:
```
================================================================================
              CRUX Test Corpus Validation Report
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total Test Cases: 15
Passed: 14 (93.3%)
Failed: 1
Errors: 0

DETECTION METRICS
--------------------------------------------------------------------------------
Precision: 0.950
Recall:    0.920
F1 Score:  0.935

================================================================================
```

## FAQ

**Q: What if my issue type isn't in the list?**
A: Use `Custom_YourIssueType` and describe it in the description field. We'll add common custom issues to the known types list.

**Q: Can I submit ARM JSON instead of Bicep?**
A: Yes! Both formats are supported.

**Q: How do I submit multiple templates?**
A: Submit each separately - this allows individual tracking and review.

**Q: What happens if the scanner doesn't detect my submission?**
A: That's valuable information! It helps us identify gaps in the model. After review, we'll use your submission to improve detection.

**Q: Can I update a submission after submitting?**
A: Currently, submit a new version with updated details. Reference the original submission ID in the description.

**Q: Who reviews submissions?**
A: The CRUX security team reviews all submissions before merging to ensure quality and validate the issues.

## Getting Help

- **List issue types**: `crux list-issue-types`
- **View submissions**: `crux list-submissions`
- **Validate corpus**: `crux validate-corpus --help`
- **Report issues**: Open a GitHub issue
