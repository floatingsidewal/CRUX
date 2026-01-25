"""
CRUX Test Corpus Module

Provides functionality for:
- Submitting new samples with known issues
- Validating scanner against test corpus
- Managing and sanitizing templates
- DLP validation for PII detection
"""

from .submission import (
    SampleSubmission,
    submit_sample,
    list_submissions,
    KNOWN_ISSUE_TYPES,
    SEVERITY_LEVELS,
)
from .validator import CorpusValidator, ValidationResult
from .sanitizer import TemplateSanitizer
from .dlp import (
    TemplateDLPValidator,
    DLPValidationResult,
    DLPValidationLevel,
    DLPFinding,
    validate_template_dlp,
    get_azure_allowed_endpoints,
    PRESIDIO_AVAILABLE,
    AZURE_ALLOWED_ENDPOINTS,
)

__all__ = [
    "SampleSubmission",
    "submit_sample",
    "list_submissions",
    "KNOWN_ISSUE_TYPES",
    "SEVERITY_LEVELS",
    "CorpusValidator",
    "ValidationResult",
    "TemplateSanitizer",
    "TemplateDLPValidator",
    "DLPValidationResult",
    "DLPValidationLevel",
    "DLPFinding",
    "validate_template_dlp",
    "get_azure_allowed_endpoints",
    "PRESIDIO_AVAILABLE",
    "AZURE_ALLOWED_ENDPOINTS",
]
