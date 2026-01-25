"""
CRUX Test Corpus Module

Provides functionality for:
- Submitting new samples with known issues
- Validating scanner against test corpus
- Managing and sanitizing templates
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

__all__ = [
    "SampleSubmission",
    "submit_sample",
    "list_submissions",
    "KNOWN_ISSUE_TYPES",
    "SEVERITY_LEVELS",
    "CorpusValidator",
    "ValidationResult",
    "TemplateSanitizer",
]
