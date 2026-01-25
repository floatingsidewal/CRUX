"""
Data Leak Prevention (DLP) Validator for Template Submissions

Uses Microsoft Presidio to detect PII and sensitive data in templates
before submission to the test corpus.

Requires: pip install presidio-analyzer presidio-anonymizer
          python -m spacy download en_core_web_lg
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Check if Presidio is available
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio not available. Install with: pip install presidio-analyzer presidio-anonymizer")


class DLPValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"        # Block submission if ANY PII detected (score >= 0.5)
    MODERATE = "moderate"    # Block only high-confidence (score >= 0.85)
    PERMISSIVE = "permissive"  # Warn only, never block


# Azure-specific patterns to detect
AZURE_PATTERNS = {
    "AZURE_STORAGE_KEY": {
        "pattern": r"[A-Za-z0-9+/]{86}==",
        "score": 0.9,
        "context": ["accountkey", "storagekey", "key1", "key2", "accesskey"],
    },
    "AZURE_SAS_TOKEN": {
        "pattern": r"\?sv=\d{4}-\d{2}-\d{2}&[^'\"\s]{20,}",
        "score": 0.95,
        "context": ["sas", "token", "sig="],
    },
    "AZURE_CONNECTION_STRING": {
        "pattern": r"(DefaultEndpointsProtocol|AccountName|AccountKey|EndpointSuffix)=[^;'\"\s]+",
        "score": 0.9,
        "context": ["connection", "connectionstring"],
    },
}

# Patterns that indicate internal/business-specific data
INTERNAL_PATTERNS = {
    "INTERNAL_DOMAIN": {
        "pattern": r"\b[\w-]+\.(internal|corp|local|intranet)\.([\w-]+\.)*[a-z]{2,}\b",
        "score": 0.8,
        "context": [],
    },
    "INTERNAL_IP_RANGE": {
        "pattern": r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
        "score": 0.6,
        "context": [],
    },
}

# Allowlist patterns that should NOT be flagged
ALLOWLIST_PATTERNS = [
    r"example\.com",
    r"contoso\.com",
    r"fabrikam\.com",
    r"adventure-works\.com",
    r"northwind\.com",
    r"00000000-0000-0000-0000-000000000000",  # Sanitized GUID
    r"user@example\.com",
    r"admin@example\.com",
    r"10\.0\.0\.[01]",  # Common example IPs
]

# Entity types to detect
DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "URL",
    "IBAN_CODE",
    "US_BANK_NUMBER",
]


@dataclass
class DLPFinding:
    """A single DLP finding."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    line_number: Optional[int] = None
    context: str = ""
    recommendation: str = ""

    def __post_init__(self):
        """Set default recommendation based on entity type."""
        if not self.recommendation:
            self.recommendation = self._get_recommendation()

    def _get_recommendation(self) -> str:
        """Get recommendation based on entity type."""
        recommendations = {
            "PERSON": "Remove personal names from comments and resource names",
            "EMAIL_ADDRESS": "Replace with example@example.com or remove",
            "PHONE_NUMBER": "Remove phone numbers from template",
            "CREDIT_CARD": "CRITICAL: Remove credit card numbers immediately",
            "US_SSN": "CRITICAL: Remove SSN immediately",
            "IP_ADDRESS": "Replace with example IP (10.0.0.1) or remove",
            "URL": "Check if URL is internal; replace with example.com if so",
            "AZURE_STORAGE_KEY": "CRITICAL: Remove storage key; use managed identity",
            "AZURE_SAS_TOKEN": "CRITICAL: Remove SAS token",
            "AZURE_CONNECTION_STRING": "CRITICAL: Remove connection string",
            "INTERNAL_DOMAIN": "Replace internal domain with example.com",
        }
        return recommendations.get(self.entity_type, "Review and remove if sensitive")

    @property
    def redacted_text(self) -> str:
        """Return redacted version of the text."""
        if len(self.text) <= 4:
            return "****"
        return self.text[:2] + "*" * (len(self.text) - 4) + self.text[-2:]

    @property
    def is_critical(self) -> bool:
        """Check if this is a critical finding."""
        return self.entity_type in [
            "CREDIT_CARD", "US_SSN", "AZURE_STORAGE_KEY",
            "AZURE_SAS_TOKEN", "AZURE_CONNECTION_STRING"
        ]


@dataclass
class DLPValidationResult:
    """Result of DLP validation."""
    is_safe: bool
    findings: List[DLPFinding] = field(default_factory=list)
    risk_score: float = 0.0
    validation_level: DLPValidationLevel = DLPValidationLevel.MODERATE
    error: Optional[str] = None

    @property
    def blocking_findings(self) -> List[DLPFinding]:
        """Findings that should block submission."""
        if self.validation_level == DLPValidationLevel.STRICT:
            threshold = 0.5
        elif self.validation_level == DLPValidationLevel.MODERATE:
            threshold = 0.85
        else:  # PERMISSIVE
            return []  # Never block

        return [f for f in self.findings if f.score >= threshold or f.is_critical]

    @property
    def warning_findings(self) -> List[DLPFinding]:
        """Findings that should warn but not block."""
        blocking = set(id(f) for f in self.blocking_findings)
        return [f for f in self.findings if id(f) not in blocking]

    @property
    def critical_findings(self) -> List[DLPFinding]:
        """Critical findings (secrets, SSN, etc.)."""
        return [f for f in self.findings if f.is_critical]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "risk_score": self.risk_score,
            "validation_level": self.validation_level.value,
            "total_findings": len(self.findings),
            "blocking_findings": len(self.blocking_findings),
            "warning_findings": len(self.warning_findings),
            "critical_findings": len(self.critical_findings),
            "findings": [
                {
                    "entity_type": f.entity_type,
                    "redacted_text": f.redacted_text,
                    "score": f.score,
                    "line_number": f.line_number,
                    "is_critical": f.is_critical,
                    "recommendation": f.recommendation,
                }
                for f in self.findings
            ],
            "error": self.error,
        }

    def print_report(self) -> None:
        """Print human-readable report."""
        print("\n" + "=" * 70)
        print("                    DLP Validation Report")
        print("=" * 70)

        status = "SAFE" if self.is_safe else "ISSUES DETECTED"
        print(f"\nStatus: {status}")
        print(f"Risk Score: {self.risk_score:.2f}")
        print(f"Validation Level: {self.validation_level.value}")

        if self.error:
            print(f"\nError: {self.error}")
            return

        if not self.findings:
            print("\nNo sensitive data detected.")
            return

        print(f"\nTotal Findings: {len(self.findings)}")
        print(f"  Critical: {len(self.critical_findings)}")
        print(f"  Blocking: {len(self.blocking_findings)}")
        print(f"  Warnings: {len(self.warning_findings)}")

        if self.critical_findings:
            print("\n" + "-" * 40)
            print("CRITICAL FINDINGS (must fix)")
            print("-" * 40)
            for f in self.critical_findings:
                print(f"\n  [{f.entity_type}] Line {f.line_number or '?'}")
                print(f"    Detected: {f.redacted_text}")
                print(f"    Confidence: {f.score:.0%}")
                print(f"    Action: {f.recommendation}")

        if self.blocking_findings:
            non_critical_blocking = [f for f in self.blocking_findings if not f.is_critical]
            if non_critical_blocking:
                print("\n" + "-" * 40)
                print("BLOCKING FINDINGS")
                print("-" * 40)
                for f in non_critical_blocking:
                    print(f"\n  [{f.entity_type}] Line {f.line_number or '?'}")
                    print(f"    Detected: {f.redacted_text}")
                    print(f"    Confidence: {f.score:.0%}")
                    print(f"    Action: {f.recommendation}")

        if self.warning_findings:
            print("\n" + "-" * 40)
            print("WARNINGS (review recommended)")
            print("-" * 40)
            for f in self.warning_findings[:5]:  # Show first 5 warnings
                print(f"\n  [{f.entity_type}] Line {f.line_number or '?'}")
                print(f"    Detected: {f.redacted_text}")
                print(f"    Confidence: {f.score:.0%}")

            if len(self.warning_findings) > 5:
                print(f"\n  ... and {len(self.warning_findings) - 5} more warnings")

        print("\n" + "=" * 70)


class TemplateDLPValidator:
    """
    Validates templates for PII and sensitive data using Microsoft Presidio.

    Usage:
        validator = TemplateDLPValidator()
        result = validator.validate(template_content)

        if not result.is_safe:
            print(result.print_report())
    """

    def __init__(
        self,
        validation_level: DLPValidationLevel = DLPValidationLevel.MODERATE,
        custom_patterns: Optional[Dict[str, Dict]] = None,
        allowlist: Optional[List[str]] = None,
    ):
        """
        Initialize the DLP validator.

        Args:
            validation_level: Strictness level for validation
            custom_patterns: Additional patterns to detect
            allowlist: Regex patterns to ignore (won't be flagged)
        """
        self.validation_level = validation_level
        self.allowlist = allowlist or ALLOWLIST_PATTERNS.copy()

        if not PRESIDIO_AVAILABLE:
            logger.warning("Presidio not available, using regex-only mode")
            self.analyzer = None
            self.anonymizer = None
        else:
            self._init_presidio(custom_patterns)

    def _init_presidio(self, custom_patterns: Optional[Dict] = None) -> None:
        """Initialize Presidio engines with custom recognizers."""
        try:
            # Try to create analyzer with spaCy
            self.analyzer = AnalyzerEngine()

            # Add Azure-specific recognizers
            self._add_pattern_recognizers(AZURE_PATTERNS)
            self._add_pattern_recognizers(INTERNAL_PATTERNS)

            # Add custom patterns if provided
            if custom_patterns:
                self._add_pattern_recognizers(custom_patterns)

            # Initialize anonymizer
            self.anonymizer = AnonymizerEngine()

            logger.info("Presidio DLP initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Presidio: {e}")
            logger.warning("Falling back to regex-only mode")
            self.analyzer = None
            self.anonymizer = None

    def _add_pattern_recognizers(self, patterns: Dict[str, Dict]) -> None:
        """Add custom pattern recognizers to the analyzer."""
        if not self.analyzer:
            return

        for name, config in patterns.items():
            pattern = Pattern(
                name=name,
                regex=config["pattern"],
                score=config.get("score", 0.8),
            )
            recognizer = PatternRecognizer(
                supported_entity=name,
                patterns=[pattern],
                context=config.get("context", []),
            )
            self.analyzer.registry.add_recognizer(recognizer)

    def validate(
        self,
        content: str,
        filename: Optional[str] = None,
    ) -> DLPValidationResult:
        """
        Validate template content for sensitive data.

        Args:
            content: Template content to validate
            filename: Optional filename for context

        Returns:
            DLPValidationResult with findings
        """
        result = DLPValidationResult(
            is_safe=True,
            validation_level=self.validation_level,
        )

        # Build line number mapping
        lines = content.split("\n")
        line_starts = [0]
        for line in lines[:-1]:
            line_starts.append(line_starts[-1] + len(line) + 1)

        def get_line_number(pos: int) -> int:
            for i, start in enumerate(line_starts):
                if start > pos:
                    return i
            return len(lines)

        findings = []

        # Use Presidio if available
        if self.analyzer:
            try:
                # Analyze for standard entities
                entities = DEFAULT_ENTITIES + list(AZURE_PATTERNS.keys()) + list(INTERNAL_PATTERNS.keys())
                presidio_results = self.analyzer.analyze(
                    text=content,
                    entities=entities,
                    language="en",
                )

                for r in presidio_results:
                    text = content[r.start:r.end]

                    # Check allowlist
                    if self._is_allowlisted(text):
                        continue

                    finding = DLPFinding(
                        entity_type=r.entity_type,
                        text=text,
                        start=r.start,
                        end=r.end,
                        score=r.score,
                        line_number=get_line_number(r.start),
                    )
                    findings.append(finding)

            except Exception as e:
                logger.error(f"Presidio analysis failed: {e}")
                result.error = str(e)

        # Always run regex patterns as backup/supplement
        findings.extend(self._regex_scan(content, get_line_number))

        # Deduplicate findings (same text, overlapping positions)
        findings = self._deduplicate_findings(findings)

        result.findings = findings
        result.risk_score = self._calculate_risk_score(findings)
        result.is_safe = len(result.blocking_findings) == 0

        return result

    def _regex_scan(self, content: str, get_line_number) -> List[DLPFinding]:
        """Scan content using regex patterns only."""
        findings = []

        all_patterns = {**AZURE_PATTERNS, **INTERNAL_PATTERNS}

        for name, config in all_patterns.items():
            pattern = config["pattern"]
            score = config.get("score", 0.8)

            for match in re.finditer(pattern, content, re.IGNORECASE):
                text = match.group()

                if self._is_allowlisted(text):
                    continue

                finding = DLPFinding(
                    entity_type=name,
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    score=score,
                    line_number=get_line_number(match.start()),
                )
                findings.append(finding)

        return findings

    def _is_allowlisted(self, text: str) -> bool:
        """Check if text matches any allowlist pattern."""
        for pattern in self.allowlist:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _deduplicate_findings(self, findings: List[DLPFinding]) -> List[DLPFinding]:
        """Remove duplicate findings (same text or overlapping positions)."""
        if not findings:
            return []

        # Sort by start position
        sorted_findings = sorted(findings, key=lambda f: (f.start, -f.score))

        deduplicated = [sorted_findings[0]]
        for finding in sorted_findings[1:]:
            last = deduplicated[-1]

            # Skip if overlapping
            if finding.start < last.end:
                # Keep the one with higher score
                if finding.score > last.score:
                    deduplicated[-1] = finding
                continue

            deduplicated.append(finding)

        return deduplicated

    def _calculate_risk_score(self, findings: List[DLPFinding]) -> float:
        """Calculate aggregate risk score from findings."""
        if not findings:
            return 0.0

        # Weight critical findings higher
        score = 0.0
        for f in findings:
            if f.is_critical:
                score += f.score * 2
            else:
                score += f.score

        # Normalize to 0-1 range (cap at 1.0)
        return min(1.0, score / 5)

    def get_safe_content(self, content: str) -> str:
        """
        Return content with detected PII redacted.

        Args:
            content: Original template content

        Returns:
            Content with PII replaced with placeholders
        """
        if not self.anonymizer:
            # Fallback: use regex replacement
            return self._regex_anonymize(content)

        result = self.validate(content)

        if not result.findings:
            return content

        # Build anonymization operators
        try:
            anonymized = self.anonymizer.anonymize(
                text=content,
                analyzer_results=[
                    type("Result", (), {
                        "entity_type": f.entity_type,
                        "start": f.start,
                        "end": f.end,
                        "score": f.score,
                    })()
                    for f in result.findings
                ],
                operators={
                    "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "user@example.com"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                    "IP_ADDRESS": OperatorConfig("replace", {"new_value": "10.0.0.1"}),
                },
            )
            return anonymized.text
        except Exception as e:
            logger.warning(f"Anonymization failed: {e}, using regex fallback")
            return self._regex_anonymize(content)

    def _regex_anonymize(self, content: str) -> str:
        """Fallback regex-based anonymization."""
        result = content

        # Replace detected patterns
        all_patterns = {**AZURE_PATTERNS, **INTERNAL_PATTERNS}
        for name, config in all_patterns.items():
            pattern = config["pattern"]
            result = re.sub(pattern, f"<{name}>", result, flags=re.IGNORECASE)

        return result


def validate_template_dlp(
    template_path: str,
    validation_level: str = "moderate",
    output_path: Optional[str] = None,
) -> DLPValidationResult:
    """
    Convenience function to validate a template file.

    Args:
        template_path: Path to template file
        validation_level: strict, moderate, or permissive
        output_path: Optional path to save report

    Returns:
        DLPValidationResult
    """
    level = DLPValidationLevel(validation_level.lower())
    validator = TemplateDLPValidator(validation_level=level)

    content = Path(template_path).read_text()
    result = validator.validate(content, filename=template_path)

    if output_path:
        import json
        Path(output_path).write_text(json.dumps(result.to_dict(), indent=2))

    return result
