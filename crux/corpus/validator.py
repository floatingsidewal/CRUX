"""
Test Corpus Validator

Runs the CRUX scanner against test corpus and measures detection accuracy.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FindingMatch:
    """Represents a match between expected and actual findings."""

    resource_type: str
    expected_labels: List[str]
    actual_labels: List[str]
    true_positives: List[str]  # Expected and found
    false_negatives: List[str]  # Expected but not found
    false_positives: List[str]  # Found but not expected
    risk_score: float
    expected_risk_level: str
    actual_risk_level: str
    risk_level_match: bool


@dataclass
class TestCaseResult:
    """Result of validating a single test case."""

    test_case_id: str
    template_path: str
    passed: bool
    error: Optional[str] = None

    # Detection metrics
    expected_findings: int = 0
    actual_findings: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0

    # Risk level accuracy
    risk_level_matches: int = 0
    risk_level_mismatches: int = 0

    # Detailed matches
    finding_matches: List[FindingMatch] = field(default_factory=list)

    # Scan output
    scan_result: Optional[Dict[str, Any]] = None

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class ValidationResult:
    """Aggregate results from validating the entire corpus."""

    timestamp: str
    model_path: str
    corpus_dir: str

    # Summary counts
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0

    # Aggregate metrics
    total_true_positives: int = 0
    total_false_negatives: int = 0
    total_false_positives: int = 0

    # Individual results
    case_results: List[TestCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Percentage of test cases that passed."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def precision(self) -> float:
        """Aggregate precision across all cases."""
        if self.total_true_positives + self.total_false_positives == 0:
            return 0.0
        return self.total_true_positives / (self.total_true_positives + self.total_false_positives)

    @property
    def recall(self) -> float:
        """Aggregate recall across all cases."""
        if self.total_true_positives + self.total_false_negatives == 0:
            return 0.0
        return self.total_true_positives / (self.total_true_positives + self.total_false_negatives)

    @property
    def f1_score(self) -> float:
        """Aggregate F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "model_path": self.model_path,
            "corpus_dir": self.corpus_dir,
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "error_cases": self.error_cases,
                "pass_rate": round(self.pass_rate, 3),
            },
            "metrics": {
                "precision": round(self.precision, 3),
                "recall": round(self.recall, 3),
                "f1_score": round(self.f1_score, 3),
                "true_positives": self.total_true_positives,
                "false_negatives": self.total_false_negatives,
                "false_positives": self.total_false_positives,
            },
            "case_results": [
                {
                    "test_case_id": r.test_case_id,
                    "passed": r.passed,
                    "error": r.error,
                    "precision": round(r.precision, 3),
                    "recall": round(r.recall, 3),
                    "f1_score": round(r.f1_score, 3),
                    "true_positives": r.true_positives,
                    "false_negatives": r.false_negatives,
                    "false_positives": r.false_positives,
                }
                for r in self.case_results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def print_report(self) -> None:
        """Print a human-readable report."""
        print("=" * 80)
        print("              CRUX Test Corpus Validation Report")
        print("=" * 80)
        print(f"\nTimestamp: {self.timestamp}")
        print(f"Model: {self.model_path}")
        print(f"Corpus: {self.corpus_dir}")

        print("\n" + "-" * 40)
        print("SUMMARY")
        print("-" * 40)
        print(f"Total Test Cases: {self.total_cases}")
        print(f"Passed: {self.passed_cases} ({self.pass_rate:.1%})")
        print(f"Failed: {self.failed_cases}")
        print(f"Errors: {self.error_cases}")

        print("\n" + "-" * 40)
        print("DETECTION METRICS")
        print("-" * 40)
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall:    {self.recall:.3f}")
        print(f"F1 Score:  {self.f1_score:.3f}")
        print(f"\nTrue Positives:  {self.total_true_positives}")
        print(f"False Negatives: {self.total_false_negatives}")
        print(f"False Positives: {self.total_false_positives}")

        # Show failed cases
        failed = [r for r in self.case_results if not r.passed]
        if failed:
            print("\n" + "-" * 40)
            print("FAILED TEST CASES")
            print("-" * 40)
            for result in failed:
                print(f"\n  {result.test_case_id}")
                if result.error:
                    print(f"    Error: {result.error}")
                if result.false_negatives > 0:
                    print(f"    Missed labels: {result.false_negatives}")
                    for match in result.finding_matches:
                        if match.false_negatives:
                            print(f"      - {match.false_negatives}")

        print("\n" + "=" * 80)


class CorpusValidator:
    """Validates CRUX scanner against test corpus."""

    def __init__(
        self,
        model_path: str,
        rules_dir: Optional[str] = None,
        corpus_dir: str = "test-corpus",
    ):
        """
        Initialize the validator.

        Args:
            model_path: Path to trained CRUX model
            rules_dir: Optional path to rules directory
            corpus_dir: Path to test corpus
        """
        self.model_path = model_path
        self.rules_dir = rules_dir
        self.corpus_dir = Path(corpus_dir)

        # Lazy load scanner
        self._scanner = None

    @property
    def scanner(self):
        """Lazy-load the scanner."""
        if self._scanner is None:
            from ..scan import TemplateScanner
            self._scanner = TemplateScanner(
                model_path=self.model_path,
                rules_dir=self.rules_dir,
            )
        return self._scanner

    def validate(self) -> ValidationResult:
        """
        Run validation against the entire test corpus.

        Returns:
            ValidationResult with aggregate metrics
        """
        result = ValidationResult(
            timestamp=datetime.utcnow().isoformat(),
            model_path=self.model_path,
            corpus_dir=str(self.corpus_dir),
        )

        # Find all test cases
        test_cases = self._discover_test_cases()
        result.total_cases = len(test_cases)

        logger.info(f"Found {len(test_cases)} test cases")

        for case_dir, expected_path in test_cases:
            case_result = self._validate_case(case_dir, expected_path)
            result.case_results.append(case_result)

            if case_result.error:
                result.error_cases += 1
            elif case_result.passed:
                result.passed_cases += 1
            else:
                result.failed_cases += 1

            # Aggregate metrics
            result.total_true_positives += case_result.true_positives
            result.total_false_negatives += case_result.false_negatives
            result.total_false_positives += case_result.false_positives

        return result

    def _discover_test_cases(self) -> List[Tuple[Path, Path]]:
        """Discover all test cases in the corpus."""
        test_cases = []

        for expected_path in self.corpus_dir.rglob("expected.json"):
            case_dir = expected_path.parent
            # Skip submissions directory
            if "submissions" in str(case_dir):
                continue
            test_cases.append((case_dir, expected_path))

        return test_cases

    def _validate_case(self, case_dir: Path, expected_path: Path) -> TestCaseResult:
        """Validate a single test case."""
        test_case_id = str(case_dir.relative_to(self.corpus_dir))

        # Load expected findings
        try:
            expected = json.loads(expected_path.read_text())
        except Exception as e:
            return TestCaseResult(
                test_case_id=test_case_id,
                template_path=str(case_dir),
                passed=False,
                error=f"Failed to load expected.json: {e}",
            )

        # Find template file
        template_path = None
        for ext in [".bicep", ".json"]:
            candidate = case_dir / f"template{ext}"
            if candidate.exists():
                template_path = candidate
                break

        if not template_path:
            return TestCaseResult(
                test_case_id=test_case_id,
                template_path=str(case_dir),
                passed=False,
                error="No template file found (template.bicep or template.json)",
            )

        # Run scan
        try:
            scan_result = self.scanner.scan(str(template_path))
        except Exception as e:
            return TestCaseResult(
                test_case_id=test_case_id,
                template_path=str(template_path),
                passed=False,
                error=f"Scan failed: {e}",
            )

        # Compare results
        return self._compare_results(
            test_case_id=test_case_id,
            template_path=str(template_path),
            expected=expected,
            scan_result=scan_result,
        )

    def _compare_results(
        self,
        test_case_id: str,
        template_path: str,
        expected: Dict[str, Any],
        scan_result,
    ) -> TestCaseResult:
        """Compare scan results against expected findings."""
        result = TestCaseResult(
            test_case_id=test_case_id,
            template_path=template_path,
            passed=True,
            scan_result=scan_result.to_dict() if scan_result else None,
        )

        expected_findings = expected.get("expected_findings", [])
        result.expected_findings = len(expected_findings)
        result.actual_findings = len(scan_result.findings) if scan_result else 0

        # Get all actual labels from scan
        actual_labels_by_type: Dict[str, List[str]] = {}
        if scan_result:
            for finding in scan_result.findings:
                resource_type = finding.resource_type
                if resource_type not in actual_labels_by_type:
                    actual_labels_by_type[resource_type] = []
                actual_labels_by_type[resource_type].extend(finding.rule_violations)

        # Compare each expected finding
        for exp in expected_findings:
            resource_type = exp.get("resource_type", "any")
            expected_labels = set(exp.get("expected_labels", []))
            expected_risk_level = exp.get("expected_risk_level", "").upper()
            must_detect = exp.get("must_detect", True)

            # Get actual labels for this resource type
            if resource_type == "auto-detect" or resource_type == "any":
                # Flatten all actual labels
                actual_labels = set()
                for labels in actual_labels_by_type.values():
                    actual_labels.update(labels)
            else:
                actual_labels = set(actual_labels_by_type.get(resource_type, []))

            # Calculate matches
            true_positives = expected_labels & actual_labels
            false_negatives = expected_labels - actual_labels
            false_positives = actual_labels - expected_labels

            result.true_positives += len(true_positives)
            result.false_negatives += len(false_negatives)
            result.false_positives += len(false_positives)

            # Check risk level
            actual_risk_level = scan_result.overall_risk_level if scan_result else "UNKNOWN"
            risk_match = (
                expected_risk_level == actual_risk_level or
                not expected_risk_level  # No expected level specified
            )

            if risk_match:
                result.risk_level_matches += 1
            else:
                result.risk_level_mismatches += 1

            # Record match details
            match = FindingMatch(
                resource_type=resource_type,
                expected_labels=list(expected_labels),
                actual_labels=list(actual_labels),
                true_positives=list(true_positives),
                false_negatives=list(false_negatives),
                false_positives=list(false_positives),
                risk_score=scan_result.overall_risk_score if scan_result else 0.0,
                expected_risk_level=expected_risk_level,
                actual_risk_level=actual_risk_level,
                risk_level_match=risk_match,
            )
            result.finding_matches.append(match)

            # Determine if this finding passed
            if must_detect and false_negatives:
                result.passed = False

        # Check false positive expectations
        false_positive_checks = expected.get("false_positive_checks", [])
        for check in false_positive_checks:
            unexpected_labels = set(check.get("unexpected_labels", []))
            all_actual = set()
            for labels in actual_labels_by_type.values():
                all_actual.update(labels)

            # If any unexpected label was found, it's a false positive
            found_unexpected = unexpected_labels & all_actual
            if found_unexpected:
                result.false_positives += len(found_unexpected)
                result.passed = False
                logger.warning(
                    f"False positive in {test_case_id}: found {found_unexpected}"
                )

        return result


def validate_corpus(
    model_path: str,
    corpus_dir: str = "test-corpus",
    rules_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> ValidationResult:
    """
    Convenience function to validate the test corpus.

    Args:
        model_path: Path to trained CRUX model
        corpus_dir: Path to test corpus
        rules_dir: Optional path to rules directory
        output_path: Optional path to save results JSON

    Returns:
        ValidationResult
    """
    validator = CorpusValidator(
        model_path=model_path,
        rules_dir=rules_dir,
        corpus_dir=corpus_dir,
    )

    result = validator.validate()

    if output_path:
        Path(output_path).write_text(result.to_json())
        logger.info(f"Results saved to: {output_path}")

    return result
