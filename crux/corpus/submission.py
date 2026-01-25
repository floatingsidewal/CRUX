"""
Sample Submission System

Allows engineers to submit templates with known misconfigurations
for inclusion in the test corpus and future model training.
"""

import json
import hashlib
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Known issue types that can be reported
KNOWN_ISSUE_TYPES = {
    # Storage
    "Storage_PublicAccess": "Public blob access enabled",
    "Storage_NoHttps": "HTTPS-only traffic not enforced",
    "Storage_WeakTLS": "TLS version below 1.2",
    "Storage_NoEncryption": "Encryption not enabled",
    "Storage_NoSoftDelete": "Soft delete not enabled",
    "Storage_NoVersioning": "Blob versioning not enabled",

    # Key Vault
    "KeyVault_NoPurgeProtection": "Purge protection not enabled",
    "KeyVault_NoSoftDelete": "Soft delete not enabled",
    "KeyVault_NoRBAC": "RBAC authorization not enabled",
    "KeyVault_PublicNetwork": "Public network access enabled",

    # Network
    "Network_NoDDoS": "DDoS protection not enabled",
    "Network_NoNSG": "Network Security Group not associated",
    "Network_OpenPorts": "Dangerous ports open to internet",
    "Network_NoPrivateEndpoint": "Using public endpoints instead of private",

    # Compute
    "VM_PasswordAuth": "Password authentication enabled (should use SSH keys)",
    "VM_NoEncryption": "Disk encryption not enabled",
    "VM_PublicIP": "Public IP directly attached",

    # Database
    "SQL_PublicAccess": "Public network access enabled",
    "SQL_NoAuditing": "Auditing not enabled",
    "SQL_NoTDE": "Transparent Data Encryption not enabled",

    # General
    "General_NoTags": "Resource missing required tags",
    "General_NoLock": "No delete lock on critical resource",
    "Custom": "Custom/other issue type",
}

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]


@dataclass
class SampleSubmission:
    """Represents a submitted sample with known issues."""

    # Submission metadata
    submission_id: str
    submitted_at: str
    submitted_by: str
    status: str = "pending"  # pending, approved, rejected, merged

    # Template info
    template_path: str
    template_hash: str
    original_filename: str

    # Issue details
    issue_types: List[str]
    severity: str
    description: str
    cis_references: List[str] = field(default_factory=list)

    # Optional context
    environment: str = "unknown"  # dev, staging, production
    discovery_method: str = "manual"  # manual, audit, incident, scan
    anonymized: bool = False

    # Review fields (populated later)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleSubmission":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "SampleSubmission":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


def generate_submission_id(template_content: str, timestamp: str) -> str:
    """Generate unique submission ID from content hash and timestamp."""
    content_hash = hashlib.sha256(template_content.encode()).hexdigest()[:8]
    time_part = timestamp.replace("-", "").replace(":", "").replace("T", "")[:12]
    return f"SUB-{time_part}-{content_hash}"


def submit_sample(
    template_path: str,
    issue_types: List[str],
    severity: str,
    description: str,
    submitted_by: str = "anonymous",
    cis_references: Optional[List[str]] = None,
    environment: str = "unknown",
    discovery_method: str = "manual",
    anonymize: bool = True,
    corpus_dir: str = "test-corpus",
) -> SampleSubmission:
    """
    Submit a new sample to the test corpus.

    Args:
        template_path: Path to the Bicep/ARM template
        issue_types: List of issue type labels (from KNOWN_ISSUE_TYPES)
        severity: Severity level (critical, high, medium, low)
        description: Description of the issue(s)
        submitted_by: Email or identifier of submitter
        cis_references: Optional CIS benchmark references
        environment: Where this was found (dev, staging, production)
        discovery_method: How it was discovered (manual, audit, incident, scan)
        anonymize: Whether to sanitize the template
        corpus_dir: Path to test corpus directory

    Returns:
        SampleSubmission object
    """
    template_path = Path(template_path)
    corpus_path = Path(corpus_dir)
    submissions_dir = corpus_path / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if severity.lower() not in SEVERITY_LEVELS:
        raise ValueError(f"Invalid severity. Must be one of: {SEVERITY_LEVELS}")

    for issue_type in issue_types:
        if issue_type not in KNOWN_ISSUE_TYPES and not issue_type.startswith("Custom_"):
            logger.warning(f"Unknown issue type: {issue_type}. Consider using 'Custom' or a known type.")

    # Read template content
    template_content = template_path.read_text()

    # Anonymize if requested
    if anonymize:
        from .sanitizer import TemplateSanitizer
        sanitizer = TemplateSanitizer()
        template_content = sanitizer.sanitize(template_content)

    # Generate submission
    timestamp = datetime.utcnow().isoformat()
    submission_id = generate_submission_id(template_content, timestamp)
    template_hash = hashlib.sha256(template_content.encode()).hexdigest()

    # Create submission directory
    submission_dir = submissions_dir / submission_id
    submission_dir.mkdir(exist_ok=True)

    # Save template
    template_ext = template_path.suffix
    saved_template_path = submission_dir / f"template{template_ext}"
    saved_template_path.write_text(template_content)

    # Create submission object
    submission = SampleSubmission(
        submission_id=submission_id,
        submitted_at=timestamp,
        submitted_by=submitted_by,
        template_path=str(saved_template_path.relative_to(corpus_path)),
        template_hash=template_hash,
        original_filename=template_path.name,
        issue_types=issue_types,
        severity=severity.lower(),
        description=description,
        cis_references=cis_references or [],
        environment=environment,
        discovery_method=discovery_method,
        anonymized=anonymize,
    )

    # Save submission metadata
    metadata_path = submission_dir / "submission.json"
    metadata_path.write_text(submission.to_json())

    # Update submissions index
    _update_submissions_index(corpus_path)

    logger.info(f"Sample submitted: {submission_id}")
    return submission


def list_submissions(
    corpus_dir: str = "test-corpus",
    status: Optional[str] = None,
) -> List[SampleSubmission]:
    """
    List all submissions in the corpus.

    Args:
        corpus_dir: Path to test corpus directory
        status: Filter by status (pending, approved, rejected, merged)

    Returns:
        List of SampleSubmission objects
    """
    corpus_path = Path(corpus_dir)
    submissions_dir = corpus_path / "submissions"

    if not submissions_dir.exists():
        return []

    submissions = []
    for submission_dir in submissions_dir.iterdir():
        if not submission_dir.is_dir():
            continue

        metadata_path = submission_dir / "submission.json"
        if metadata_path.exists():
            submission = SampleSubmission.from_json(metadata_path.read_text())
            if status is None or submission.status == status:
                submissions.append(submission)

    # Sort by submission date (newest first)
    submissions.sort(key=lambda s: s.submitted_at, reverse=True)
    return submissions


def approve_submission(
    submission_id: str,
    reviewed_by: str,
    review_notes: str = "",
    corpus_dir: str = "test-corpus",
    target_category: Optional[str] = None,
) -> SampleSubmission:
    """
    Approve a submission and optionally move it to the main corpus.

    Args:
        submission_id: ID of the submission to approve
        reviewed_by: Email or identifier of reviewer
        review_notes: Optional notes from the reviewer
        corpus_dir: Path to test corpus directory
        target_category: If provided, move to this category (e.g., "storage/new-case")

    Returns:
        Updated SampleSubmission object
    """
    corpus_path = Path(corpus_dir)
    submission_dir = corpus_path / "submissions" / submission_id
    metadata_path = submission_dir / "submission.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_id}")

    submission = SampleSubmission.from_json(metadata_path.read_text())
    submission.status = "approved"
    submission.reviewed_by = reviewed_by
    submission.reviewed_at = datetime.utcnow().isoformat()
    submission.review_notes = review_notes

    # Save updated metadata
    metadata_path.write_text(submission.to_json())

    # Move to main corpus if target specified
    if target_category:
        target_dir = corpus_path / target_category
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy template
        for template_file in submission_dir.glob("template.*"):
            shutil.copy(template_file, target_dir / template_file.name)

        # Create expected.json from submission
        expected = {
            "test_case_id": target_category,
            "version": "1.0",
            "created": submission.submitted_at,
            "contributor": submission.submitted_by,
            "source": "submission",
            "submission_id": submission_id,
            "description": submission.description,
            "expected_findings": [
                {
                    "resource_type": "auto-detect",
                    "expected_labels": submission.issue_types,
                    "expected_risk_level": submission.severity.upper(),
                    "must_detect": True,
                }
            ],
            "cis_references": submission.cis_references,
            "environment": submission.environment,
            "discovery_method": submission.discovery_method,
        }
        (target_dir / "expected.json").write_text(json.dumps(expected, indent=2))

        submission.status = "merged"
        metadata_path.write_text(submission.to_json())

        logger.info(f"Submission {submission_id} merged to {target_category}")

    _update_submissions_index(corpus_path)
    return submission


def reject_submission(
    submission_id: str,
    reviewed_by: str,
    review_notes: str,
    corpus_dir: str = "test-corpus",
) -> SampleSubmission:
    """
    Reject a submission.

    Args:
        submission_id: ID of the submission to reject
        reviewed_by: Email or identifier of reviewer
        review_notes: Reason for rejection
        corpus_dir: Path to test corpus directory

    Returns:
        Updated SampleSubmission object
    """
    corpus_path = Path(corpus_dir)
    submission_dir = corpus_path / "submissions" / submission_id
    metadata_path = submission_dir / "submission.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Submission not found: {submission_id}")

    submission = SampleSubmission.from_json(metadata_path.read_text())
    submission.status = "rejected"
    submission.reviewed_by = reviewed_by
    submission.reviewed_at = datetime.utcnow().isoformat()
    submission.review_notes = review_notes

    metadata_path.write_text(submission.to_json())
    _update_submissions_index(corpus_path)

    logger.info(f"Submission {submission_id} rejected: {review_notes}")
    return submission


def _update_submissions_index(corpus_path: Path) -> None:
    """Update the submissions count in manifest.json."""
    manifest_path = corpus_path / "manifest.json"
    if not manifest_path.exists():
        return

    manifest = json.loads(manifest_path.read_text())
    submissions = list_submissions(str(corpus_path), status="pending")
    manifest["statistics"]["pending_submissions"] = len(submissions)
    manifest["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d")
    manifest_path.write_text(json.dumps(manifest, indent=2))
