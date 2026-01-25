"""
Template Scanner

Scans Azure Bicep/ARM templates for misconfigurations using trained ML models.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..templates.compiler import BicepCompiler
from ..templates.extractor import ResourceExtractor
from ..rules.evaluator import RuleEvaluator
from ..ml.features import FeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class ResourceFinding:
    """A finding for a single resource."""

    resource_id: str
    resource_type: str
    resource_name: str
    risk_score: float  # 0.0 - 1.0
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    rule_violations: List[str]  # Labels from rule evaluation
    model_prediction: bool  # True if model predicts misconfiguration
    model_confidence: float  # Model probability
    recommendations: List[str]  # Human-readable recommendations


@dataclass
class ScanResult:
    """Result of scanning a template."""

    # Metadata
    template_path: str
    scan_timestamp: str
    model_name: str
    model_path: str

    # Summary
    overall_risk_score: float  # 0.0 - 1.0 (max of all resources)
    overall_risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    total_resources: int
    resources_at_risk: int

    # Detailed findings
    findings: List[ResourceFinding] = field(default_factory=list)

    # Rule-based findings (independent of model)
    rule_violations_count: int = 0
    unique_rule_violations: List[str] = field(default_factory=list)

    # CI/CD support
    exit_code: int = 0  # 0 = pass, 1 = fail (if threshold exceeded)
    threshold_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "template_path": self.template_path,
                "scan_timestamp": self.scan_timestamp,
                "model_name": self.model_name,
                "model_path": self.model_path,
            },
            "summary": {
                "overall_risk_score": round(self.overall_risk_score, 4),
                "overall_risk_level": self.overall_risk_level,
                "total_resources": self.total_resources,
                "resources_at_risk": self.resources_at_risk,
                "rule_violations_count": self.rule_violations_count,
                "unique_rule_violations": self.unique_rule_violations,
            },
            "findings": [
                {
                    "resource_id": f.resource_id,
                    "resource_type": f.resource_type,
                    "resource_name": f.resource_name,
                    "risk_score": round(f.risk_score, 4),
                    "risk_level": f.risk_level,
                    "rule_violations": f.rule_violations,
                    "model_prediction": f.model_prediction,
                    "model_confidence": round(f.model_confidence, 4),
                    "recommendations": f.recommendations,
                }
                for f in self.findings
            ],
            "ci_cd": {
                "exit_code": self.exit_code,
                "threshold_exceeded": self.threshold_exceeded,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        """Convert to human-readable text format."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("CRUX Template Scan Report")
        lines.append("=" * 70)
        lines.append("")

        # Metadata
        lines.append(f"Template: {self.template_path}")
        lines.append(f"Scanned:  {self.scan_timestamp}")
        lines.append(f"Model:    {self.model_name}")
        lines.append("")

        # Overall risk
        risk_emoji = {
            "CRITICAL": "[!!!]",
            "HIGH": "[!!]",
            "MEDIUM": "[!]",
            "LOW": "[OK]",
        }
        lines.append("-" * 70)
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Overall Risk: {risk_emoji.get(self.overall_risk_level, '')} {self.overall_risk_level} (score: {self.overall_risk_score:.2%})")
        lines.append(f"Resources Scanned: {self.total_resources}")
        lines.append(f"Resources at Risk: {self.resources_at_risk}")
        lines.append(f"Rule Violations: {self.rule_violations_count}")
        lines.append("")

        # Findings (sorted by risk score descending)
        if self.findings:
            sorted_findings = sorted(self.findings, key=lambda f: f.risk_score, reverse=True)

            lines.append("-" * 70)
            lines.append("FINDINGS")
            lines.append("-" * 70)

            for i, finding in enumerate(sorted_findings, 1):
                if finding.risk_score > 0.3 or finding.rule_violations:  # Only show risky resources
                    lines.append("")
                    lines.append(f"{i}. {finding.resource_type}")
                    lines.append(f"   Name: {finding.resource_name}")
                    lines.append(f"   Risk: {finding.risk_level} ({finding.risk_score:.2%})")

                    if finding.rule_violations:
                        lines.append(f"   Rule Violations:")
                        for violation in finding.rule_violations:
                            lines.append(f"     - {violation}")

                    if finding.recommendations:
                        lines.append(f"   Recommendations:")
                        for rec in finding.recommendations[:3]:  # Top 3 recommendations
                            lines.append(f"     - {rec}")

        # Rule violations summary
        if self.unique_rule_violations:
            lines.append("")
            lines.append("-" * 70)
            lines.append("UNIQUE RULE VIOLATIONS")
            lines.append("-" * 70)
            for violation in self.unique_rule_violations:
                lines.append(f"  - {violation}")

        # CI/CD result
        lines.append("")
        lines.append("-" * 70)
        if self.threshold_exceeded:
            lines.append(f"CI/CD Result: FAIL (exit code {self.exit_code})")
        else:
            lines.append(f"CI/CD Result: PASS (exit code {self.exit_code})")
        lines.append("-" * 70)

        return "\n".join(lines)


# Risk factor recommendations based on research paper findings
RISK_RECOMMENDATIONS = {
    # High-risk factors (OR > 5)
    "any_versioning_disabled": "Enable blob versioning on storage accounts (10x risk reduction)",
    "any_soft_delete_disabled": "Enable soft delete on storage accounts (10x risk reduction)",
    "any_no_identity": "Use managed identities instead of shared keys (4x risk reduction)",
    "any_no_service_endpoints": "Enable service endpoints for secure network access (3.5x risk reduction)",
    "any_no_availability_set": "Place VMs in availability sets for high availability (3x risk reduction)",
    "pct_secure_boot": "Enable secure boot on all virtual machines (6x risk reduction)",

    # Protective factors (OR < 0.3)
    "all_encryption_enabled": "Ensure encryption is enabled across all resources (93% risk reduction)",
    "all_auto_patch": "Enable automatic patching on all VMs (92% risk reduction)",
    "all_managed_disks": "Use managed disks for all VMs (70% risk reduction)",

    # Medium-risk factors
    "any_public_access": "Disable public blob access on storage accounts",
    "any_http_allowed": "Enforce HTTPS-only traffic",
    "any_weak_tls": "Upgrade to TLS 1.2 minimum",
    "any_open_ssh": "Restrict SSH access to known IP ranges",
    "any_open_rdp": "Restrict RDP access to known IP ranges",
    "any_ddos_disabled": "Enable DDoS protection on virtual networks",
    "any_no_patching": "Enable automatic OS patching",
    "any_diagnostics_disabled": "Enable boot diagnostics for troubleshooting",
    "any_no_encryption": "Enable encryption at rest",
    "any_password_auth": "Disable password authentication, use SSH keys",
}

# CIS benchmark references for common labels
CIS_REFERENCES = {
    "Storage_PublicAccess": "CIS 3.7 - Ensure public access is disabled for storage accounts",
    "Storage_NoHttps": "CIS 3.1 - Ensure secure transfer required is enabled",
    "Storage_WeakTLS": "CIS 3.10 - Ensure minimum TLS version is set to 1.2",
    "Storage_NoEncryption": "CIS 3.2 - Ensure storage account default access tier is set to Hot",
    "VM_NoSecureBoot": "CIS 7.6 - Ensure secure boot is enabled for VMs",
    "VM_PasswordAuth": "CIS 7.4 - Ensure SSH key authentication is used",
    "Network_OpenSSH": "CIS 6.1 - Ensure SSH access is restricted",
    "Network_OpenRDP": "CIS 6.2 - Ensure RDP access is restricted",
    "KeyVault_NoPurgeProtection": "CIS 8.4 - Ensure purge protection is enabled",
    "KeyVault_NoSoftDelete": "CIS 8.5 - Ensure soft delete is enabled",
}


class TemplateScanner:
    """
    Scans Azure templates for misconfigurations using trained ML models.

    Combines rule-based detection with ML model predictions to provide
    comprehensive risk assessment.
    """

    def __init__(
        self,
        model_path: str,
        rules_dir: Optional[str] = None,
        fail_threshold: Optional[float] = None,
    ):
        """
        Initialize the template scanner.

        Args:
            model_path: Path to trained model (.pkl file)
            rules_dir: Directory containing YAML rule files (optional)
            fail_threshold: Risk score threshold for CI/CD failure (None = never fail)
        """
        self.model_path = Path(model_path)
        self.rules_dir = Path(rules_dir) if rules_dir else None
        self.fail_threshold = fail_threshold

        # Load model
        self._load_model()

        # Initialize components
        self.compiler = BicepCompiler(verify_installation=False)
        self.extractor = ResourceExtractor()

        # Load rule evaluator if rules directory provided
        if self.rules_dir and self.rules_dir.exists():
            self.rule_evaluator = RuleEvaluator(str(self.rules_dir))
            logger.info(f"Loaded {len(self.rule_evaluator.rules)} rules from {self.rules_dir}")
        else:
            self.rule_evaluator = None
            logger.info("No rules directory provided - rule-based detection disabled")

    def _load_model(self) -> None:
        """Load the trained model and feature extractor."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model_name = model_data.get("model_name", "Unknown")
        self.model = model_data.get("model")
        self.feature_names = model_data.get("feature_names", [])
        self.label_names = model_data.get("label_names", [])

        logger.info(f"Loaded model: {self.model_name}")
        logger.info(f"Features: {len(self.feature_names)}, Labels: {len(self.label_names)}")

        # Load feature extractor
        feature_path = self.model_path.parent / f"{self.model_path.stem}_features.pkl"
        if feature_path.exists():
            with open(feature_path, "rb") as f:
                self.feature_extractor = pickle.load(f)
            logger.info(f"Loaded feature extractor from {feature_path}")
        else:
            raise FileNotFoundError(
                f"Feature extractor not found: {feature_path}\n"
                "Ensure the feature extractor was saved alongside the model."
            )

    def scan(self, template_path: str) -> ScanResult:
        """
        Scan a template for misconfigurations.

        Args:
            template_path: Path to Bicep or ARM JSON template

        Returns:
            ScanResult with detailed findings
        """
        template_path = Path(template_path)

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        logger.info(f"Scanning template: {template_path}")

        # Step 1: Load/compile template
        arm_template = self._load_template(template_path)

        # Step 2: Extract resources
        resources = self.extractor.extract_resources(arm_template)
        logger.info(f"Extracted {len(resources)} resources")

        if not resources:
            return self._create_empty_result(template_path)

        # Step 3: Rule-based evaluation
        rule_labels = {}
        if self.rule_evaluator:
            rule_labels = self.rule_evaluator.evaluate_resources(resources)

        # Step 4: ML model prediction
        predictions, probabilities = self._predict(resources)

        # Step 5: Combine results into findings
        findings = self._create_findings(resources, rule_labels, predictions, probabilities)

        # Step 6: Calculate overall risk
        overall_risk_score = max(f.risk_score for f in findings) if findings else 0.0
        overall_risk_level = self._score_to_level(overall_risk_score)

        # Step 7: Aggregate rule violations
        all_violations = []
        for labels in rule_labels.values():
            all_violations.extend(labels)
        unique_violations = list(set(all_violations))

        # Step 8: Determine CI/CD exit code
        threshold_exceeded = False
        exit_code = 0
        if self.fail_threshold is not None and overall_risk_score >= self.fail_threshold:
            threshold_exceeded = True
            exit_code = 1

        # Create result
        result = ScanResult(
            template_path=str(template_path),
            scan_timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            model_path=str(self.model_path),
            overall_risk_score=overall_risk_score,
            overall_risk_level=overall_risk_level,
            total_resources=len(resources),
            resources_at_risk=sum(1 for f in findings if f.risk_score > 0.5),
            findings=findings,
            rule_violations_count=len(all_violations),
            unique_rule_violations=unique_violations,
            exit_code=exit_code,
            threshold_exceeded=threshold_exceeded,
        )

        return result

    def _load_template(self, template_path: Path) -> Dict[str, Any]:
        """Load and optionally compile a template."""
        if template_path.suffix == ".bicep":
            logger.info("Compiling Bicep template...")
            try:
                return self.compiler.compile(template_path)
            except Exception as e:
                logger.warning(f"Bicep compilation failed: {e}")
                raise RuntimeError(f"Failed to compile Bicep template: {e}")
        elif template_path.suffix == ".json":
            with open(template_path) as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported template format: {template_path.suffix}")

    def _predict(
        self, resources: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ML model prediction on resources."""
        # Extract features
        X, _ = self.feature_extractor.transform(resources)

        # Predict
        predictions = self.model.predict(X)

        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            # For MultiOutputClassifier, get probability of positive class
            if isinstance(probas, list):
                probabilities = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probas])
            else:
                probabilities = probas
        else:
            probabilities = predictions.astype(float)

        return predictions, probabilities

    def _create_findings(
        self,
        resources: List[Dict[str, Any]],
        rule_labels: Dict[str, List[str]],
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> List[ResourceFinding]:
        """Create detailed findings for each resource."""
        findings = []

        for i, resource in enumerate(resources):
            resource_id = resource.get("id", f"resource-{i}")
            resource_type = resource.get("type", "Unknown")
            resource_name = resource.get("name", "Unknown")

            # Get rule violations
            violations = rule_labels.get(resource_id, [])

            # Get model prediction
            if len(predictions.shape) == 1:
                # Binary classification
                model_pred = bool(predictions[i])
                model_conf = float(probabilities[i]) if probabilities.ndim == 1 else float(probabilities[i, 0])
            else:
                # Multi-label: any positive prediction
                model_pred = bool(np.any(predictions[i]))
                # Average probability or max
                model_conf = float(np.max(probabilities[i]))

            # Calculate risk score (combine rule violations and model confidence)
            rule_score = min(len(violations) * 0.2, 0.5)  # Up to 0.5 from rules
            model_score = model_conf * 0.5  # Up to 0.5 from model
            risk_score = min(rule_score + model_score, 1.0)

            # Adjust risk if both rules and model agree
            if violations and model_pred:
                risk_score = min(risk_score + 0.2, 1.0)

            risk_level = self._score_to_level(risk_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                resource_type, violations, model_pred
            )

            findings.append(ResourceFinding(
                resource_id=resource_id,
                resource_type=resource_type,
                resource_name=resource_name,
                risk_score=risk_score,
                risk_level=risk_level,
                rule_violations=violations,
                model_prediction=model_pred,
                model_confidence=model_conf,
                recommendations=recommendations,
            ))

        return findings

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to level."""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendations(
        self,
        resource_type: str,
        violations: List[str],
        model_pred: bool,
    ) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []

        # Add CIS references for violations
        for violation in violations:
            if violation in CIS_REFERENCES:
                recommendations.append(CIS_REFERENCES[violation])

        # Add general recommendations based on resource type
        if "Storage" in resource_type:
            if model_pred or any("Storage" in v for v in violations):
                recommendations.extend([
                    "Enable soft delete for blob recovery",
                    "Enable versioning for data protection",
                    "Enforce HTTPS-only traffic",
                ])
        elif "VirtualMachine" in resource_type or "/virtualMachines" in resource_type:
            if model_pred or any("VM" in v for v in violations):
                recommendations.extend([
                    "Enable secure boot and vTPM",
                    "Use SSH key authentication instead of passwords",
                    "Enable boot diagnostics for troubleshooting",
                ])
        elif "KeyVault" in resource_type or "vaults" in resource_type:
            if model_pred or any("KeyVault" in v for v in violations):
                recommendations.extend([
                    "Enable purge protection",
                    "Enable soft delete with 90-day retention",
                    "Use RBAC authorization instead of access policies",
                ])
        elif "Network" in resource_type or "securityGroups" in resource_type:
            if model_pred or any("Network" in v for v in violations):
                recommendations.extend([
                    "Restrict SSH (22) and RDP (3389) to known IPs",
                    "Enable DDoS protection",
                    "Use service endpoints for Azure services",
                ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)

        return unique_recs[:5]  # Top 5 recommendations

    def _create_empty_result(self, template_path: Path) -> ScanResult:
        """Create result for template with no resources."""
        return ScanResult(
            template_path=str(template_path),
            scan_timestamp=datetime.now().isoformat(),
            model_name=self.model_name,
            model_path=str(self.model_path),
            overall_risk_score=0.0,
            overall_risk_level="LOW",
            total_resources=0,
            resources_at_risk=0,
            findings=[],
            rule_violations_count=0,
            unique_rule_violations=[],
            exit_code=0,
            threshold_exceeded=False,
        )


def scan_template(
    template_path: str,
    model_path: str,
    rules_dir: Optional[str] = None,
    fail_threshold: Optional[float] = None,
) -> ScanResult:
    """
    Convenience function to scan a template.

    Args:
        template_path: Path to Bicep or ARM JSON template
        model_path: Path to trained model
        rules_dir: Directory containing YAML rules (optional)
        fail_threshold: Risk threshold for CI/CD failure (optional)

    Returns:
        ScanResult with detailed findings
    """
    scanner = TemplateScanner(
        model_path=model_path,
        rules_dir=rules_dir,
        fail_threshold=fail_threshold,
    )
    return scanner.scan(template_path)
