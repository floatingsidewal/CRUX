"""
Template Sanitizer

Removes sensitive information from ARM/Bicep templates before submission.
"""

import re
import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TemplateSanitizer:
    """Sanitizes ARM/Bicep templates by removing sensitive information."""

    # Patterns to detect and replace
    PATTERNS = {
        # Azure subscription IDs
        "subscription_id": (
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "00000000-0000-0000-0000-000000000000",
        ),
        # Azure tenant IDs (same format)
        "tenant_id": (
            r"tenantId:\s*'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'",
            "tenantId: '00000000-0000-0000-0000-000000000000'",
        ),
        # Resource group names (common patterns)
        "resource_group": (
            r"resourceGroup\(\)\.name",
            "resourceGroup().name",  # Keep as-is, it's a function
        ),
        # IP addresses
        "ip_address": (
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
            "10.0.0.1",
        ),
        # Email addresses
        "email": (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "user@example.com",
        ),
        # Connection strings
        "connection_string": (
            r"(Server|Data Source)=[^;]+;[^'\"]+",
            "Server=sanitized;Database=sanitized;",
        ),
        # Storage account keys (base64-like patterns in key contexts)
        "storage_key": (
            r"(accountKey|storageAccountKey|key1|key2):\s*'[A-Za-z0-9+/=]{40,}'",
            r"\1: 'REDACTED_KEY'",
        ),
        # Passwords in various formats
        "password": (
            r"(password|pwd|secret|apikey|api_key|token):\s*'[^']+'" ,
            r"\1: 'REDACTED'",
        ),
        # SAS tokens
        "sas_token": (
            r"\?sv=[^'\"&\s]+",
            "?sv=REDACTED",
        ),
    }

    # Property paths that should have their values redacted
    SENSITIVE_PROPERTIES = [
        "administratorLoginPassword",
        "adminPassword",
        "password",
        "sshPublicKey",
        "secretValue",
        "value",  # In secrets context
        "connectionString",
        "primaryKey",
        "secondaryKey",
        "storageAccountKey",
        "accessKey",
        "apiKey",
        "token",
        "certificate",
        "privateKey",
    ]

    # Resource name patterns to generalize
    NAME_PATTERNS = {
        "storageAccounts": "sanitizedstorage",
        "vaults": "sanitized-keyvault",
        "virtualNetworks": "sanitized-vnet",
        "virtualMachines": "sanitized-vm",
        "networkSecurityGroups": "sanitized-nsg",
        "sqlServers": "sanitized-sql",
        "webSites": "sanitized-webapp",
    }

    def __init__(
        self,
        preserve_structure: bool = True,
        preserve_resource_types: bool = True,
        redact_names: bool = True,
    ):
        """
        Initialize the sanitizer.

        Args:
            preserve_structure: Keep the template structure intact
            preserve_resource_types: Keep Azure resource type identifiers
            redact_names: Replace resource names with generic ones
        """
        self.preserve_structure = preserve_structure
        self.preserve_resource_types = preserve_resource_types
        self.redact_names = redact_names

    def sanitize(self, content: str) -> str:
        """
        Sanitize a template (Bicep or ARM JSON).

        Args:
            content: Template content as string

        Returns:
            Sanitized template content
        """
        # Detect format
        is_json = content.strip().startswith("{")

        if is_json:
            return self._sanitize_arm_json(content)
        else:
            return self._sanitize_bicep(content)

    def _sanitize_bicep(self, content: str) -> str:
        """Sanitize Bicep template."""
        result = content

        # Apply regex patterns
        for pattern_name, (pattern, replacement) in self.PATTERNS.items():
            if pattern_name == "resource_group":
                continue  # Skip, keep function references

            # Handle backreferences in replacement
            if r"\1" in replacement:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(pattern, replacement, result)

        # Redact param default values that look sensitive
        result = re.sub(
            r"@secure\(\)\s*param\s+(\w+)\s+string\s*=\s*'[^']+'",
            r"@secure()\nparam \1 string = 'REDACTED'",
            result,
        )

        # Add sanitization notice
        notice = "// SANITIZED: Sensitive values have been redacted\n"
        if not result.startswith("// SANITIZED"):
            result = notice + result

        return result

    def _sanitize_arm_json(self, content: str) -> str:
        """Sanitize ARM JSON template."""
        try:
            template = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse as JSON, applying text-based sanitization")
            return self._sanitize_bicep(content)

        # Recursively sanitize the template
        template = self._sanitize_dict(template)

        # Add sanitization metadata
        if "metadata" not in template:
            template["metadata"] = {}
        template["metadata"]["_sanitized"] = True
        template["metadata"]["_sanitized_at"] = "TIMESTAMP"

        return json.dumps(template, indent=2)

    def _sanitize_dict(self, obj: Any, path: str = "") -> Any:
        """Recursively sanitize a dictionary."""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Check if this is a sensitive property
                if key.lower() in [p.lower() for p in self.SENSITIVE_PROPERTIES]:
                    if isinstance(value, str) and value:
                        result[key] = "REDACTED"
                    else:
                        result[key] = value
                else:
                    result[key] = self._sanitize_dict(value, current_path)

            return result
        elif isinstance(obj, list):
            return [self._sanitize_dict(item, f"{path}[]") for item in obj]
        elif isinstance(obj, str):
            return self._sanitize_string(obj)
        else:
            return obj

    def _sanitize_string(self, value: str) -> str:
        """Sanitize a string value."""
        result = value

        # Apply patterns
        for pattern_name, (pattern, replacement) in self.PATTERNS.items():
            if r"\1" in replacement:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(pattern, replacement, result)

        return result


def sanitize_template(
    input_path: str,
    output_path: str = None,
    **kwargs,
) -> str:
    """
    Convenience function to sanitize a template file.

    Args:
        input_path: Path to input template
        output_path: Optional path to save sanitized template
        **kwargs: Arguments passed to TemplateSanitizer

    Returns:
        Sanitized template content
    """
    from pathlib import Path

    input_path = Path(input_path)
    content = input_path.read_text()

    sanitizer = TemplateSanitizer(**kwargs)
    sanitized = sanitizer.sanitize(content)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(sanitized)
        logger.info(f"Sanitized template saved to: {output_path}")

    return sanitized
