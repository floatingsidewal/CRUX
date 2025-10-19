"""
Resource Extractor

Extracts resource information from ARM template JSON.
"""

from typing import Dict, Any, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class ResourceExtractor:
    """Extracts and normalizes resources from ARM templates."""

    def extract_resources(self, arm_template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all resources from an ARM template.

        Args:
            arm_template: ARM template as a dictionary

        Returns:
            List of resource dictionaries with normalized structure

        Note:
            This handles both top-level resources and nested child resources.
        """
        resources = arm_template.get("resources", [])
        if not resources:
            logger.warning("No resources found in ARM template")
            return []

        extracted = []
        for resource in resources:
            extracted.append(self._normalize_resource(resource))

            # Handle nested resources (child resources)
            if "resources" in resource:
                for child in resource["resources"]:
                    extracted.append(self._normalize_resource(child, parent=resource))

        logger.debug(f"Extracted {len(extracted)} resources")
        return extracted

    def _normalize_resource(
        self,
        resource: Dict[str, Any],
        parent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize a resource to a standard structure.

        Args:
            resource: Raw resource from ARM template
            parent: Parent resource if this is a nested resource

        Returns:
            Normalized resource dictionary
        """
        # Extract type and API version
        resource_type = resource.get("type", "Unknown")
        api_version = resource.get("apiVersion", "Unknown")

        # Build full type (including parent if nested)
        if parent:
            parent_type = parent.get("type", "")
            resource_type = f"{parent_type}/{resource_type}"

        # Extract resource name
        name = resource.get("name", "Unknown")

        # Generate a unique resource ID (simulate what Azure would create)
        # In real ARM templates, this would be constructed with resourceId() function
        resource_id = self._construct_resource_id(resource_type, name, parent)

        # Extract properties
        properties = resource.get("properties", {})

        # Extract dependencies
        depends_on = resource.get("dependsOn", [])

        # Extract location
        location = resource.get("location", None)

        # Extract tags
        tags = resource.get("tags", {})

        # Extract SKU if present
        sku = resource.get("sku", None)

        # Extract kind if present
        kind = resource.get("kind", None)

        return {
            "id": resource_id,
            "type": resource_type,
            "apiVersion": api_version,
            "name": name,
            "location": location,
            "properties": properties,
            "dependsOn": depends_on,
            "tags": tags,
            "sku": sku,
            "kind": kind,
            "_raw": resource,  # Keep original for debugging
        }

    def _construct_resource_id(
        self,
        resource_type: str,
        name: str,
        parent: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct a simulated resource ID.

        Args:
            resource_type: Azure resource type
            name: Resource name
            parent: Parent resource if nested

        Returns:
            Simulated resource ID string
        """
        # This is a simplified version - real Azure resource IDs are more complex
        # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/{type}/{name}
        base_id = f"/subscriptions/template-sim/resourceGroups/template-rg/providers/{resource_type}"

        if parent:
            parent_name = parent.get("name", "parent")
            return f"{base_id}/{parent_name}/{name}"
        else:
            return f"{base_id}/{name}"

    def extract_resource_dependencies(
        self, arm_template: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extract dependency relationships between resources.

        Args:
            arm_template: ARM template as dictionary

        Returns:
            Dictionary mapping resource IDs to lists of dependent resource IDs
        """
        resources = self.extract_resources(arm_template)
        dependencies = {}

        for resource in resources:
            resource_id = resource["id"]
            depends_on = resource.get("dependsOn", [])

            # Resolve dependsOn references
            resolved_deps = []
            for dep in depends_on:
                # dependsOn can be resource IDs or ARM expressions like [resourceId('...')]
                if isinstance(dep, str):
                    # Try to resolve ARM expression
                    resolved_dep = self._resolve_dependency_reference(dep, resources)
                    if resolved_dep:
                        resolved_deps.append(resolved_dep)

            dependencies[resource_id] = resolved_deps

        return dependencies

    def _resolve_dependency_reference(
        self, dep_ref: str, all_resources: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Resolve a dependsOn reference to an actual resource ID.

        Args:
            dep_ref: Dependency reference (could be ARM expression or resource ID)
            all_resources: List of all resources for lookup

        Returns:
            Resolved resource ID or None if not found
        """
        # If it's already a full resource ID, return it
        if dep_ref.startswith("/subscriptions"):
            return dep_ref

        # Try to extract resource type and name from ARM expression
        # Example: [resourceId('Microsoft.Storage/storageAccounts', 'mystorageaccount')]
        resource_id_match = re.search(
            r"resourceId\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]", dep_ref
        )
        if resource_id_match:
            resource_type = resource_id_match.group(1)
            resource_name = resource_id_match.group(2)

            # Find matching resource
            for resource in all_resources:
                if (
                    resource["type"] == resource_type
                    and resource["name"] == resource_name
                ):
                    return resource["id"]

        # If we can't resolve it, just return the original reference
        logger.debug(f"Could not resolve dependency reference: {dep_ref}")
        return dep_ref

    def get_resources_by_type(
        self, arm_template: Dict[str, Any], resource_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get all resources of a specific type.

        Args:
            arm_template: ARM template dictionary
            resource_type: Azure resource type (e.g., 'Microsoft.Storage/storageAccounts')

        Returns:
            List of resources matching the type
        """
        resources = self.extract_resources(arm_template)
        return [r for r in resources if r["type"] == resource_type]


def extract_resources(arm_template: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function to extract resources from an ARM template.

    Args:
        arm_template: ARM template as dictionary

    Returns:
        List of normalized resource dictionaries
    """
    extractor = ResourceExtractor()
    return extractor.extract_resources(arm_template)
