"""
Dependency Graph Builder

Builds dependency graphs from ARM template resources using NetworkX.
"""

from typing import Dict, Any, List, Optional
import logging
import networkx as nx

from .extractor import ResourceExtractor

logger = logging.getLogger(__name__)


class DependencyGraphBuilder:
    """Builds dependency graphs from ARM template resources."""

    def __init__(self):
        """Initialize the graph builder."""
        self.extractor = ResourceExtractor()

    def build_graph(
        self,
        arm_template: Dict[str, Any],
        include_properties: bool = True,
    ) -> nx.DiGraph:
        """
        Build a directed dependency graph from an ARM template.

        Args:
            arm_template: ARM template as dictionary
            include_properties: Include resource properties as node attributes

        Returns:
            NetworkX directed graph where:
            - Nodes represent resources
            - Edges represent dependencies (A depends on B)
            - Node attributes contain resource metadata
        """
        resources = self.extractor.extract_resources(arm_template)
        dependencies = self.extractor.extract_resource_dependencies(arm_template)

        graph = nx.DiGraph()

        # Add nodes (resources)
        for resource in resources:
            node_id = resource["id"]
            node_attrs = {
                "type": resource["type"],
                "name": resource["name"],
                "location": resource.get("location"),
                "api_version": resource.get("apiVersion"),
            }

            # Optionally include full properties
            if include_properties:
                node_attrs["properties"] = resource.get("properties", {})
                node_attrs["sku"] = resource.get("sku")
                node_attrs["kind"] = resource.get("kind")
                node_attrs["tags"] = resource.get("tags", {})

            graph.add_node(node_id, **node_attrs)

        # Add edges (dependencies)
        for resource_id, deps in dependencies.items():
            for dep_id in deps:
                # Edge from resource TO its dependency (resource depends on dep)
                if dep_id in graph:
                    graph.add_edge(resource_id, dep_id, relationship="dependsOn")
                else:
                    logger.warning(
                        f"Dependency {dep_id} not found in graph for {resource_id}"
                    )

        logger.info(
            f"Built graph with {graph.number_of_nodes()} nodes and "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    def get_resource_subgraph(
        self, graph: nx.DiGraph, resource_types: List[str]
    ) -> nx.DiGraph:
        """
        Extract a subgraph containing only specific resource types.

        Args:
            graph: Full dependency graph
            resource_types: List of resource types to include

        Returns:
            Subgraph containing only specified resource types
        """
        nodes_to_keep = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get("type") in resource_types
        ]
        return graph.subgraph(nodes_to_keep).copy()

    def get_downstream_dependencies(
        self, graph: nx.DiGraph, resource_id: str
    ) -> List[str]:
        """
        Get all resources that a given resource depends on (downstream).

        Args:
            graph: Dependency graph
            resource_id: ID of the resource

        Returns:
            List of resource IDs that this resource depends on
        """
        if resource_id not in graph:
            return []

        # Successors in a directed graph (nodes this resource points to)
        return list(graph.successors(resource_id))

    def get_upstream_dependents(
        self, graph: nx.DiGraph, resource_id: str
    ) -> List[str]:
        """
        Get all resources that depend on a given resource (upstream).

        Args:
            graph: Dependency graph
            resource_id: ID of the resource

        Returns:
            List of resource IDs that depend on this resource
        """
        if resource_id not in graph:
            return []

        # Predecessors in a directed graph (nodes that point to this resource)
        return list(graph.predecessors(resource_id))

    def get_connected_component(
        self, graph: nx.DiGraph, resource_id: str
    ) -> nx.DiGraph:
        """
        Get the weakly connected component containing a resource.

        Args:
            graph: Dependency graph
            resource_id: ID of the resource

        Returns:
            Subgraph of the connected component
        """
        if resource_id not in graph:
            return nx.DiGraph()

        # Convert to undirected for connectivity analysis
        undirected = graph.to_undirected()
        component_nodes = nx.node_connected_component(undirected, resource_id)

        return graph.subgraph(component_nodes).copy()

    def export_to_graphml(
        self, graph: nx.DiGraph, output_file: str
    ) -> None:
        """
        Export graph to GraphML format (for Gephi, Cytoscape, etc.).

        Args:
            graph: Dependency graph
            output_file: Path to output GraphML file
        """
        # Convert complex attributes to strings for GraphML compatibility
        graph_copy = graph.copy()
        for node, attrs in graph_copy.nodes(data=True):
            for key, value in attrs.items():
                if isinstance(value, (dict, list)):
                    attrs[key] = str(value)

        nx.write_graphml(graph_copy, output_file)
        logger.info(f"Exported graph to {output_file}")

    def export_to_json(
        self, graph: nx.DiGraph, output_file: str
    ) -> None:
        """
        Export graph to JSON format (node-link format).

        Args:
            graph: Dependency graph
            output_file: Path to output JSON file
        """
        import json

        data = nx.node_link_data(graph)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported graph to {output_file}")

    def get_graph_statistics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Compute basic statistics about the dependency graph.

        Args:
            graph: Dependency graph

        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "num_weakly_connected_components": nx.number_weakly_connected_components(graph),
            "is_dag": nx.is_directed_acyclic_graph(graph),
        }

        if graph.number_of_nodes() > 0:
            stats["avg_in_degree"] = sum(dict(graph.in_degree()).values()) / graph.number_of_nodes()
            stats["avg_out_degree"] = sum(dict(graph.out_degree()).values()) / graph.number_of_nodes()

        # Resource type distribution
        type_counts = {}
        for node, attrs in graph.nodes(data=True):
            resource_type = attrs.get("type", "Unknown")
            type_counts[resource_type] = type_counts.get(resource_type, 0) + 1

        stats["resource_types"] = type_counts

        return stats


def build_dependency_graph(arm_template: Dict[str, Any]) -> nx.DiGraph:
    """
    Convenience function to build a dependency graph from an ARM template.

    Args:
        arm_template: ARM template as dictionary

    Returns:
        NetworkX directed graph
    """
    builder = DependencyGraphBuilder()
    return builder.build_graph(arm_template)
