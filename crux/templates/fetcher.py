"""
Template Fetcher

Downloads Azure Quickstart Templates from GitHub and filters Bicep files.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TemplateFetcher:
    """Fetches Azure templates from GitHub repositories."""

    DEFAULT_REPO = "https://github.com/Azure/azure-quickstart-templates.git"
    DEFAULT_BRANCH = "master"

    def __init__(self, output_dir: str = "templates"):
        """
        Initialize the template fetcher.

        Args:
            output_dir: Directory to store downloaded templates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_repo(
        self,
        repo_url: Optional[str] = None,
        branch: Optional[str] = None,
        shallow: bool = True,
    ) -> Path:
        """
        Clone or update a GitHub repository.

        Args:
            repo_url: GitHub repository URL (default: Azure Quickstart Templates)
            branch: Git branch to clone (default: master)
            shallow: Use shallow clone for faster downloads

        Returns:
            Path to the cloned repository directory

        Raises:
            subprocess.CalledProcessError: If git clone/pull fails
        """
        repo_url = repo_url or self.DEFAULT_REPO
        branch = branch or self.DEFAULT_BRANCH

        # Extract repo name from URL
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = self.output_dir / repo_name

        if repo_path.exists():
            logger.info(f"Repository already exists at {repo_path}, updating...")
            subprocess.run(
                ["git", "-C", str(repo_path), "pull"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            logger.info(f"Cloning {repo_url} to {repo_path}...")
            cmd = ["git", "clone"]
            if shallow:
                cmd.extend(["--depth", "1"])
            cmd.extend(["--branch", branch, repo_url, str(repo_path)])

            subprocess.run(cmd, check=True, capture_output=True, text=True)

        logger.info(f"Repository ready at {repo_path}")
        return repo_path

    def discover_bicep_files(
        self,
        repo_path: Path,
        pattern: str = "**/*.bicep",
        exclude_patterns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Path]:
        """
        Discover Bicep template files in a repository.

        Args:
            repo_path: Path to the repository
            pattern: Glob pattern to match files (default: **/*.bicep)
            exclude_patterns: List of patterns to exclude
            limit: Maximum number of templates to return

        Returns:
            List of paths to Bicep files
        """
        exclude_patterns = exclude_patterns or [
            "**/test/**",
            "**/tests/**",
            "**/.test/**",
            "**/modules/**",  # Exclude standalone modules, focus on complete templates
        ]

        logger.info(f"Discovering Bicep files in {repo_path} with pattern {pattern}...")

        bicep_files = []
        for bicep_file in repo_path.glob(pattern):
            # Check if file should be excluded
            should_exclude = any(bicep_file.match(pattern) for pattern in exclude_patterns)
            if should_exclude:
                continue

            # Check if it's a regular file
            if bicep_file.is_file():
                bicep_files.append(bicep_file)

        logger.info(f"Found {len(bicep_files)} Bicep files")

        if limit:
            bicep_files = bicep_files[:limit]
            logger.info(f"Limited to {limit} files")

        return bicep_files

    def fetch_templates(
        self,
        repo_url: Optional[str] = None,
        pattern: str = "quickstarts/**/*.bicep",
        limit: Optional[int] = None,
    ) -> List[Path]:
        """
        High-level method to fetch templates from a repository.

        Args:
            repo_url: GitHub repository URL
            pattern: Glob pattern to match Bicep files
            limit: Maximum number of templates to return

        Returns:
            List of paths to Bicep template files
        """
        repo_path = self.fetch_repo(repo_url)
        templates = self.discover_bicep_files(repo_path, pattern=pattern, limit=limit)
        return templates


def fetch_azure_quickstart_templates(
    output_dir: str = "templates",
    limit: Optional[int] = None,
) -> List[Path]:
    """
    Convenience function to fetch Azure Quickstart Templates.

    Args:
        output_dir: Directory to store templates
        limit: Maximum number of templates to fetch

    Returns:
        List of paths to Bicep template files
    """
    fetcher = TemplateFetcher(output_dir=output_dir)
    return fetcher.fetch_templates(
        pattern="quickstarts/**/*.bicep",
        limit=limit,
    )
