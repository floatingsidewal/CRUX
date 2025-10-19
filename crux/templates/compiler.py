"""
Template Compiler

Compiles Bicep templates to ARM JSON using Azure CLI.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BicepCompiler:
    """Compiles Bicep templates to ARM JSON."""

    def __init__(self):
        """Initialize the Bicep compiler and verify installation."""
        self._verify_bicep_installation()

    def _verify_bicep_installation(self) -> None:
        """Verify that Bicep CLI is installed and available."""
        try:
            result = subprocess.run(
                ["az", "bicep", "version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Bicep CLI version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Bicep CLI not found or not working. "
                "Please ensure Azure CLI and Bicep are installed. "
                f"Error: {e}"
            )

    def compile(
        self,
        bicep_file: Path,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Compile a Bicep file to ARM JSON.

        Args:
            bicep_file: Path to the Bicep template file
            output_file: Optional path for output JSON (default: temp file)

        Returns:
            Dictionary containing the ARM template JSON

        Raises:
            subprocess.CalledProcessError: If compilation fails
            json.JSONDecodeError: If output is not valid JSON
        """
        if not bicep_file.exists():
            raise FileNotFoundError(f"Bicep file not found: {bicep_file}")

        logger.info(f"Compiling {bicep_file}...")

        # Use stdout if no output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                "az",
                "bicep",
                "build",
                "--file",
                str(bicep_file),
                "--outfile",
                str(output_file),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            with open(output_file, "r") as f:
                arm_template = json.load(f)
        else:
            cmd = ["az", "bicep", "build", "--file", str(bicep_file), "--stdout"]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            arm_template = json.loads(result.stdout)

        logger.debug(
            f"Compiled successfully: {len(arm_template.get('resources', []))} resources"
        )
        return arm_template

    def compile_string(self, bicep_content: str) -> Dict[str, Any]:
        """
        Compile Bicep content from a string.

        Args:
            bicep_content: Bicep template content as a string

        Returns:
            Dictionary containing the ARM template JSON

        Raises:
            subprocess.CalledProcessError: If compilation fails
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bicep", delete=False) as f:
            f.write(bicep_content)
            temp_file = Path(f.name)

        try:
            return self.compile(temp_file)
        finally:
            temp_file.unlink()  # Clean up temp file

    def compile_batch(
        self,
        bicep_files: list[Path],
        output_dir: Optional[Path] = None,
        continue_on_error: bool = True,
    ) -> Dict[Path, Dict[str, Any]]:
        """
        Compile multiple Bicep files in batch.

        Args:
            bicep_files: List of Bicep file paths
            output_dir: Directory for output files (optional)
            continue_on_error: Continue if individual files fail

        Returns:
            Dictionary mapping file paths to their compiled ARM templates

        Raises:
            subprocess.CalledProcessError: If compilation fails and continue_on_error is False
        """
        results = {}
        errors = {}

        for bicep_file in bicep_files:
            try:
                if output_dir:
                    output_file = output_dir / f"{bicep_file.stem}.json"
                else:
                    output_file = None

                arm_template = self.compile(bicep_file, output_file)
                results[bicep_file] = arm_template

            except Exception as e:
                logger.warning(f"Failed to compile {bicep_file}: {e}")
                errors[bicep_file] = str(e)

                if not continue_on_error:
                    raise

        if errors:
            logger.warning(
                f"Compilation errors: {len(errors)}/{len(bicep_files)} files failed"
            )
            for file, error in list(errors.items())[:5]:  # Show first 5 errors
                logger.warning(f"  {file.name}: {error}")

        logger.info(
            f"Compiled {len(results)}/{len(bicep_files)} files successfully"
        )
        return results


def compile_bicep_file(bicep_file: Path) -> Dict[str, Any]:
    """
    Convenience function to compile a single Bicep file.

    Args:
        bicep_file: Path to Bicep template

    Returns:
        ARM template as dictionary
    """
    compiler = BicepCompiler()
    return compiler.compile(bicep_file)
