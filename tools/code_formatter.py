"""Code formatter for generated code - supports multiple languages and formatters."""

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class FormatterType(Enum):
    """Supported code formatters."""
    BLACK = "black"           # Python
    ISORT = "isort"           # Python imports
    PRETTIER = "prettier"     # JavaScript/TypeScript/JSON/YAML/etc
    GOFMT = "gofmt"          # Go
    RUSTFMT = "rustfmt"      # Rust
    CLANG_FORMAT = "clang-format"  # C/C++/Java


@dataclass
class FormatResult:
    """Result of a formatting operation."""
    success: bool
    files_formatted: List[str]
    errors: List[str]
    output: str
    formatter: str


class CodeFormatter:
    """
    Multi-language code formatter.
    Supports Python (black, isort), JavaScript/TypeScript (prettier),
    Go (gofmt), Rust (rustfmt), and C/C++/Java (clang-format).
    """

    # File extensions to formatter mapping
    EXTENSION_TO_FORMATTER = {
        ".py": [FormatterType.BLACK, FormatterType.ISORT],
        ".pyi": [FormatterType.BLACK, FormatterType.ISORT],
        ".js": [FormatterType.PRETTIER],
        ".jsx": [FormatterType.PRETTIER],
        ".ts": [FormatterType.PRETTIER],
        ".tsx": [FormatterType.PRETTIER],
        ".json": [FormatterType.PRETTIER],
        ".yaml": [FormatterType.PRETTIER],
        ".yml": [FormatterType.PRETTIER],
        ".md": [FormatterType.PRETTIER],
        ".css": [FormatterType.PRETTIER],
        ".scss": [FormatterType.PRETTIER],
        ".html": [FormatterType.PRETTIER],
        ".go": [FormatterType.GOFMT],
        ".rs": [FormatterType.RUSTFMT],
        ".c": [FormatterType.CLANG_FORMAT],
        ".cpp": [FormatterType.CLANG_FORMAT],
        ".h": [FormatterType.CLANG_FORMAT],
        ".hpp": [FormatterType.CLANG_FORMAT],
        ".java": [FormatterType.CLANG_FORMAT],
    }

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._check_available_formatters()

    def _check_available_formatters(self) -> None:
        """Check which formatters are available on the system."""
        self.available_formatters = {}

        formatter_commands = {
            FormatterType.BLACK: "black",
            FormatterType.ISORT: "isort",
            FormatterType.PRETTIER: "prettier",
            FormatterType.GOFMT: "gofmt",
            FormatterType.RUSTFMT: "rustfmt",
            FormatterType.CLANG_FORMAT: "clang-format",
        }

        for formatter, cmd in formatter_commands.items():
            self.available_formatters[formatter] = shutil.which(cmd) is not None

    def is_formatter_available(self, formatter: FormatterType) -> bool:
        """Check if a specific formatter is available."""
        return self.available_formatters.get(formatter, False)

    def get_formatters_for_file(self, file_path: str) -> List[FormatterType]:
        """Get the appropriate formatters for a file based on extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_TO_FORMATTER.get(ext, [])

    def format_file(
        self,
        file_path: str,
        formatters: Optional[List[FormatterType]] = None
    ) -> FormatResult:
        """
        Format a single file.

        Args:
            file_path: Path to the file to format
            formatters: List of formatters to use (auto-detect if not specified)

        Returns:
            FormatResult with details about the formatting
        """
        path = Path(file_path)

        if not path.exists():
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[f"File not found: {file_path}"],
                output="",
                formatter=""
            )

        # Auto-detect formatters if not specified
        if formatters is None:
            formatters = self.get_formatters_for_file(str(path))

        if not formatters:
            return FormatResult(
                success=True,
                files_formatted=[],
                errors=[f"No formatter available for: {path.suffix}"],
                output="",
                formatter=""
            )

        errors = []
        formatted = False
        output_parts = []

        for formatter in formatters:
            if not self.is_formatter_available(formatter):
                errors.append(f"{formatter.value} not installed")
                continue

            result = self._run_formatter(formatter, str(path))
            if result.success:
                formatted = True
                output_parts.append(result.output)
            else:
                errors.extend(result.errors)

        return FormatResult(
            success=formatted or len(errors) == 0,
            files_formatted=[str(path)] if formatted else [],
            errors=errors,
            output="\n".join(output_parts),
            formatter=", ".join(f.value for f in formatters if self.is_formatter_available(f))
        )

    def format_directory(
        self,
        directory: str,
        recursive: bool = True,
        exclude_patterns: Optional[List[str]] = None
    ) -> FormatResult:
        """
        Format all supported files in a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to format files in subdirectories
            exclude_patterns: Patterns to exclude (e.g., ["node_modules", "__pycache__"])

        Returns:
            FormatResult with details about the formatting
        """
        path = Path(directory)

        if not path.exists():
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[f"Directory not found: {directory}"],
                output="",
                formatter=""
            )

        exclude_patterns = exclude_patterns or ["node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"]

        # Find all supported files
        files_to_format: Dict[FormatterType, List[Path]] = {}

        pattern = "**/*" if recursive else "*"
        for ext, formatters in self.EXTENSION_TO_FORMATTER.items():
            for file_path in path.glob(f"{pattern}{ext}"):
                # Check exclusions
                should_exclude = False
                for exclude in exclude_patterns:
                    if exclude in str(file_path):
                        should_exclude = True
                        break

                if should_exclude:
                    continue

                # Group files by formatter
                for formatter in formatters:
                    if formatter not in files_to_format:
                        files_to_format[formatter] = []
                    files_to_format[formatter].append(file_path)

        # Run formatters
        all_formatted = []
        all_errors = []
        all_output = []

        for formatter, files in files_to_format.items():
            if not files:
                continue

            if not self.is_formatter_available(formatter):
                all_errors.append(f"{formatter.value} not installed, skipping {len(files)} files")
                continue

            result = self._run_formatter_batch(formatter, files)
            all_formatted.extend(result.files_formatted)
            all_errors.extend(result.errors)
            if result.output:
                all_output.append(result.output)

        return FormatResult(
            success=len(all_errors) == 0,
            files_formatted=all_formatted,
            errors=all_errors,
            output="\n".join(all_output),
            formatter="multiple"
        )

    def _run_formatter(self, formatter: FormatterType, file_path: str) -> FormatResult:
        """Run a specific formatter on a file."""
        commands = {
            FormatterType.BLACK: ["black", file_path],
            FormatterType.ISORT: ["isort", file_path],
            FormatterType.PRETTIER: ["prettier", "--write", file_path],
            FormatterType.GOFMT: ["gofmt", "-w", file_path],
            FormatterType.RUSTFMT: ["rustfmt", file_path],
            FormatterType.CLANG_FORMAT: ["clang-format", "-i", file_path],
        }

        cmd = commands.get(formatter)
        if not cmd:
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[f"Unknown formatter: {formatter}"],
                output="",
                formatter=formatter.value
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                return FormatResult(
                    success=True,
                    files_formatted=[file_path],
                    errors=[],
                    output=result.stdout,
                    formatter=formatter.value
                )
            else:
                return FormatResult(
                    success=False,
                    files_formatted=[],
                    errors=[result.stderr or f"Formatter {formatter.value} failed"],
                    output=result.stdout,
                    formatter=formatter.value
                )

        except subprocess.TimeoutExpired:
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[f"Formatter timed out after {self.timeout}s"],
                output="",
                formatter=formatter.value
            )
        except Exception as e:
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[str(e)],
                output="",
                formatter=formatter.value
            )

    def _run_formatter_batch(
        self,
        formatter: FormatterType,
        files: List[Path]
    ) -> FormatResult:
        """Run a formatter on multiple files at once."""
        file_paths = [str(f) for f in files]

        # Some formatters support multiple files at once
        batch_commands = {
            FormatterType.BLACK: ["black"] + file_paths,
            FormatterType.ISORT: ["isort"] + file_paths,
            FormatterType.PRETTIER: ["prettier", "--write"] + file_paths,
            FormatterType.GOFMT: ["gofmt", "-w"] + file_paths,
        }

        cmd = batch_commands.get(formatter)

        # Fall back to individual formatting if batch not supported
        if not cmd:
            all_formatted = []
            all_errors = []
            for file_path in file_paths:
                result = self._run_formatter(formatter, file_path)
                all_formatted.extend(result.files_formatted)
                all_errors.extend(result.errors)
            return FormatResult(
                success=len(all_errors) == 0,
                files_formatted=all_formatted,
                errors=all_errors,
                output="",
                formatter=formatter.value
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout * len(files)  # Scale timeout with file count
            )

            if result.returncode == 0:
                return FormatResult(
                    success=True,
                    files_formatted=file_paths,
                    errors=[],
                    output=result.stdout,
                    formatter=formatter.value
                )
            else:
                return FormatResult(
                    success=False,
                    files_formatted=[],
                    errors=[result.stderr or f"Formatter {formatter.value} failed"],
                    output=result.stdout,
                    formatter=formatter.value
                )

        except subprocess.TimeoutExpired:
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[f"Formatter timed out"],
                output="",
                formatter=formatter.value
            )
        except Exception as e:
            return FormatResult(
                success=False,
                files_formatted=[],
                errors=[str(e)],
                output="",
                formatter=formatter.value
            )


def format_project(project_path: str, recursive: bool = True) -> FormatResult:
    """Convenience function to format a project."""
    formatter = CodeFormatter()
    return formatter.format_directory(project_path, recursive=recursive)


def format_file(file_path: str) -> FormatResult:
    """Convenience function to format a single file."""
    formatter = CodeFormatter()
    return formatter.format_file(file_path)
