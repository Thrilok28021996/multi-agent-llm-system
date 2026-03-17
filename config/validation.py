"""Configuration validation for Company AGI startup."""

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Critical - prevents startup
    WARNING = "warning"  # Non-critical - should be addressed
    INFO = "info"        # Informational


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        prefix = {
            ValidationSeverity.ERROR: "[ERROR]",
            ValidationSeverity.WARNING: "[WARN]",
            ValidationSeverity.INFO: "[INFO]"
        }[self.severity]

        result = f"{prefix} [{self.category}] {self.message}"
        if self.suggestion:
            result += f"\n         Suggestion: {self.suggestion}"
        return result


class ConfigValidator:
    """
    Validates Company AGI configuration on startup.

    Checks:
    - Ollama availability (package installed + server reachable)
    - Required models pulled in Ollama
    - Directory permissions
    - Environment variables
    - Memory and storage configuration
    """

    def __init__(self, settings: Any = None):
        if settings is None:
            from config.settings import settings as default_settings
            self.settings = default_settings
        else:
            self.settings = settings

        self.issues: List[ValidationIssue] = []

    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if no critical errors."""
        self.issues.clear()
        self._validate_directories()
        self._validate_llm_backend()
        self._validate_models()
        self._validate_settings_values()
        self._validate_environment()
        errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
        return len(errors) == 0

    def _validate_directories(self) -> None:
        """Validate output directories exist and are writable."""
        from pathlib import Path
        directories = [
            ("output_dir", self.settings.output_dir),
            ("solutions_dir", self.settings.solutions_dir),
            ("reports_dir", self.settings.reports_dir),
            ("logs_dir", self.settings.logs_dir),
            ("memory_dir", self.settings.memory.chroma_persist_dir),
        ]

        for name, path_str in directories:
            path = Path(path_str)
            try:
                path.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="Directory",
                    message=f"Cannot create directory: {path}",
                    suggestion="Check filesystem permissions"
                ))
                continue
            except Exception as e:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="Directory",
                    message=f"Error creating {name}: {e}",
                ))
                continue

            test_file = path / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="Directory",
                    message=f"Directory not writable: {path}",
                    suggestion="Check filesystem permissions"
                ))

    def _validate_llm_backend(self) -> None:
        """Validate that Ollama is installed and the server is reachable."""
        # Check package
        try:
            import ollama  # noqa
        except ImportError:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="LLMBackend",
                message="ollama Python package not installed",
                suggestion="Run: pip install ollama"
            ))
            return

        # Check server reachability
        try:
            import ollama as _ollama
            from config.llm_client import get_ollama_host
            host = get_ollama_host()
            client = _ollama.Client(host=host)
            client.list()  # lightweight ping
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="LLMBackend",
                message=f"Ollama server reachable at {host}"
            ))
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="LLMBackend",
                message=f"Ollama server not reachable: {e}",
                suggestion="Start Ollama with: ollama serve"
            ))

    def _validate_models(self) -> None:
        """Validate that required Ollama models are pulled."""
        try:
            import ollama as _ollama
            from config.llm_client import get_ollama_host
            from config.models import MODEL_CONFIGS

            host = get_ollama_host()
            client = _ollama.Client(host=host)
            pulled = {m["name"] for m in client.list().get("models", [])}

            required = {spec.ollama_model for spec in MODEL_CONFIGS.values()}
            missing = []
            for tag in required:
                # Ollama list may include digest suffix; match by base tag
                if not any(tag in p for p in pulled):
                    missing.append(tag)

            if missing:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="Models",
                    message=f"Models not pulled: {', '.join(missing)}",
                    suggestion="Run: " + "  &&  ".join(f"ollama pull {m}" for m in missing)
                ))
        except Exception:
            pass  # Server unreachable — already reported above

    def _validate_settings_values(self) -> None:
        """Validate configuration values are within acceptable ranges."""
        if self.settings.llm.timeout < 10:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Settings",
                message="LLM timeout is very low (< 10s)",
                suggestion="Consider increasing timeout for complex prompts"
            ))

        if self.settings.llm.temperature < 0 or self.settings.llm.temperature > 2:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="Settings",
                message="Invalid temperature value (must be 0-2)",
            ))

        if self.settings.llm.num_ctx < 1024:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Settings",
                message="Context window is very small (< 1024)",
                suggestion="Consider increasing num_ctx for better context"
            ))

        if self.settings.research.rate_limit_delay < 0.5:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Settings",
                message="Rate limit delay is very low (< 0.5s)",
                suggestion="Low delay may cause rate limiting from sources"
            ))

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.settings.workflow.log_level.upper() not in valid_log_levels:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Settings",
                message=f"Invalid log level: {self.settings.workflow.log_level}",
                suggestion=f"Use one of: {', '.join(valid_log_levels)}"
            ))

    def _validate_environment(self) -> None:
        """Validate environment variables and system requirements."""
        if sys.version_info < (3, 9):
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="Environment",
                message=f"Python {sys.version_info.major}.{sys.version_info.minor} detected",
                suggestion="Python 3.9+ is recommended"
            ))

        optional_deps = [
            ("loguru", "Enhanced logging"),
            ("chromadb", "Vector memory storage"),
            ("aiohttp", "Async HTTP requests"),
            ("beautifulsoup4", "Web scraping"),
        ]

        for package, description in optional_deps:
            try:
                __import__(package)
            except ImportError:
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="Dependencies",
                    message=f"Optional: {package} not installed ({description})",
                    suggestion=f"pip install {package}"
                ))

    def get_report(self) -> str:
        """Generate a validation report."""
        lines = [
            "=" * 60,
            "Company AGI Configuration Validation Report",
            "=" * 60,
            ""
        ]

        if not self.issues:
            lines.append("All validations passed successfully.")
        else:
            errors = [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
            warnings = [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
            infos = [i for i in self.issues if i.severity == ValidationSeverity.INFO]

            if errors:
                lines.append(f"ERRORS ({len(errors)}):")
                lines.append("-" * 40)
                for issue in errors:
                    lines.append(str(issue))
                lines.append("")

            if warnings:
                lines.append(f"WARNINGS ({len(warnings)}):")
                lines.append("-" * 40)
                for issue in warnings:
                    lines.append(str(issue))
                lines.append("")

            if infos:
                lines.append(f"INFO ({len(infos)}):")
                lines.append("-" * 40)
                for issue in infos:
                    lines.append(str(issue))
                lines.append("")

        lines.extend([
            "",
            "=" * 60,
            f"Summary: {len([i for i in self.issues if i.severity == ValidationSeverity.ERROR])} errors, "
            f"{len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])} warnings, "
            f"{len([i for i in self.issues if i.severity == ValidationSeverity.INFO])} info",
            "=" * 60
        ])

        return "\n".join(lines)

    def print_report(self) -> None:
        """Print the validation report to stdout."""
        logger.info(self.get_report())


def validate_config_on_startup(exit_on_error: bool = True) -> bool:
    """Validate configuration on startup."""
    validator = ConfigValidator()
    is_valid = validator.validate_all()
    validator.print_report()

    if not is_valid and exit_on_error:
        logger.error("Startup aborted due to configuration errors.")
        sys.exit(1)

    return is_valid


__all__ = [
    "ConfigValidator",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_config_on_startup",
]
