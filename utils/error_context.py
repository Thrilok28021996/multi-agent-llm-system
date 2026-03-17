"""
Error Context Collection for Company AGI.

Provides comprehensive error context capture and reporting:
- Detailed error information with stack traces
- System state at time of error
- Recent actions leading to error
- Remediation suggestions
"""

import json
import os
import platform
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    MODEL_ERROR = "model_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


@dataclass
class SystemState:
    """System state snapshot at time of error."""
    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    working_directory: str = ""
    memory_usage_mb: float = 0.0
    available_memory_mb: float = 0.0
    cpu_count: int = 0
    env_vars: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, include_env: bool = False) -> "SystemState":
        """Capture current system state."""
        state = cls(
            python_version=sys.version,
            platform=platform.platform(),
            hostname=platform.node(),
            working_directory=os.getcwd(),
            cpu_count=os.cpu_count() or 0,
        )

        # Try to get memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            state.memory_usage_mb = mem.used / (1024 * 1024)
            state.available_memory_mb = mem.available / (1024 * 1024)
        except ImportError:
            pass

        # Include safe env vars if requested
        if include_env:
            safe_env_keys = ["OLLAMA_HOST", "HOME", "USER", "PATH"]
            state.env_vars = {
                k: v for k, v in os.environ.items()
                if k in safe_env_keys or k.startswith("COMPANY_AGI_")
            }

        return state

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionRecord:
    """Record of a recent action."""
    timestamp: str
    action: str
    details: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorContext:
    """Comprehensive error context."""
    error_id: str
    timestamp: str
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    stack_trace: str
    system_state: SystemState
    recent_actions: List[ActionRecord] = field(default_factory=list)
    attempted_remediation: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    extra_context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        recent_actions: Optional[List[ActionRecord]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> "ErrorContext":
        """Create error context from an exception."""
        import uuid

        # Auto-detect category if not provided
        if category is None:
            category = cls._detect_category(error)

        # Auto-detect severity if not provided
        if severity is None:
            severity = cls._detect_severity(error, category)

        # Generate suggestions based on error
        suggestions = cls._generate_suggestions(error, category)

        return cls(
            error_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=category,
            severity=severity,
            stack_trace=traceback.format_exc(),
            system_state=SystemState.capture(),
            recent_actions=recent_actions or [],
            suggestions=suggestions,
            extra_context=extra_context or {},
        )

    @staticmethod
    def _detect_category(error: Exception) -> ErrorCategory:
        """Auto-detect error category from exception type."""
        error_type = type(error).__name__.lower()
        error_msg = str(error).lower()

        if "timeout" in error_type or "timeout" in error_msg:
            return ErrorCategory.TIMEOUT_ERROR
        if "connection" in error_type or "connect" in error_msg:
            return ErrorCategory.CONNECTION_ERROR
        if "permission" in error_type or "denied" in error_msg:
            return ErrorCategory.PERMISSION_ERROR
        if "model" in error_msg or "llm" in error_msg or "ollama" in error_msg:
            return ErrorCategory.MODEL_ERROR
        if "validation" in error_type or "invalid" in error_msg:
            return ErrorCategory.VALIDATION_ERROR
        if "memory" in error_msg or "resource" in error_msg:
            return ErrorCategory.RESOURCE_ERROR
        if "config" in error_msg or "setting" in error_msg:
            return ErrorCategory.CONFIGURATION_ERROR

        return ErrorCategory.UNKNOWN

    @staticmethod
    def _detect_severity(
        error: Exception,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """Auto-detect error severity."""
        # Error can be used for more detailed severity detection
        _ = error
        # Critical categories
        if category in [ErrorCategory.RESOURCE_ERROR]:
            return ErrorSeverity.CRITICAL

        # High severity categories
        if category in [
            ErrorCategory.CONNECTION_ERROR,
            ErrorCategory.MODEL_ERROR,
            ErrorCategory.PERMISSION_ERROR,
        ]:
            return ErrorSeverity.HIGH

        # Medium severity
        if category in [
            ErrorCategory.TIMEOUT_ERROR,
            ErrorCategory.CONFIGURATION_ERROR,
        ]:
            return ErrorSeverity.MEDIUM

        return ErrorSeverity.LOW

    @staticmethod
    def _generate_suggestions(
        error: Exception,
        category: ErrorCategory,
    ) -> List[str]:
        """Generate remediation suggestions based on error."""
        suggestions = []
        error_msg = str(error).lower()

        if category == ErrorCategory.MODEL_ERROR:
            suggestions.extend([
                "Ensure Ollama is running: ollama serve",
                "Pull required models: ollama pull qwen3:8b",
            ])

        if category == ErrorCategory.CONNECTION_ERROR:
            suggestions.extend([
                "Check network connectivity",
                "Verify Ollama server URL (default: http://localhost:11434)",
                "Check if firewall is blocking connections",
            ])

        if category == ErrorCategory.TIMEOUT_ERROR:
            suggestions.extend([
                "Increase timeout settings",
                "Check system load and available resources",
                "Try a smaller model if using large models",
            ])

        if category == ErrorCategory.PERMISSION_ERROR:
            suggestions.extend([
                "Check file/directory permissions",
                "Run with appropriate user permissions",
                f"Current working directory: {os.getcwd()}",
            ])

        if category == ErrorCategory.RESOURCE_ERROR:
            suggestions.extend([
                "Close other applications to free memory",
                "Check available disk space",
                "Consider using a smaller model",
            ])

        if category == ErrorCategory.CONFIGURATION_ERROR:
            suggestions.extend([
                "Validate configuration file syntax",
                "Check for required environment variables",
                "Review configuration documentation",
            ])

        # Add specific suggestions based on error message
        if "not found" in error_msg:
            suggestions.append("Verify the resource exists and path is correct")

        if "memory" in error_msg:
            suggestions.append("Try reducing batch size or context length")

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "stack_trace": self.stack_trace,
            "system_state": self.system_state.to_dict(),
            "recent_actions": [a.to_dict() for a in self.recent_actions],
            "attempted_remediation": self.attempted_remediation,
            "suggestions": self.suggestions,
            "extra_context": self.extra_context,
        }

    def to_terminal(self, color: bool = True, verbose: bool = False) -> str:
        """Format error context for terminal display."""
        colors = {
            "red": "\033[31m",
            "yellow": "\033[33m",
            "cyan": "\033[36m",
            "green": "\033[32m",
            "bold": "\033[1m",
            "reset": "\033[0m",
        }

        if not color:
            colors = {k: "" for k in colors}

        severity_colors = {
            ErrorSeverity.LOW: colors["cyan"],
            ErrorSeverity.MEDIUM: colors["yellow"],
            ErrorSeverity.HIGH: colors["red"],
            ErrorSeverity.CRITICAL: colors["bold"] + colors["red"],
        }

        lines = [
            f"{colors['bold']}Error Report [{self.error_id}]{colors['reset']}",
            f"",
            f"{colors['red']}Error:{colors['reset']} {self.error_type}: {self.error_message}",
            f"Category: {self.category.value}",
            f"Severity: {severity_colors[self.severity]}{self.severity.value}{colors['reset']}",
            f"Time: {self.timestamp}",
        ]

        if self.recent_actions:
            lines.append(f"\n{colors['bold']}Recent Actions:{colors['reset']}")
            for action in self.recent_actions[-5:]:
                status = f"{colors['green']}+{colors['reset']}" if action.success else f"{colors['red']}-{colors['reset']}"
                lines.append(f"  {status} {action.action}")
                if action.details:
                    lines.append(f"      {action.details}")

        if self.suggestions:
            lines.append(f"\n{colors['bold']}Suggested Fixes:{colors['reset']}")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {suggestion}")

        if verbose:
            lines.append(f"\n{colors['bold']}Stack Trace:{colors['reset']}")
            lines.append(self.stack_trace)

            lines.append(f"\n{colors['bold']}System State:{colors['reset']}")
            lines.append(f"  Python: {self.system_state.python_version.split()[0]}")
            lines.append(f"  Platform: {self.system_state.platform}")
            if self.system_state.memory_usage_mb > 0:
                lines.append(f"  Memory: {self.system_state.memory_usage_mb:.0f}MB used / {self.system_state.available_memory_mb:.0f}MB available")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save error context to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ErrorCollector:
    """
    Collects and tracks errors across the application.

    Features:
    - Track recent actions for context
    - Collect and aggregate errors
    - Generate error reports
    """

    MAX_ACTIONS = 100
    MAX_ERRORS = 50

    def __init__(self, report_dir: Optional[Path] = None):
        self.report_dir = report_dir or Path("output/errors")
        self._actions: List[ActionRecord] = []
        self._errors: List[ErrorContext] = []

    def record_action(
        self,
        action: str,
        details: Optional[str] = None,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an action for error context."""
        record = ActionRecord(
            timestamp=datetime.now().isoformat(),
            action=action,
            details=details,
            success=success,
            duration_ms=duration_ms,
        )
        self._actions.append(record)

        # Trim old actions
        if len(self._actions) > self.MAX_ACTIONS:
            self._actions = self._actions[-self.MAX_ACTIONS:]

    def capture_error(
        self,
        error: Exception,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        save_report: bool = True,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ErrorContext:
        """Capture an error with full context."""
        context = ErrorContext.from_exception(
            error=error,
            category=category,
            severity=severity,
            recent_actions=self._actions.copy(),
            extra_context=extra_context,
        )

        self._errors.append(context)

        # Trim old errors
        if len(self._errors) > self.MAX_ERRORS:
            self._errors = self._errors[-self.MAX_ERRORS:]

        # Save report if requested
        if save_report:
            report_path = self.report_dir / f"error_{context.error_id}.json"
            context.save(report_path)

        return context

    def get_recent_errors(
        self,
        count: int = 10,
        category: Optional[ErrorCategory] = None,
    ) -> List[ErrorContext]:
        """Get recent errors."""
        errors = self._errors
        if category:
            errors = [e for e in errors if e.category == category]
        return errors[-count:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all collected errors."""
        by_category: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for error in self._errors:
            cat = error.category.value
            sev = error.severity.value
            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_errors": len(self._errors),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_actions_count": len(self._actions),
        }

    def clear(self) -> None:
        """Clear all collected data."""
        self._actions.clear()
        self._errors.clear()


# Global error collector instance
_error_collector: Optional[ErrorCollector] = None


def get_error_collector(
    report_dir: Optional[Path] = None,
) -> ErrorCollector:
    """Get or create the global error collector."""
    global _error_collector
    if _error_collector is None:
        _error_collector = ErrorCollector(report_dir=report_dir)
    return _error_collector


def reset_error_collector() -> None:
    """Reset the global error collector."""
    global _error_collector
    _error_collector = None


def capture_error(
    error: Exception,
    print_report: bool = True,
    verbose: bool = False,
    **kwargs: Any,
) -> ErrorContext:
    """Convenience function to capture an error."""
    collector = get_error_collector()
    context = collector.capture_error(error, **kwargs)

    if print_report:
        print(context.to_terminal(verbose=verbose))

    return context
