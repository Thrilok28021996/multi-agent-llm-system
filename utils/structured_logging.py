"""
Structured Logging with Correlation IDs for Company AGI.

Provides Claude Code-style logging with:
- Correlation IDs for request tracing across agents
- Structured JSON output for analysis
- Context propagation through workflows
- Terminal-friendly colored output
"""

import json
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogContext:
    """Logging context with correlation tracking."""
    correlation_id: str
    agent_name: Optional[str] = None
    phase: Optional[str] = None
    step_id: Optional[str] = None
    workflow_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_agent(self, agent_name: str) -> "LogContext":
        """Create new context with agent name."""
        return LogContext(
            correlation_id=self.correlation_id,
            agent_name=agent_name,
            phase=self.phase,
            step_id=self.step_id,
            workflow_id=self.workflow_id,
            extra=self.extra.copy(),
        )

    def with_phase(self, phase: str, step_id: Optional[str] = None) -> "LogContext":
        """Create new context with phase."""
        return LogContext(
            correlation_id=self.correlation_id,
            agent_name=self.agent_name,
            phase=phase,
            step_id=step_id or self.step_id,
            workflow_id=self.workflow_id,
            extra=self.extra.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "correlation_id": self.correlation_id,
        }
        if self.agent_name:
            result["agent_name"] = self.agent_name
        if self.phase:
            result["phase"] = self.phase
        if self.step_id:
            result["step_id"] = self.step_id
        if self.workflow_id:
            result["workflow_id"] = self.workflow_id
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class LogRecord:
    """A structured log record."""
    timestamp: str
    level: str
    message: str
    context: Dict[str, Any]
    source: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)

    def to_text(self, color: bool = True) -> str:
        """Convert to human-readable text."""
        colors = {
            "DEBUG": "\033[36m",     # Cyan
            "INFO": "\033[32m",      # Green
            "WARNING": "\033[33m",   # Yellow
            "ERROR": "\033[31m",     # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",
        }

        level_color = colors.get(self.level, "") if color else ""
        reset = colors["RESET"] if color else ""

        parts = [
            f"[{self.timestamp}]",
            f"{level_color}{self.level:8}{reset}",
        ]

        if self.context.get("agent_name"):
            parts.append(f"[{self.context['agent_name']}]")
        if self.context.get("phase"):
            parts.append(f"({self.context['phase']})")

        parts.append(self.message)

        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.1f}ms)")

        if self.error:
            parts.append(f"- {self.error.get('type', 'Error')}: {self.error.get('message', '')}")

        return " ".join(parts)


class StructuredLogger:
    """
    Structured logger with correlation ID support.

    Features:
    - Automatic correlation ID propagation
    - JSON output for analysis
    - Terminal-friendly colored output
    - Context-aware logging
    """

    def __init__(
        self,
        name: str = "company_agi",
        level: LogLevel = LogLevel.INFO,
        output: Optional[TextIO] = None,
        json_output: Optional[Path] = None,
        color_enabled: bool = True,
    ):
        self.name = name
        self.level = level
        self.output = output or sys.stderr
        self.json_output = json_output
        self.color_enabled = color_enabled

        # Thread-local context storage
        self._local = threading.local()
        self._handlers: List[Callable[[LogRecord], None]] = []
        self._json_file: Optional[TextIO] = None

        if json_output:
            json_output.parent.mkdir(parents=True, exist_ok=True)
            self._json_file = open(json_output, "a")

    @property
    def context(self) -> Optional[LogContext]:
        """Get current thread's log context."""
        return getattr(self._local, "context", None)

    @context.setter
    def context(self, ctx: Optional[LogContext]) -> None:
        """Set current thread's log context."""
        self._local.context = ctx

    def new_context(
        self,
        agent_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> LogContext:
        """Create a new logging context with correlation ID."""
        return LogContext(
            correlation_id=str(uuid.uuid4())[:8],
            agent_name=agent_name,
            workflow_id=workflow_id or str(uuid.uuid4())[:8],
        )

    @contextmanager
    def context_scope(self, ctx: LogContext):
        """Context manager for scoped logging context."""
        old_context = self.context
        self.context = ctx
        try:
            yield ctx
        finally:
            self.context = old_context

    @contextmanager
    def phase_scope(self, phase: str, step_id: Optional[str] = None):
        """Context manager for logging within a phase."""
        if self.context:
            new_ctx = self.context.with_phase(phase, step_id)
            old_context = self.context
            self.context = new_ctx
            try:
                yield new_ctx
            finally:
                self.context = old_context
        else:
            yield None

    @contextmanager
    def agent_scope(self, agent_name: str):
        """Context manager for logging within an agent."""
        if self.context:
            new_ctx = self.context.with_agent(agent_name)
            old_context = self.context
            self.context = new_ctx
            try:
                yield new_ctx
            finally:
                self.context = old_context
        else:
            yield None

    @contextmanager
    def timed_operation(self, operation: str, level: LogLevel = LogLevel.DEBUG):
        """Context manager that logs operation duration."""
        start = time.perf_counter()
        try:
            yield
            duration_ms = (time.perf_counter() - start) * 1000
            self._log(level, f"{operation} completed", duration_ms=duration_ms)
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            self._log(
                LogLevel.ERROR,
                f"{operation} failed",
                duration_ms=duration_ms,
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise

    def add_handler(self, handler: Callable[[LogRecord], None]) -> None:
        """Add a custom log handler."""
        self._handlers.append(handler)

    def _log(
        self,
        level: LogLevel,
        message: str,
        source: Optional[str] = None,
        duration_ms: Optional[float] = None,
        error: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        """Internal logging method."""
        if level.value < self.level.value:
            return

        context_dict = self.context.to_dict() if self.context else {}
        if extra:
            context_dict["extra"] = {**context_dict.get("extra", {}), **extra}

        record = LogRecord(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            message=message,
            context=context_dict,
            source=source or self.name,
            duration_ms=duration_ms,
            error=error,
        )

        # Write to console
        try:
            print(record.to_text(self.color_enabled), file=self.output)
        except Exception as e:
            import sys
            print(f"[StructuredLogger] Console write failed: {e}", file=sys.stderr)

        # Write to JSON file
        if self._json_file:
            try:
                self._json_file.write(record.to_json() + "\n")
                self._json_file.flush()
            except Exception as e:
                import sys
                print(f"[StructuredLogger] JSON write failed: {e}", file=sys.stderr)

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(record)
            except Exception as e:
                import sys
                print(f"[StructuredLogger] Handler failed: {e}", file=sys.stderr)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self._log(LogLevel.ERROR, message, error=error_dict, **kwargs)

    def critical(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log critical message."""
        error_dict = None
        if error:
            error_dict = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self._log(LogLevel.CRITICAL, message, error=error_dict, **kwargs)

    def close(self) -> None:
        """Close the logger and its resources."""
        if self._json_file:
            self._json_file.close()
            self._json_file = None


class LogAggregator:
    """Aggregates logs for analysis and reporting."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._records: List[LogRecord] = []

    def load_logs(self, pattern: str = "*.json") -> List[LogRecord]:
        """Load logs from JSON files."""
        records = []
        for log_file in self.log_dir.glob(pattern):
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            records.append(LogRecord(**data))
                        except (json.JSONDecodeError, TypeError):
                            continue  # Skip malformed log lines
            except OSError:
                continue  # Skip unreadable log files
        return records

    def filter_by_correlation(self, correlation_id: str) -> List[LogRecord]:
        """Filter logs by correlation ID."""
        return [
            r for r in self._records
            if r.context.get("correlation_id") == correlation_id
        ]

    def filter_by_agent(self, agent_name: str) -> List[LogRecord]:
        """Filter logs by agent name."""
        return [
            r for r in self._records
            if r.context.get("agent_name") == agent_name
        ]

    def filter_by_level(self, level: LogLevel) -> List[LogRecord]:
        """Filter logs by level."""
        return [
            r for r in self._records
            if LogLevel[r.level].value >= level.value
        ]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        errors = [r for r in self._records if r.error]
        by_type: Dict[str, int] = {}
        for error in errors:
            err_type = error.error.get("type", "Unknown") if error.error else "Unknown"
            by_type[err_type] = by_type.get(err_type, 0) + 1

        return {
            "total_errors": len(errors),
            "by_type": by_type,
            "recent_errors": [asdict(e) for e in errors[-10:]],
        }

    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics by phase."""
        by_phase: Dict[str, List[float]] = {}
        for record in self._records:
            phase = record.context.get("phase")
            if phase and record.duration_ms:
                if phase not in by_phase:
                    by_phase[phase] = []
                by_phase[phase].append(record.duration_ms)

        stats = {}
        for phase, durations in by_phase.items():
            stats[phase] = {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "total_ms": sum(durations),
            }

        return stats


# Global logger instance
_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger(
    name: str = "company_agi",
    level: LogLevel = LogLevel.INFO,
    json_output: Optional[Path] = None,
) -> StructuredLogger:
    """Get or create the global structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(
            name=name,
            level=level,
            json_output=json_output,
        )
    return _structured_logger


def reset_structured_logger() -> None:
    """Reset the global structured logger."""
    global _structured_logger
    if _structured_logger:
        _structured_logger.close()
    _structured_logger = None
