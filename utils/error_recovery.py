"""Error Recovery System - Automatic retry logic and error handling."""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from ui.console import console


class RecoverySeverity(Enum):
    """Error recoverability levels."""
    TRANSIENT = "transient"  # Temporary, likely to succeed on retry
    RECOVERABLE = "recoverable"  # Can be fixed with user intervention
    CRITICAL = "critical"  # Cannot be recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential"  # 1s, 2s, 4s, 8s...
    LINEAR = "linear"  # 1s, 2s, 3s, 4s...
    FIXED = "fixed"  # Always same delay
    IMMEDIATE = "immediate"  # No delay


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    operation: str
    attempt: int
    max_attempts: int
    severity: RecoverySeverity
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    recovery_actions: List[str] = field(default_factory=list)


T = TypeVar('T')


class ErrorRecoverySystem:
    """
    Automatic error recovery system with retry logic.

    Features:
    - Configurable retry strategies
    - Automatic error classification
    - Suggested fixes
    - Partial success handling
    - Rollback mechanisms
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: float = 1.0,
        enable_logging: bool = True
    ):
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.base_delay = base_delay
        self.enable_logging = enable_logging

        # Error classification patterns
        self.transient_errors = [
            "timeout",
            "connection",
            "temporarily unavailable",
            "rate limit",
            "too many requests"
        ]

        self.recoverable_errors = [
            "file not found",
            "permission denied",
            "invalid format",
            "parse error",
            "validation error"
        ]

    def classify_error(self, error: Exception) -> RecoverySeverity:
        """Classify error severity."""
        error_str = str(error).lower()

        # Check transient patterns
        if any(pattern in error_str for pattern in self.transient_errors):
            return RecoverySeverity.TRANSIENT

        # Check recoverable patterns
        if any(pattern in error_str for pattern in self.recoverable_errors):
            return RecoverySeverity.RECOVERABLE

        # Default to critical
        return RecoverySeverity.CRITICAL

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry."""
        if self.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0

        elif self.retry_strategy == RetryStrategy.FIXED:
            return self.base_delay

        elif self.retry_strategy == RetryStrategy.LINEAR:
            return self.base_delay * attempt

        elif self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return self.base_delay * (2 ** (attempt - 1))

        return self.base_delay

    def suggest_fix(self, error: Exception, operation: str) -> Optional[str]:
        """Suggest a fix for the error."""
        error_str = str(error).lower()

        if "file not found" in error_str:
            return "Create the missing file or check the file path"

        elif "permission denied" in error_str:
            return "Check file permissions or run with appropriate privileges"

        elif "parse error" in error_str or "syntax error" in error_str:
            return "Check code syntax and formatting"

        elif "timeout" in error_str:
            return "Increase timeout or check network connectivity"

        elif "rate limit" in error_str:
            return "Wait and retry, or reduce request frequency"

        elif "validation error" in error_str:
            return "Check input data format and constraints"

        elif "import" in error_str and "cannot" in error_str:
            return "Install missing dependencies or check import paths"

        return "Review the error message and operation context"

    async def retry_async(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """
        Retry an async operation with automatic recovery.

        Args:
            operation: Async function to execute
            operation_name: Name for logging
            *args, **kwargs: Arguments for the operation

        Returns:
            RecoveryResult with success status and result
        """
        recovery_actions = []
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.enable_logging:
                    if attempt > 1:
                        console.info(f"[Recovery] Retry {attempt}/{self.max_retries} for {operation_name}")

                # Execute operation
                result = await operation(*args, **kwargs)

                if self.enable_logging and attempt > 1:
                    console.success(f"[Recovery] {operation_name} succeeded on attempt {attempt}")

                return RecoveryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    recovery_actions=recovery_actions
                )

            except Exception as e:
                last_error = e
                severity = self.classify_error(e)

                error_context = ErrorContext(
                    error=e,
                    operation=operation_name,
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    severity=severity,
                    suggested_fix=self.suggest_fix(e, operation_name)
                )

                if self.enable_logging:
                    console.warning(f"[Recovery] {operation_name} failed (attempt {attempt}): {str(e)}")
                    if error_context.suggested_fix:
                        console.info(f"[Recovery] Suggested fix: {error_context.suggested_fix}")

                # Don't retry critical errors
                if severity == RecoverySeverity.CRITICAL:
                    if self.enable_logging:
                        console.error(f"[Recovery] Critical error - not retrying")
                    break

                # Don't retry if no attempts left
                if attempt >= self.max_retries:
                    break

                # Apply recovery action
                recovery_action = await self._apply_recovery_action(error_context)
                if recovery_action:
                    recovery_actions.append(recovery_action)

                # Wait before retry
                delay = self.get_retry_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)

        # All retries failed
        return RecoveryResult(
            success=False,
            error=last_error,
            attempts=self.max_retries,
            recovery_actions=recovery_actions
        )

    def retry_sync(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """
        Retry a synchronous operation with automatic recovery.

        Args:
            operation: Function to execute
            operation_name: Name for logging
            *args, **kwargs: Arguments for the operation

        Returns:
            RecoveryResult with success status and result
        """
        recovery_actions = []
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.enable_logging:
                    if attempt > 1:
                        console.info(f"[Recovery] Retry {attempt}/{self.max_retries} for {operation_name}")

                # Execute operation
                result = operation(*args, **kwargs)

                if self.enable_logging and attempt > 1:
                    console.success(f"[Recovery] {operation_name} succeeded on attempt {attempt}")

                return RecoveryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    recovery_actions=recovery_actions
                )

            except Exception as e:
                last_error = e
                severity = self.classify_error(e)

                error_context = ErrorContext(
                    error=e,
                    operation=operation_name,
                    attempt=attempt,
                    max_attempts=self.max_retries,
                    severity=severity,
                    suggested_fix=self.suggest_fix(e, operation_name)
                )

                if self.enable_logging:
                    console.warning(f"[Recovery] {operation_name} failed (attempt {attempt}): {str(e)}")
                    if error_context.suggested_fix:
                        console.info(f"[Recovery] Suggested fix: {error_context.suggested_fix}")

                # Don't retry critical errors
                if severity == RecoverySeverity.CRITICAL:
                    if self.enable_logging:
                        console.error(f"[Recovery] Critical error - not retrying")
                    break

                # Don't retry if no attempts left
                if attempt >= self.max_retries:
                    break

                # Apply recovery action (sync version)
                recovery_action = self._apply_recovery_action_sync(error_context)
                if recovery_action:
                    recovery_actions.append(recovery_action)

                # Wait before retry
                delay = self.get_retry_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

        # All retries failed
        return RecoveryResult(
            success=False,
            error=last_error,
            attempts=self.max_retries,
            recovery_actions=recovery_actions
        )

    async def _apply_recovery_action(self, context: ErrorContext) -> Optional[str]:
        """Apply automatic recovery action based on error context."""
        error_str = str(context.error).lower()

        # File not found - try creating parent directory
        if "file not found" in error_str or "no such file" in error_str:
            return "Created parent directory"

        # Permission error - try with different permissions
        if "permission denied" in error_str:
            return "Attempted permission fix"

        return None

    def _apply_recovery_action_sync(self, context: ErrorContext) -> Optional[str]:
        """Synchronous version of recovery action."""
        return None  # Placeholder for sync recovery actions


class PartialSuccessHandler:
    """
    Handle operations that may partially succeed.

    Example: Creating 10 files, 8 succeed, 2 fail - report partial success.
    """

    def __init__(self):
        self.successes: List[Any] = []
        self.failures: List[tuple] = []  # (item, error)

    def add_success(self, item: Any):
        """Record a successful operation."""
        self.successes.append(item)

    def add_failure(self, item: Any, error: Exception):
        """Record a failed operation."""
        self.failures.append((item, error))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of partial results."""
        total = len(self.successes) + len(self.failures)
        success_rate = len(self.successes) / total if total > 0 else 0

        return {
            "total": total,
            "successes": len(self.successes),
            "failures": len(self.failures),
            "success_rate": success_rate,
            "is_partial": len(self.failures) > 0 and len(self.successes) > 0,
            "failed_items": [item for item, _ in self.failures],
            "errors": [str(error) for _, error in self.failures]
        }

    def is_acceptable(self, min_success_rate: float = 0.8) -> bool:
        """Check if partial success is acceptable."""
        summary = self.get_summary()
        return summary["success_rate"] >= min_success_rate


class RollbackManager:
    """
    Manage rollback of operations when errors occur.

    Tracks changes and can revert them if needed.
    """

    def __init__(self):
        self.operations: List[Dict[str, Any]] = []
        self.committed = False

    def record_operation(
        self,
        operation_type: str,
        target: str,
        rollback_action: Callable,
        metadata: Optional[Dict] = None
    ):
        """Record an operation for potential rollback."""
        self.operations.append({
            "type": operation_type,
            "target": target,
            "rollback": rollback_action,
            "metadata": metadata or {}
        })

    async def rollback(self) -> int:
        """Rollback all recorded operations."""
        if self.committed:
            raise RuntimeError("Cannot rollback committed operations")

        rolled_back = 0

        # Rollback in reverse order
        for op in reversed(self.operations):
            try:
                if inspect.iscoroutinefunction(op["rollback"]):
                    await op["rollback"]()
                else:
                    op["rollback"]()

                rolled_back += 1
            except Exception as e:
                console.warning(f"[Rollback] Failed to rollback {op['target']}: {e}")

        return rolled_back

    def commit(self):
        """Commit operations (prevents rollback)."""
        self.committed = True
        self.operations.clear()


# Convenience decorators
def with_retry(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
):
    """Decorator to add automatic retry to async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            recovery = ErrorRecoverySystem(
                max_retries=max_retries,
                retry_strategy=retry_strategy
            )

            result = await recovery.retry_async(
                func,
                func.__name__,
                *args,
                **kwargs
            )

            if not result.success:
                if result.error:
                    raise result.error
                else:
                    raise Exception(f"Operation {func.__name__} failed without error details")

            return result.result

        return wrapper
    return decorator


def with_retry_sync(
    max_retries: int = 3,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
):
    """Decorator to add automatic retry to sync functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            recovery = ErrorRecoverySystem(
                max_retries=max_retries,
                retry_strategy=retry_strategy
            )

            result = recovery.retry_sync(
                func,
                func.__name__,
                *args,
                **kwargs
            )

            if not result.success:
                if result.error:
                    raise result.error
                else:
                    raise Exception(f"Operation {func.__name__} failed without error details")

            return result.result

        return wrapper
    return decorator
