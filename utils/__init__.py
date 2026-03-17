"""Utility modules for Company AGI."""

from .input_validation import (
    sanitize_prompt_input,
    validate_problem_description,
    truncate_to_token_limit,
    InputValidationError,
)

from .file_lock import (
    FileLock,
    FileLockError,
    FileLockTimeoutError,
    file_lock,
    atomic_write_json,
    safe_read_json,
)

from .rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    get_global_rate_limiter,
    create_default_rate_limiter,
)

from .logging import (
    logger,
    setup_logging,
    get_logger,
    LogConfig,
    AgentLogger,
    WorkflowLogger,
    log_function_call,
    ensure_logging_initialized,
)

from .hooks import (
    Hook,
    HookEvent,
    HookDecision,
    HookContext,
    HookResult,
    HooksManager,
    CommandHook,
    CallableHook,
    ToolValidatorHook,
    FileProtectionHook,
    AuditLogHook,
    get_hooks_manager,
    create_bash_validator,
    create_auto_formatter_hook,
)

from .sandbox import (
    SandboxMode,
    SandboxConfig,
    SandboxResult,
    SandboxExecutor,
    SandboxedBashTool,
    CommandValidator,
    get_sandbox,
    create_readonly_sandbox,
    create_development_sandbox,
    create_ci_sandbox,
)

from .permissions import (
    PermissionLevel,
    PermissionScope,
    PermissionRule,
    PermissionRequest,
    PermissionDecision,
    PermissionManager,
    PermissionAuditLog,
    get_permission_manager,
    create_readonly_permissions,
    create_development_permissions,
    create_agent_permissions,
)

from .structured_logging import (
    StructuredLogger,
    LogLevel,
    LogContext,
    LogRecord,
    LogAggregator,
    get_structured_logger,
    reset_structured_logger,
)

from .error_context import (
    ErrorContext,
    ErrorCollector,
    ErrorSeverity,
    ErrorCategory,
    SystemState,
    ActionRecord,
    get_error_collector,
    reset_error_collector,
    capture_error,
)

from .progress_tracker import (
    ProgressTracker,
    ProgressItem,
    ProgressStatus,
    ProgressBar,
    PhaseMetrics,
    get_progress_tracker,
    reset_progress_tracker,
)

from .health_check import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    get_health_checker,
    reset_health_checker,
    check_health,
)

from .cost_tracker import (
    CostTracker,
    ModelPricing,
    ModelProvider,
    UsageRecord,
    AgentCostSummary,
    SessionCostSummary,
    get_cost_tracker,
    reset_cost_tracker,
    track_usage,
)

from .memory_monitor import (
    MemoryMonitor,
    MemoryThresholds,
    MemoryLevel,
    SystemMemoryInfo,
    AgentMemoryInfo,
    MemoryWarning,
    get_memory_monitor,
    reset_memory_monitor,
)

__all__ = [
    # Input validation
    "sanitize_prompt_input",
    "validate_problem_description",
    "truncate_to_token_limit",
    "InputValidationError",
    # File locking
    "FileLock",
    "FileLockError",
    "FileLockTimeoutError",
    "file_lock",
    "atomic_write_json",
    "safe_read_json",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "get_global_rate_limiter",
    "create_default_rate_limiter",
    # Logging
    "logger",
    "setup_logging",
    "get_logger",
    "LogConfig",
    "AgentLogger",
    "WorkflowLogger",
    "log_function_call",
    "ensure_logging_initialized",
    # Hooks
    "Hook",
    "HookEvent",
    "HookDecision",
    "HookContext",
    "HookResult",
    "HooksManager",
    "CommandHook",
    "CallableHook",
    "ToolValidatorHook",
    "FileProtectionHook",
    "AuditLogHook",
    "get_hooks_manager",
    "create_bash_validator",
    "create_auto_formatter_hook",
    # Sandbox
    "SandboxMode",
    "SandboxConfig",
    "SandboxResult",
    "SandboxExecutor",
    "SandboxedBashTool",
    "CommandValidator",
    "get_sandbox",
    "create_readonly_sandbox",
    "create_development_sandbox",
    "create_ci_sandbox",
    # Permissions
    "PermissionLevel",
    "PermissionScope",
    "PermissionRule",
    "PermissionRequest",
    "PermissionDecision",
    "PermissionManager",
    "PermissionAuditLog",
    "get_permission_manager",
    "create_readonly_permissions",
    "create_development_permissions",
    "create_agent_permissions",
    # Structured Logging
    "StructuredLogger",
    "LogLevel",
    "LogContext",
    "LogRecord",
    "LogAggregator",
    "get_structured_logger",
    "reset_structured_logger",
    # Error Context
    "ErrorContext",
    "ErrorCollector",
    "ErrorSeverity",
    "ErrorCategory",
    "SystemState",
    "ActionRecord",
    "get_error_collector",
    "reset_error_collector",
    "capture_error",
    # Progress Tracking
    "ProgressTracker",
    "ProgressItem",
    "ProgressStatus",
    "ProgressBar",
    "PhaseMetrics",
    "get_progress_tracker",
    "reset_progress_tracker",
    # Health Checks
    "HealthChecker",
    "HealthCheckResult",
    "HealthStatus",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "get_health_checker",
    "reset_health_checker",
    "check_health",
    # Cost Tracking
    "CostTracker",
    "ModelPricing",
    "ModelProvider",
    "UsageRecord",
    "AgentCostSummary",
    "SessionCostSummary",
    "get_cost_tracker",
    "reset_cost_tracker",
    "track_usage",
    # Memory Monitoring
    "MemoryMonitor",
    "MemoryThresholds",
    "MemoryLevel",
    "SystemMemoryInfo",
    "AgentMemoryInfo",
    "MemoryWarning",
    "get_memory_monitor",
    "reset_memory_monitor",
]
