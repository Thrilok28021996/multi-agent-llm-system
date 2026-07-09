"""Logging configuration using loguru for Company AGI."""

import sys
import logging as stdlib_logging
from pathlib import Path
from typing import Optional, Any
from functools import wraps
import time

# Try to import loguru, fall back to standard logging
LOGURU_AVAILABLE = False
logger: Any = None

try:
    from loguru import logger as loguru_logger
    logger = loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    # Fallback to standard logging if loguru not installed
    logger = stdlib_logging.getLogger("company_agi")


# Default log directory
DEFAULT_LOG_DIR = Path("./output/logs")

# Log format templates
CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

JSON_FORMAT = "{message}"


class LogConfig:
    """Logging configuration."""

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        rotation: str = "10 MB",
        retention: str = "7 days",
        enable_json: bool = False,
        enable_console: bool = True,
        enable_file: bool = True
    ):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.console_level = console_level
        self.file_level = file_level
        self.rotation = rotation
        self.retention = retention
        self.enable_json = enable_json
        self.enable_console = enable_console
        self.enable_file = enable_file


def setup_logging(config: Optional[LogConfig] = None) -> None:
    """
    Set up logging with loguru.

    Args:
        config: Logging configuration
    """
    if not LOGURU_AVAILABLE:
        # Fallback configuration for standard logging
        stdlib_logging.basicConfig(
            level=stdlib_logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        return

    config = config or LogConfig()

    # Remove default handler
    logger.remove()

    # Create log directory
    config.log_dir.mkdir(parents=True, exist_ok=True)

    # Add console handler
    if config.enable_console:
        logger.add(
            sys.stderr,
            format=CONSOLE_FORMAT,
            level=config.console_level,
            colorize=True
        )

    # Add file handler for general logs
    if config.enable_file:
        logger.add(
            config.log_dir / "company_agi_{time:YYYY-MM-DD}.log",
            format=FILE_FORMAT,
            level=config.file_level,
            rotation=config.rotation,
            retention=config.retention,
            compression="zip"
        )

        # Add separate error log
        logger.add(
            config.log_dir / "errors_{time:YYYY-MM-DD}.log",
            format=FILE_FORMAT,
            level="ERROR",
            rotation=config.rotation,
            retention=config.retention,
            compression="zip"
        )

    # Add JSON log for structured logging (useful for log aggregation)
    if config.enable_json:
        logger.add(
            config.log_dir / "company_agi_{time:YYYY-MM-DD}.json",
            format=JSON_FORMAT,
            level=config.file_level,
            rotation=config.rotation,
            retention=config.retention,
            serialize=True
        )


def get_logger(name: str = "company_agi"):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually module name)

    Returns:
        Logger instance
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    else:
        return stdlib_logging.getLogger(name)


def log_function_call(level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments and return values.

    Args:
        level: Log level for the messages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            log_func = getattr(logger, level.lower(), logger.debug)

            # Log entry
            log_func(f"Entering {func_name} with args={args[:3]}... kwargs={list(kwargs.keys())}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_func(f"Exiting {func_name} after {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(f"Error in {func_name} after {elapsed:.3f}s: {e}")
                raise

        return wrapper
    return decorator


def log_async_function_call(level: str = "DEBUG"):
    """
    Decorator to log async function calls.

    Args:
        level: Log level for the messages
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            log_func = getattr(logger, level.lower(), logger.debug)

            log_func(f"Entering async {func_name}")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_func(f"Exiting async {func_name} after {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(f"Error in async {func_name} after {elapsed:.3f}s: {e}")
                raise

        return wrapper
    return decorator


class AgentLogger:
    """
    Specialized logger for agent operations.

    Provides structured logging for agent activities with context.
    """

    def __init__(self, agent_name: str, agent_role: str):
        self.agent_name = agent_name
        self.agent_role = agent_role
        self._logger = get_logger(f"agent.{agent_name}")

    def _format_message(self, message: str) -> str:
        return f"[{self.agent_role}] {message}"

    def debug(self, message: str, **kwargs):
        self._logger.debug(self._format_message(message), **kwargs)

    def info(self, message: str, **kwargs):
        self._logger.info(self._format_message(message), **kwargs)

    def warning(self, message: str, **kwargs):
        self._logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, **kwargs):
        self._logger.error(self._format_message(message), **kwargs)

    def critical(self, message: str, **kwargs):
        self._logger.critical(self._format_message(message), **kwargs)

    def task_start(self, task_type: str, task_description: str):
        """Log the start of a task."""
        self.info(f"Starting task: {task_type} - {task_description}")

    def task_complete(self, task_type: str, success: bool, duration: float):
        """Log task completion."""
        status = "SUCCESS" if success else "FAILED"
        self.info(f"Task {task_type} {status} in {duration:.2f}s")

    def llm_call(self, model: str, prompt_length: int):
        """Log an LLM API call."""
        self.debug(f"LLM call to {model} with {prompt_length} chars")

    def llm_response(self, model: str, response_length: int, duration: float):
        """Log LLM response received."""
        self.debug(f"LLM response from {model}: {response_length} chars in {duration:.2f}s")

    def tool_use(self, tool_name: str, success: bool):
        """Log tool usage."""
        status = "success" if success else "failed"
        self.debug(f"Tool {tool_name}: {status}")

    def message_sent(self, recipient: str, message_type: str):
        """Log message sent to another agent."""
        self.debug(f"Message sent to {recipient} ({message_type})")

    def message_received(self, sender: str, message_type: str):
        """Log message received from another agent."""
        self.debug(f"Message received from {sender} ({message_type})")


class WorkflowLogger:
    """
    Specialized logger for workflow orchestration.
    """

    def __init__(self):
        self._logger = get_logger("workflow")

    def phase_start(self, phase_name: str):
        """Log workflow phase start."""
        self._logger.info(f"=== PHASE: {phase_name} ===")

    def phase_complete(self, phase_name: str, duration: float):
        """Log workflow phase completion."""
        self._logger.info(f"=== PHASE COMPLETE: {phase_name} ({duration:.2f}s) ===")

    def state_change(self, old_state: str, new_state: str):
        """Log workflow state transition."""
        self._logger.info(f"State: {old_state} -> {new_state}")

    def problem_discovered(self, problem_id: str, severity: str):
        """Log problem discovery."""
        self._logger.info(f"Problem discovered [{severity}]: {problem_id}")

    def decision_made(self, decision_type: str, outcome: str):
        """Log decision made."""
        self._logger.info(f"Decision [{decision_type}]: {outcome}")

    def error(self, message: str, exc_info: bool = True):
        """Log workflow error."""
        self._logger.error(message, exc_info=exc_info)


# Initialize default logging on import
_initialized = False


def ensure_logging_initialized():
    """Ensure logging is initialized (call this in main entry points)."""
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True


# Export main logger for direct use
__all__ = [
    "logger",
    "setup_logging",
    "get_logger",
    "LogConfig",
    "AgentLogger",
    "WorkflowLogger",
    "log_function_call",
    "log_async_function_call",
    "ensure_logging_initialized",
    "LOGURU_AVAILABLE",
]
