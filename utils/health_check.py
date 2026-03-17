"""
Health Checks and Circuit Breaker for Company AGI.

Provides system health monitoring and fault tolerance:
- Ollama server connectivity checks
- Model availability verification
- Circuit breaker pattern for failing services
- Health dashboard for terminal
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    success_threshold: int = 2           # Successes to close from half-open
    timeout_seconds: float = 60.0        # Time before trying half-open
    half_open_max_calls: int = 3         # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, all requests pass
    - OPEN: Service failing, requests blocked
    - HALF_OPEN: Testing recovery, limited requests
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        # Reset counters on state change
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0

        if self.on_state_change:
            self.on_state_change(old_state, new_state)

    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
        return False

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": datetime.fromtimestamp(self._last_failure_time).isoformat()
            if self._last_failure_time else None,
        }


class HealthChecker:
    """
    Health checking system.

    Features:
    - Multiple health checks
    - Periodic background checking
    - Circuit breakers for services
    - Health dashboard
    """

    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        **_kwargs,  # absorb deprecated keyword args
    ):
        self.check_interval_seconds = check_interval_seconds

        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("llm_backend", self._check_llm_backend)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_fn

    def get_circuit_breaker(
        self,
        service_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker(
                name=service_name,
                config=config,
            )
        return self._circuit_breakers[service_name]

    def _check_llm_backend(self) -> HealthCheckResult:
        """Check Ollama server availability."""
        start = time.perf_counter()
        try:
            import ollama as _ollama  # noqa
            from config.llm_client import get_ollama_host
            host = get_ollama_host()
            client = _ollama.Client(host=host)
            client.list()
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_backend",
                status=HealthStatus.HEALTHY,
                message=f"Ollama server reachable at {host}",
                latency_ms=latency_ms,
                details={"backend": "ollama", "host": host},
            )
        except ImportError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_backend",
                status=HealthStatus.UNHEALTHY,
                message="ollama Python package not installed",
                latency_ms=latency_ms,
                error=str(e),
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="llm_backend",
                status=HealthStatus.UNHEALTHY,
                message=f"Ollama server not reachable: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        import shutil

        start = time.perf_counter()
        try:
            stat = shutil.disk_usage(".")
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_pct = ((stat.total - stat.free) / stat.total) * 100
            latency_ms = (time.perf_counter() - start) * 1000

            if free_gb < 1.0:
                status = HealthStatus.UNHEALTHY
                message = f"Very low disk space: {free_gb:.1f}GB"
            elif free_gb < 5.0:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb:.1f}GB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.1f}GB free"

            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_pct": round(used_pct, 1),
                },
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check disk space: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def _check_memory(self) -> HealthCheckResult:
        """Check available memory."""
        start = time.perf_counter()
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            used_pct = mem.percent
            latency_ms = (time.perf_counter() - start) * 1000

            if available_gb < 0.5:
                status = HealthStatus.UNHEALTHY
                message = f"Very low memory: {available_gb:.1f}GB available"
            elif available_gb < 2.0:
                status = HealthStatus.DEGRADED
                message = f"Low memory: {available_gb:.1f}GB available"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {available_gb:.1f}GB available"

            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details={
                    "available_gb": round(available_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_pct": round(used_pct, 1),
                },
            )

        except ImportError:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed - cannot check memory",
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check memory: {e}",
                latency_ms=latency_ms,
                error=str(e),
            )

    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown health check: {name}",
            )

        result = self._checks[name]()
        with self._lock:
            self._results[name] = result
        return result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results

    def get_results(self) -> Dict[str, HealthCheckResult]:
        """Get cached health check results."""
        with self._lock:
            return dict(self._results)

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = self.get_results()
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def start_background_checks(self) -> None:
        """Start background health checks."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._background_check_loop, daemon=True)
        self._thread.start()

    def stop_background_checks(self) -> None:
        """Stop background health checks."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _background_check_loop(self) -> None:
        """Background check loop."""
        while self._running:
            self.run_all_checks()
            time.sleep(self.check_interval_seconds)

    def get_dashboard(self, color: bool = True) -> str:
        """Get health dashboard as formatted string."""
        colors = {
            HealthStatus.HEALTHY: "\033[32m",    # Green
            HealthStatus.DEGRADED: "\033[33m",   # Yellow
            HealthStatus.UNHEALTHY: "\033[31m",  # Red
            HealthStatus.UNKNOWN: "\033[90m",    # Gray
            "reset": "\033[0m",
            "bold": "\033[1m",
        }

        if not color:
            colors = {k: "" for k in colors}

        lines = [
            f"{colors['bold']}System Health Dashboard{colors['reset']}",
            "=" * 50,
        ]

        # Overall status
        overall = self.get_overall_status()
        overall_color = colors.get(overall, "")
        lines.append(f"Overall: {overall_color}[{overall.value.upper()}]{colors['reset']}")
        lines.append("")

        # Individual checks
        results = self.get_results()
        for name, result in sorted(results.items()):
            status_color = colors.get(result.status, "")
            icon = {
                HealthStatus.HEALTHY: "[+]",
                HealthStatus.DEGRADED: "[!]",
                HealthStatus.UNHEALTHY: "[-]",
                HealthStatus.UNKNOWN: "[?]",
            }.get(result.status, "[?]")

            lines.append(
                f"  {status_color}{icon}{colors['reset']} {name:15} "
                f"{result.message} ({result.latency_ms:.0f}ms)"
            )

            # Show details for non-healthy checks
            if result.status != HealthStatus.HEALTHY and result.details:
                for key, value in result.details.items():
                    lines.append(f"       {key}: {value}")

        # Circuit breaker status
        if self._circuit_breakers:
            lines.extend(["", f"{colors['bold']}Circuit Breakers:{colors['reset']}"])
            for name, cb in self._circuit_breakers.items():
                state_colors = {
                    CircuitState.CLOSED: colors.get(HealthStatus.HEALTHY, ""),
                    CircuitState.OPEN: colors.get(HealthStatus.UNHEALTHY, ""),
                    CircuitState.HALF_OPEN: colors.get(HealthStatus.DEGRADED, ""),
                }
                state_color = state_colors.get(cb.state, "")
                lines.append(f"  {name}: {state_color}{cb.state.value}{colors['reset']}")

        return "\n".join(lines)

    def save_report(self, path: Path) -> None:
        """Save health report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status().value,
            "checks": {k: v.to_dict() for k, v in self.get_results().items()},
            "circuit_breakers": {
                k: v.get_stats() for k, v in self._circuit_breakers.items()
            },
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(**_kwargs) -> HealthChecker:
    """Get or create the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset the global health checker."""
    global _health_checker
    if _health_checker:
        _health_checker.stop_background_checks()
    _health_checker = None


def check_health(print_dashboard: bool = True) -> Dict[str, HealthCheckResult]:
    """Convenience function to run health checks."""
    checker = get_health_checker()
    results = checker.run_all_checks()

    if print_dashboard:
        print(checker.get_dashboard())

    return results
