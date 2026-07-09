"""Rate limiting utilities for web requests and API calls."""

import asyncio
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Any


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 1.0  # Max requests per second
    requests_per_minute: float = 30.0  # Max requests per minute
    burst_size: int = 5  # Max burst requests
    backoff_base: float = 2.0  # Exponential backoff base
    max_backoff: float = 60.0  # Max backoff delay in seconds
    retry_after_429: bool = True  # Automatically retry after 429 errors


@dataclass
class DomainState:
    """State tracking for a specific domain."""
    tokens: float = 5.0  # Current token bucket level
    last_request: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    backoff_until: float = 0.0  # Unix timestamp until which to back off
    total_requests: int = 0
    total_errors: int = 0
    last_429_time: Optional[float] = None


class RateLimiter:
    """
    Token bucket rate limiter with per-domain tracking and adaptive backoff.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Per-domain rate limit tracking
    - Automatic exponential backoff on errors
    - 429 (Too Many Requests) handling
    - Thread-safe for sync operations
    - Async-safe for async operations
    """

    def __init__(
        self,
        default_config: Optional[RateLimitConfig] = None,
        domain_configs: Optional[Dict[str, RateLimitConfig]] = None
    ):
        """
        Initialize the rate limiter.

        Args:
            default_config: Default rate limit configuration
            domain_configs: Per-domain configurations
        """
        self.default_config = default_config or RateLimitConfig()
        self.domain_configs: Dict[str, RateLimitConfig] = domain_configs or {}
        self.domain_states: Dict[str, DomainState] = defaultdict(DomainState)
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    def get_config(self, domain: str) -> RateLimitConfig:
        """Get rate limit config for a domain."""
        return self.domain_configs.get(domain, self.default_config)

    def set_domain_config(self, domain: str, config: RateLimitConfig) -> None:
        """Set custom rate limit config for a domain."""
        self.domain_configs[domain] = config

    def _refill_tokens(self, domain: str) -> None:
        """Refill tokens based on elapsed time (token bucket algorithm)."""
        config = self.get_config(domain)
        state = self.domain_states[domain]

        now = time.time()
        elapsed = now - state.last_request

        # Refill tokens based on time elapsed
        tokens_to_add = elapsed * config.requests_per_second
        state.tokens = min(config.burst_size, state.tokens + tokens_to_add)
        state.last_request = now

    def _calculate_wait_time(self, domain: str) -> float:
        """Calculate how long to wait before the next request."""
        config = self.get_config(domain)
        state = self.domain_states[domain]

        now = time.time()

        # Check if we're in backoff period
        if state.backoff_until > now:
            return state.backoff_until - now

        # Refill tokens
        self._refill_tokens(domain)

        if state.tokens >= 1.0:
            return 0.0

        # Calculate wait time for next token
        return (1.0 - state.tokens) / config.requests_per_second

    def wait_sync(self, domain: str) -> None:
        """
        Wait until a request can be made (synchronous version).

        Args:
            domain: The domain to rate limit
        """
        with self._lock:
            wait_time = self._calculate_wait_time(domain)

        if wait_time > 0:
            time.sleep(wait_time)

        with self._lock:
            state = self.domain_states[domain]
            state.tokens -= 1.0
            state.total_requests += 1

    async def wait_async(self, domain: str) -> None:
        """
        Wait until a request can be made (asynchronous version).

        Args:
            domain: The domain to rate limit
        """
        async with self._async_lock:
            wait_time = self._calculate_wait_time(domain)

        if wait_time > 0:
            await asyncio.sleep(wait_time)

        async with self._async_lock:
            state = self.domain_states[domain]
            state.tokens -= 1.0
            state.total_requests += 1

    def record_success(self, domain: str) -> None:
        """Record a successful request, resetting error counters."""
        with self._lock:
            state = self.domain_states[domain]
            state.consecutive_errors = 0

    def record_error(self, domain: str, status_code: Optional[int] = None) -> float:
        """
        Record a failed request and calculate backoff.

        Args:
            domain: The domain that had an error
            status_code: HTTP status code if available

        Returns:
            Recommended wait time before retry
        """
        with self._lock:
            config = self.get_config(domain)
            state = self.domain_states[domain]

            state.consecutive_errors += 1
            state.total_errors += 1

            # Calculate exponential backoff
            backoff = min(
                config.backoff_base ** state.consecutive_errors,
                config.max_backoff
            )

            # Handle 429 specifically
            if status_code == 429:
                state.last_429_time = time.time()
                # Use longer backoff for rate limit errors
                backoff = min(backoff * 2, config.max_backoff)

            state.backoff_until = time.time() + backoff
            return backoff

    def record_429(self, domain: str, retry_after: Optional[int] = None) -> float:
        """
        Record a 429 Too Many Requests error.

        Args:
            domain: The domain that returned 429
            retry_after: Value from Retry-After header if present

        Returns:
            Recommended wait time before retry
        """
        with self._lock:
            config = self.get_config(domain)
            state = self.domain_states[domain]

            state.last_429_time = time.time()
            state.consecutive_errors += 1
            state.total_errors += 1

            if retry_after:
                wait_time = float(retry_after)
            else:
                # Default exponential backoff for 429
                wait_time = min(
                    config.backoff_base ** state.consecutive_errors * 2,
                    config.max_backoff
                )

            state.backoff_until = time.time() + wait_time
            return wait_time

    def get_stats(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiting statistics.

        Args:
            domain: Specific domain to get stats for, or None for all

        Returns:
            Dictionary with rate limiting statistics
        """
        with self._lock:
            if domain:
                state = self.domain_states[domain]
                return {
                    "domain": domain,
                    "total_requests": state.total_requests,
                    "total_errors": state.total_errors,
                    "consecutive_errors": state.consecutive_errors,
                    "current_tokens": state.tokens,
                    "in_backoff": state.backoff_until > time.time(),
                    "backoff_remaining": max(0, state.backoff_until - time.time())
                }
            else:
                return {
                    domain: {
                        "total_requests": state.total_requests,
                        "total_errors": state.total_errors,
                        "error_rate": (
                            state.total_errors / state.total_requests
                            if state.total_requests > 0 else 0
                        )
                    }
                    for domain, state in self.domain_states.items()
                }

    def reset(self, domain: Optional[str] = None) -> None:
        """
        Reset rate limiter state.

        Args:
            domain: Specific domain to reset, or None for all
        """
        with self._lock:
            if domain:
                self.domain_states[domain] = DomainState()
            else:
                self.domain_states.clear()


# Pre-configured rate limiters for common services
COMMON_RATE_LIMITS = {
    "reddit.com": RateLimitConfig(
        requests_per_second=0.5,  # Reddit recommends 1 request per 2 seconds
        requests_per_minute=30,
        burst_size=2
    ),
    "www.reddit.com": RateLimitConfig(
        requests_per_second=0.5,
        requests_per_minute=30,
        burst_size=2
    ),
    "api.github.com": RateLimitConfig(
        requests_per_second=1.0,
        requests_per_minute=60,
        burst_size=10
    ),
    "hacker-news.firebaseio.com": RateLimitConfig(
        requests_per_second=2.0,
        requests_per_minute=100,
        burst_size=10
    ),
}


def create_default_rate_limiter() -> RateLimiter:
    """Create a rate limiter with common service configurations."""
    return RateLimiter(
        default_config=RateLimitConfig(),
        domain_configs=COMMON_RATE_LIMITS.copy()
    )


# Global rate limiter instance for shared use
_global_rate_limiter: Optional[RateLimiter] = None


def get_global_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = create_default_rate_limiter()
    return _global_rate_limiter
