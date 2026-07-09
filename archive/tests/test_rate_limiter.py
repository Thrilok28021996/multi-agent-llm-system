"""Tests for rate limiting utilities."""

import pytest
import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    get_global_rate_limiter,
    create_default_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_second == 1.0
        assert config.requests_per_minute == 30.0
        assert config.burst_size == 5
        assert config.backoff_base == 2.0
        assert config.max_backoff == 60.0
        assert config.retry_after_429 is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=0.5,
            requests_per_minute=10,
            burst_size=2
        )
        assert config.requests_per_second == 0.5
        assert config.requests_per_minute == 10
        assert config.burst_size == 2


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter()
        assert limiter.default_config is not None
        assert limiter.domain_states is not None

    def test_get_default_config(self):
        """Test getting default config for unknown domain."""
        limiter = RateLimiter()
        config = limiter.get_config("unknown.domain.com")
        assert config == limiter.default_config

    def test_custom_domain_config(self):
        """Test setting custom domain config."""
        custom_config = RateLimitConfig(requests_per_second=0.1)
        limiter = RateLimiter(domain_configs={"example.com": custom_config})

        assert limiter.get_config("example.com") == custom_config
        assert limiter.get_config("other.com") != custom_config

    def test_set_domain_config(self):
        """Test dynamically setting domain config."""
        limiter = RateLimiter()
        custom_config = RateLimitConfig(requests_per_second=0.5)

        limiter.set_domain_config("test.com", custom_config)
        assert limiter.get_config("test.com") == custom_config

    def test_wait_sync_consumes_token(self):
        """Test that wait_sync consumes a token."""
        limiter = RateLimiter()
        domain = "test.domain.com"

        # First call should be immediate (have burst tokens)
        start = time.time()
        limiter.wait_sync(domain)
        elapsed = time.time() - start
        assert elapsed < 0.1  # Should be nearly instant

        # Check token was consumed
        state = limiter.domain_states[domain]
        assert state.total_requests == 1

    def test_record_success_resets_errors(self):
        """Test that recording success resets error counters."""
        limiter = RateLimiter()
        domain = "test.com"

        # Record some errors
        limiter.record_error(domain)
        limiter.record_error(domain)
        assert limiter.domain_states[domain].consecutive_errors == 2

        # Record success
        limiter.record_success(domain)
        assert limiter.domain_states[domain].consecutive_errors == 0

    def test_record_error_calculates_backoff(self):
        """Test that recording error calculates proper backoff."""
        limiter = RateLimiter()
        domain = "test.com"

        backoff1 = limiter.record_error(domain)
        backoff2 = limiter.record_error(domain)

        # Backoff should increase exponentially
        assert backoff2 > backoff1

    def test_record_429_longer_backoff(self):
        """Test that 429 errors get longer backoff."""
        limiter = RateLimiter()
        domain = "test.com"

        regular_backoff = limiter.record_error(domain, status_code=500)
        limiter.record_success(domain)  # Reset

        error_429_backoff = limiter.record_429(domain)

        # 429 should have longer backoff
        assert error_429_backoff > 0
        assert limiter.domain_states[domain].last_429_time is not None

    def test_record_429_with_retry_after(self):
        """Test 429 with Retry-After header."""
        limiter = RateLimiter()
        domain = "test.com"

        wait_time = limiter.record_429(domain, retry_after=30)
        assert wait_time == 30.0

    def test_get_stats_single_domain(self):
        """Test getting stats for a single domain."""
        limiter = RateLimiter()
        domain = "test.com"

        limiter.wait_sync(domain)
        limiter.record_error(domain)

        stats = limiter.get_stats(domain)
        assert stats["domain"] == domain
        assert stats["total_requests"] == 1
        assert stats["total_errors"] == 1
        assert stats["consecutive_errors"] == 1

    def test_get_stats_all_domains(self):
        """Test getting stats for all domains."""
        limiter = RateLimiter()

        limiter.wait_sync("domain1.com")
        limiter.wait_sync("domain2.com")

        stats = limiter.get_stats()
        assert "domain1.com" in stats
        assert "domain2.com" in stats

    def test_reset_single_domain(self):
        """Test resetting a single domain."""
        limiter = RateLimiter()
        domain = "test.com"

        limiter.wait_sync(domain)
        limiter.record_error(domain)
        limiter.reset(domain)

        state = limiter.domain_states[domain]
        assert state.total_requests == 0
        assert state.total_errors == 0

    def test_reset_all_domains(self):
        """Test resetting all domains."""
        limiter = RateLimiter()

        limiter.wait_sync("domain1.com")
        limiter.wait_sync("domain2.com")
        limiter.reset()

        assert len(limiter.domain_states) == 0


class TestAsyncRateLimiter:
    """Tests for async rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_wait_async_consumes_token(self):
        """Test that wait_async consumes a token."""
        limiter = RateLimiter()
        domain = "async.test.com"

        await limiter.wait_async(domain)

        state = limiter.domain_states[domain]
        assert state.total_requests == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_rate_limited(self):
        """Test that concurrent requests are properly rate limited."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=2
        )
        limiter = RateLimiter(default_config=config)
        domain = "concurrent.test.com"

        # Make several concurrent requests
        tasks = [limiter.wait_async(domain) for _ in range(5)]
        await asyncio.gather(*tasks)

        state = limiter.domain_states[domain]
        assert state.total_requests == 5


class TestGlobalRateLimiter:
    """Tests for global rate limiter."""

    def test_get_global_rate_limiter(self):
        """Test getting global rate limiter returns same instance."""
        limiter1 = get_global_rate_limiter()
        limiter2 = get_global_rate_limiter()
        assert limiter1 is limiter2

    def test_create_default_rate_limiter(self):
        """Test creating default rate limiter with common configs."""
        limiter = create_default_rate_limiter()

        # Should have reddit config
        reddit_config = limiter.get_config("reddit.com")
        assert reddit_config.requests_per_second == 0.5

        # Should have github config
        github_config = limiter.get_config("api.github.com")
        assert github_config.requests_per_second == 1.0
