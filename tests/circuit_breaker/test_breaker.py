from __future__ import annotations

import pytest
import pybreaker

from rag.circuit_breaker.breaker import CircuitBreakerRegistry
from rag.config import CircuitBreakerConfig


def _make_config(failure_threshold: int = 3) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout_ms=30000,
        half_open_max_calls=1,
    )


def _fail() -> None:
    raise Exception("simulated failure")


class TestRegistryInitialState:
    def test_unknown_provider_is_available(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        assert registry.is_available("anthropic") is True

    def test_unknown_provider_state_is_closed(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        assert registry.state("anthropic") == "closed"

    def test_get_creates_breaker_on_first_access(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        breaker = registry.get("anthropic")
        assert isinstance(breaker, pybreaker.CircuitBreaker)

    def test_get_returns_same_instance_for_same_provider(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        assert registry.get("anthropic") is registry.get("anthropic")

    def test_get_returns_different_instances_for_different_providers(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        assert registry.get("anthropic") is not registry.get("openai")

    def test_new_breaker_state_is_closed(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        registry.get("anthropic")
        assert registry.state("anthropic") == "closed"

    def test_new_breaker_is_available(self) -> None:
        registry = CircuitBreakerRegistry(_make_config())
        registry.get("anthropic")
        assert registry.is_available("anthropic") is True


class TestCircuitOpening:
    def test_circuit_opens_after_failure_threshold(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)

        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                registry.get("anthropic").call(_fail)

        assert registry.state("anthropic") == "open"

    def test_open_circuit_is_not_available(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)

        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                registry.get("anthropic").call(_fail)

        assert registry.is_available("anthropic") is False

    def test_open_circuit_raises_circuit_breaker_error(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)

        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                registry.get("anthropic").call(_fail)

        # Next call hits the open circuit — no function execution
        with pytest.raises(pybreaker.CircuitBreakerError):
            registry.get("anthropic").call(_fail)

    def test_failures_on_one_provider_do_not_affect_another(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)

        for _ in range(config.failure_threshold):
            with pytest.raises(Exception):
                registry.get("anthropic").call(_fail)

        # openai breaker is independent
        assert registry.is_available("openai") is True
        assert registry.state("openai") == "closed"

    def test_circuit_does_not_open_before_threshold(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)

        for _ in range(config.failure_threshold - 1):
            with pytest.raises(Exception):
                registry.get("anthropic").call(_fail)

        assert registry.state("anthropic") == "closed"
        assert registry.is_available("anthropic") is True

    def test_success_resets_failure_count(self) -> None:
        config = _make_config(failure_threshold=3)
        registry = CircuitBreakerRegistry(config)
        breaker = registry.get("anthropic")

        # Two failures — not enough to open
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(_fail)

        # One success — failure count resets
        breaker.call(lambda: None)

        # Two more failures — should still not open (count was reset)
        for _ in range(2):
            with pytest.raises(Exception):
                breaker.call(_fail)

        assert registry.state("anthropic") == "closed"
