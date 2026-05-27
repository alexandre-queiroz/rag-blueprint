from __future__ import annotations

import pybreaker

from rag.config import CircuitBreakerConfig


class CircuitBreakerRegistry:
    """Per-provider circuit breaker registry backed by pybreaker.

    One CircuitBreaker per provider — provider-scoped, not model-scoped.
    A provider in the Open state is skipped across all complexity tiers.

    States (per provider):
        closed    — normal operation; all calls pass through
        open      — opened after `failure_threshold` consecutive failures;
                    calls raise CircuitBreakerError immediately
        half-open — after `recovery_timeout_ms`, one test call is allowed;
                    success closes the circuit, failure re-opens it

    NOTE: pybreaker always allows exactly 1 test call in the half-open state.
    The `half_open_max_calls` config field documents the intended value but
    pybreaker does not expose this as a constructor parameter in v1.x.

    Gateway usage:
        breaker = registry.get(provider)
        try:
            result = await breaker.call_async(litellm.acompletion, **kwargs)
        except pybreaker.CircuitBreakerError:
            # circuit is open — skip to next provider
        except Exception:
            # call failed — pybreaker already recorded the failure
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._breakers: dict[str, pybreaker.CircuitBreaker] = {}

    def get(self, provider: str) -> pybreaker.CircuitBreaker:
        """Return the circuit breaker for a provider, creating it on first access."""
        if provider not in self._breakers:
            self._breakers[provider] = pybreaker.CircuitBreaker(
                fail_max=self._config.failure_threshold,
                reset_timeout=self._config.recovery_timeout_ms / 1000,
                name=provider,
            )
        return self._breakers[provider]

    def is_available(self, provider: str) -> bool:
        """Return True if the provider can accept calls (circuit not open)."""
        if provider not in self._breakers:
            return True  # never failed — no breaker created yet
        return self._breakers[provider].current_state != "open"

    def state(self, provider: str) -> str:
        """Return the current circuit state: 'closed', 'open', or 'half-open'."""
        if provider not in self._breakers:
            return "closed"
        return self._breakers[provider].current_state
