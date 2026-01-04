"""
Cost Tracker Service
Version: 1.1 - Token Usage Persistence & Cost Alerts (FIXED)

PROBLEM SOLVED:
- Token counts reset on container restart
- No visibility into cost trends
- No alerts when budget exceeded

FEATURES:
- Prometheus metrics for Grafana
- Redis persistence (survives restarts)
- Per-tenant cost tracking
- Budget alerts
- Cost estimation (USD)
- Health check endpoint
- Token validation
- Retry mechanism for Redis

PRICING (Azure OpenAI, 2026 - configurable via env):
- Input:  $0.00015 per 1K tokens (gpt-4o-mini default)
- Output: $0.0006 per 1K tokens
"""

import os
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, List, Callable, Awaitable, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Try to import prometheus_client - check at usage time, not module load
_prometheus_client = None
_prometheus_available = None


def _check_prometheus() -> bool:
    """Lazily check if prometheus_client is available."""
    global _prometheus_client, _prometheus_available
    if _prometheus_available is None:
        try:
            import prometheus_client
            _prometheus_client = prometheus_client
            _prometheus_available = True
        except ImportError:
            _prometheus_available = False
            logger.warning("prometheus_client not available - metrics disabled")
    return _prometheus_available


# ============================================================================
# PRICING CONFIGURATION
# ============================================================================

class ModelPricing:
    """
    Azure OpenAI pricing per 1K tokens (USD).

    Pricing can be overridden via environment variables:
    - PRICING_GPT4O_MINI_INPUT=0.00015
    - PRICING_GPT4O_MINI_OUTPUT=0.0006
    """

    # Default pricing (can be overridden via env)
    _DEFAULT_PRICING = {
        "gpt-4o-mini": {
            "input": 0.00015,
            "output": 0.0006
        },
        "gpt-4o": {
            "input": 0.0025,
            "output": 0.01
        },
        "gpt-4-turbo": {
            "input": 0.01,
            "output": 0.03
        },
        "text-embedding-ada-002": {
            "input": 0.0001,
            "output": 0.0
        },
        "text-embedding-3-small": {
            "input": 0.00002,
            "output": 0.0
        }
    }

    _pricing_cache: Dict[str, Dict[str, float]] = None

    @classmethod
    def _load_pricing(cls) -> Dict[str, Dict[str, float]]:
        """Load pricing from env vars with defaults."""
        if cls._pricing_cache is not None:
            return cls._pricing_cache

        pricing = cls._DEFAULT_PRICING.copy()

        # Allow env var overrides
        env_overrides = {
            "gpt-4o-mini": ("PRICING_GPT4O_MINI_INPUT", "PRICING_GPT4O_MINI_OUTPUT"),
            "gpt-4o": ("PRICING_GPT4O_INPUT", "PRICING_GPT4O_OUTPUT"),
        }

        for model, (input_env, output_env) in env_overrides.items():
            input_price = os.getenv(input_env)
            output_price = os.getenv(output_env)
            if input_price:
                try:
                    pricing[model]["input"] = float(input_price)
                except ValueError:
                    logger.warning(f"Invalid pricing env var {input_env}: {input_price}")
            if output_price:
                try:
                    pricing[model]["output"] = float(output_price)
                except ValueError:
                    logger.warning(f"Invalid pricing env var {output_env}: {output_price}")

        cls._pricing_cache = pricing
        return pricing

    @classmethod
    def get_model_pricing(cls, model: str) -> Dict[str, float]:
        """Get pricing for a model with validation."""
        pricing = cls._load_pricing()
        if model not in pricing:
            logger.warning(f"Unknown model '{model}', using gpt-4o-mini pricing as fallback")
            return pricing["gpt-4o-mini"]
        return pricing[model]

    @classmethod
    def calculate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost in USD using Decimal for precision.

        Args:
            model: Model name
            prompt_tokens: Number of input tokens (must be >= 0)
            completion_tokens: Number of output tokens (must be >= 0)

        Returns:
            Cost in USD
        """
        # Validate tokens
        if prompt_tokens < 0 or completion_tokens < 0:
            logger.warning(
                f"Invalid token counts: prompt={prompt_tokens}, completion={completion_tokens}"
            )
            prompt_tokens = max(0, prompt_tokens)
            completion_tokens = max(0, completion_tokens)

        pricing = cls.get_model_pricing(model)

        # Use Decimal for precision
        input_cost = Decimal(str(prompt_tokens)) / Decimal("1000") * Decimal(str(pricing["input"]))
        output_cost = Decimal(str(completion_tokens)) / Decimal("1000") * Decimal(str(pricing["output"]))
        total = input_cost + output_cost

        # Round to 8 decimal places for microcents
        return float(total.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models."""
        return list(cls._load_pricing().keys())


# ============================================================================
# PROMETHEUS METRICS (lazy initialization)
# ============================================================================

_metrics_initialized = False
_TOKENS_PROMPT = None
_TOKENS_COMPLETION = None
_COST_USD = None
_REQUESTS_TOTAL = None
_REQUEST_LATENCY = None
_BUDGET_REMAINING = None


def _init_prometheus_metrics():
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized, _TOKENS_PROMPT, _TOKENS_COMPLETION
    global _COST_USD, _REQUESTS_TOTAL, _REQUEST_LATENCY, _BUDGET_REMAINING

    if _metrics_initialized or not _check_prometheus():
        return

    from prometheus_client import Counter, Gauge, Histogram

    # Token counters
    _TOKENS_PROMPT = Counter(
        'llm_tokens_prompt_total',
        'Total prompt tokens used',
        ['model', 'tenant_id']
    )
    _TOKENS_COMPLETION = Counter(
        'llm_tokens_completion_total',
        'Total completion tokens used',
        ['model', 'tenant_id']
    )

    # Cost counter (not gauge!) - use Counter for cumulative values
    _COST_USD = Counter(
        'llm_cost_usd_total',
        'Total estimated cost in USD',
        ['model', 'tenant_id']
    )

    # Request counter
    _REQUESTS_TOTAL = Counter(
        'llm_requests_total',
        'Total LLM API requests',
        ['model', 'tenant_id', 'status']
    )

    # Latency histogram
    _REQUEST_LATENCY = Histogram(
        'llm_request_duration_seconds',
        'LLM request latency',
        ['model'],
        buckets=[0.5, 1, 2, 5, 10, 30, 60]
    )

    # Budget alert gauge
    _BUDGET_REMAINING = Gauge(
        'llm_budget_remaining_usd',
        'Remaining daily budget in USD',
        ['tenant_id']
    )

    _metrics_initialized = True


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UsageRecord:
    """Single usage record."""
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    tenant_id: Optional[str] = None
    request_id: Optional[str] = None
    latency_ms: Optional[int] = None


@dataclass
class DailyStats:
    """Aggregated daily statistics."""
    date: str
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    by_model: Dict[str, Dict] = field(default_factory=dict)
    by_tenant: Dict[str, Dict] = field(default_factory=dict)


# ============================================================================
# COST TRACKER SERVICE
# ============================================================================

class CostTracker:
    """
    Track and persist LLM usage costs.

    Storage:
    - Redis: Real-time counters (fast, volatile)
    - Prometheus: Metrics for Grafana
    - Daily aggregates for analysis
    """

    REDIS_KEY_PREFIX = "cost:"
    REDIS_KEY_DAILY = "cost:daily:{date}"
    REDIS_KEY_TOTAL = "cost:total"
    REDIS_TTL_DAYS = 90  # Keep 90 days of data

    # Budget configuration
    DEFAULT_DAILY_BUDGET_USD = 50.0  # $50/day default
    MAX_EXPORT_DAYS = 365  # Maximum days for billing export

    def __init__(
        self,
        redis_client,
        daily_budget_usd: float = None,
        alert_callback: Callable[[Dict], Awaitable[None]] = None
    ):
        """
        Initialize cost tracker.

        Args:
            redis_client: Async Redis client
            daily_budget_usd: Daily budget limit (triggers alert if exceeded)
            alert_callback: Async function to call when budget exceeded
        """
        self.redis = redis_client
        self.alert_callback = alert_callback

        # Parse daily budget with validation
        self.daily_budget = self._parse_daily_budget(daily_budget_usd)

        # In-memory cache for current session (bounded)
        self._session_tokens = {
            "prompt": 0,
            "completion": 0,
            "cost": 0.0,
            "requests": 0
        }
        self._session_start = datetime.now(timezone.utc)

        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 0.1  # seconds

        logger.info(f"CostTracker initialized: daily_budget=${self.daily_budget}")

    def _parse_daily_budget(self, budget: float = None) -> float:
        """Parse and validate daily budget."""
        if budget is not None:
            return max(0.0, float(budget))

        env_budget = os.getenv("DAILY_COST_BUDGET_USD")
        if env_budget:
            try:
                return max(0.0, float(env_budget))
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid DAILY_COST_BUDGET_USD: {env_budget}, using default")

        return self.DEFAULT_DAILY_BUDGET_USD

    def reset_session(self) -> Dict:
        """
        Reset session counters and return the previous values.

        Returns:
            Previous session statistics
        """
        previous = self._session_tokens.copy()
        previous["session_duration_seconds"] = (
            datetime.now(timezone.utc) - self._session_start
        ).total_seconds()

        self._session_tokens = {
            "prompt": 0,
            "completion": 0,
            "cost": 0.0,
            "requests": 0
        }
        self._session_start = datetime.now(timezone.utc)

        return previous

    async def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-4o-mini",
        tenant_id: str = "default",
        request_id: str = None,
        latency_ms: int = None,
        success: bool = True
    ) -> UsageRecord:
        """
        Record token usage.

        Args:
            prompt_tokens: Number of prompt tokens (must be >= 0)
            completion_tokens: Number of completion tokens (must be >= 0)
            model: Model name
            tenant_id: Tenant identifier
            request_id: Optional request ID for tracing
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded

        Returns:
            UsageRecord with cost estimation
        """
        # Validate tokens (reject negative values)
        if prompt_tokens < 0:
            logger.warning(f"Negative prompt_tokens rejected: {prompt_tokens}")
            prompt_tokens = 0
        if completion_tokens < 0:
            logger.warning(f"Negative completion_tokens rejected: {completion_tokens}")
            completion_tokens = 0

        # Calculate cost with precision
        cost_usd = ModelPricing.calculate_cost(model, prompt_tokens, completion_tokens)
        total_tokens = prompt_tokens + completion_tokens

        # Create record
        record = UsageRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=cost_usd,
            tenant_id=tenant_id,
            request_id=request_id,
            latency_ms=latency_ms
        )

        # Update in-memory counters
        self._session_tokens["prompt"] += prompt_tokens
        self._session_tokens["completion"] += completion_tokens
        self._session_tokens["cost"] += cost_usd
        self._session_tokens["requests"] += 1

        # Update Prometheus metrics
        self._update_prometheus_metrics(record, success)

        # Persist to Redis with retry
        await self._persist_usage_with_retry(record)

        # Check budget
        await self._check_budget(tenant_id)

        return record

    def _update_prometheus_metrics(self, record: UsageRecord, success: bool) -> None:
        """Update Prometheus metrics."""
        if not _check_prometheus():
            return

        _init_prometheus_metrics()

        try:
            _TOKENS_PROMPT.labels(
                model=record.model,
                tenant_id=record.tenant_id
            ).inc(record.prompt_tokens)

            _TOKENS_COMPLETION.labels(
                model=record.model,
                tenant_id=record.tenant_id
            ).inc(record.completion_tokens)

            # Use Counter.inc() properly
            _COST_USD.labels(
                model=record.model,
                tenant_id=record.tenant_id
            ).inc(record.estimated_cost_usd)

            _REQUESTS_TOTAL.labels(
                model=record.model,
                tenant_id=record.tenant_id,
                status="success" if success else "error"
            ).inc()

            if record.latency_ms:
                _REQUEST_LATENCY.labels(model=record.model).observe(record.latency_ms / 1000)
        except Exception as e:
            logger.warning(f"Failed to update Prometheus metrics: {e}")

    async def _persist_usage_with_retry(self, record: UsageRecord) -> bool:
        """Persist usage record to Redis with retry logic."""
        for attempt in range(self._max_retries):
            try:
                await self._persist_usage(record)
                return True
            except Exception as e:
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to persist usage after {self._max_retries} retries: {e}")
        return False

    async def _persist_usage(self, record: UsageRecord) -> None:
        """Persist usage record to Redis."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_key = self.REDIS_KEY_DAILY.format(date=today)

        # Use pipeline for atomicity
        pipe = self.redis.pipeline()
        pipe.hincrby(daily_key, "total_requests", 1)
        pipe.hincrby(daily_key, "prompt_tokens", record.prompt_tokens)
        pipe.hincrby(daily_key, "completion_tokens", record.completion_tokens)

        # Use string for cost to avoid floating point issues
        # Store as integer microcents (cost * 100000000)
        cost_microcents = int(record.estimated_cost_usd * 100000000)
        pipe.hincrby(daily_key, "cost_microcents", cost_microcents)

        pipe.hincrby(daily_key, f"model:{record.model}:tokens", record.total_tokens)
        pipe.hincrby(daily_key, f"tenant:{record.tenant_id}:tokens", record.total_tokens)
        pipe.expire(daily_key, self.REDIS_TTL_DAYS * 86400)

        # Increment total counters
        pipe.hincrby(self.REDIS_KEY_TOTAL, "prompt_tokens", record.prompt_tokens)
        pipe.hincrby(self.REDIS_KEY_TOTAL, "completion_tokens", record.completion_tokens)
        pipe.hincrby(self.REDIS_KEY_TOTAL, "cost_microcents", cost_microcents)

        results = await pipe.execute()

        # Verify pipeline succeeded
        if not all(r is not None for r in results):
            logger.warning("Some Redis pipeline commands may have failed")

    async def _check_budget(self, tenant_id: str) -> None:
        """Check if daily budget exceeded."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            daily_key = self.REDIS_KEY_DAILY.format(date=today)

            # Get cost in microcents
            cost_microcents = await self.redis.hget(daily_key, "cost_microcents")
            daily_cost = int(cost_microcents) / 100000000 if cost_microcents else 0.0

            remaining = self.daily_budget - daily_cost

            # Update Prometheus gauge
            if _check_prometheus():
                _init_prometheus_metrics()
                _BUDGET_REMAINING.labels(tenant_id=tenant_id).set(max(0, remaining))

            # Trigger alert if budget exceeded
            if daily_cost >= self.daily_budget:
                logger.warning(
                    f"BUDGET ALERT: Daily cost ${daily_cost:.2f} exceeds "
                    f"budget ${self.daily_budget:.2f}!"
                )

                if self.alert_callback:
                    try:
                        await self.alert_callback({
                            "type": "budget_exceeded",
                            "tenant_id": tenant_id,
                            "daily_cost": daily_cost,
                            "budget": self.daily_budget,
                            "date": today
                        })
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")

        except Exception as e:
            logger.error(f"Budget check failed: {e}")

    async def get_daily_stats(self, date: str = None) -> DailyStats:
        """
        Get daily statistics.

        Args:
            date: Date in YYYY-MM-DD format (default: today)

        Returns:
            DailyStats object
        """
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        daily_key = self.REDIS_KEY_DAILY.format(date=date)

        try:
            data = await self.redis.hgetall(daily_key)

            if not data:
                return DailyStats(date=date)

            # Parse by-model and by-tenant stats
            by_model = {}
            by_tenant = {}

            for key, value in data.items():
                # Handle bytes from Redis
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                if isinstance(value, bytes):
                    value = value.decode('utf-8')

                if key.startswith("model:"):
                    parts = key.split(":")
                    if len(parts) >= 3:
                        model = parts[1]
                        by_model[model] = {"tokens": int(value)}
                elif key.startswith("tenant:"):
                    parts = key.split(":")
                    if len(parts) >= 3:
                        tenant = parts[1]
                        by_tenant[tenant] = {"tokens": int(value)}

            # Convert microcents back to USD
            cost_microcents = data.get("cost_microcents", data.get(b"cost_microcents", 0))
            if isinstance(cost_microcents, bytes):
                cost_microcents = int(cost_microcents.decode('utf-8'))
            else:
                cost_microcents = int(cost_microcents) if cost_microcents else 0
            cost_usd = cost_microcents / 100000000

            prompt = int(data.get("prompt_tokens", data.get(b"prompt_tokens", 0)))
            completion = int(data.get("completion_tokens", data.get(b"completion_tokens", 0)))

            return DailyStats(
                date=date,
                total_requests=int(data.get("total_requests", data.get(b"total_requests", 0))),
                total_prompt_tokens=prompt,
                total_completion_tokens=completion,
                total_tokens=prompt + completion,
                estimated_cost_usd=cost_usd,
                by_model=by_model,
                by_tenant=by_tenant
            )

        except Exception as e:
            logger.error(f"Failed to get daily stats: {e}")
            return DailyStats(date=date)

    async def get_total_stats(self) -> Dict:
        """Get all-time statistics."""
        try:
            data = await self.redis.hgetall(self.REDIS_KEY_TOTAL)

            if not data:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0
                }

            # Handle bytes
            prompt = data.get("prompt_tokens", data.get(b"prompt_tokens", 0))
            completion = data.get("completion_tokens", data.get(b"completion_tokens", 0))
            cost_microcents = data.get("cost_microcents", data.get(b"cost_microcents", 0))

            if isinstance(prompt, bytes):
                prompt = int(prompt.decode('utf-8'))
            if isinstance(completion, bytes):
                completion = int(completion.decode('utf-8'))
            if isinstance(cost_microcents, bytes):
                cost_microcents = int(cost_microcents.decode('utf-8'))

            prompt = int(prompt) if prompt else 0
            completion = int(completion) if completion else 0
            cost_microcents = int(cost_microcents) if cost_microcents else 0

            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "estimated_cost_usd": cost_microcents / 100000000
            }

        except Exception as e:
            logger.error(f"Failed to get total stats: {e}")
            return {}

    async def get_session_stats(self) -> Dict:
        """Get current session statistics (in-memory)."""
        return {
            "session_prompt_tokens": self._session_tokens["prompt"],
            "session_completion_tokens": self._session_tokens["completion"],
            "session_total_tokens": self._session_tokens["prompt"] + self._session_tokens["completion"],
            "session_cost_usd": self._session_tokens["cost"],
            "session_requests": self._session_tokens["requests"],
            "session_duration_seconds": (
                datetime.now(timezone.utc) - self._session_start
            ).total_seconds()
        }

    async def get_weekly_trend(self) -> List[DailyStats]:
        """Get last 7 days of statistics using batch Redis calls."""
        dates = [
            (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(7)
        ]

        # Batch fetch using pipeline
        pipe = self.redis.pipeline()
        for date in dates:
            daily_key = self.REDIS_KEY_DAILY.format(date=date)
            pipe.hgetall(daily_key)

        try:
            results = await pipe.execute()
            stats = []
            for date, data in zip(dates, results):
                if not data:
                    stats.append(DailyStats(date=date))
                    continue

                # Parse similar to get_daily_stats
                prompt = int(data.get(b"prompt_tokens", data.get("prompt_tokens", 0)) or 0)
                completion = int(data.get(b"completion_tokens", data.get("completion_tokens", 0)) or 0)
                cost_microcents = int(data.get(b"cost_microcents", data.get("cost_microcents", 0)) or 0)

                stats.append(DailyStats(
                    date=date,
                    total_requests=int(data.get(b"total_requests", data.get("total_requests", 0)) or 0),
                    total_prompt_tokens=prompt,
                    total_completion_tokens=completion,
                    total_tokens=prompt + completion,
                    estimated_cost_usd=cost_microcents / 100000000
                ))
            return stats
        except Exception as e:
            logger.error(f"Failed to get weekly trend: {e}")
            return [DailyStats(date=date) for date in dates]

    async def export_for_billing(
        self,
        start_date: str,
        end_date: str,
        max_days: int = None
    ) -> List[Dict]:
        """
        Export usage data for billing.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_days: Maximum days to export (default: 365)

        Returns:
            List of daily usage records
        """
        max_days = max_days or self.MAX_EXPORT_DAYS

        try:
            current = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return []

        # Limit the export range
        day_count = (end - current).days + 1
        if day_count > max_days:
            logger.warning(f"Export limited to {max_days} days (requested: {day_count})")
            end = current + timedelta(days=max_days - 1)

        if day_count <= 0:
            return []

        # Collect all dates
        dates = []
        temp = current
        while temp <= end:
            dates.append(temp.strftime("%Y-%m-%d"))
            temp += timedelta(days=1)

        # Batch fetch using pipeline
        pipe = self.redis.pipeline()
        for date in dates:
            daily_key = self.REDIS_KEY_DAILY.format(date=date)
            pipe.hgetall(daily_key)

        try:
            results = await pipe.execute()
            records = []
            for date, data in zip(dates, results):
                stats = DailyStats(date=date)
                if data:
                    stats.total_requests = int(data.get(b"total_requests", data.get("total_requests", 0)) or 0)
                    stats.total_prompt_tokens = int(data.get(b"prompt_tokens", data.get("prompt_tokens", 0)) or 0)
                    stats.total_completion_tokens = int(data.get(b"completion_tokens", data.get("completion_tokens", 0)) or 0)
                    stats.total_tokens = stats.total_prompt_tokens + stats.total_completion_tokens
                    cost_microcents = int(data.get(b"cost_microcents", data.get("cost_microcents", 0)) or 0)
                    stats.estimated_cost_usd = cost_microcents / 100000000
                records.append(asdict(stats))
            return records
        except Exception as e:
            logger.error(f"Failed to export billing data: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for monitoring.

        Returns:
            Health status with Redis connectivity and stats
        """
        health = {
            "healthy": True,
            "redis_connected": False,
            "prometheus_available": _check_prometheus(),
            "session_stats": await self.get_session_stats(),
            "daily_budget": self.daily_budget,
            "checked_at": datetime.now(timezone.utc).isoformat()
        }

        try:
            # Check Redis connectivity
            await self.redis.ping()
            health["redis_connected"] = True

            # Get today's stats
            today_stats = await self.get_daily_stats()
            health["today_cost_usd"] = today_stats.estimated_cost_usd
            health["budget_remaining_usd"] = max(0, self.daily_budget - today_stats.estimated_cost_usd)
            health["budget_used_percent"] = (
                (today_stats.estimated_cost_usd / self.daily_budget * 100)
                if self.daily_budget > 0 else 0
            )

        except Exception as e:
            health["healthy"] = False
            health["error"] = str(e)

        return health


# Global instance with lock for thread safety
_cost_tracker: Optional[CostTracker] = None
_cost_tracker_lock = asyncio.Lock()


async def get_cost_tracker(redis_client) -> CostTracker:
    """Get or create cost tracker singleton (thread-safe async)."""
    global _cost_tracker

    if _cost_tracker is not None:
        return _cost_tracker

    async with _cost_tracker_lock:
        # Double-check after acquiring lock
        if _cost_tracker is None:
            _cost_tracker = CostTracker(redis_client)
        return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset the singleton (for testing)."""
    global _cost_tracker
    _cost_tracker = None
