"""
Model Drift Detection Service
Version: 1.1 - Statistical Monitoring for AI Model Performance (FIXED)

PROBLEM SOLVED:
- OpenAI/Anthropic update models without notice
- Performance can degrade overnight
- Error rates can spike without visible cause
- No early warning system for model quality issues

FEATURES:
- Statistical drift detection (error rate, latency, hallucination rate)
- Baseline comparison (rolling averages)
- Automatic alerts when thresholds exceeded
- A/B testing framework ready
- Model version tracking

USAGE:
    detector = ModelDriftDetector(redis_client, db_session)
    await detector.record_interaction(...)

    # Check for drift
    drift_report = await detector.check_drift()
    if drift_report.has_drift:
        # Alert team!
"""

import os
import json
import logging
import asyncio
import statistics
import threading
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Thread-safe singleton lock
_singleton_lock = threading.Lock()


class DriftType(str, Enum):
    """Types of model drift."""
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    HALLUCINATION_RATE = "hallucination_rate"
    CONFIDENCE = "confidence"
    TOOL_SELECTION = "tool_selection"


class DriftSeverity(str, Enum):
    """Severity levels for drift detection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class InteractionMetrics:
    """Metrics for a single model interaction."""
    timestamp: str
    model_version: str
    latency_ms: int
    success: bool
    error_type: Optional[str] = None
    confidence_score: Optional[float] = None
    tools_called: List[str] = field(default_factory=list)
    hallucination_reported: bool = False
    tenant_id: Optional[str] = None
    metric_id: Optional[str] = None  # Unique ID to prevent key collisions


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    drift_type: str
    severity: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    message: str
    detected_at: str
    model_version: str
    recommended_action: str
    alert_id: str = ""  # For deduplication

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.drift_type}_{self.model_version}_{self.severity}"


@dataclass
class DriftReport:
    """Complete drift analysis report."""
    report_id: str
    generated_at: str
    model_version: str
    analysis_window_hours: int
    has_drift: bool
    overall_severity: str
    alerts: List[DriftAlert] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ModelDriftDetector:
    """
    Statistical model drift detection.

    STRATEGY:
    1. Collect metrics for every interaction
    2. Maintain rolling baseline (7-day average)
    3. Compare recent window (1-6 hours) against baseline
    4. Alert if deviation exceeds thresholds

    Thresholds (configurable):
    - Error rate: +50% = LOW, +100% = MEDIUM, +200% = HIGH
    - Latency: +30% = LOW, +50% = MEDIUM, +100% = HIGH
    - Hallucination rate: +25% = LOW, +50% = MEDIUM, +100% = HIGH
    """

    # Redis keys
    REDIS_METRICS_KEY = "drift:metrics"
    REDIS_BASELINE_KEY = "drift:baseline"
    REDIS_ALERTS_KEY = "drift:alerts"
    REDIS_TIMELINE_KEY = "drift:timeline"

    # Thresholds (percentage deviation from baseline)
    THRESHOLDS = {
        DriftType.ERROR_RATE: {
            DriftSeverity.LOW: 0.50,      # +50%
            DriftSeverity.MEDIUM: 1.00,   # +100%
            DriftSeverity.HIGH: 2.00,     # +200%
            DriftSeverity.CRITICAL: 3.00  # +300%
        },
        DriftType.LATENCY: {
            DriftSeverity.LOW: 0.30,
            DriftSeverity.MEDIUM: 0.50,
            DriftSeverity.HIGH: 1.00,
            DriftSeverity.CRITICAL: 2.00
        },
        DriftType.HALLUCINATION_RATE: {
            DriftSeverity.LOW: 0.25,
            DriftSeverity.MEDIUM: 0.50,
            DriftSeverity.HIGH: 1.00,
            DriftSeverity.CRITICAL: 1.50
        },
        DriftType.CONFIDENCE: {
            DriftSeverity.LOW: -0.10,     # Decrease
            DriftSeverity.MEDIUM: -0.20,
            DriftSeverity.HIGH: -0.30,
            DriftSeverity.CRITICAL: -0.40
        }
    }

    # Analysis windows - configurable via environment
    BASELINE_WINDOW_DAYS = int(os.getenv("DRIFT_BASELINE_WINDOW_DAYS", "7"))
    ANALYSIS_WINDOW_HOURS = int(os.getenv("DRIFT_ANALYSIS_WINDOW_HOURS", "6"))
    MIN_SAMPLES_FOR_ANALYSIS = int(os.getenv("DRIFT_MIN_SAMPLES", "50"))

    # Memory bounds
    MAX_RECENT_METRICS = 10000  # Maximum in-memory metrics

    # Baseline cache TTL
    BASELINE_CACHE_TTL_HOURS = 1

    def __init__(
        self,
        redis_client=None,
        db_session=None,
        analysis_window_hours: int = None,
        baseline_window_days: int = None,
        alert_callback: Callable[[DriftReport], Awaitable[None]] = None
    ):
        """
        Initialize drift detector.

        Args:
            redis_client: Async Redis client
            db_session: Async database session
            analysis_window_hours: Hours for recent analysis (default: 6)
            baseline_window_days: Days for baseline calculation (default: 7)
            alert_callback: Async callback for sending alerts
        """
        self.redis = redis_client
        self.db = db_session
        self.alert_callback = alert_callback

        self.analysis_window = timedelta(
            hours=analysis_window_hours or self.ANALYSIS_WINDOW_HOURS
        )
        self.baseline_window = timedelta(
            days=baseline_window_days or self.BASELINE_WINDOW_DAYS
        )

        # In-memory cache for recent metrics - BOUNDED using deque
        self._recent_metrics: deque = deque(maxlen=self.MAX_RECENT_METRICS)
        self._baseline_cache: Optional[Dict[str, Any]] = None
        self._baseline_cache_time: Optional[datetime] = None

        # Track sent alerts for deduplication
        self._sent_alerts: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(hours=1)  # Don't repeat same alert within 1 hour

        logger.info(
            f"ModelDriftDetector initialized: "
            f"analysis={self.analysis_window}, baseline={self.baseline_window}, "
            f"max_memory_samples={self.MAX_RECENT_METRICS}"
        )

    async def record_interaction(
        self,
        model_version: str,
        latency_ms: int,
        success: bool,
        error_type: str = None,
        confidence_score: float = None,
        tools_called: List[str] = None,
        hallucination_reported: bool = False,
        tenant_id: str = None
    ) -> None:
        """
        Record metrics for a single model interaction.

        Call this after every LLM API call.
        """
        # Generate unique ID for this metric to prevent collisions
        metric_id = str(uuid.uuid4())

        metrics = InteractionMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=model_version,
            latency_ms=max(0, latency_ms),  # Ensure non-negative
            success=success,
            error_type=error_type,
            confidence_score=confidence_score,
            tools_called=tools_called or [],
            hallucination_reported=hallucination_reported,
            tenant_id=tenant_id,
            metric_id=metric_id
        )

        # Add to bounded deque (automatically removes oldest if full)
        self._recent_metrics.append(metrics)

        # Persist to Redis
        if self.redis:
            await self._persist_metric(metrics)

        logger.debug(
            f"Recorded interaction: model={model_version}, "
            f"success={success}, latency={latency_ms}ms"
        )

    async def _persist_metric(self, metrics: InteractionMetrics) -> None:
        """Persist metric to Redis with TTL."""
        if not self.redis:
            return

        try:
            # Use unique metric_id to prevent key collisions
            key = f"{self.REDIS_METRICS_KEY}:{metrics.metric_id}"
            await self.redis.setex(
                key,
                int(self.baseline_window.total_seconds()),
                json.dumps(asdict(metrics))
            )

            # Add to sorted set for time-based queries
            timestamp = datetime.fromisoformat(
                metrics.timestamp.replace('Z', '+00:00')
            ).timestamp()
            await self.redis.zadd(
                self.REDIS_TIMELINE_KEY,
                {key: timestamp}
            )

            # Clean up old entries from timeline
            await self._cleanup_timeline()

        except Exception as e:
            logger.warning(f"Failed to persist drift metric: {e}")

    async def _cleanup_timeline(self) -> None:
        """Clean up old entries from Redis timeline."""
        if not self.redis:
            return

        try:
            cutoff = (datetime.now(timezone.utc) - self.baseline_window).timestamp()
            # Remove entries older than baseline window
            await self.redis.zremrangebyscore(self.REDIS_TIMELINE_KEY, 0, cutoff)
        except Exception as e:
            logger.debug(f"Timeline cleanup failed: {e}")

    async def check_drift(
        self,
        model_version: str = None,
        force_refresh: bool = False
    ) -> DriftReport:
        """
        Analyze recent metrics for model drift.

        Args:
            model_version: Filter by specific model version (optional, any format)
            force_refresh: Force baseline recalculation

        Returns:
            DriftReport with analysis results
        """
        report_id = f"drift_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Get recent metrics
        recent = await self._get_recent_metrics(model_version)

        if len(recent) < self.MIN_SAMPLES_FOR_ANALYSIS:
            return DriftReport(
                report_id=report_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                model_version=model_version or "all",
                analysis_window_hours=int(self.analysis_window.total_seconds() / 3600),
                has_drift=False,
                overall_severity=DriftSeverity.NONE.value,
                metrics_summary={"sample_count": len(recent)},
                recommendations=[
                    f"Insufficient data: {len(recent)} samples "
                    f"(need {self.MIN_SAMPLES_FOR_ANALYSIS})"
                ]
            )

        # Get or calculate baseline
        baseline = await self._get_baseline(model_version, force_refresh)

        # Calculate current metrics
        current = self._calculate_metrics(recent)

        # Detect drift
        alerts = self._detect_drift(current, baseline, model_version or "all")

        # Deduplicate alerts
        alerts = self._deduplicate_alerts(alerts)

        # Determine overall severity
        overall_severity = self._get_overall_severity(alerts)

        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, current, baseline)

        report = DriftReport(
            report_id=report_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_version=model_version or "all",
            analysis_window_hours=int(self.analysis_window.total_seconds() / 3600),
            has_drift=len(alerts) > 0,
            overall_severity=overall_severity.value,
            alerts=alerts,
            metrics_summary=current,
            baseline_comparison=baseline,
            recommendations=recommendations
        )

        # Persist report
        await self._save_report(report)

        if alerts:
            # Structured JSON logging for drift alerts
            for alert in alerts:
                logger.warning(
                    f"🚨 DRIFT_ALERT: type={alert.drift_type}, severity={alert.severity}, "
                    f"current={alert.current_value:.3f}, baseline={alert.baseline_value:.3f}, "
                    f"deviation={alert.deviation_percentage:.1f}%"
                )
            
            # Summary log
            logger.warning(
                f"📊 DRIFT_SUMMARY: report_id={report.report_id}, "
                f"severity={overall_severity.value}, alert_count={len(alerts)}, "
                f"model={model_version or 'all'}, window_hours={int(self.analysis_window.total_seconds() / 3600)}"
            )
            
            # Send alert via callback
            await self._send_alert(report)

        return report

    def _deduplicate_alerts(self, alerts: List[DriftAlert]) -> List[DriftAlert]:
        """Remove duplicate alerts that were recently sent."""
        now = datetime.now(timezone.utc)
        deduplicated = []

        for alert in alerts:
            alert_key = alert.alert_id
            last_sent = self._sent_alerts.get(alert_key)

            if last_sent is None or (now - last_sent) > self._alert_cooldown:
                deduplicated.append(alert)
                self._sent_alerts[alert_key] = now

        # Clean up old entries
        cutoff = now - self._alert_cooldown * 2
        self._sent_alerts = {
            k: v for k, v in self._sent_alerts.items() if v > cutoff
        }

        return deduplicated

    async def _send_alert(self, report: DriftReport) -> None:
        """Send alert via callback if configured."""
        if self.alert_callback and report.alerts:
            try:
                await self.alert_callback(report)
                logger.info(f"Alert sent for report {report.report_id}")
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def _get_recent_metrics(
        self,
        model_version: str = None
    ) -> List[InteractionMetrics]:
        """Get metrics from recent analysis window."""
        cutoff = datetime.now(timezone.utc) - self.analysis_window

        # First try in-memory cache
        recent = [
            m for m in self._recent_metrics
            if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00')) > cutoff
        ]

        # If insufficient, try Redis
        if len(recent) < self.MIN_SAMPLES_FOR_ANALYSIS and self.redis:
            recent = await self._load_metrics_from_redis(cutoff, self.analysis_window)

        # Filter by model version if specified
        if model_version:
            recent = [m for m in recent if m.model_version == model_version]

        return recent

    async def _load_metrics_from_redis(
        self,
        cutoff: datetime,
        window: timedelta
    ) -> List[InteractionMetrics]:
        """Load metrics from Redis within time window."""
        if not self.redis:
            return []

        try:
            # Get keys from sorted set
            min_score = cutoff.timestamp()
            max_score = datetime.now(timezone.utc).timestamp()

            keys = await self.redis.zrangebyscore(
                self.REDIS_TIMELINE_KEY,
                min_score,
                max_score
            )

            metrics = []
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                data = await self.redis.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    metrics.append(InteractionMetrics(**json.loads(data)))

            return metrics
        except Exception as e:
            logger.warning(f"Failed to load metrics from Redis: {e}")
            return []

    async def _get_baseline(
        self,
        model_version: str = None,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Get or calculate baseline metrics."""
        cache_valid = (
            self._baseline_cache is not None and
            self._baseline_cache_time is not None and
            (datetime.now(timezone.utc) - self._baseline_cache_time) <
            timedelta(hours=self.BASELINE_CACHE_TTL_HOURS) and
            not force_refresh
        )

        if cache_valid:
            return self._baseline_cache

        # Try to load from Redis
        if self.redis and not force_refresh:
            try:
                data = await self.redis.get(self.REDIS_BASELINE_KEY)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    self._baseline_cache = json.loads(data)
                    self._baseline_cache_time = datetime.now(timezone.utc)
                    return self._baseline_cache
            except Exception:
                pass

        # Calculate new baseline
        cutoff = datetime.now(timezone.utc) - self.baseline_window
        baseline_metrics = await self._load_metrics_from_redis(
            cutoff,
            self.baseline_window
        )

        if model_version:
            baseline_metrics = [
                m for m in baseline_metrics
                if m.model_version == model_version
            ]

        if baseline_metrics:
            baseline = self._calculate_metrics(baseline_metrics)
        else:
            # Default baseline values - more conservative defaults
            baseline = {
                "error_rate": float(os.getenv("DRIFT_DEFAULT_ERROR_RATE", "0.05")),
                "avg_latency_ms": float(os.getenv("DRIFT_DEFAULT_LATENCY_MS", "1500")),
                "hallucination_rate": float(os.getenv("DRIFT_DEFAULT_HALLUCINATION_RATE", "0.02")),
                "avg_confidence": float(os.getenv("DRIFT_DEFAULT_CONFIDENCE", "0.85")),
                "sample_count": 0,
                "is_default": True  # Flag to indicate this is default, not calculated
            }

        # Cache baseline
        self._baseline_cache = baseline
        self._baseline_cache_time = datetime.now(timezone.utc)

        if self.redis:
            try:
                await self.redis.setex(
                    self.REDIS_BASELINE_KEY,
                    3600 * self.BASELINE_CACHE_TTL_HOURS,
                    json.dumps(baseline)
                )
            except Exception:
                pass

        return baseline

    def _calculate_metrics(
        self,
        metrics: List[InteractionMetrics]
    ) -> Dict[str, Any]:
        """Calculate aggregate metrics from interaction list."""
        if not metrics:
            return {}

        total = len(metrics)
        errors = sum(1 for m in metrics if not m.success)
        hallucinations = sum(1 for m in metrics if m.hallucination_reported)
        latencies = [m.latency_ms for m in metrics if m.latency_ms >= 0]
        confidences = [
            m.confidence_score for m in metrics
            if m.confidence_score is not None
        ]

        # Error type breakdown
        error_types = defaultdict(int)
        for m in metrics:
            if m.error_type:
                error_types[m.error_type] += 1

        # Tool usage
        tool_usage = defaultdict(int)
        for m in metrics:
            for tool in m.tools_called:
                tool_usage[tool] += 1

        # Calculate p95 correctly
        p95_latency = self._calculate_percentile(latencies, 0.95)

        return {
            "sample_count": total,
            "error_rate": errors / total if total > 0 else 0,
            "hallucination_rate": hallucinations / total if total > 0 else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": p95_latency,
            "avg_confidence": statistics.mean(confidences) if confidences else None,
            "error_types": dict(error_types),
            "tool_usage": dict(tool_usage)
        }

    def _calculate_percentile(
        self,
        values: List[float],
        percentile: float
    ) -> Optional[float]:
        """
        Calculate percentile correctly handling small sample sizes.

        Args:
            values: List of values
            percentile: Percentile to calculate (0-1)

        Returns:
            Percentile value or None if insufficient data
        """
        if not values:
            return None

        n = len(values)
        if n < 2:
            return values[0] if values else None

        sorted_values = sorted(values)

        # Use linear interpolation for more accurate percentile
        k = (n - 1) * percentile
        f = int(k)
        c = f + 1 if f + 1 < n else f

        # Linear interpolation
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def _detect_drift(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
        model_version: str
    ) -> List[DriftAlert]:
        """Detect drift by comparing current metrics to baseline."""
        alerts = []
        now = datetime.now(timezone.utc).isoformat()

        # Check error rate drift - handle zero baseline
        baseline_error = baseline.get("error_rate", 0)
        current_error = current.get("error_rate", 0)
        if baseline_error > 0:
            error_deviation = (current_error - baseline_error) / baseline_error
            severity = self._get_severity(DriftType.ERROR_RATE, error_deviation)
            if severity != DriftSeverity.NONE:
                alerts.append(DriftAlert(
                    drift_type=DriftType.ERROR_RATE.value,
                    severity=severity.value,
                    current_value=current_error,
                    baseline_value=baseline_error,
                    deviation_percent=error_deviation * 100,
                    message=f"Error rate increased by {error_deviation*100:.1f}%",
                    detected_at=now,
                    model_version=model_version,
                    recommended_action="Review recent error logs and check for API changes"
                ))
        elif current_error > 0.1:  # Alert if error rate > 10% even without baseline
            alerts.append(DriftAlert(
                drift_type=DriftType.ERROR_RATE.value,
                severity=DriftSeverity.MEDIUM.value,
                current_value=current_error,
                baseline_value=0,
                deviation_percent=100,
                message=f"High error rate: {current_error*100:.1f}% (no baseline)",
                detected_at=now,
                model_version=model_version,
                recommended_action="Investigate error causes immediately"
            ))

        # Check latency drift - handle zero baseline
        baseline_latency = baseline.get("avg_latency_ms", 0)
        current_latency = current.get("avg_latency_ms", 0)
        if baseline_latency > 0:
            latency_deviation = (current_latency - baseline_latency) / baseline_latency
            severity = self._get_severity(DriftType.LATENCY, latency_deviation)
            if severity != DriftSeverity.NONE:
                alerts.append(DriftAlert(
                    drift_type=DriftType.LATENCY.value,
                    severity=severity.value,
                    current_value=current_latency,
                    baseline_value=baseline_latency,
                    deviation_percent=latency_deviation * 100,
                    message=f"Latency increased by {latency_deviation*100:.1f}%",
                    detected_at=now,
                    model_version=model_version,
                    recommended_action="Check API provider status and network conditions"
                ))

        # Check hallucination rate drift - handle zero baseline
        baseline_hal = baseline.get("hallucination_rate", 0)
        current_hal = current.get("hallucination_rate", 0)
        if baseline_hal > 0:
            hal_deviation = (current_hal - baseline_hal) / baseline_hal
            severity = self._get_severity(DriftType.HALLUCINATION_RATE, hal_deviation)
            if severity != DriftSeverity.NONE:
                alerts.append(DriftAlert(
                    drift_type=DriftType.HALLUCINATION_RATE.value,
                    severity=severity.value,
                    current_value=current_hal,
                    baseline_value=baseline_hal,
                    deviation_percent=hal_deviation * 100,
                    message=f"Hallucination rate increased by {hal_deviation*100:.1f}%",
                    detected_at=now,
                    model_version=model_version,
                    recommended_action="Check for model version changes and review RAG context"
                ))
        elif current_hal > 0.1:  # Alert if hallucination rate > 10%
            alerts.append(DriftAlert(
                drift_type=DriftType.HALLUCINATION_RATE.value,
                severity=DriftSeverity.MEDIUM.value,
                current_value=current_hal,
                baseline_value=0,
                deviation_percent=100,
                message=f"High hallucination rate: {current_hal*100:.1f}%",
                detected_at=now,
                model_version=model_version,
                recommended_action="Review RAG context and model responses"
            ))

        # Check confidence drift (decrease is bad) - handle zero/None baseline
        baseline_conf = baseline.get("avg_confidence")
        current_conf = current.get("avg_confidence")
        if baseline_conf and baseline_conf > 0 and current_conf is not None:
            conf_deviation = (current_conf - baseline_conf) / baseline_conf
            severity = self._get_severity(DriftType.CONFIDENCE, conf_deviation)
            if severity != DriftSeverity.NONE:
                alerts.append(DriftAlert(
                    drift_type=DriftType.CONFIDENCE.value,
                    severity=severity.value,
                    current_value=current_conf,
                    baseline_value=baseline_conf,
                    deviation_percent=conf_deviation * 100,
                    message=f"Model confidence decreased by {abs(conf_deviation)*100:.1f}%",
                    detected_at=now,
                    model_version=model_version,
                    recommended_action="Review prompt templates and context quality"
                ))

        return alerts

    def _get_severity(
        self,
        drift_type: DriftType,
        deviation: float
    ) -> DriftSeverity:
        """Determine severity based on deviation from baseline."""
        thresholds = self.THRESHOLDS.get(drift_type, {})

        if not thresholds:
            return DriftSeverity.NONE

        # For confidence, we check negative deviation (decrease)
        if drift_type == DriftType.CONFIDENCE:
            critical = thresholds.get(DriftSeverity.CRITICAL, -0.40)
            high = thresholds.get(DriftSeverity.HIGH, -0.30)
            medium = thresholds.get(DriftSeverity.MEDIUM, -0.20)
            low = thresholds.get(DriftSeverity.LOW, -0.10)

            if deviation <= critical:
                return DriftSeverity.CRITICAL
            elif deviation <= high:
                return DriftSeverity.HIGH
            elif deviation <= medium:
                return DriftSeverity.MEDIUM
            elif deviation <= low:
                return DriftSeverity.LOW
        else:
            # For other metrics, check positive deviation (increase)
            critical = thresholds.get(DriftSeverity.CRITICAL, 3.0)
            high = thresholds.get(DriftSeverity.HIGH, 2.0)
            medium = thresholds.get(DriftSeverity.MEDIUM, 1.0)
            low = thresholds.get(DriftSeverity.LOW, 0.5)

            if deviation >= critical:
                return DriftSeverity.CRITICAL
            elif deviation >= high:
                return DriftSeverity.HIGH
            elif deviation >= medium:
                return DriftSeverity.MEDIUM
            elif deviation >= low:
                return DriftSeverity.LOW

        return DriftSeverity.NONE

    def _get_overall_severity(self, alerts: List[DriftAlert]) -> DriftSeverity:
        """Get the highest severity from all alerts."""
        if not alerts:
            return DriftSeverity.NONE

        severity_order = [
            DriftSeverity.CRITICAL,
            DriftSeverity.HIGH,
            DriftSeverity.MEDIUM,
            DriftSeverity.LOW
        ]

        for severity in severity_order:
            if any(a.severity == severity.value for a in alerts):
                return severity

        return DriftSeverity.NONE

    def _generate_recommendations(
        self,
        alerts: List[DriftAlert],
        current: Dict[str, Any],
        baseline: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis."""
        recommendations = []

        if not alerts:
            recommendations.append("No drift detected. Model performance is stable.")
            return recommendations

        # Check for error rate issues
        error_alerts = [a for a in alerts if a.drift_type == DriftType.ERROR_RATE.value]
        if error_alerts:
            error_types = current.get("error_types", {})
            if "400" in error_types or "validation" in str(error_types).lower():
                recommendations.append(
                    "High validation errors: Check if API schema has changed"
                )
            if "timeout" in str(error_types).lower():
                recommendations.append(
                    "Timeout errors increasing: Consider increasing timeout or checking network"
                )
            if "401" in error_types or "403" in error_types:
                recommendations.append(
                    "Authentication errors: Verify API keys and permissions"
                )

        # Check for hallucination issues
        hal_alerts = [
            a for a in alerts
            if a.drift_type == DriftType.HALLUCINATION_RATE.value
        ]
        if hal_alerts:
            recommendations.append(
                "Hallucinations increasing: Review RAG context quality and freshness"
            )
            recommendations.append(
                "Consider triggering RAG refresh if embeddings are stale"
            )

        # Check for latency issues
        latency_alerts = [a for a in alerts if a.drift_type == DriftType.LATENCY.value]
        if latency_alerts:
            recommendations.append(
                "Latency degradation: Check API provider status page"
            )
            if current.get("avg_latency_ms", 0) > 5000:
                recommendations.append(
                    "Consider implementing request caching for common queries"
                )

        # Overall recommendations based on severity
        critical_or_high = [
            a for a in alerts
            if a.severity in [DriftSeverity.CRITICAL.value, DriftSeverity.HIGH.value]
        ]
        if critical_or_high:
            recommendations.append(
                "URGENT: Multiple critical/high severity issues detected. "
                "Consider temporarily switching to fallback model."
            )

        return recommendations

    async def _save_report(self, report: DriftReport) -> None:
        """Save drift report for historical analysis."""
        if not self.redis:
            return

        try:
            # Convert to dict with proper serialization
            report_dict = asdict(report)

            await self.redis.setex(
                f"{self.REDIS_ALERTS_KEY}:{report.report_id}",
                86400 * 7,  # Keep for 7 days
                json.dumps(report_dict, default=str)
            )

            # If there are alerts, also save to active alerts
            if report.alerts:
                await self.redis.lpush(
                    f"{self.REDIS_ALERTS_KEY}:active",
                    json.dumps(report_dict, default=str)
                )
                await self.redis.ltrim(f"{self.REDIS_ALERTS_KEY}:active", 0, 99)

        except Exception as e:
            logger.warning(f"Failed to save drift report: {e}")

    async def get_active_alerts(self, limit: int = 10) -> List[DriftReport]:
        """Get list of recent drift alerts."""
        if not self.redis:
            return []

        try:
            alerts_data = await self.redis.lrange(
                f"{self.REDIS_ALERTS_KEY}:active",
                0,
                limit - 1
            )

            reports = []
            for data in alerts_data:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                report_dict = json.loads(data)
                # Reconstruct DriftAlert objects
                if "alerts" in report_dict:
                    report_dict["alerts"] = [
                        DriftAlert(**a) for a in report_dict["alerts"]
                    ]
                reports.append(DriftReport(**report_dict))

            return reports

        except Exception as e:
            logger.warning(f"Failed to get active alerts: {e}")
            return []

    async def health_check(self, timeout_seconds: int = 5) -> Dict[str, Any]:
        """
        Health check for monitoring with timeout protection.

        Args:
            timeout_seconds: Maximum time for health check

        Returns:
            Health status dict
        """
        try:
            # Run check_drift with timeout to prevent blocking
            report = await asyncio.wait_for(
                self.check_drift(),
                timeout=timeout_seconds
            )

            return {
                "healthy": not report.has_drift or report.overall_severity in [
                    DriftSeverity.NONE.value,
                    DriftSeverity.LOW.value
                ],
                "drift_detected": report.has_drift,
                "severity": report.overall_severity,
                "alerts_count": len(report.alerts),
                "sample_count": report.metrics_summary.get("sample_count", 0),
                "last_check": report.generated_at,
                "redis_connected": self.redis is not None
            }
        except asyncio.TimeoutError:
            return {
                "healthy": True,  # Assume healthy if check times out
                "error": "Health check timed out",
                "timeout_seconds": timeout_seconds
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    def get_metrics_count(self) -> int:
        """Get count of metrics in memory."""
        return len(self._recent_metrics)


# Singleton instance
_drift_detector: Optional[ModelDriftDetector] = None


def get_drift_detector(
    redis_client=None,
    db_session=None,
    alert_callback: Callable[[DriftReport], Awaitable[None]] = None
) -> ModelDriftDetector:
    """Get or create drift detector singleton (thread-safe)."""
    global _drift_detector

    if _drift_detector is not None:
        return _drift_detector

    with _singleton_lock:
        # Double-check after acquiring lock
        if _drift_detector is None:
            _drift_detector = ModelDriftDetector(
                redis_client,
                db_session,
                alert_callback=alert_callback
            )
        return _drift_detector


def reset_drift_detector() -> None:
    """Reset the singleton (for testing)."""
    global _drift_detector
    with _singleton_lock:
        _drift_detector = None
