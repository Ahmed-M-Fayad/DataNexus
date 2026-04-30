"""
DataNexus — Score Calculator
Computes the Quality Score (0.0 – 100.0) from a list of check results.

This module is pure computation — no database access, no I/O.
It can be imported and tested in complete isolation.

Formula
-------
    score = (sum of severity weights for PASSING checks
             / sum of severity weights for ALL checks) × 100

Severity weights (from the SRS):
    Critical → 1.00   (a failed critical check costs the most)
    High     → 0.75
    Medium   → 0.50
    Low      → 0.25

Status rules:
    "pass"  → weight counts toward passed_weight ✓
    "skip"  → treated as pass (check was intentionally not run)
    "fail"  → weight does NOT count toward passed_weight ✗
    "error" → treated as fail (check could not run = unknown = unsafe to ignore) ✗

Usage
-----
    from src.validator.score_calculator import calculate_quality_score

    results = [
        {"status": "pass", "severity": "high"},
        {"status": "fail", "severity": "critical"},
    ]
    score = calculate_quality_score(results)   # → 42.86
"""

import logging

logger = logging.getLogger(__name__)

# ── Severity → numeric weight ────────────────────────────────────────────────
# Defined once here. If the SRS changes these weights, change them here only.
SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 1.00,
    "high":     0.75,
    "medium":   0.50,
    "low":      0.25,
}

# Statuses that contribute to the passed_weight numerator.
# "skip" is included because a skipped check was not run by design —
# it should not penalise the score.
_PASSING_STATUSES: frozenset[str] = frozenset({"pass", "skip"})

# Default weight used when an unrecognised severity string is encountered.
# Falls back to "medium" so unknown severities still cause a moderate penalty.
_DEFAULT_WEIGHT: float = SEVERITY_WEIGHTS["medium"]


def calculate_quality_score(results: list[dict]) -> float:
    """
    Computes the weighted Quality Score from a list of check result dicts.

    Args:
        results: list of dicts, each must contain at minimum:
                   "status"   → "pass" | "fail" | "error" | "skip"
                   "severity" → "critical" | "high" | "medium" | "low"
                 Extra keys are silently ignored.

    Returns:
        float in [0.0, 100.0].
        Returns 100.0 when results is empty (nothing ran = nothing failed).

    Examples:
        All checks pass                          → 100.0
        All checks fail                          → 0.0
        1 critical fail + 1 medium pass          → (0.50 / 1.50) × 100 = 33.33
        1 high pass + 1 high fail                → (0.75 / 1.50) × 100 = 50.00
        1 critical pass + 1 low fail             → (1.00 / 1.25) × 100 = 80.00
    """
    # ── edge case: nothing ran ────────────────────────────────────────────────
    if not results:
        logger.warning(
            "calculate_quality_score received an empty results list. "
            "Returning 100.0 (nothing failed = perfect score by default)."
        )
        return 100.0

    total_weight:  float = 0.0   # denominator — sum of ALL check weights
    passed_weight: float = 0.0   # numerator   — sum of PASSING check weights

    for r in results:
        # Normalise to lowercase strings to tolerate capitalisation differences
        severity: str = str(r.get("severity", "medium")).lower().strip()
        status:   str = str(r.get("status",   "error")).lower().strip()

        # Look up weight; fall back to medium for unrecognised severities.
        # This prevents a crash from a config typo while still applying a penalty.
        weight: float = SEVERITY_WEIGHTS.get(severity, _DEFAULT_WEIGHT)
        if severity not in SEVERITY_WEIGHTS:
            logger.warning(
                f"Unrecognised severity '{severity}' — using weight {_DEFAULT_WEIGHT} "
                f"(medium default). Check your validation config."
            )

        total_weight += weight

        if status in _PASSING_STATUSES:
            passed_weight += weight
        # "fail" and "error" add to total_weight but NOT to passed_weight,
        # which is correct — they lower the score proportionally.

    # ── guard: prevents ZeroDivisionError if somehow all weights were 0 ──────
    if total_weight == 0.0:
        return 100.0

    # ── compute, clamp to [0, 100], round to 2 decimal places ────────────────
    raw: float   = (passed_weight / total_weight) * 100.0
    score: float = round(max(0.0, min(100.0, raw)), 2)

    logger.info(
        f"Quality score: {score:.2f} | "
        f"passed_weight={passed_weight:.2f} / total_weight={total_weight:.2f} | "
        f"checks: {len(results)} total, "
        f"{sum(1 for r in results if str(r.get('status','')).lower() == 'pass')} pass, "
        f"{sum(1 for r in results if str(r.get('status','')).lower() == 'fail')} fail, "
        f"{sum(1 for r in results if str(r.get('status','')).lower() == 'error')} error."
    )
    return score


def score_breakdown(results: list[dict]) -> dict:
    """
    Returns a detailed breakdown of the quality score computation.
    Useful for the dashboard's Run Details page and for debugging.

    Args:
        results: same list of dicts as calculate_quality_score().

    Returns:
        {
            "score":          float,   # final 0–100 quality score
            "total_checks":   int,
            "passed_checks":  int,
            "failed_checks":  int,
            "error_checks":   int,
            "skipped_checks": int,
            "by_severity": {
                "critical": { "total": int, "passed": int, "failed": int,
                              "weight_contributed": float },
                "high":     { ... },
                "medium":   { ... },
                "low":      { ... },
            }
        }
    """
    score = calculate_quality_score(results)

    # Initialise the breakdown structure with zero counters for every severity.
    breakdown: dict = {
        "score":          score,
        "total_checks":   len(results),
        "passed_checks":  0,
        "failed_checks":  0,
        "error_checks":   0,
        "skipped_checks": 0,
        "by_severity": {
            sev: {"total": 0, "passed": 0, "failed": 0, "weight_contributed": 0.0}
            for sev in SEVERITY_WEIGHTS    # "critical", "high", "medium", "low"
        },
    }

    for r in results:
        severity: str = str(r.get("severity", "medium")).lower().strip()
        status:   str = str(r.get("status",   "error")).lower().strip()

        # ── top-level counters ────────────────────────────────────────────────
        if   status == "pass":  breakdown["passed_checks"]  += 1
        elif status == "fail":  breakdown["failed_checks"]  += 1
        elif status == "error": breakdown["error_checks"]   += 1
        elif status == "skip":  breakdown["skipped_checks"] += 1

        # ── per-severity counters ─────────────────────────────────────────────
        if severity in breakdown["by_severity"]:
            bucket = breakdown["by_severity"][severity]
            bucket["total"] += 1
            if status in _PASSING_STATUSES:
                bucket["passed"]             += 1
                bucket["weight_contributed"] += SEVERITY_WEIGHTS[severity]
            elif status in ("fail", "error"):
                bucket["failed"] += 1

    return breakdown
