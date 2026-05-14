import logging

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 1.00,
    "high":     0.75,
    "medium":   0.50,
    "low":      0.25,
}

_PASSING_STATUSES: frozenset[str] = frozenset({"pass", "skip"})

_DEFAULT_WEIGHT: float = SEVERITY_WEIGHTS["medium"]


def calculate_quality_score(results: list[dict]) -> float:
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

        weight: float = SEVERITY_WEIGHTS.get(severity, _DEFAULT_WEIGHT)
        if severity not in SEVERITY_WEIGHTS:
            logger.warning(
                f"Unrecognised severity '{severity}' — using weight {_DEFAULT_WEIGHT} "
                f"(medium default). Check your validation config."
            )

        total_weight += weight

        if status in _PASSING_STATUSES:
            passed_weight += weight

    # ── guard: prevents ZeroDivisionError if somehow all weights were 0 ──────
    if total_weight == 0.0:
        return 100.0

    # ── compute, clamp to [0, 100], round to 2 decimal places ────────────────
    raw: float   = (passed_weight / total_weight) * 100.0
    score: float = round(max(0.0, min(100.0, raw)), 2)

    return score

# Dead code: Has no use yet (streamlit dashboard)
def score_breakdown(results: list[dict]) -> dict:
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