"""
tests/integration/test_full_run.py
====================================
End-to-end integration test for the DataNexus core workflow.

What this test exercises (in order)
-------------------------------------
1.  Reads a ValidationConfig from the database.
2.  Loads test_data.test_customers as a pandas DataFrame (via _load_dataframe).
3.  Runs the Data Profiler → writes a DataProfile row.
4.  Runs 3 checks via CheckRunner:
        - email_not_null  (not_null)   → expected FAIL  (2/10 rows null)
        - age_in_range    (range)      → expected PASS  (9/10 rows valid)
        - id_unique       (unique)     → expected PASS  (all 10 ids distinct)
5.  Computes quality score via ScoreCalculator.
6.  Writes ValidationRun status (FAIL) + quality_score to DB.
7.  Writes 3 ValidationResult rows (one per check) to DB.

Modules under test
-------------------
    src.validator.validation_engine   (orchestrator)
    src.validator.check_runner        (individual checks)
    src.validator.score_calculator    (scoring)
    src.profiler.profiler_engine      (data profiling)
    src.database.models               (ORM reads for assertions)

Modules NOT under test (gracefully skipped by the engine)
----------------------------------------------------------
    src.alert_manager  — not yet built, engine logs a warning and continues
    src.ge_adapter     — use_ge=False (default), so GE is never called

Prerequisite
------------
    Run Alembic migrations first:
        alembic upgrade head
"""

import json
import logging

import pytest

from src.database import get_db_session
from src.database.models import (
    CheckStatus,
    DataProfile,
    RunStatus,
    ValidationResult,
    ValidationRun,
)
from src.validator.validation_engine import ValidationEngine
from src.validator.score_calculator import calculate_quality_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: fetch rows from the DB inside the test
# ─────────────────────────────────────────────────────────────────────────────

def _get_run(run_id: int) -> ValidationRun:
    with get_db_session() as session:
        run = session.query(ValidationRun).filter_by(id=run_id).first()
        # Detach so it's usable after the session closes
        session.expunge(run)
    return run


def _get_results(run_id: int) -> list:
    with get_db_session() as session:
        rows = session.query(ValidationResult).filter_by(run_id=run_id).all()
        for r in rows:
            session.expunge(r)
    return rows


def _get_profile(dataset_id: int) -> DataProfile:
    """Returns the most recent DataProfile for a dataset."""
    with get_db_session() as session:
        profile = (
            session.query(DataProfile)
            .filter_by(dataset_id=dataset_id)
            .order_by(DataProfile.profiled_at.desc())
            .first()
        )
        if profile:
            session.expunge(profile)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Main integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestFullValidationRun:
    """
    One test class, several assertion methods, all sharing a single
    `datanexus_setup` fixture call — so the engine runs only once and
    every assertion inspects the same run.
    """

    @pytest.fixture(autouse=True)
    def _run_engine(self, datanexus_setup):
        """
        Runs the ValidationEngine once for this entire test class and stores
        the results as instance attributes so every test method can read them.
        """
        self.config_id      = datanexus_setup["config_id"]
        self.dataset_id     = datanexus_setup["dataset_id"]
        self.expected_checks = datanexus_setup["expected_checks"]

        engine      = ValidationEngine()
        self.run_id = engine.run(config_id=self.config_id, triggered_by="pytest")

        # Load all result data once — shared by every assertion below
        self.run     = _get_run(self.run_id)
        self.results = _get_results(self.run_id)
        self.profile = _get_profile(self.dataset_id)

    # ── 1. ValidationRun row ─────────────────────────────────────────────────

    def test_run_was_created(self):
        """A ValidationRun row must exist with the correct config_id."""
        assert self.run is not None, \
            f"No ValidationRun found for run_id={self.run_id}"
        assert self.run.config_id == self.config_id

    def test_run_status_is_fail(self):
        """
        With email_not_null failing (high severity), the quality score should
        fall below the 80% threshold → run status = FAIL.
        """
        assert self.run.status == RunStatus.fail, (
            f"Expected status=fail but got status={self.run.status}. "
            f"quality_score={self.run.quality_score}"
        )

    def test_run_has_quality_score(self):
        """Quality score must be a float in [0.0, 100.0]."""
        assert self.run.quality_score is not None, \
            "quality_score was not written to the ValidationRun row."
        assert 0.0 <= self.run.quality_score <= 100.0, \
            f"quality_score={self.run.quality_score} is out of the valid 0–100 range."

    def test_run_quality_score_matches_expected(self):
        """
        Manually reproduce the ScoreCalculator formula and verify the stored
        score matches.

        Expected results from the test data:
            email_not_null  → FAIL  (severity=high,     weight=0.75)
            age_in_range    → PASS  (severity=medium,   weight=0.50)
            id_unique       → PASS  (severity=critical, weight=1.00)

        Score = (0.50 + 1.00) / (0.75 + 0.50 + 1.00) × 100
              = 1.50 / 2.25 × 100
              ≈ 66.67
        """
        expected_inputs = [
            {"status": "fail", "severity": "high"},
            {"status": "pass", "severity": "medium"},
            {"status": "pass", "severity": "critical"},
        ]
        expected_score = calculate_quality_score(expected_inputs)

        assert abs(self.run.quality_score - expected_score) < 0.5, (
            f"Stored quality_score={self.run.quality_score:.2f} differs from "
            f"expected {expected_score:.2f} by more than 0.5."
        )

    def test_run_timestamps_are_set(self):
        """started_at and finished_at must both be populated after a completed run."""
        assert self.run.started_at  is not None, "started_at was not set."
        assert self.run.finished_at is not None, "finished_at was not set."
        assert self.run.finished_at >= self.run.started_at, \
            "finished_at is earlier than started_at — something is wrong with timing."

    def test_run_triggered_by(self):
        """triggered_by must record 'pytest' as we passed it."""
        assert self.run.triggered_by == "pytest"

    # ── 2. ValidationResult rows ─────────────────────────────────────────────

    def test_correct_number_of_results(self):
        """One ValidationResult row must exist per check in the config."""
        assert len(self.results) == len(self.expected_checks), (
            f"Expected {len(self.expected_checks)} result rows "
            f"but found {len(self.results)}."
        )

    def test_all_check_names_present(self):
        """Every check name from the YAML must appear in the results."""
        stored_names = {r.check_name for r in self.results}
        for expected_name in self.expected_checks:
            assert expected_name in stored_names, (
                f"Check '{expected_name}' was not found in validation_results. "
                f"Stored names: {stored_names}"
            )

    def test_email_not_null_failed(self):
        """
        email_not_null should FAIL because 2/10 rows have NULL email,
        giving an 80% pass rate which is below the 0.95 threshold.
        """
        result = next(
            (r for r in self.results if r.check_name == "email_not_null"), None
        )
        assert result is not None, "No result row for 'email_not_null'."
        assert result.status == CheckStatus.fail, (
            f"email_not_null should be FAIL (80% pass < 95% threshold) "
            f"but got status={result.status}."
        )

    def test_email_not_null_failing_rows(self):
        """email_not_null should report exactly 2 failing rows (the two NULLs)."""
        result = next(
            (r for r in self.results if r.check_name == "email_not_null"), None
        )
        assert result is not None
        assert result.failing_rows == 2, (
            f"email_not_null should report 2 failing rows but got {result.failing_rows}."
        )
        assert result.total_rows == 10, (
            f"total_rows should be 10 but got {result.total_rows}."
        )

    def test_age_in_range_passed(self):
        """
        age_in_range should PASS: 9/10 rows have age in [18, 120].
        90% pass rate >= 85% threshold.
        """
        result = next(
            (r for r in self.results if r.check_name == "age_in_range"), None
        )
        assert result is not None, "No result row for 'age_in_range'."
        assert result.status == CheckStatus.pass_, (
            f"age_in_range should be PASS (90% >= 85%) but got {result.status}."
        )

    def test_age_in_range_failing_rows(self):
        """age_in_range should report exactly 1 failing row (age=200)."""
        result = next(
            (r for r in self.results if r.check_name == "age_in_range"), None
        )
        assert result is not None
        assert result.failing_rows == 1, (
            f"age_in_range should report 1 failing row but got {result.failing_rows}."
        )

    def test_id_unique_passed(self):
        """
        id_unique should PASS: all 10 ids are distinct.
        100% pass rate >= 100% threshold.
        """
        result = next(
            (r for r in self.results if r.check_name == "id_unique"), None
        )
        assert result is not None, "No result row for 'id_unique'."
        assert result.status == CheckStatus.pass_, (
            f"id_unique should be PASS (all ids unique) but got {result.status}."
        )
        assert result.failing_rows == 0, (
            f"id_unique should have 0 failing rows but got {result.failing_rows}."
        )

    def test_results_have_correct_severities(self):
        """Severities stored in the DB must match what the YAML config declares."""
        from src.database.models import Severity

        expected_severities = {
            "email_not_null": Severity.high,
            "age_in_range":   Severity.medium,
            "id_unique":      Severity.critical,
        }
        for result in self.results:
            if result.check_name in expected_severities:
                expected = expected_severities[result.check_name]
                assert result.severity == expected, (
                    f"Check '{result.check_name}': expected severity={expected} "
                    f"but got {result.severity}."
                )

    def test_no_error_status_results(self):
        """
        No check should end up in ERROR status — errors indicate exceptions
        in CheckRunner, not data failures. All checks should be pass or fail.
        """
        error_results = [r for r in self.results if r.status == CheckStatus.error]
        assert len(error_results) == 0, (
            f"{len(error_results)} check(s) ended with ERROR status: "
            f"{[r.check_name for r in error_results]}. "
            f"Check the error_message column for details."
        )

    # ── 3. DataProfile row ────────────────────────────────────────────────────

    def test_profile_was_created(self):
        """A DataProfile row must exist for the dataset after the run."""
        assert self.profile is not None, (
            f"No DataProfile found for dataset_id={self.dataset_id}. "
            f"The Profiler did not run or did not save its result."
        )

    def test_profile_row_count(self):
        """Profile must report 10 rows (matches the 10 rows we inserted)."""
        assert self.profile.row_count == 10, (
            f"profile.row_count should be 10 but got {self.profile.row_count}."
        )

    def test_profile_column_count(self):
        """Profile must report 4 columns (id, name, email, age)."""
        assert self.profile.column_count == 4, (
            f"profile.column_count should be 4 but got {self.profile.column_count}."
        )

    def test_profile_json_is_valid(self):
        """profile_json must be parseable JSON with the required top-level keys."""
        try:
            profile_data = json.loads(self.profile.profile_json)
        except json.JSONDecodeError as exc:
            pytest.fail(f"profile_json is not valid JSON: {exc}")

        for required_key in ("row_count", "column_count", "columns"):
            assert required_key in profile_data, (
                f"profile_json is missing required key '{required_key}'. "
                f"Keys present: {list(profile_data.keys())}"
            )

    def test_profile_json_contains_all_columns(self):
        """profile_json must contain stats for every column in the test table."""
        profile_data = json.loads(self.profile.profile_json)
        columns_in_profile = set(profile_data["columns"].keys())
        expected_columns   = {"id", "name", "email", "age"}

        assert expected_columns.issubset(columns_in_profile), (
            f"Some columns are missing from profile_json. "
            f"Expected: {expected_columns}, got: {columns_in_profile}"
        )

    def test_profile_email_null_pct(self):
        """
        Profile should report 20% null for the email column
        (2 nulls / 10 total rows = 20.0%).
        """
        profile_data = json.loads(self.profile.profile_json)
        email_stats  = profile_data["columns"].get("email", {})
        null_pct     = email_stats.get("null_pct")

        assert null_pct is not None, \
            "email column stats are missing 'null_pct'."
        assert abs(null_pct - 20.0) < 0.1, (
            f"email null_pct should be 20.0 but got {null_pct}."
        )

    def test_profile_email_detected_as_email_pattern(self):
        """
        The 8 non-null email values match the email regex pattern.
        PatternDetector should label the column as 'email'.
        (80% match > 75% threshold → pattern detected.)
        """
        profile_data     = json.loads(self.profile.profile_json)
        detected_pattern = profile_data["columns"]["email"].get("detected_pattern")

        assert detected_pattern == "email", (
            f"email column should be detected as 'email' pattern "
            f"but got '{detected_pattern}'."
        )

    def test_profile_age_numeric_stats(self):
        """
        Age column must have numeric stats populated (min, max, mean, median).
        min should be 19, max should be 200 (we inserted age=200 deliberately).
        """
        profile_data = json.loads(self.profile.profile_json)
        age_stats    = profile_data["columns"].get("age", {})

        assert age_stats.get("min") == 19.0,  \
            f"age min should be 19 but got {age_stats.get('min')}."
        assert age_stats.get("max") == 200.0, \
            f"age max should be 200 but got {age_stats.get('max')}."
        assert age_stats.get("mean")   is not None, "age mean is missing."
        assert age_stats.get("median") is not None, "age median is missing."


# ─────────────────────────────────────────────────────────────────────────────
# Isolated unit-style checks (no fixture needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreCalculatorIsolated:
    """
    Tests ScoreCalculator in isolation — pure math, no DB.
    These run fast and catch scoring regressions without touching the DB.
    """

    def test_all_pass_returns_100(self):
        results = [
            {"status": "pass", "severity": "critical"},
            {"status": "pass", "severity": "high"},
        ]
        assert calculate_quality_score(results) == 100.0

    def test_all_fail_returns_0(self):
        results = [
            {"status": "fail", "severity": "critical"},
            {"status": "fail", "severity": "medium"},
        ]
        assert calculate_quality_score(results) == 0.0

    def test_empty_results_returns_100(self):
        """Empty results → nothing failed → perfect score."""
        assert calculate_quality_score([]) == 100.0

    def test_skip_counts_as_pass(self):
        """Skipped checks should not penalise the score."""
        results = [
            {"status": "skip", "severity": "critical"},
            {"status": "fail", "severity": "low"},
        ]
        # passed_weight = 1.00 (critical skip), total = 1.00 + 0.25 = 1.25
        # score = 1.00 / 1.25 * 100 = 80.0
        assert calculate_quality_score(results) == 80.0

    def test_error_counts_as_fail(self):
        """Error status should penalise the score just like fail."""
        results_with_fail  = [{"status": "fail",  "severity": "high"}]
        results_with_error = [{"status": "error", "severity": "high"}]
        assert calculate_quality_score(results_with_fail) == \
               calculate_quality_score(results_with_error)

    def test_mixed_scenario_formula(self):
        """
        1 critical fail + 1 medium pass
        passed_weight = 0.50
        total_weight  = 1.00 + 0.50 = 1.50
        score = 0.50 / 1.50 * 100 = 33.33
        """
        results = [
            {"status": "fail", "severity": "critical"},
            {"status": "pass", "severity": "medium"},
        ]
        assert abs(calculate_quality_score(results) - 33.33) < 0.01


class TestCheckRunnerIsolated:
    """
    Tests individual CheckRunner checks against in-memory DataFrames.
    No database, no fixtures — just pandas and logic.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        from src.validator.check_runner import CheckRunner
        self.runner = CheckRunner()

    def _make_check(self, check_type, column, **kwargs):
        """Convenience builder for a minimal check dict."""
        return {
            "name":       f"test_{check_type}",
            "check_type": check_type,
            "column":     column,
            "severity":   "medium",
            "threshold":  1.0,
            **kwargs,
        }

    def test_not_null_all_pass(self):
        import pandas as pd
        df = pd.DataFrame({"email": ["a@b.com", "c@d.com", "e@f.com"]})
        result = self.runner.run(df, self._make_check("not_null", "email"))
        assert result["success"] is True
        assert result["failing_count"] == 0

    def test_not_null_with_nulls(self):
        import pandas as pd
        df = pd.DataFrame({"email": ["a@b.com", None, "e@f.com"]})
        result = self.runner.run(
            df,
            self._make_check("not_null", "email", threshold=0.5)
        )
        # 2/3 pass = 66.7% >= 50% → PASS, but failing_count = 1
        assert result["failing_count"] == 1
        assert result["success"] is True   # passes threshold

    def test_range_check_fails_out_of_bound(self):
        import pandas as pd
        df = pd.DataFrame({"age": [25, 30, 200, 19, 45]})
        result = self.runner.run(
            df,
            self._make_check("range", "age", min=18, max=120, threshold=1.0)
        )
        assert result["failing_count"] == 1
        assert result["success"] is False  # 4/5 = 80% < 100% threshold

    def test_unique_check_detects_duplicates(self):
        import pandas as pd
        df = pd.DataFrame({"id": [1, 2, 2, 3, 4]})
        result = self.runner.run(
            df,
            self._make_check("unique", "id", threshold=1.0)
        )
        # Both duplicate '2' rows are marked failing → failing_count = 2
        assert result["failing_count"] == 2
        assert result["success"] is False

    def test_regex_check(self):
        import pandas as pd
        df = pd.DataFrame({"email": ["valid@test.com", "not-an-email", "also@valid.com"]})
        result = self.runner.run(
            df,
            self._make_check(
                "regex", "email",
                regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
                threshold=0.6
            )
        )
        assert result["failing_count"] == 1   # "not-an-email"
        assert result["success"] is True       # 2/3 ≈ 66.7% >= 60%

    def test_unknown_check_type_raises(self):
        import pandas as pd
        df = pd.DataFrame({"col": [1, 2, 3]})
        with pytest.raises(ValueError, match="unknown check_type"):
            self.runner.run(df, self._make_check("does_not_exist", "col"))

    def test_missing_column_raises(self):
        import pandas as pd
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in DataFrame"):
            self.runner.run(df, self._make_check("not_null", "col_b"))