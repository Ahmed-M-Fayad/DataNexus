"""
validation_engine.py — DataNexus Validation Engine
===================================================
The central orchestrator of DataNexus.

This module is the ONLY place that writes to validation_runs and
validation_results tables. All other modules (API, CLI, Dashboard)
only READ those tables.

Execution flow for one run:
    1.  Create ValidationRun row  (status = pending)
    2.  Load ValidationConfig from DB
    3.  Mark run as RUNNING
    4.  Load dataset → pandas DataFrame
    5.  Run Data Profiler    (graceful if Step 3 not yet built)
    6.  Parse YAML checks from config
    7.  For each check:
            a. Run via CheckRunner
            b. Write ValidationResult row immediately
    8.  Compute Quality Score via ScoreCalculator
    9.  Mark run as PASS or FAIL + write score
    10. If FAIL → fire Alert Manager  (graceful if Step 7 not yet built)
    11. Return run_id to caller

Error handling philosophy:
    - An unhandled exception in steps 2–10 marks the run as ERROR and re-raises.
    - Data Profiler and Alert Manager failures are LOGGED but never abort the run.
    - A single check failure never stops other checks from running.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yaml

from src.database import get_db_session
from src.database.models import (
    Alert, AlertChannel, AlertStatus, CheckStatus,
    DataSource, Dataset, RunStatus, Severity,
    ValidationConfig, ValidationResult, ValidationRun,
)
from .check_runner import CheckRunner
from .score_calculator import SEVERITY_WEIGHTS, calculate_quality_score

logger = logging.getLogger(__name__)


# ── Internal snapshot dataclass ───────────────────────────────────────────────
# SQLAlchemy ORM objects become "detached" once their session closes.
# Instead of keeping the session open for the whole run (bad: holds a DB
# connection for minutes), we snapshot the fields we need into a plain dataclass
# immediately after loading, then close the session. Pure Python from then on.

@dataclass
class _ConfigSnapshot:
    """Immutable snapshot of a ValidationConfig row + its dataset/source info."""
    config_id:         int
    config_yaml:       str
    quality_threshold: float   # stored as 0.0–1.0 in the DB
    alert_on_failure:  bool
    alert_channels:    Optional[str]   # e.g. "email,slack"
    dataset_id:        int
    source_type:       str    # "csv" | "postgresql" | "mysql" | "json" | "parquet"
    connection_string: Optional[str]
    schema_name:       Optional[str]
    table_name:        str


# ── ValidationEngine ──────────────────────────────────────────────────────────

class ValidationEngine:
    """
    Orchestrates a complete validation run for a given config_id.

    Usage:
        engine = ValidationEngine()
        run_id = engine.run(config_id=5, triggered_by="api")

    Thread safety:
        Each call to run() opens and closes its own DB sessions.
        One ValidationEngine instance can safely handle concurrent runs
        (but the same run_id must not be triggered twice simultaneously).
    """

    def __init__(self):
        # One CheckRunner instance, reused across all runs (it is stateless)
        self._runner = CheckRunner()

    # ─────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────

    def run(self, config_id: int, triggered_by: str = "api") -> int:
        """
        Executes a full validation run for the given config and returns its run_id.

        Args:
            config_id:    ID of the ValidationConfig row to execute
            triggered_by: who started this run — one of:
                          "api" | "cli" | "airflow" | "github_actions"

        Returns:
            run_id (int) — ID of the ValidationRun row that was created.
            The caller can use this to query results from the DB.

        Raises:
            ValueError:   config_id not found, config is inactive, or YAML is malformed
            RuntimeError: fatal error during execution (DB failure, load error, etc.)
            Any exception marks the run as ERROR in the database before re-raising.
        """
        run_id = self._create_run_record(config_id, triggered_by)

        try:
            self._execute(run_id, config_id)
        except Exception as exc:
            # Always mark the run as ERROR before propagating, so the DB
            # accurately reflects what happened.
            self._mark_error(run_id, error_message=str(exc))
            logger.error(
                f"ValidationRun id={run_id} ended with ERROR: {exc}", exc_info=True
            )
            raise

        return run_id

    # ─────────────────────────────────────────────────────────────────────
    # Step 1 — Create the run record
    # ─────────────────────────────────────────────────────────────────────

    def _create_run_record(self, config_id: int, triggered_by: str) -> int:
        """
        Writes the initial ValidationRun row with status=pending.

        We verify the config exists and is active BEFORE creating the run row.
        This prevents orphan run records for invalid config IDs.

        Returns:
            run_id (int)
        """
        with get_db_session() as session:
            config = session.query(ValidationConfig).filter_by(id=config_id).first()

            if config is None:
                raise ValueError(
                    f"ValidationConfig id={config_id} does not exist in the database."
                )
            if not config.is_active:
                raise ValueError(
                    f"ValidationConfig id={config_id} is inactive (is_active=False). "
                    f"Activate it before running."
                )

            run = ValidationRun(
                config_id    = config_id,
                status       = RunStatus.pending,
                triggered_by = triggered_by,
                created_at   = datetime.utcnow(),
            )
            session.add(run)
            session.flush()    # assigns run.id without committing the transaction
            run_id = run.id    # capture the assigned PK before session closes

        logger.info(
            f"Created ValidationRun id={run_id} | config_id={config_id} | "
            f"triggered_by='{triggered_by}'"
        )
        return run_id

    # ─────────────────────────────────────────────────────────────────────
    # Step 2–10 — Main execution flow
    # ─────────────────────────────────────────────────────────────────────

    def _execute(self, run_id: int, config_id: int) -> None:
        """
        Runs the full validation flow in sequence.
        Any unhandled exception here propagates up to run(), which marks ERROR.
        """

        # ── Step 3: mark as RUNNING ───────────────────────────────────────
        self._update_status(run_id, RunStatus.running, started_at=datetime.utcnow())

        # ── Step 4: load config + dataset ────────────────────────────────
        snapshot, df = self._load_data(config_id)
        logger.info(
            f"[Run {run_id}] Loaded dataset '{snapshot.schema_name}.{snapshot.table_name}': "
            f"{len(df)} rows × {len(df.columns)} columns."
        )

        # ── Step 5: run Data Profiler (Step 3 module — optional) ─────────
        self._run_profiler(snapshot, df)

        # ── Step 6: parse checks from YAML ───────────────────────────────
        checks = self._parse_checks(snapshot)
        logger.info(f"[Run {run_id}] Parsed {len(checks)} checks from config YAML.")

        if not checks:
            logger.warning(
                f"[Run {run_id}] Config id={config_id} has no checks — "
                f"run will finish with quality_score=100 (PASS)."
            )

        # ── Step 7: run each check + write ValidationResult immediately ──
        scored_results: List[dict] = []
        for check in checks:
            scored_result = self._run_and_persist_check(run_id, df, check)
            scored_results.append(scored_result)

        # ── Step 8: compute quality score ────────────────────────────────
        quality_score = calculate_quality_score(scored_results)

        # ── Step 9: determine PASS / FAIL against the configured threshold ─
        # quality_score is 0–100; quality_threshold is stored as 0.0–1.0
        # Convert quality_score to the same 0.0–1.0 scale for comparison.
        pass_threshold = float(snapshot.quality_threshold)   # e.g. 0.95
        final_status   = (
            RunStatus.pass_
            if (quality_score / 100.0) >= pass_threshold
            else RunStatus.fail
        )

        self._update_status(
            run_id,
            final_status,
            quality_score = quality_score,
            finished_at   = datetime.utcnow(),
        )

        logger.info(
            f"[Run {run_id}] Finished | status={final_status.value} | "
            f"score={quality_score:.2f} | threshold={pass_threshold:.2f}"
        )

        # ── Step 10: fire alerts if FAIL ─────────────────────────────────
        if final_status == RunStatus.fail:
            self._dispatch_alerts(run_id, snapshot)

    # ─────────────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_data(self, config_id: int):
        """
        Loads the ValidationConfig and its associated Dataset + DataSource
        from the DB, snapshots the values we need, then loads the actual
        dataset as a pandas DataFrame.

        Why snapshot instead of keeping the session open?
            We don't want a DB connection held open during potentially long
            validation runs. We read everything we need in one quick session,
            close it, then work in memory.

        Returns:
            (_ConfigSnapshot, pd.DataFrame)
        """
        with get_db_session() as session:
            config  = session.query(ValidationConfig).filter_by(id=config_id).first()
            dataset = session.query(Dataset).filter_by(id=config.dataset_id).first()
            source  = session.query(DataSource).filter_by(id=dataset.source_id).first()

            # Snapshot all values we'll need BEFORE the session closes
            snapshot = _ConfigSnapshot(
                config_id         = config_id,
                config_yaml       = config.config_yaml,
                quality_threshold = config.quality_threshold,
                alert_on_failure  = config.alert_on_failure,
                alert_channels    = config.alert_channels,
                dataset_id        = dataset.id,
                source_type       = source.source_type.value,   # str, e.g. "csv"
                connection_string = source.connection_string,
                schema_name       = dataset.schema_name,
                table_name        = dataset.table_name,
            )
        # Session is now closed — snapshot is a plain Python dataclass, safe to use

        df = _load_dataframe(
            source_type       = snapshot.source_type,
            connection_string = snapshot.connection_string,
            schema_name       = snapshot.schema_name,
            table_name        = snapshot.table_name,
        )
        return snapshot, df

    # ─────────────────────────────────────────────────────────────────────
    # Data Profiler integration  (Step 3 — optional until built)
    # ─────────────────────────────────────────────────────────────────────

    def _run_profiler(self, snapshot: _ConfigSnapshot, df: pd.DataFrame) -> None:
        """
        Attempts to run the Data Profiler (Step 3 of the build roadmap).

        Behaviour:
          - If src/profiler is not yet built → logs a WARNING and continues.
          - If the profiler raises any exception → logs a WARNING and continues.
          - A profiler failure never aborts a validation run.
        """
        try:
            from src.profiler import ProfilerEngine    # noqa: PLC0415
            profiler = ProfilerEngine()
            profiler.profile(dataset_id=snapshot.dataset_id, df=df)
            logger.info("Data Profiler completed successfully.")

        except ImportError:
            logger.warning(
                "src/profiler not yet available — skipping profiling. "
                "Build Step 3 (Data Profiler) to enable this."
            )
        except Exception as exc:
            # Profiler crashing must NEVER abort a validation run.
            logger.warning(f"Data Profiler raised an exception — skipping: {exc}")

    # ─────────────────────────────────────────────────────────────────────
    # Config YAML parsing
    # ─────────────────────────────────────────────────────────────────────

    def _parse_checks(self, snapshot: _ConfigSnapshot) -> List[dict]:
        """
        Parses the raw YAML text from the config into a list of check dicts,
        each guaranteed to have the keys CheckRunner expects.

        This is an inline minimal parser. Step 4 (Config Parser module) will
        replace or extend this with full schema validation, richer error
        messages, and support for global dataset/schedule keys.

        Each output check dict has:
            name, check_type, column, threshold, severity
            (plus optional: min, max, regex, values, use_ge)
        """
        # ── parse YAML text ───────────────────────────────────────────────
        try:
            raw = yaml.safe_load(snapshot.config_yaml)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Config id={snapshot.config_id} contains invalid YAML: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config id={snapshot.config_id} YAML must be a mapping "
                f"(dict) at the top level, got {type(raw).__name__}."
            )

        checks_raw = raw.get("checks", [])
        if not isinstance(checks_raw, list):
            raise ValueError(
                f"Config id={snapshot.config_id}: 'checks' key must be a list, "
                f"got {type(checks_raw).__name__}."
            )

        # ── normalise each check ──────────────────────────────────────────
        checks = []
        for idx, item in enumerate(checks_raw):
            try:
                check = self._normalise_check(item, index=idx)
                checks.append(check)
            except ValueError as exc:
                # Include config_id and index in the error for easy debugging
                raise ValueError(
                    f"Config id={snapshot.config_id}, check[{idx}]: {exc}"
                ) from exc

        return checks

    def _normalise_check(self, item: dict, index: int) -> dict:
        """
        Validates and normalises one check dict from the parsed YAML.

        Rules:
          - name, check_type, column, severity are REQUIRED.
          - threshold defaults to 1.0 (every row must pass) if omitted.
          - check_type and severity are lower-cased and stripped.
          - Unrecognised extra keys are passed through unchanged so
            check_type-specific keys (min, max, regex, values) survive.

        Raises:
            ValueError with a clear human-readable message.
        """
        if not isinstance(item, dict):
            raise ValueError(
                f"Check at index {index} must be a dict, got {type(item).__name__}."
            )

        required = ["name", "check_type", "column", "severity"]
        missing  = [k for k in required if k not in item]
        if missing:
            raise ValueError(
                f"Check at index {index} is missing required keys: {missing}. "
                f"Keys present: {list(item.keys())}"
            )

        # Build the normalised check dict
        check: dict = {
            "name":       str(item["name"]),
            "check_type": str(item["check_type"]).lower().strip(),
            "column":     str(item["column"]),
            "severity":   str(item["severity"]).lower().strip(),
            "threshold":  float(item.get("threshold", 1.0)),
        }

        # Pass through optional check-type-specific keys
        for optional_key in ("min", "max", "regex", "values", "use_ge"):
            if optional_key in item:
                check[optional_key] = item[optional_key]

        # Validate severity is a known value
        valid_severities = set(SEVERITY_WEIGHTS.keys())
        if check["severity"] not in valid_severities:
            raise ValueError(
                f"Check '{check['name']}': severity '{check['severity']}' is not valid. "
                f"Must be one of: {sorted(valid_severities)}"
            )

        # Validate threshold is in range
        if not (0.0 <= check["threshold"] <= 1.0):
            raise ValueError(
                f"Check '{check['name']}': threshold {check['threshold']} is out of range. "
                f"Must be between 0.0 and 1.0 inclusive."
            )

        return check

    # ─────────────────────────────────────────────────────────────────────
    # Single check execution + DB persistence
    # ─────────────────────────────────────────────────────────────────────

    def _run_and_persist_check(
        self, run_id: int, df: pd.DataFrame, check: dict
    ) -> dict:
        """
        Runs one check, writes a ValidationResult row to the DB immediately,
        and returns a scored-result dict for the ScoreCalculator.

        "Write immediately" is intentional: if the engine crashes halfway
        through, we still have partial results in the DB rather than nothing.

        On check-level exception:
          - status is set to CheckStatus.error
          - error_message is stored in the ValidationResult row
          - run continues to the next check  (per the architecture spec:
            "all checks always run")
          - the check counts as FAILED for scoring purposes (success=False)

        Returns:
            {"success": bool, "severity": str}
            — the minimum information ScoreCalculator needs.
        """
        check_name = check.get("name", "unnamed")
        error_msg: Optional[str] = None

        try:
            raw    = self._runner.run(df, check)
            status = CheckStatus.pass_ if raw["success"] else CheckStatus.fail
            logger.info(
                f"[Check '{check_name}'] "
                f"{'PASSED' if raw['success'] else 'FAILED'} | "
                f"failing_rows={raw['failing_count']}/{raw['total_count']}"
            )

        except Exception as exc:
            # Check-level error: log it, mark as error, do NOT stop the run
            error_msg = str(exc)[:2000]   # cap length for DB column
            logger.error(
                f"[Check '{check_name}'] Raised exception — marking as ERROR: {exc}",
                exc_info=True,
            )
            raw = {
                "success":            False,
                "failing_count":      0,
                "total_count":        len(df),
                "unexpected_samples": [],
            }
            status = CheckStatus.error

        # Persist the result row immediately, before moving to the next check
        self._write_result_row(run_id, check, raw, status, error_message=error_msg)

        # Return the lightweight scored-result for ScoreCalculator.
        # ── BUG FIX: was {"success": bool} — score_calculator reads "status" (str),
        # not "success" (bool). Without this fix r.get("status", "error") defaults
        # to "error" for every check, making passed_weight always 0 and the
        # quality score always 0.0 regardless of actual results.
        # status.value converts CheckStatus.pass_ → "pass", .fail → "fail", etc.
        return {
            "status":   status.value,
            "severity": check.get("severity", "low"),
        }

    def _write_result_row(
        self,
        run_id:        int,
        check:         dict,
        raw:           dict,
        status:        CheckStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Writes ONE ValidationResult row to the database.

        actual_value: human-readable summary of what the data actually contained.
            e.g. "42/1000 rows failed" or "all rows passed"

        expected_value: the threshold the check was configured with.
            e.g. "threshold=0.95 (95% of rows must pass)"

        severity string → Severity enum:
            We use getattr with a fallback so an unexpected string never
            causes a crash; it degrades to Severity.low instead.
        """
        severity_str  = check.get("severity", "low").lower().strip()
        severity_enum = getattr(Severity, severity_str, Severity.low)

        failing_count = raw.get("failing_count", 0)
        total_count   = raw.get("total_count", 0)

        actual_value = (
            f"{failing_count}/{total_count} rows failed"
            if failing_count > 0
            else "all rows passed"
        )
        expected_value = (
            f"threshold={check.get('threshold', 1.0):.2f} "
            f"({check.get('threshold', 1.0)*100:.0f}% of rows must pass)"
        )

        with get_db_session() as session:
            result_row = ValidationResult(
                run_id         = run_id,
                check_name     = check.get("name", "unnamed"),
                column_name    = check.get("column"),
                check_type     = check.get("check_type", "unknown"),
                status         = status,
                severity       = severity_enum,
                expected_value = expected_value,
                actual_value   = actual_value,
                failing_rows   = failing_count,
                total_rows     = total_count,
                error_message  = error_message,
                executed_at    = datetime.utcnow(),
            )
            session.add(result_row)

    # ─────────────────────────────────────────────────────────────────────
    # DB status update helpers
    # ─────────────────────────────────────────────────────────────────────

    def _update_status(
        self,
        run_id:        int,
        status:        RunStatus,
        quality_score: Optional[float]    = None,
        started_at:    Optional[datetime] = None,
        finished_at:   Optional[datetime] = None,
        error_message: Optional[str]      = None,
    ) -> None:
        """
        Updates the ValidationRun row's status and optional fields.
        Called three times during a run: pending→running, then →pass/fail.
        """
        with get_db_session() as session:
            run = session.query(ValidationRun).filter_by(id=run_id).first()
            if run is None:
                # This should never happen, but we guard defensively
                raise RuntimeError(
                    f"ValidationRun id={run_id} disappeared from the DB "
                    f"during a status update. This indicates a data consistency issue."
                )
            run.status = status
            if quality_score is not None: run.quality_score = quality_score
            if started_at    is not None: run.started_at    = started_at
            if finished_at   is not None: run.finished_at   = finished_at
            if error_message is not None: run.error_message = error_message

    def _mark_error(self, run_id: int, error_message: str) -> None:
        """
        Marks a run as ERROR with the exception message.
        Wrapped in its own try/except so that a DB failure here doesn't
        mask the original exception that caused the run to fail.
        """
        try:
            self._update_status(
                run_id,
                RunStatus.error,
                error_message = error_message[:2000],
                finished_at   = datetime.utcnow(),
            )
        except Exception as db_exc:
            logger.error(
                f"Could not mark ValidationRun {run_id} as ERROR in DB: {db_exc}. "
                f"Original error was: {error_message}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Alert Manager integration  (Step 7 — optional until built)
    # ─────────────────────────────────────────────────────────────────────

    def _dispatch_alerts(self, run_id: int, snapshot: _ConfigSnapshot) -> None:
        """
        Calls the Alert Manager (Step 7) when a run fails.

        Behaviour:
          - If alert_on_failure=False in the config → skip silently.
          - If src/alert_manager not yet built → log WARNING and continue.
          - If AlertManager raises → log WARNING and continue.
          - Alert failure NEVER aborts the run or changes its DB status.
        """
        if not snapshot.alert_on_failure:
            logger.info(
                f"[Run {run_id}] Alert skipped — alert_on_failure=False in config."
            )
            return

        try:
            from src.alert_manager import AlertManager    # noqa: PLC0415
            manager = AlertManager()
            manager.dispatch(run_id)
            logger.info(f"[Run {run_id}] Alert Manager dispatched successfully.")

        except ImportError:
            logger.warning(
                "src/alert_manager not yet available — skipping alerts. "
                "Build Step 7 (Alert Manager) to enable notifications."
            )
        except Exception as exc:
            # Alert failure must NEVER crash or change a completed run's status
            logger.warning(
                f"[Run {run_id}] Alert Manager raised an exception — skipping: {exc}"
            )


# ── Module-level helper: dataset loader ──────────────────────────────────────
# Kept as a plain function (not a method) because it has no dependency on
# ValidationEngine state and is easier to test in isolation.

def _load_dataframe(
    source_type:       str,
    connection_string: Optional[str],
    schema_name:       Optional[str],
    table_name:        str,
) -> pd.DataFrame:
    """
    Loads a dataset into a pandas DataFrame based on its source type.

    Supported source types (matching SourceType enum in models.py):
        "csv"        — reads the file at connection_string (path or URL)
        "json"       — reads the file at connection_string
        "parquet"    — reads the file at connection_string
        "postgresql" — reads the table from the PostgreSQL DB
        "mysql"      — reads the table from the MySQL DB

    For database sources, connection_string must be a valid SQLAlchemy URL
    (e.g. "postgresql://user:pass@host:5432/dbname").

    Raises:
        ValueError: if source_type is not supported
        Any pandas/SQLAlchemy exception if the read fails
    """
    if source_type == "csv":
        # connection_string is the file path or URL to the CSV
        return pd.read_csv(connection_string)

    elif source_type == "json":
        return pd.read_json(connection_string)

    elif source_type == "parquet":
        return pd.read_parquet(connection_string)

    elif source_type in ("postgresql", "mysql"):
        # Use a separate engine just for this read — don't reuse the app engine.
        # try/finally guarantees dispose() runs even if read_sql_table raises,
        # so the connection pool is never left open after this function returns.
        from sqlalchemy import create_engine as _sa_engine    # noqa: PLC0415
        read_engine = _sa_engine(connection_string)
        try:
            # pd.read_sql_table loads the entire table; for large tables this
            # should be replaced with sampling in a future optimisation pass.
            return pd.read_sql_table(
                table_name  = table_name,
                con         = read_engine,
                schema      = schema_name,   # None is valid for the default schema
            )
        finally:
            read_engine.dispose()   # always close the pool — prevents connection leaks

    else:
        raise ValueError(
            f"Unsupported source_type: '{source_type}'. "
            f"Supported types: csv, json, parquet, postgresql, mysql."
        )