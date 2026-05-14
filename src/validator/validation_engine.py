import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from src.database import get_db_session
from src.database.models import (
    CheckStatus, DataSource, Dataset, RunStatus, Severity,
    ValidationConfig, ValidationResult, ValidationRun,
)
from src.config_parser import from_string, ConfigParseError
from .check_runner import CheckRunner
from .score_calculator import calculate_quality_score

logger = logging.getLogger(__name__)


@dataclass
class _ConfigSnapshot:
    """Immutable snapshot of a ValidationConfig row + its dataset/source info.

    Taken immediately after DB load so the session can be closed before the
    (potentially long) validation run starts — avoids holding a connection open.
    """
    config_id:         int
    config_yaml:       str
    quality_threshold: float   # 0.0–1.0 as stored in the DB
    alert_on_failure:  bool
    alert_channels:    Optional[str]
    dataset_id:        int
    source_type:       str
    connection_string: Optional[str]
    schema_name:       Optional[str]
    table_name:        str


class ValidationEngine:

    def __init__(self):
        self._runner = CheckRunner()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, config_id: int, triggered_by: str = "api") -> int:
        run_id = self._create_run_record(config_id, triggered_by)
        try:
            self._execute(run_id, config_id)
        except Exception as exc:
            self._mark_error(run_id, error_message=str(exc))
            logger.error(f"ValidationRun id={run_id} ended with ERROR: {exc}", exc_info=True)
            raise
        return run_id

    # ── Run record creation ───────────────────────────────────────────────────

    def _create_run_record(self, config_id: int, triggered_by: str) -> int:
        with get_db_session() as session:
            config = session.query(ValidationConfig).filter_by(id=config_id).first()
            if config is None:
                raise ValueError(f"ValidationConfig id={config_id} does not exist in the database.")
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
            session.flush()
            run_id = run.id

        logger.info(
            f"Created ValidationRun id={run_id} | config_id={config_id} | "
            f"triggered_by='{triggered_by}'"
        )
        return run_id

    # ── Main execution flow ───────────────────────────────────────────────────

    def _execute(self, run_id: int, config_id: int) -> None:
        self._update_status(run_id, RunStatus.running, started_at=datetime.utcnow())

        snapshot, df = self._load_data(config_id)
        logger.info(
            f"[Run {run_id}] Loaded dataset '{snapshot.schema_name}.{snapshot.table_name}': "
            f"{len(df)} rows × {len(df.columns)} columns."
        )

        self._run_profiler(snapshot, df)

        checks = self._parse_checks(snapshot)
        logger.info(f"[Run {run_id}] Parsed {len(checks)} checks from config YAML.")

        if not checks:
            logger.warning(
                f"[Run {run_id}] Config id={config_id} has no checks — "
                f"run will finish with quality_score=100 (PASS)."
            )

        scored_results: List[dict] = []
        for check in checks:
            scored_results.append(self._run_and_persist_check(run_id, df, check))

        quality_score  = calculate_quality_score(scored_results)
        pass_threshold = float(snapshot.quality_threshold)
        final_status   = (
            RunStatus.pass_
            if (quality_score / 100.0) >= pass_threshold
            else RunStatus.fail
        )

        self._update_status(
            run_id, final_status,
            quality_score=quality_score,
            finished_at=datetime.utcnow(),
        )
        logger.info(
            f"[Run {run_id}] Finished | status={final_status.value} | "
            f"score={quality_score:.2f} | threshold={pass_threshold:.2f}"
        )

        if final_status == RunStatus.fail:
            self._dispatch_alerts(run_id, snapshot)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self, config_id: int):
        with get_db_session() as session:
            config  = session.query(ValidationConfig).filter_by(id=config_id).first()
            dataset = session.query(Dataset).filter_by(id=config.dataset_id).first()
            source  = session.query(DataSource).filter_by(id=dataset.source_id).first()

            snapshot = _ConfigSnapshot(
                config_id         = config_id,
                config_yaml       = config.config_yaml,
                quality_threshold = config.quality_threshold,
                alert_on_failure  = config.alert_on_failure,
                alert_channels    = config.alert_channels,
                dataset_id        = dataset.id,
                source_type       = source.source_type.value,
                connection_string = source.connection_string,
                schema_name       = dataset.schema_name,
                table_name        = dataset.table_name,
            )

        df = _load_dataframe(
            source_type       = snapshot.source_type,
            connection_string = snapshot.connection_string,
            schema_name       = snapshot.schema_name,
            table_name        = snapshot.table_name,
        )
        return snapshot, df

    # ── Profiler integration (graceful — failure never aborts a run) ──────────

    def _run_profiler(self, snapshot: _ConfigSnapshot, df: pd.DataFrame) -> None:
        try:
            from src.profiler import ProfilerEngine  # noqa: PLC0415
            ProfilerEngine().profile(dataset_id=snapshot.dataset_id, df=df)
            logger.info("Data Profiler completed successfully.")
        except ImportError:
            logger.warning("src/profiler not yet available — skipping profiling.")
        except Exception as exc:
            logger.warning(f"Data Profiler raised an exception — skipping: {exc}")

    # ── Check parsing — routed through ConfigParser ───────────────────────────

    def _parse_checks(self, snapshot: _ConfigSnapshot) -> List[dict]:
        """
        Parse the stored YAML via ConfigParser (single validation path shared
        with the CLI), then convert each CheckConfig to the dict format that
        CheckRunner.run() expects.
        """
        try:
            parsed = from_string(snapshot.config_yaml)
        except ConfigParseError as exc:
            raise ValueError(
                f"Config id={snapshot.config_id} failed validation: {exc}"
            ) from exc

        checks = []
        for c in parsed.checks:
            check = {
                "name":       c.name,
                "check_type": c.check_type,
                "column":     c.column or "",
                "severity":   c.severity,
                "threshold":  c.threshold if c.threshold is not None else 1.0,
            }
            # Pass through optional check-type-specific keys.
            if c.min_value      is not None: check["min"]    = c.min_value
            if c.max_value      is not None: check["max"]    = c.max_value
            if c.pattern        is not None: check["regex"]  = c.pattern
            if c.accepted_values is not None: check["values"] = c.accepted_values
            checks.append(check)

        return checks

    # ── Single check execution + immediate DB write ───────────────────────────

    def _run_and_persist_check(self, run_id: int, df: pd.DataFrame, check: dict) -> dict:
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
            # Check-level errors are logged and marked ERROR; the run continues.
            error_msg = str(exc)[:2000]
            logger.error(
                f"[Check '{check_name}'] Raised exception — marking as ERROR: {exc}",
                exc_info=True,
            )
            raw = {"success": False, "failing_count": 0, "total_count": len(df),
                   "unexpected_samples": []}
            status = CheckStatus.error

        self._write_result_row(run_id, check, raw, status, error_message=error_msg)
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
        severity_str  = check.get("severity", "low").lower().strip()
        severity_enum = getattr(Severity, severity_str, Severity.low)

        failing_count = raw.get("failing_count", 0)
        total_count   = raw.get("total_count", 0)
        threshold     = check.get("threshold", 1.0)

        actual_value   = (f"{failing_count}/{total_count} rows failed"
                          if failing_count > 0 else "all rows passed")
        expected_value = f"threshold={threshold:.2f} ({threshold*100:.0f}% of rows must pass)"

        with get_db_session() as session:
            session.add(ValidationResult(
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
            ))

    # ── DB status helpers ─────────────────────────────────────────────────────

    def _update_status(
        self,
        run_id:        int,
        status:        RunStatus,
        quality_score: Optional[float]    = None,
        started_at:    Optional[datetime] = None,
        finished_at:   Optional[datetime] = None,
        error_message: Optional[str]      = None,
    ) -> None:
        with get_db_session() as session:
            run = session.query(ValidationRun).filter_by(id=run_id).first()
            if run is None:
                raise RuntimeError(
                    f"ValidationRun id={run_id} disappeared from the DB during a status update."
                )
            run.status = status
            if quality_score is not None: run.quality_score = quality_score
            if started_at    is not None: run.started_at    = started_at
            if finished_at   is not None: run.finished_at   = finished_at
            if error_message is not None: run.error_message = error_message

    def _mark_error(self, run_id: int, error_message: str) -> None:
        # Wrapped separately so a DB failure here doesn't mask the original exception.
        try:
            self._update_status(
                run_id, RunStatus.error,
                error_message=error_message[:2000],
                finished_at=datetime.utcnow(),
            )
        except Exception as db_exc:
            logger.error(
                f"Could not mark ValidationRun {run_id} as ERROR in DB: {db_exc}. "
                f"Original error was: {error_message}"
            )

    # ── Alert Manager integration (graceful — failure never changes run status) ─

    def _dispatch_alerts(self, run_id: int, snapshot: _ConfigSnapshot) -> None:
        if not snapshot.alert_on_failure:
            logger.info(f"[Run {run_id}] Alert skipped — alert_on_failure=False in config.")
            return
        try:
            from src.alert_manager import AlertManager  # noqa: PLC0415
            AlertManager().dispatch(run_id)
            logger.info(f"[Run {run_id}] Alert Manager dispatched successfully.")
        except ImportError:
            logger.warning("src/alert_manager not yet available — skipping alerts.")
        except Exception as exc:
            logger.warning(f"[Run {run_id}] Alert Manager raised an exception — skipping: {exc}")


# ── Module-level helper ───────────────────────────────────────────────────────

def _load_dataframe(
    source_type:       str,
    connection_string: Optional[str],
    schema_name:       Optional[str],
    table_name:        str,
) -> pd.DataFrame:
    if source_type == "csv":
        return pd.read_csv(connection_string)
    elif source_type == "json":
        return pd.read_json(connection_string)
    elif source_type == "parquet":
        return pd.read_parquet(connection_string)
    elif source_type in ("postgresql", "mysql"):
        from sqlalchemy import create_engine as _sa_engine  # noqa: PLC0415
        read_engine = _sa_engine(connection_string)
        try:
            # Loads the full table; replace with sampling for very large datasets.
            return pd.read_sql_table(
                table_name=table_name,
                con=read_engine,
                schema=schema_name,
            )
        finally:
            read_engine.dispose()
    else:
        raise ValueError(
            f"Unsupported source_type: '{source_type}'. "
            f"Supported types: csv, json, parquet, postgresql, mysql."
        )
