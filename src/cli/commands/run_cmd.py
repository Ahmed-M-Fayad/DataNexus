"""
src/cli/commands/run_cmd.py
============================
Two top-level commands:

  datanexus run <config-id> [--triggered-by TEXT]
      Triggers a full validation run against the given config.
      Prints a live progress line for each check, then a final summary
      with quality score and per-check results table.

  datanexus profile <dataset-id>
      Loads the dataset and runs the ProfilerEngine.
      Prints a column-level statistics table.
"""

import sys
from pathlib import Path
from datetime import datetime

import click

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_db_session
from src.database.models import (
    Dataset, DataSource, ValidationConfig,
    ValidationRun, ValidationResult,
)
from src.cli.formatting import (
    print_table, print_run_card, print_section,
    status_label, severity_label, score_bar, mask_connection_string,
)


# ─────────────────────────────────────────────────────────────────────────────
# datanexus run <config-id>
# ─────────────────────────────────────────────────────────────────────────────

@click.command("run")
@click.argument("config_id", type=int)
@click.option("--triggered-by", default="cli", show_default=True,
              help="Label stored in the DB to show who triggered this run.")
def run_cmd(config_id: int, triggered_by: str) -> None:
    """
    Run a validation config against its dataset.

    CONFIG_ID is the integer id from the validation_configs table.
    Use `datanexus config list` to see all available config ids.

    \b
    Examples:
      datanexus run 1
      datanexus run 2 --triggered-by instructor_demo
    """
    # ── Resolve the config name for a helpful banner ──────────────────────────
    config_name = _get_config_name(config_id)
    if config_name is None:
        click.echo(f"\n  Error: No ValidationConfig with id={config_id} found in the database.")
        click.echo("  Run `datanexus config list` to see available configs.")
        click.echo("  Run `datanexus seed` if the database is empty.\n")
        sys.exit(1)

    click.echo()
    click.echo(f"  DataNexus — Validation Run")
    click.echo(f"  Config   : [{config_id}] {config_name}")
    click.echo(f"  Trigger  : {triggered_by}")
    click.echo(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo()
    click.echo("  Running validation engine...")

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        from src.validator import ValidationEngine
        engine = ValidationEngine()
        run_id = engine.run(config_id=config_id, triggered_by=triggered_by)
    except ValueError as exc:
        click.echo(f"\n  Configuration error: {exc}\n")
        sys.exit(1)
    except Exception as exc:
        click.echo(f"\n  Validation engine error: {exc}\n")
        sys.exit(1)

    # ── Fetch results and display ─────────────────────────────────────────────
    _display_run_results(run_id)


def _get_config_name(config_id: int):
    try:
        with get_db_session() as session:
            cfg = session.query(ValidationConfig).filter_by(id=config_id).first()
            return cfg.name if cfg else None
    except Exception:
        return None


def _display_run_results(run_id: int) -> None:
    """Query the completed run + its results and print them."""
    with get_db_session() as session:
        run = session.query(ValidationRun).filter_by(id=run_id).first()
        if run is None:
            click.echo(f"  Could not find run id={run_id} in the database.")
            return

        cfg  = run.config
        results = (
            session.query(ValidationResult)
            .filter_by(run_id=run_id)
            .order_by(ValidationResult.id)
            .all()
        )

        # Build the run card dict
        run_dict = {
            "id":            run.id,
            "status":        run.status.value if run.status else "unknown",
            "quality_score": run.quality_score,
            "config_id":     run.config_id,
            "config_name":   cfg.name if cfg else None,
            "triggered_by":  run.triggered_by,
            "started_at":    run.started_at,
            "finished_at":   run.finished_at,
            "error_message": run.error_message,
        }
        print_run_card(run_dict)

        # Per-check results table
        if results:
            rows = []
            for r in results:
                rows.append([
                    r.check_name,
                    r.check_type,
                    r.column_name or "—",
                    status_label(r.status.value if r.status else "unknown"),
                    severity_label(r.severity.value if r.severity else "?"),
                    f"{r.failing_rows}/{r.total_rows}" if r.total_rows else "—",
                ])
            print_table(
                headers=["Check Name", "Type", "Column", "Status", "Severity", "Failing/Total"],
                rows=rows,
                title="Per-Check Results",
            )
        else:
            click.echo("  (no check results recorded)\n")

    click.echo()


# ─────────────────────────────────────────────────────────────────────────────
# datanexus profile <dataset-id>
# ─────────────────────────────────────────────────────────────────────────────

@click.command("profile")
@click.argument("dataset_id", type=int)
def profile_cmd(dataset_id: int) -> None:
    """
    Run data profiling on a dataset and display column statistics.

    DATASET_ID is the integer id from the datasets table.
    Use `datanexus datasets list` to see available dataset ids.

    \b
    Examples:
      datanexus profile 1
    """
    import json
    import pandas as pd

    # ── Resolve dataset + source info ─────────────────────────────────────────
    try:
        with get_db_session() as session:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if dataset is None:
                click.echo(f"\n  Error: No Dataset with id={dataset_id} found.")
                click.echo("  Run `datanexus datasets list` to see available datasets.\n")
                sys.exit(1)

            source = session.query(DataSource).filter_by(id=dataset.source_id).first()
            source_type       = source.source_type.value
            connection_string = source.connection_string
            schema_name       = dataset.schema_name
            table_name        = dataset.table_name
            dataset_name      = table_name
    except Exception as exc:
        click.echo(f"\n  Database error: {exc}\n")
        sys.exit(1)

    click.echo()
    click.echo(f"  DataNexus — Data Profiler")
    click.echo(f"  Dataset  : [{dataset_id}] {dataset_name}")
    click.echo(f"  Source   : {source_type}  →  {mask_connection_string(connection_string) or '(db table)'}")
    click.echo()
    click.echo("  Loading data...")

    # ── Load DataFrame ────────────────────────────────────────────────────────
    try:
        df = _load_df(source_type, connection_string, schema_name, table_name)
    except Exception as exc:
        click.echo(f"\n  Error loading data: {exc}\n")
        sys.exit(1)

    click.echo(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    click.echo("  Profiling...")

    # ── Run profiler ──────────────────────────────────────────────────────────
    try:
        from src.profiler import ProfilerEngine
        record = ProfilerEngine().profile(df=df, dataset_id=dataset_id)
        profile = json.loads(record.profile_json)
    except Exception as exc:
        click.echo(f"\n  Profiler error: {exc}\n")
        sys.exit(1)

    # ── Display ───────────────────────────────────────────────────────────────
    print_section("Profile Summary")
    click.echo(f"  Rows    : {profile.get('row_count', '?')}")
    click.echo(f"  Columns : {profile.get('column_count', '?')}")

    col_stats = profile.get("columns", {})
    rows = []
    for col_name, stats in col_stats.items():
        rows.append([
            col_name,
            f"{stats.get('null_pct', 0):.1f}%",
            str(stats.get("distinct_count", "?")),
            str(stats.get("min", "—")),
            str(stats.get("max", "—")),
            f"{stats.get('mean', '—'):.2f}" if isinstance(stats.get("mean"), (int, float)) else "—",
            stats.get("detected_pattern", "—") or "—",
        ])

    print_table(
        headers=["Column", "Null%", "Distinct", "Min", "Max", "Mean", "Pattern"],
        rows=rows,
        title="Column Statistics",
    )
    click.echo(f"  Profile saved to data_profiles table (dataset_id={dataset_id})\n")


# ── Internal: load a DataFrame from any supported source type ─────────────────

def _load_df(source_type: str, connection_string: str, schema_name, table_name: str):
    import pandas as pd
    if source_type == "csv":
        return pd.read_csv(connection_string)
    elif source_type == "json":
        return pd.read_json(connection_string)
    elif source_type == "parquet":
        return pd.read_parquet(connection_string)
    elif source_type in ("postgresql", "mysql"):
        from sqlalchemy import create_engine as _sa
        eng = _sa(connection_string)
        try:
            return pd.read_sql_table(table_name, con=eng, schema=schema_name)
        finally:
            eng.dispose()
    else:
        raise ValueError(f"Unsupported source_type '{source_type}'.")
