"""
src/cli/commands/sources_cmd.py
================================
Two top-level command groups for inspecting registered data sources
and datasets:

  datanexus sources list
      Shows all entries in the data_sources table.

  datanexus datasets list [--source-id N]
      Shows all entries in the datasets table, optionally filtered
      by source id.
"""

import sys
from pathlib import Path

import click

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.database import get_db_session
from src.database.models import DataSource, Dataset
from src.cli.formatting import print_table, mask_connection_string


# ─────────────────────────────────────────────────────────────────────────────
# datanexus sources list
# ─────────────────────────────────────────────────────────────────────────────

@click.group("sources")
def sources_group() -> None:
    """List and inspect registered data sources."""
    pass


@sources_group.command("list")
@click.option("--all", "show_all", is_flag=True, default=False,
              help="Include inactive sources.")
def sources_list(show_all: bool) -> None:
    """
    List all data sources registered in the database.

    \b
    Examples:
      datanexus sources list
    """
    try:
        with get_db_session() as session:
            query = session.query(DataSource)
            if not show_all:
                query = query.filter_by(is_active=True)
            sources = query.order_by(DataSource.id).all()

            if not sources:
                click.echo("\n  No data sources found.")
                click.echo("  Run `datanexus seed` to register the demo CSV source.\n")
                return

            rows = []
            for s in sources:
                conn = mask_connection_string(s.connection_string) or "—"
                if len(conn) > 50:
                    conn = "…" + conn[-47:]
                rows.append([
                    s.id,
                    s.source_type.value,
                    s.name[:40],
                    conn,
                    "yes" if s.is_active else "no",
                    s.created_at.strftime("%Y-%m-%d") if s.created_at else "—",
                ])

        print_table(
            headers=["ID", "Type", "Name", "Connection / Path", "Active", "Created"],
            rows=rows,
            title=f"Data Sources  ({len(rows)} found)",
        )
        click.echo()

    except Exception as exc:
        click.echo(f"\n  Error: {exc}\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# datanexus datasets list
# ─────────────────────────────────────────────────────────────────────────────

@click.group("datasets")
def datasets_group() -> None:
    """List and inspect registered datasets."""
    pass


@datasets_group.command("list")
@click.option("--source-id", default=None, type=int,
              help="Filter datasets by source id.")
@click.option("--all", "show_all", is_flag=True, default=False,
              help="Include inactive datasets.")
def datasets_list(source_id: int, show_all: bool) -> None:
    """
    List all datasets registered in the database.

    \b
    Examples:
      datanexus datasets list
      datanexus datasets list --source-id 1
    """
    try:
        with get_db_session() as session:
            query = session.query(Dataset)
            if source_id is not None:
                query = query.filter_by(source_id=source_id)
            if not show_all:
                query = query.filter_by(is_active=True)
            datasets = query.order_by(Dataset.id).all()

            if not datasets:
                click.echo("\n  No datasets found.")
                click.echo("  Run `datanexus seed` to register the demo dataset.\n")
                return

            rows = []
            for d in datasets:
                source = session.query(DataSource).filter_by(id=d.source_id).first()
                source_name = source.name[:30] if source else "?"
                full_name   = f"{d.schema_name}.{d.table_name}" if d.schema_name else d.table_name
                rows.append([
                    d.id,
                    d.source_id,
                    source_name,
                    full_name,
                    "yes" if d.is_active else "no",
                    d.created_at.strftime("%Y-%m-%d") if d.created_at else "—",
                ])

        print_table(
            headers=["ID", "Source ID", "Source Name", "Dataset (schema.table)", "Active", "Created"],
            rows=rows,
            title=f"Datasets  ({len(rows)} found)",
        )
        click.echo()

    except Exception as exc:
        click.echo(f"\n  Error: {exc}\n")
        sys.exit(1)
