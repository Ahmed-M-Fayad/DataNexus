"""Initial schema — all 8 DataNexus tables

Revision ID: 0001
Revises:
Create Date: 2026-03-14
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── 1. data_sources ───────────────────────────────────────────────────────
    op.create_table(
        "data_sources",
        sa.Column("id",   sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("source_type", sa.Enum(
            "postgresql", "mysql", "csv", "json", "parquet",
            name="sourcetype"
        ), nullable=False),
        sa.Column("connection_string", sa.Text(), nullable=True),
        sa.Column("description",       sa.Text(), nullable=True),
        sa.Column("is_active",  sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )

    # ── 2. datasets ───────────────────────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id",          sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id",   sa.Integer(),
                  sa.ForeignKey("data_sources.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("schema_name", sa.String(255), nullable=True),
        sa.Column("table_name",  sa.String(255), nullable=False),
        sa.Column("description", sa.Text(),    nullable=True),
        sa.Column("is_active",   sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at",  sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
        sa.UniqueConstraint("source_id", "schema_name", "table_name",
                            name="uq_dataset_source_schema_table"),
    )
    op.create_index("ix_datasets_source_id", "datasets", ["source_id"])

    # ── 3. data_profiles ──────────────────────────────────────────────────────
    op.create_table(
        "data_profiles",
        sa.Column("id",           sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dataset_id",   sa.Integer(),
                  sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("row_count",    sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.Column("profile_json", sa.Text(),    nullable=False),
        sa.Column("profiled_at",  sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_data_profiles_dataset_id", "data_profiles", ["dataset_id"])
    op.create_index("ix_data_profiles_profiled_at", "data_profiles",
                    ["dataset_id", sa.text("profiled_at DESC")])

    # ── 4. validation_configs ─────────────────────────────────────────────────
    op.create_table(
        "validation_configs",
        sa.Column("id",                sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("dataset_id",        sa.Integer(),
                  sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("name",              sa.String(255), nullable=False),
        sa.Column("config_yaml",       sa.Text(),      nullable=False),
        sa.Column("schedule_cron",     sa.String(100), nullable=True),
        sa.Column("quality_threshold", sa.Float(),     nullable=False, server_default="0.95"),
        sa.Column("alert_on_failure",  sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("alert_channels",    sa.String(255), nullable=True),
        sa.Column("is_active",         sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("created_at",        sa.DateTime(),  nullable=False,
                  server_default=sa.text("NOW()")),
        sa.Column("updated_at",        sa.DateTime(),  nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_validation_configs_dataset_id",
                    "validation_configs", ["dataset_id"])

    # ── 5. test_definitions ───────────────────────────────────────────────────
    op.create_table(
        "test_definitions",
        sa.Column("id",                  sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name",                sa.String(255), nullable=False, unique=True),
        sa.Column("category",            sa.String(100), nullable=False),
        sa.Column("description",         sa.Text(),    nullable=True),
        sa.Column("implementation_code", sa.Text(),    nullable=True),
        sa.Column("is_builtin",          sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at",          sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )

    # ── 6. validation_runs ────────────────────────────────────────────────────
    op.create_table(
        "validation_runs",
        sa.Column("id",            sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("config_id",     sa.Integer(),
                  sa.ForeignKey("validation_configs.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("status", sa.Enum(
            "pending", "running", "pass", "fail", "error",
            name="runstatus"
        ), nullable=False, server_default="pending"),
        sa.Column("quality_score", sa.Float(),    nullable=True),
        sa.Column("triggered_by",  sa.String(100), nullable=True),
        sa.Column("started_at",    sa.DateTime(), nullable=True),
        sa.Column("finished_at",   sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(),     nullable=True),
        sa.Column("created_at",    sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_validation_runs_config_id",
                    "validation_runs", ["config_id"])
    op.create_index("ix_validation_runs_status_created",
                    "validation_runs", ["status", sa.text("created_at DESC")])

    # ── 7. validation_results ─────────────────────────────────────────────────
    op.create_table(
        "validation_results",
        sa.Column("id",             sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id",         sa.Integer(),
                  sa.ForeignKey("validation_runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("check_name",    sa.String(255), nullable=False),
        sa.Column("column_name",   sa.String(255), nullable=True),
        sa.Column("check_type",    sa.String(100), nullable=False),
        sa.Column("status", sa.Enum(
            "pass", "fail", "error", "skip",
            name="checkstatus"
        ), nullable=False),
        sa.Column("severity", sa.Enum(
            "critical", "high", "medium", "low",
            name="severity"
        ), nullable=False),
        sa.Column("expected_value", sa.Text(),    nullable=True),
        sa.Column("actual_value",   sa.Text(),    nullable=True),
        sa.Column("failing_rows",   sa.Integer(), nullable=True),
        sa.Column("total_rows",     sa.Integer(), nullable=True),
        sa.Column("error_message",  sa.Text(),    nullable=True),
        sa.Column("executed_at",    sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_validation_results_run_id",
                    "validation_results", ["run_id"])
    op.create_index("ix_validation_results_run_status",
                    "validation_results", ["run_id", "status"])

    # ── 8. alerts ─────────────────────────────────────────────────────────────
    op.create_table(
        "alerts",
        sa.Column("id",              sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id",          sa.Integer(),
                  sa.ForeignKey("validation_runs.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("channel", sa.Enum(
            "email", "slack",
            name="alertchannel"
        ), nullable=False),
        sa.Column("alert_type",      sa.String(100), nullable=False),
        sa.Column("severity", sa.Enum(
            "critical", "high", "medium", "low",
            name="alertseverity"
        ), nullable=False),
        sa.Column("message",         sa.Text(),    nullable=False),
        sa.Column("status", sa.Enum(
            "pending", "sent", "failed", "partially_sent",
            name="alertstatus"
        ), nullable=False, server_default="pending"),
        sa.Column("acknowledged",    sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("acknowledged_by", sa.String(255), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(), nullable=True),
        sa.Column("sent_at",         sa.DateTime(), nullable=True),
        sa.Column("created_at",      sa.DateTime(), nullable=False,
                  server_default=sa.text("NOW()")),
    )
    op.create_index("ix_alerts_run_id", "alerts", ["run_id"])
    op.create_index("ix_alerts_severity_ack",
                    "alerts", ["severity", "acknowledged"])

    # ── Seed built-in test definitions ───────────────────────────────────────
    op.execute("""
        INSERT INTO test_definitions (name, category, description, is_builtin) VALUES
        ('not_null',   'completeness', 'Column must not contain NULL values',             true),
        ('unique',     'uniqueness',   'Column values must be unique across all rows',    true),
        ('range',      'validity',     'Numeric value must fall within min/max bounds',   true),
        ('regex',      'validity',     'String value must match a regular expression',    true),
        ('not_empty',  'completeness', 'String column must not contain empty strings',    true),
        ('referential_integrity', 'consistency',
                       'Foreign key values must exist in the referenced table',          true)
    """)


def downgrade() -> None:
    # Drop in reverse order to respect foreign key constraints
    op.drop_table("alerts")
    op.drop_table("validation_results")
    op.drop_table("validation_runs")
    op.drop_table("test_definitions")
    op.drop_table("validation_configs")
    op.drop_table("data_profiles")
    op.drop_table("datasets")
    op.drop_table("data_sources")

    # Drop custom enum types
    op.execute("DROP TYPE IF EXISTS alertstatus")
    op.execute("DROP TYPE IF EXISTS alertseverity")
    op.execute("DROP TYPE IF EXISTS alertchannel")
    op.execute("DROP TYPE IF EXISTS checkstatus")
    op.execute("DROP TYPE IF EXISTS severity")
    op.execute("DROP TYPE IF EXISTS runstatus")
    op.execute("DROP TYPE IF EXISTS sourcetype")
