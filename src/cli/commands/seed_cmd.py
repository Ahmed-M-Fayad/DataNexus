"""
src/cli/commands/seed_cmd.py
============================
`datanexus seed [--target csv|db|all] [--reset]`

Seeds the database with demo data so every other CLI command has
something real to run against without needing a live external database.

--target csv (default — unchanged from before)
-----------------------------------------------
  data/sample_customers.csv    — 20 rows with deliberate quality issues
  data_sources       1 row     — CSV source pointing at the file above
  datasets           1 row     — "sample_customers" table entry
  validation_configs 2 rows    — "Basic" (3 checks) and "Strict" (5 checks)

--target db — validates real PostgreSQL tables, not files
-----------------------------------------------------------
  demo_departments table  — 5-row reference table (id, name)
  demo_employees   table  — 25-row table with deliberately planted issues
                             spanning every supported check_type: unique,
                             not_empty, range, regex, in_set, foreign_key
                             (referential-integrity-style), and freshness
  data_sources       1 row     — postgresql source, same DATABASE_URL
  datasets           1 row     — "demo_employees" table entry
  validation_configs 2 rows    — "PostgreSQL Basic Checks" (3 checks) and
                                  "PostgreSQL Full Data Quality Suite" (7 checks)

--target all seeds both. Idempotent by default: safe to run twice;
use --reset to wipe and re-seed whatever --target points at.
"""

import csv
import sys
from pathlib import Path

import click

# ── Resolve project root so imports work however the CLI is invoked ──────────
_HERE        = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]   # src/cli/commands/ → src/cli/ → src/ → project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import text, inspect

from src.database import get_db_session, engine
from src.database.connection import DATABASE_URL
from src.database.models import DataSource, Dataset, SourceType, ValidationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Sample CSV rows — 20 rows with deliberate quality issues baked in
# ─────────────────────────────────────────────────────────────────────────────
#
# Issues planted for demo visibility:
#   - 3 rows have empty email      → email_not_empty check will FAIL
#   - 3 rows have age outside 18–120 → age_in_range check will FAIL
#   - 1 duplicate customer_id (18) → id_unique check will FAIL
#   - 2 rows have invalid status   → status_valid check (strict) will FAIL

_SAMPLE_CSV_ROWS = [
    ["customer_id", "name",           "email",                   "age", "status"],
    [1,  "Alice Johnson",  "alice@example.com",       25,  "active"],
    [2,  "Bob Smith",      "bob@example.com",         30,  "active"],
    [3,  "Carol White",    "carol@example.com",       22,  "inactive"],
    [4,  "Dave Brown",     "dave@example.com",        45,  "active"],
    [5,  "Eve Davis",      "eve@example.com",         28,  "pending"],
    [6,  "Frank Miller",   "frank@example.com",       35,  "active"],
    [7,  "Grace Wilson",   "grace@example.com",       19,  "inactive"],
    [8,  "Hank Moore",     "hank@example.com",        52,  "active"],
    [9,  "Iris Taylor",    "",                        33,  "active"],    # empty email
    [10, "Jack Anderson",  "",                        200, "active"],    # empty email + age=200
    [11, "Karen Thomas",   "karen@example.com",       41,  "active"],
    [12, "Leo Jackson",    "leo@example.com",         16,  "active"],    # age=16 (underage)
    [13, "Mia Harris",     "mia@example.com",         29,  "active"],
    [14, "Nick Martin",    "nick@example.com",        38,  "unknown"],   # invalid status
    [15, "Olivia Garcia",  "olivia@example.com",      24,  "active"],
    [16, "Paul Martinez",  "",                        31,  "pending"],   # empty email
    [17, "Quinn Robinson", "quinn@example.com",       27,  "inactive"],
    [18, "Rachel Clark",   "rachel@example.com",      44,  "active"],
    [19, "Sam Rodriguez",  "sam@example.com",         36,  "suspended"], # invalid status
    [18, "Tom Lewis",      "tom@example.com",         50,  "active"],    # duplicate id=18
]

_BASIC_CONFIG_YAML = """\
dataset: sample_customers
name: Basic Customer Checks
description: Checks email completeness, age range, and id uniqueness.
quality_threshold: 80.0
alert_channels: []
checks:
  - name: email_not_empty
    column: email
    check_type: not_empty
    threshold: 0.90
    severity: high

  - name: age_in_range
    column: age
    check_type: range
    min_value: 18
    max_value: 120
    threshold: 0.85
    severity: medium

  - name: id_unique
    column: customer_id
    check_type: unique
    threshold: 1.0
    severity: critical
"""

_STRICT_CONFIG_YAML = """\
dataset: sample_customers
name: Strict Customer Checks
description: All basic checks plus status validation and name completeness.
quality_threshold: 90.0
alert_channels: []
checks:
  - name: email_not_empty
    column: email
    check_type: not_empty
    threshold: 0.95
    severity: high

  - name: age_in_range
    column: age
    check_type: range
    min_value: 18
    max_value: 120
    threshold: 0.90
    severity: high

  - name: id_unique
    column: customer_id
    check_type: unique
    threshold: 1.0
    severity: critical

  - name: status_valid
    column: status
    check_type: in_set
    accepted_values: [active, inactive, pending]
    threshold: 1.0
    severity: high

  - name: name_not_empty
    column: name
    check_type: not_empty
    threshold: 1.0
    severity: medium
"""


# ─────────────────────────────────────────────────────────────────────────────
# DB demo tables — "demo_departments" (reference) + "demo_employees" (validated)
# ─────────────────────────────────────────────────────────────────────────────
#
# Unlike the CSV demo (not_empty, range, unique, in_set only), this dataset is
# built to exercise every check_type the engine supports, including the two
# not touched by the CSV demo: regex and freshness — plus foreign_key, which
# is demonstrated here in its real intended role: checking that department_id
# values reference a real row in demo_departments (via accepted_values, since
# foreign_key/referential_integrity are in_set under the hood).
#
# Issues planted for demo visibility:
#   - 1 duplicate employee_id (5)         → employee_id_unique will FAIL
#   - 2 rows have empty full_name         → full_name_not_empty will FAIL
#   - 2 rows have malformed email         → email_format (regex) will FAIL
#   - 3 rows have salary outside 30k–250k → salary_in_range will FAIL
#   - 3 rows reference a department_id
#     that doesn't exist in demo_departments → department_reference_check will FAIL
#   - 2 rows have an invalid status       → status_valid will FAIL
#   - 3 rows have hire_date outside the
#     2015-01-01 .. 2026-07-01 window     → hire_date_freshness will FAIL

_DEMO_DEPARTMENTS_COLUMNS = ["department_id", "department_name"]
_DEMO_DEPARTMENTS_ROWS = [
    (100, "Engineering"),
    (200, "Sales"),
    (300, "Marketing"),
    (400, "Human Resources"),
    (500, "Finance"),
]

_DEMO_EMPLOYEES_COLUMNS = [
    "employee_id", "full_name", "email", "salary", "department_id", "status", "hire_date",
]
_DEMO_EMPLOYEES_ROWS = [
    (1,  "Yasmin Adel",             "yasmin.adel@company.com",     62000,  100, "active",     "2021-03-14"),
    (2,  "Omar Farouk",             "omar.farouk@company.com",     58000,  200, "active",     "2020-07-01"),
    (3,  "Salma Nabil",             "salma.nabil@company.com",     71000,  300, "on_leave",   "2019-11-23"),
    (4,  "Karim Adly",              "karim.adly@company.com",      49000,  400, "active",     "2022-01-10"),
    (5,  "Hana Fathy",              "hana.fathy@company.com",      83000,  500, "active",     "2018-06-05"),
    (6,  "Tarek Mostafa",           "tarek.mostafa@company.com",   67000,  100, "active",     "2023-09-19"),
    (7,  "Nourhan Samy",            "nourhan.samy@company.com",    45000,  200, "terminated", "2020-02-28"),
    (8,  "Mostafa Kamal",           "mostafa.kamal@company.com",   39000,  300, "active",     "2024-04-02"),
    (9,  "",                        "rana.hesham@company.com",     52000,  400, "active",     "2021-08-17"),  # empty name
    (10, "Ziad Younes",             "ziad.younescompany.com",      61000,  500, "active",     "2019-12-30"),  # malformed email
    (11, "Dina Sherif",             "dina.sherif@company.com",     -5000,  100, "active",     "2022-05-22"),  # negative salary
    (12, "Ahmed Lotfy",             "ahmed.lotfy@company.com",     58000,  999, "active",     "2020-10-11"),  # unknown department
    (13, "Mai Gaber",               "mai.gaber@company.com",       64000,  200, "unknown",    "2021-02-14"),  # invalid status
    (14, "Sherif Adel",             "sherif.adel@company.com",     72000,  300, "active",     "2009-05-01"),  # stale hire_date
    (15, "",                        "nada.wael@company.com",       55000,  400, "active",     "2023-03-08"),  # empty name
    (16, "Youssef Emad",            "youssef.emad@company.com",    480000, 500, "active",     "2022-07-19"),  # salary too high
    (17, "Passant Ali",             "passant.ali@company.com",     63000,  100, "active",     "2027-01-15"),  # future hire_date
    (18, "Bassel Nagy",             "bassel.nagy@company.com",     59000,  0,   "active",     "2020-09-09"),  # unknown department
    (19, "Aya Fouad",               "aya.fouad@company.com",       47000,  200, "archived",   "2018-12-25"),  # invalid status
    (20, "Khaled Reda",             "khaled.reda@company.com",     68000,  300, "active",     "2021-04-30"),
    (21, "Menna Tarek",             "menna.tarek @ company.com",   51000,  400, "active",     "2022-11-11"),  # malformed email
    (22, "Amr Hisham",              "amr.hisham@company.com",      45000,  500, "on_leave",   "2020-06-06"),
    (23, "Lobna Samir",             "lobna.samir@company.com",     300000, 100, "active",     "2011-03-03"),  # salary + stale
    (24, "Waleed Fathy",            "waleed.fathy@company.com",    56000,  150, "active",     "2023-01-01"),  # unknown department
    (5,  "Hana Fathy (duplicate)",  "hana.fathy2@company.com",     60000,  200, "active",     "2022-02-02"),  # duplicate id=5
]

_DB_BASIC_CONFIG_YAML = """\
dataset: demo_employees
name: PostgreSQL Basic Checks
description: Quick 3-check triage against the live demo_employees table.
quality_threshold: 80.0
alert_channels: []
checks:
  - name: employee_id_unique
    column: employee_id
    check_type: unique
    threshold: 1.0
    severity: critical

  - name: full_name_not_empty
    column: full_name
    check_type: not_empty
    threshold: 0.90
    severity: high

  - name: salary_in_range
    column: salary
    check_type: range
    min_value: 30000
    max_value: 250000
    threshold: 0.85
    severity: medium
"""

_DB_FULL_CONFIG_YAML = """\
dataset: demo_employees
name: PostgreSQL Full Data Quality Suite
description: >
  Exercises every supported check_type against a real Postgres table,
  including a referential-integrity-style foreign_key check against
  demo_departments and a freshness check on hire_date.
quality_threshold: 90.0
alert_channels: []
checks:
  - name: employee_id_unique
    column: employee_id
    check_type: unique
    threshold: 1.0
    severity: critical

  - name: full_name_not_empty
    column: full_name
    check_type: not_empty
    threshold: 0.90
    severity: medium

  - name: email_format
    column: email
    check_type: regex
    pattern: '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'
    threshold: 0.90
    severity: high

  - name: salary_in_range
    column: salary
    check_type: range
    min_value: 30000
    max_value: 250000
    threshold: 0.85
    severity: medium

  - name: department_reference_check
    column: department_id
    check_type: foreign_key
    accepted_values: [100, 200, 300, 400, 500]
    threshold: 0.95
    severity: critical

  - name: status_valid
    column: status
    check_type: in_set
    accepted_values: [active, on_leave, terminated]
    threshold: 0.90
    severity: high

  - name: hire_date_freshness
    column: hire_date
    check_type: freshness
    min_value: 2015-01-01
    max_value: 2026-07-01
    threshold: 0.85
    severity: low
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(_SAMPLE_CSV_ROWS)
    click.echo(f"  [CSV]    Created {path}  ({len(_SAMPLE_CSV_ROWS) - 1} data rows)")


def _reset(session) -> None:
    """Delete all rows created by a previous seed run and reset all ID sequences to 1."""
    for name in ("Basic Customer Checks", "Strict Customer Checks"):
        for c in session.query(ValidationConfig).filter_by(name=name).all():
            session.delete(c)
    for d in session.query(Dataset).filter_by(table_name="sample_customers").all():
        session.delete(d)
    for s in session.query(DataSource).filter_by(name="DataNexus Demo — CSV").all():
        session.delete(s)
    session.flush()

    sequences = [
        "data_sources_id_seq",
        "datasets_id_seq",
        "validation_configs_id_seq",
        "validation_runs_id_seq",
        "validation_results_id_seq",
        "data_profiles_id_seq",
        "alerts_id_seq",
        "test_definitions_id_seq",
    ]
    for seq in sequences:
        session.execute(text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))

    click.echo("  [RESET]  Existing seed data removed and ID sequences restarted.")


def _seed(csv_path: Path) -> dict:
    with get_db_session() as session:
        existing = session.query(DataSource).filter_by(name="DataNexus Demo — CSV").first()
        if existing:
            click.echo("  [SKIP]   DataSource already exists — seed data is already present.")
            click.echo("           Run with --reset to wipe and re-seed.")
            ds  = session.query(Dataset).filter_by(source_id=existing.id,
                                                   table_name="sample_customers").first()
            cfs = session.query(ValidationConfig).filter_by(
                dataset_id=ds.id).all() if ds else []
            return {"source_id": existing.id,
                    "dataset_id": ds.id if ds else None,
                    "config_ids": [c.id for c in cfs],
                    "config_names": [c.name for c in cfs]}

        source = DataSource(
            name              = "DataNexus Demo — CSV",
            source_type       = SourceType.csv,
            connection_string = str(csv_path),
            description       = "Sample customer CSV for CLI demo and integration testing.",
            is_active         = True,
        )
        session.add(source)
        session.flush()
        click.echo(f"  [CREATE] DataSource  id={source.id}  →  {csv_path.name}")

        dataset = Dataset(
            source_id   = source.id,
            schema_name = None,
            table_name  = "sample_customers",
            description = "20-row sample customer dataset with deliberate quality issues.",
            is_active   = True,
        )
        session.add(dataset)
        session.flush()
        click.echo(f"  [CREATE] Dataset     id={dataset.id}  →  sample_customers")

        cfg_basic = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "Basic Customer Checks",
            config_yaml       = _BASIC_CONFIG_YAML,
            schedule_cron     = "0 */6 * * *",
            quality_threshold = 0.80,
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(cfg_basic)
        session.flush()
        click.echo(f"  [CREATE] Config      id={cfg_basic.id}  →  'Basic Customer Checks'  "
                   f"(3 checks, threshold=80%)")

        cfg_strict = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "Strict Customer Checks",
            config_yaml       = _STRICT_CONFIG_YAML,
            schedule_cron     = "0 0 * * *",
            quality_threshold = 0.90,
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(cfg_strict)
        session.flush()
        click.echo(f"  [CREATE] Config      id={cfg_strict.id}  →  'Strict Customer Checks'  "
                   f"(5 checks, threshold=90%)")

        return {
            "source_id":    source.id,
            "dataset_id":   dataset.id,
            "config_ids":   [cfg_basic.id, cfg_strict.id],
            "config_names": [cfg_basic.name, cfg_strict.name],
        }


# ─────────────────────────────────────────────────────────────────────────────
# DB-target helpers — writes real tables into the Postgres database itself,
# then registers them exactly like any other data source/dataset.
# ─────────────────────────────────────────────────────────────────────────────

_DB_SOURCE_NAME = "DataNexus Demo — PostgreSQL"


def _write_db_tables(reset: bool) -> None:
    """
    Create (or drop+recreate) demo_departments and demo_employees as real
    tables in the same Postgres database DataNexus already talks to, then
    populate them with pandas.to_sql — no separate DB/credentials needed.
    """
    import pandas as pd

    inspector    = inspect(engine)
    has_dept_tbl = inspector.has_table("demo_departments")
    has_emp_tbl  = inspector.has_table("demo_employees")

    if reset:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS demo_employees"))
            conn.execute(text("DROP TABLE IF EXISTS demo_departments"))
        has_dept_tbl = has_emp_tbl = False
        click.echo("  [RESET]  Dropped demo_departments / demo_employees (if they existed).")

    if has_dept_tbl and has_emp_tbl:
        click.echo("  [SKIP]   demo_departments / demo_employees already exist in Postgres.")
        return

    departments_df = pd.DataFrame(_DEMO_DEPARTMENTS_ROWS, columns=_DEMO_DEPARTMENTS_COLUMNS)
    employees_df   = pd.DataFrame(_DEMO_EMPLOYEES_ROWS,   columns=_DEMO_EMPLOYEES_COLUMNS)
    employees_df["hire_date"] = pd.to_datetime(employees_df["hire_date"]).dt.date

    departments_df.to_sql("demo_departments", con=engine, if_exists="replace", index=False)
    click.echo(f"  [CREATE] Table       demo_departments   ({len(departments_df)} rows)  "
               f"— reference table for the foreign_key check")

    employees_df.to_sql("demo_employees", con=engine, if_exists="replace", index=False)
    click.echo(f"  [CREATE] Table       demo_employees      ({len(employees_df)} rows, "
               f"deliberate quality issues planted)")


def _reset_db_registrations(session) -> None:
    """Remove the DataSource/Dataset/ValidationConfig rows for the DB demo (not the tables)."""
    for name in ("PostgreSQL Basic Checks", "PostgreSQL Full Data Quality Suite"):
        for c in session.query(ValidationConfig).filter_by(name=name).all():
            session.delete(c)
    for d in session.query(Dataset).filter_by(table_name="demo_employees").all():
        session.delete(d)
    for s in session.query(DataSource).filter_by(name=_DB_SOURCE_NAME).all():
        session.delete(s)
    session.flush()
    click.echo("  [RESET]  Existing PostgreSQL demo registrations removed.")


def _seed_db() -> dict:
    with get_db_session() as session:
        existing = session.query(DataSource).filter_by(name=_DB_SOURCE_NAME).first()
        if existing:
            click.echo("  [SKIP]   DataSource already exists — DB seed data is already present.")
            click.echo("           Run with --reset to wipe and re-seed.")
            ds  = session.query(Dataset).filter_by(source_id=existing.id,
                                                    table_name="demo_employees").first()
            cfs = session.query(ValidationConfig).filter_by(
                dataset_id=ds.id).all() if ds else []
            return {"source_id": existing.id,
                    "dataset_id": ds.id if ds else None,
                    "config_ids": [c.id for c in cfs],
                    "config_names": [c.name for c in cfs]}

        source = DataSource(
            name              = _DB_SOURCE_NAME,
            source_type       = SourceType.postgresql,
            connection_string = DATABASE_URL,
            description       = "Live Postgres tables (demo_employees + demo_departments) "
                                 "for the DB-table validation demo.",
            is_active         = True,
        )
        session.add(source)
        session.flush()
        click.echo(f"  [CREATE] DataSource  id={source.id}  →  postgresql (demo_employees table)")

        dataset = Dataset(
            source_id   = source.id,
            schema_name = None,
            table_name  = "demo_employees",
            description = "25-row employees table with deliberate quality issues, "
                           "referencing demo_departments for the foreign_key check.",
            is_active   = True,
        )
        session.add(dataset)
        session.flush()
        click.echo(f"  [CREATE] Dataset     id={dataset.id}  →  demo_employees")

        cfg_basic = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "PostgreSQL Basic Checks",
            config_yaml       = _DB_BASIC_CONFIG_YAML,
            schedule_cron     = "0 */6 * * *",
            quality_threshold = 0.80,
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(cfg_basic)
        session.flush()
        click.echo(f"  [CREATE] Config      id={cfg_basic.id}  →  'PostgreSQL Basic Checks'  "
                   f"(3 checks, threshold=80%)")

        cfg_full = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "PostgreSQL Full Data Quality Suite",
            config_yaml       = _DB_FULL_CONFIG_YAML,
            schedule_cron     = "0 0 * * *",
            quality_threshold = 0.90,
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(cfg_full)
        session.flush()
        click.echo(f"  [CREATE] Config      id={cfg_full.id}  →  'PostgreSQL Full Data Quality Suite'  "
                   f"(7 checks, threshold=90%)")

        return {
            "source_id":    source.id,
            "dataset_id":   dataset.id,
            "config_ids":   [cfg_basic.id, cfg_full.id],
            "config_names": [cfg_basic.name, cfg_full.name],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Click command
# ─────────────────────────────────────────────────────────────────────────────

@click.command("seed")
@click.option("--reset", is_flag=True, default=False,
              help="Wipe existing seed data first, then re-seed from scratch.")
@click.option("--target", type=click.Choice(["csv", "db", "all"]), default="csv",
              show_default=True,
              help="What to seed: 'csv' (file-based demo, unchanged default), "
                   "'db' (real Postgres tables demo_departments/demo_employees), "
                   "or 'all' (both).")
def seed_cmd(reset: bool, target: str) -> None:
    """
    Seed the database with demo data for CLI testing.

    \b
    --target csv (default):
      data/sample_customers.csv  — 20 rows with deliberate quality issues
      data_sources               — 1 CSV source entry
      datasets                   — 1 dataset entry
      validation_configs         — 2 configs (Basic 3-check, Strict 5-check)

    \b
    --target db:
      demo_departments / demo_employees — real tables written into Postgres
      data_sources                      — 1 postgresql source entry
      datasets                          — 1 dataset entry (demo_employees)
      validation_configs                — 2 configs (Basic 3-check, Full 7-check,
                                           covering every supported check_type)

    \b
    --target all seeds both of the above in one run.

    Safe to run multiple times — idempotent unless --reset is passed.
    --reset only wipes the target(s) you asked for.

    \b
    Examples:
      datanexus seed                 # csv demo only (same as before)
      datanexus seed --target db     # Postgres table validation demo
      datanexus seed --target all    # both, one command
      datanexus seed --target db --reset
    """
    click.echo()
    click.echo("DataNexus — Seed Demo Data")
    click.echo(f"Target: {target}")
    click.echo("=" * 50)

    do_csv = target in ("csv", "all")
    do_db  = target in ("db", "all")

    csv_ids: dict = {}
    db_ids:  dict = {}
    csv_path = _PROJECT_ROOT / "data" / "sample_customers.csv"

    # ── CSV target ───────────────────────────────────────────────────────────
    if do_csv:
        click.echo("\n  -- CSV demo --")
        if reset:
            with get_db_session() as session:
                _reset(session)

        if not csv_path.exists():
            _write_csv(csv_path)
        else:
            click.echo(f"  [SKIP]   CSV already exists at {csv_path}")

        csv_ids = _seed(csv_path)

    # ── DB target ────────────────────────────────────────────────────────────
    if do_db:
        click.echo("\n  -- PostgreSQL table demo --")
        if reset:
            with get_db_session() as session:
                _reset_db_registrations(session)
        _write_db_tables(reset=reset)
        db_ids = _seed_db()

    # ── Summary ──────────────────────────────────────────────────────────────
    click.echo()
    click.echo("─" * 50)
    click.echo("  Seed complete. Try these commands next:\n")

    if do_csv:
        config_ids = csv_ids.get("config_ids", [])
        if len(config_ids) >= 2:
            click.echo(f"  # CSV demo — list configs / run them:")
            click.echo(f"  datanexus config list")
            click.echo(f"  datanexus run {config_ids[0]}   # Basic Customer Checks")
            click.echo(f"  datanexus run {config_ids[1]}   # Strict Customer Checks\n")

    if do_db:
        config_ids = db_ids.get("config_ids", [])
        if len(config_ids) >= 2:
            click.echo(f"  # PostgreSQL demo — list configs / run them:")
            click.echo(f"  datanexus config list")
            click.echo(f"  datanexus run {config_ids[0]}   # PostgreSQL Basic Checks")
            click.echo(f"  datanexus run {config_ids[1]}   # PostgreSQL Full Data Quality Suite")
            click.echo(f"  datanexus profile {db_ids.get('dataset_id')}   # profile demo_employees\n")

    click.echo(f"  # All runs so far:")
    click.echo(f"  datanexus runs list\n")

    if do_csv:
        click.echo(f"  CSV  → Source id={csv_ids.get('source_id')}  "
                   f"Dataset id={csv_ids.get('dataset_id')}  "
                   f"Config ids={csv_ids.get('config_ids')}")
    if do_db:
        click.echo(f"  DB   → Source id={db_ids.get('source_id')}  "
                   f"Dataset id={db_ids.get('dataset_id')}  "
                   f"Config ids={db_ids.get('config_ids')}")
    click.echo("─" * 50)
    click.echo()