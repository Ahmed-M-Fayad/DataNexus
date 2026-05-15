"""
scripts/seed.py — DataNexus Demo Seed Script
=============================================
Populates the database with one complete chain of demo rows:
    DataSource → Dataset → ValidationConfig

This gives every CLI command something real to run against without
needing a live external database. The seeded dataset uses a local
CSV file (data/sample_customers.csv) that this script also creates.

Usage
-----
    # From the project root, with venv activated:
    python scripts/seed.py

    # To wipe everything and re-seed from scratch:
    python scripts/seed.py --reset

What gets created
-----------------
    data_sources        1 row  — CSV source pointing at data/sample_customers.csv
    datasets            1 row  — "sample_customers" dataset
    validation_configs  2 rows — "basic" config (3 checks) and "strict" config (5 checks)
    data/sample_customers.csv  — 20 rows with deliberate quality issues for demo

Nothing is created if the seed data already exists (idempotent by default).
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# ── Make sure project root is on the path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from src.database import get_db_session
from src.database.models import (
    DataSource,
    Dataset,
    SourceType,
    ValidationConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sample CSV content — 20 rows with deliberate quality issues
# ─────────────────────────────────────────────────────────────────────────────
#
# Issues planted for demo:
#   - 3 rows have NULL email      → email_not_null will FAIL
#   - 2 rows have age outside [18, 120] → age_in_range will flag them
#   - 1 duplicate customer_id     → id_unique will FAIL
#   - 2 rows have invalid status  → status_valid will FAIL

SAMPLE_CSV_ROWS = [
    ["customer_id", "name",            "email",                    "age", "status"],
    [1,  "Alice Johnson",   "alice@example.com",        25,  "active"],
    [2,  "Bob Smith",       "bob@example.com",          30,  "active"],
    [3,  "Carol White",     "carol@example.com",        22,  "inactive"],
    [4,  "Dave Brown",      "dave@example.com",         45,  "active"],
    [5,  "Eve Davis",       "eve@example.com",          28,  "pending"],
    [6,  "Frank Miller",    "frank@example.com",        35,  "active"],
    [7,  "Grace Wilson",    "grace@example.com",        19,  "inactive"],
    [8,  "Hank Moore",      "hank@example.com",         52,  "active"],
    [9,  "Iris Taylor",     "",                         33,  "active"],   # empty email
    [10, "Jack Anderson",   "",                         200, "active"],   # empty email + age=200
    [11, "Karen Thomas",    "karen@example.com",        41,  "active"],
    [12, "Leo Jackson",     "leo@example.com",          16,  "active"],   # age=16 (underage)
    [13, "Mia Harris",      "mia@example.com",          29,  "active"],
    [14, "Nick Martin",     "nick@example.com",         38,  "unknown"], # invalid status
    [15, "Olivia Garcia",   "olivia@example.com",       24,  "active"],
    [16, "Paul Martinez",   "",                         31,  "pending"], # empty email
    [17, "Quinn Robinson",  "quinn@example.com",        27,  "inactive"],
    [18, "Rachel Clark",    "rachel@example.com",       44,  "active"],
    [19, "Sam Rodriguez",   "sam@example.com",          36,  "suspended"], # invalid status
    [18, "Tom Lewis",       "tom@example.com",          50,  "active"],   # duplicate id=18
]

# ─────────────────────────────────────────────────────────────────────────────
# Validation config YAML strings
# ─────────────────────────────────────────────────────────────────────────────

BASIC_CONFIG_YAML = """\
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

STRICT_CONFIG_YAML = """\
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
    values: [active, inactive, pending]
    threshold: 1.0
    severity: high

  - name: name_not_empty
    column: name
    check_type: not_empty
    threshold: 1.0
    severity: medium
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_sample_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(SAMPLE_CSV_ROWS)
    print(f"  [CSV]    Created {path}  ({len(SAMPLE_CSV_ROWS) - 1} data rows)")


def _reset_seed_data(session) -> None:
    """Delete all rows created by a previous seed run, in FK-safe order."""
    configs = session.query(ValidationConfig).filter(
        ValidationConfig.name.in_(["Basic Customer Checks", "Strict Customer Checks"])
    ).all()
    for c in configs:
        session.delete(c)

    datasets = session.query(Dataset).filter_by(table_name="sample_customers").all()
    for d in datasets:
        session.delete(d)

    sources = session.query(DataSource).filter_by(name="DataNexus Demo — CSV").all()
    for s in sources:
        session.delete(s)

    session.flush()
    print("  [RESET]  Existing seed data removed.")


def _seed(csv_path: Path) -> dict:
    """
    Insert the seed rows and return a dict with the created IDs.
    Idempotent: if the DataSource already exists, skip creation and
    return the existing IDs.
    """
    with get_db_session() as session:

        # ── DataSource ────────────────────────────────────────────────────────
        existing_source = session.query(DataSource).filter_by(
            name="DataNexus Demo — CSV"
        ).first()

        if existing_source:
            print("  [SKIP]   DataSource already exists — seed data is already present.")
            print("           Run with --reset to wipe and re-seed.")

            # Still fetch config IDs so we can print them
            dataset = session.query(Dataset).filter_by(
                source_id=existing_source.id, table_name="sample_customers"
            ).first()
            configs = session.query(ValidationConfig).filter_by(
                dataset_id=dataset.id
            ).all() if dataset else []

            return {
                "source_id":  existing_source.id,
                "dataset_id": dataset.id if dataset else None,
                "config_ids": [c.id for c in configs],
            }

        source = DataSource(
            name              = "DataNexus Demo — CSV",
            source_type       = SourceType.csv,
            connection_string = str(csv_path),
            description       = "Sample customer CSV for CLI demo and integration testing.",
            is_active         = True,
        )
        session.add(source)
        session.flush()
        print(f"  [CREATE] DataSource  id={source.id}  →  {csv_path.name}")

        # ── Dataset ───────────────────────────────────────────────────────────
        dataset = Dataset(
            source_id   = source.id,
            schema_name = None,         # not applicable for CSV sources
            table_name  = "sample_customers",
            description = "20-row sample customer dataset with deliberate quality issues.",
            is_active   = True,
        )
        session.add(dataset)
        session.flush()
        print(f"  [CREATE] Dataset     id={dataset.id}  →  sample_customers")

        # ── ValidationConfig 1 — Basic ────────────────────────────────────────
        config_basic = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "Basic Customer Checks",
            config_yaml       = BASIC_CONFIG_YAML,
            schedule_cron     = "0 */6 * * *",
            quality_threshold = 0.80,   # stored as 0.0–1.0 in DB
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(config_basic)
        session.flush()
        print(f"  [CREATE] Config      id={config_basic.id}  →  'Basic Customer Checks'  (3 checks, threshold=80%)")

        # ── ValidationConfig 2 — Strict ───────────────────────────────────────
        config_strict = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "Strict Customer Checks",
            config_yaml       = STRICT_CONFIG_YAML,
            schedule_cron     = "0 0 * * *",
            quality_threshold = 0.90,
            alert_on_failure  = False,
            alert_channels    = None,
            is_active         = True,
        )
        session.add(config_strict)
        session.flush()
        print(f"  [CREATE] Config      id={config_strict.id}  →  'Strict Customer Checks'  (5 checks, threshold=90%)")

        return {
            "source_id":  source.id,
            "dataset_id": dataset.id,
            "config_ids": [config_basic.id, config_strict.id],
        }


def _print_summary(ids: dict, csv_path: Path) -> None:
    basic_id, strict_id = ids["config_ids"][0], ids["config_ids"][1]
    print()
    print("─" * 60)
    print("  Seed complete. Run a validation from the project root:")
    print()
    print(f"  # Basic config (3 checks, 80% threshold):")
    print(f"  python -m src.cli run {basic_id}")
    print()
    print(f"  # Strict config (5 checks, 90% threshold):")
    print(f"  python -m src.cli run {strict_id}")
    print()
    print(f"  # Or call the engine directly:")
    print(f"  python -c \"")
    print(f"  from src.validator import ValidationEngine")
    print(f"  engine = ValidationEngine()")
    print(f"  run_id = engine.run(config_id={basic_id}, triggered_by='seed_test')")
    print(f"  print('Run ID:', run_id)")
    print(f"  \"")
    print()
    print(f"  CSV location : {csv_path}")
    print(f"  Source ID    : {ids['source_id']}")
    print(f"  Dataset ID   : {ids['dataset_id']}")
    print(f"  Config IDs   : {ids['config_ids']}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed the DataNexus database with demo data for CLI testing."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all existing seed data before re-seeding.",
    )
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "data" / "sample_customers.csv"

    print()
    print("DataNexus — Demo Seed Script")
    print("=" * 60)

    # ── Reset if requested ────────────────────────────────────────────────────
    if args.reset:
        with get_db_session() as session:
            _reset_seed_data(session)

    # ── Create the sample CSV ─────────────────────────────────────────────────
    if not csv_path.exists():
        _create_sample_csv(csv_path)
    else:
        print(f"  [SKIP]   CSV already exists at {csv_path}")

    # ── Seed the database ─────────────────────────────────────────────────────
    ids = _seed(csv_path)

    # ── Print what to do next ─────────────────────────────────────────────────
    _print_summary(ids, csv_path)


if __name__ == "__main__":
    main()