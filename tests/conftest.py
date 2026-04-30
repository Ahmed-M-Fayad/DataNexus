"""
tests/conftest.py
=================
Shared pytest fixtures for the DataNexus integration test suite.

What this file sets up
-----------------------
1. A real SQLAlchemy session connected to your local DataNexus PostgreSQL DB.
2. A `test_data` schema with a `test_customers` table populated with 10 rows
   that include deliberate data quality issues (nulls, out-of-range values).
3. DataNexus metadata rows (DataSource → Dataset → ValidationConfig) that
   point the Validation Engine at that test table.
4. Full teardown after every test: removes all DataNexus rows and the test
   schema so each test run starts from a clean slate.

Requirements
------------
- Your local PostgreSQL must be running.
- You must have run: alembic upgrade head  (DataNexus schema must exist).
- DATABASE_URL must be set in your .env file.
  Example: DATABASE_URL=postgresql://datanexus:secret@localhost:5432/datanexus
"""

import os

import pytest
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load .env so DATABASE_URL is available
load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]

from src.database.models import (
    DataSource,
    Dataset,
    SourceType,
    ValidationConfig,
)


# ── Session fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    """
    Creates one SQLAlchemy engine for the entire test session.
    Disposed at the end. 'scope=session' means it is created once and shared.
    """
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """
    Yields a fresh SQLAlchemy session for each test function.
    The session is rolled back and closed after the test — this means any
    DataNexus ORM rows inserted in a test do NOT persist unless committed.

    Note: the integration test fixture (datanexus_setup) uses its OWN
    sessions via get_db_session() so the engine can see those committed rows.
    This fixture is provided for manual assertions inside test functions.
    """
    Session = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)
    session = Session()
    yield session
    session.close()


# ── Test data + DataNexus metadata fixture ─────────────────────────────────────

@pytest.fixture(scope="function")
def datanexus_setup(db_engine):
    """
    Full integration fixture — sets up everything the Validation Engine needs
    and tears it all down after the test.

    Yields
    ------
    dict with keys:
        config_id    (int) — ID of the ValidationConfig row to pass to engine.run()
        dataset_id   (int) — ID of the Dataset row
        source_id    (int) — ID of the DataSource row
        expected_checks (list[dict]) — the checks list parsed from the YAML,
                         for asserting against validation_results rows
    """
    Session = sessionmaker(bind=db_engine, autoflush=False, autocommit=False)

    # ── STEP 1: Create the test_data schema and test_customers table ──────────
    #
    # We put test data in a separate `test_data` schema inside the DataNexus DB.
    # This is the simplest setup — no second database, no Docker needed.
    #
    # Table shape:
    #   id    INTEGER   — all unique
    #   name  TEXT      — all populated
    #   email TEXT      — 2 NULLs to trigger email_not_null failure
    #   age   INTEGER   — 1 out-of-range (200) to test range check
    #
    with db_engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS test_data"))
        conn.execute(text("DROP TABLE IF EXISTS test_data.test_customers"))
        conn.execute(text("""
            CREATE TABLE test_data.test_customers (
                id    INTEGER,
                name  TEXT,
                email TEXT,
                age   INTEGER
            )
        """))
        conn.execute(text("""
            INSERT INTO test_data.test_customers (id, name, email, age) VALUES
            (1,  'Alice',   'alice@example.com',   25),
            (2,  'Bob',     'bob@example.com',     30),
            (3,  'Carol',   'carol@example.com',   22),
            (4,  'Dave',    'dave@example.com',    45),
            (5,  'Eve',     'eve@example.com',     28),
            (6,  'Frank',   'frank@example.com',   35),
            (7,  'Grace',   'grace@example.com',   19),
            (8,  'Hank',    'hank@example.com',    52),
            (9,  'Iris',    NULL,                  33),   -- NULL email  (fail not_null)
            (10, 'Jack',    NULL,                  200)   -- NULL email + age=200 (two failures)
        """))
        conn.commit()

    # ── STEP 2: Register DataNexus metadata rows ──────────────────────────────
    #
    # DataSource → Dataset → ValidationConfig
    # The DataSource's connection_string points to the SAME database so the
    # Validation Engine can read test_data.test_customers via pd.read_sql_table.

    # Load the test YAML config from disk so it matches what's in the test file
    config_yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "examples", "test_customers.yaml"
    )
    with open(config_yaml_path, "r") as fh:
        config_yaml_text = fh.read()

    # Parse out the checks list so the test can assert on specific check names
    raw_yaml = yaml.safe_load(config_yaml_text)
    expected_checks = [c["name"] for c in raw_yaml.get("checks", [])]

    session = Session()
    try:
        # 1. DataSource — points to this same PostgreSQL database
        source = DataSource(
            name              = "_test_source_customers",
            source_type       = SourceType.postgresql,
            connection_string = DATABASE_URL,
            description       = "Integration test source — auto-deleted after test",
            is_active         = True,
        )
        session.add(source)
        session.flush()   # assigns source.id

        # 2. Dataset — points to test_data.test_customers
        dataset = Dataset(
            source_id   = source.id,
            schema_name = "test_data",
            table_name  = "test_customers",
            description = "Integration test dataset — auto-deleted after test",
            is_active   = True,
        )
        session.add(dataset)
        session.flush()   # assigns dataset.id

        # 3. ValidationConfig — stores the YAML and threshold
        #    quality_threshold=0.80 (stored as 0.0–1.0 in the DB)
        config = ValidationConfig(
            dataset_id        = dataset.id,
            name              = "_test_config_customers",
            config_yaml       = config_yaml_text,
            quality_threshold = 0.80,   # run FAILS if score < 80 / 100
            alert_on_failure  = False,  # no alerts needed in tests
            alert_channels    = None,
            is_active         = True,
        )
        session.add(config)
        session.flush()   # assigns config.id

        # Capture IDs before committing
        source_id  = source.id
        dataset_id = dataset.id
        config_id  = config.id

        session.commit()

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # ── YIELD to the test ─────────────────────────────────────────────────────
    yield {
        "config_id":       config_id,
        "dataset_id":      dataset_id,
        "source_id":       source_id,
        "expected_checks": expected_checks,
    }

    # ── TEARDOWN — runs after the test, even if it failed ────────────────────
    #
    # Delete all DataNexus rows we created (cascade handles child rows).
    # Drop the test schema entirely.

    teardown_session = Session()
    try:
        # Delete ValidationConfig first — cascades to ValidationRuns → Results + Alerts
        cfg = teardown_session.query(ValidationConfig).filter_by(id=config_id).first()
        if cfg:
            teardown_session.delete(cfg)

        # Delete Dataset (cascades to DataProfile rows)
        ds = teardown_session.query(Dataset).filter_by(id=dataset_id).first()
        if ds:
            teardown_session.delete(ds)

        # Delete DataSource
        src = teardown_session.query(DataSource).filter_by(id=source_id).first()
        if src:
            teardown_session.delete(src)

        teardown_session.commit()
    except Exception:
        teardown_session.rollback()
        raise
    finally:
        teardown_session.close()

    # Drop the test schema (raw SQL — ORM does not manage it)
    with db_engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS test_data CASCADE"))
        conn.commit()