"""
Alembic migration environment.

This file is run by Alembic before every migration command.
It connects to the database using DATABASE_URL from your .env file
and tells Alembic which models to track for schema changes.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# ── Make sure src/ is on the path so we can import our models ────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Load .env so DATABASE_URL is available ───────────────────────────────────
load_dotenv()

# ── Import all models so Alembic can detect schema changes ───────────────────
from src.database.models import Base  # noqa: E402  (import after sys.path setup)

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Inject DATABASE_URL from environment into Alembic config
# This means alembic.ini does NOT need a hardcoded URL
config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

# Set up Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# This is the MetaData object that holds all our table definitions.
# Alembic compares this against the live database to detect differences.
target_metadata = Base.metadata


# ── Offline mode — generates SQL without connecting to DB ────────────────────
def run_migrations_offline() -> None:
    """
    Generate migration SQL without an active DB connection.
    Useful for reviewing what will change before applying it.
    Run with: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode — applies migrations directly to the database ─────────────────
def run_migrations_online() -> None:
    """
    Apply migrations to the live database.
    This is the mode used by: alembic upgrade head
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # don't pool connections during migrations
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,    # detect column type changes
            compare_server_default=True,  # detect default value changes
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
