"""
Alembic Environment Configuration
MobilityOne Database Migrations

IMPORTANT: Migrations MUST run with admin_user (ADMIN_DATABASE_URL)
This ensures migrations have full privileges to create/alter tables.

Usage:
    # Apply migrations
    alembic upgrade head

    # Rollback
    alembic downgrade -1
"""

import os
from logging.config import fileConfig

from sqlalchemy import pool, create_engine
from sqlalchemy.engine import Connection

from alembic import context

# Import shared Base and models
from base import Base
import models  # noqa: F401 - registers models with Base

# Alembic Config object
config = context.config

# Setup logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate
target_metadata = Base.metadata


def get_url() -> str:
    """
    Get database URL for migrations.

    CRITICAL: Always use ADMIN_DATABASE_URL for migrations!
    This user has full privileges to create/alter tables.
    """
    url = os.getenv("ADMIN_DATABASE_URL") or os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "No database URL configured. "
            "Set ADMIN_DATABASE_URL or DATABASE_URL environment variable."
        )

    # Convert async URL to sync for alembic (psycopg2)
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")

    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL scripts)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to database)."""
    connectable = create_engine(
        get_url(),
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
