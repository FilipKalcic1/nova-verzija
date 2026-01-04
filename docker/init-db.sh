#!/bin/bash
set -e

# =============================================================================
# MobilityOne Database Initialization
# Version: 3.0 - Production Security (Alembic Migrations)
#
# SECURITY MODEL:
# - bot_user: Limited access (conversations, messages, INSERT on hallucinations)
#   NO CREATE privilege - cannot create or alter tables!
# - admin_user: Full access (all tables, can run migrations)
#
# FLOW:
# 1. This script creates database and users (runs on postgres startup)
# 2. Alembic migrations create tables (runs separately with admin_user)
# 3. Bot/Worker use bot_user for runtime (limited access)
# =============================================================================

echo "=== MobilityOne Database Initialization v3.0 ==="
echo "Setting up dual-user security model..."

# Get passwords from environment
BOT_PASSWORD="${BOT_DB_PASSWORD:-bot_secure_password}"
ADMIN_PASSWORD="${ADMIN_DB_PASSWORD:-admin_secure_password}"

# =============================================================================
# 1. CREATE DATABASE
# =============================================================================
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<-EOSQL
    SELECT 'CREATE DATABASE mobility_db'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mobility_db')\gexec
EOSQL

echo "Database mobility_db created"

# =============================================================================
# 2. CREATE USERS AND SET BASE PERMISSIONS
# =============================================================================
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "mobility_db" <<-EOSQL
    -- Create users if not exist
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bot_user') THEN
            CREATE USER bot_user WITH PASSWORD '${BOT_PASSWORD}';
            RAISE NOTICE 'Created bot_user';
        ELSE
            ALTER USER bot_user WITH PASSWORD '${BOT_PASSWORD}';
            RAISE NOTICE 'Updated bot_user password';
        END IF;

        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'admin_user') THEN
            CREATE USER admin_user WITH PASSWORD '${ADMIN_PASSWORD}';
            RAISE NOTICE 'Created admin_user';
        ELSE
            ALTER USER admin_user WITH PASSWORD '${ADMIN_PASSWORD}';
            RAISE NOTICE 'Updated admin_user password';
        END IF;
    END
    \$\$;

    -- ==========================================================================
    -- ADMIN USER: Full access (for migrations and admin API)
    -- ==========================================================================
    GRANT CONNECT ON DATABASE mobility_db TO admin_user;
    GRANT ALL PRIVILEGES ON SCHEMA public TO admin_user;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_user;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin_user;

    -- Admin can create tables (for alembic migrations)
    GRANT CREATE ON SCHEMA public TO admin_user;

    -- Future tables created by admin_user will have full privileges
    ALTER DEFAULT PRIVILEGES FOR ROLE admin_user IN SCHEMA public
        GRANT ALL PRIVILEGES ON TABLES TO admin_user;
    ALTER DEFAULT PRIVILEGES FOR ROLE admin_user IN SCHEMA public
        GRANT ALL PRIVILEGES ON SEQUENCES TO admin_user;

    -- ==========================================================================
    -- BOT USER: Limited access (NO CREATE privilege!)
    -- ==========================================================================
    GRANT CONNECT ON DATABASE mobility_db TO bot_user;
    GRANT USAGE ON SCHEMA public TO bot_user;

    -- NOTE: bot_user CANNOT create tables!
    -- Tables are created by alembic with admin_user

    -- Default privileges for tables created by admin_user
    -- Bot gets SELECT/INSERT/UPDATE/DELETE on most tables
    ALTER DEFAULT PRIVILEGES FOR ROLE admin_user IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO bot_user;
    ALTER DEFAULT PRIVILEGES FOR ROLE admin_user IN SCHEMA public
        GRANT USAGE ON SEQUENCES TO bot_user;

EOSQL

echo "Users created with appropriate permissions"
echo ""
echo "=== Database Ready ==="
echo "  - admin_user: Full access (for alembic migrations)"
echo "  - bot_user: Limited access (NO CREATE privilege!)"
echo ""
echo "NEXT STEP: Run 'alembic upgrade head' with ADMIN_DATABASE_URL"
echo ""
