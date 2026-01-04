import os
from alembic.config import Config
from alembic import command
import sys
import logging

# Configure logging to show output from Alembic
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    """
    Programmatically runs Alembic migrations to reset the database.
    This script is a workaround for environments where 'docker-compose'
    or 'alembic' CLI commands are not directly runnable.
    """
    log.info("--- Starting database reset script ---")

    # Alembic configuration is loaded from alembic.ini
    # The associated env.py will read the DB URL from environment variables
    alembic_cfg = Config("alembic.ini")

    try:
        # Step 1: Downgrade to base (drop all tables)
        log.info("Running Alembic downgrade to base...")
        command.downgrade(alembic_cfg, "base")
        log.info("Downgrade to base successful.")

        # Step 2: Upgrade to head (recreate all tables)
        log.info("Running Alembic upgrade to head...")
        command.upgrade(alembic_cfg, "head")
        log.info("Upgrade to head successful.")

        log.info("--- Database has been successfully reset. ---")

    except Exception as e:
        log.error(f"An error occurred during the migration: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
