"""
SQLAlchemy Base - Shared declarative base for all models.

This file exists to avoid circular imports between models.py and database.py.
Both files import Base from here.
"""

from sqlalchemy.orm import declarative_base

Base = declarative_base()
