"""
Database connection and session management for PostgreSQL

Provides connection pooling, session creation, and database initialization
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from sqlalchemy.pool import QueuePool, StaticPool
from contextlib import contextmanager
from typing import Generator
from pathlib import Path
from loguru import logger

from .models import Base
from ..config.settings import get_settings


# Global engine and session factory
_engine = None
_session_factory = None


def init_database(database_url: str = None, echo: bool = False) -> None:
    """
    Initialize database engine and session factory

    Args:
        database_url: Database connection URL (SQLite or PostgreSQL)
        echo: Enable SQL query logging

    Example:
        init_database("sqlite:///data/trading.db")
        init_database("postgresql://user:pass@localhost:5432/trading_db")
    """
    global _engine, _session_factory

    settings = get_settings()
    db_url = database_url or settings.database_url

    # Detect database type and configure appropriately
    is_sqlite = db_url.startswith("sqlite")
    
    if is_sqlite:
        # For SQLite: ensure directory exists
        if ":///" in db_url:
            db_path = db_url.split("///")[-1]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing SQLite database: {db_url}")
        
        # SQLite configuration - use StaticPool for thread safety
        _engine = create_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=echo,
        )
        
        # Enable foreign keys for SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # PostgreSQL configuration with connection pooling
        logger.info(f"Initializing PostgreSQL connection to {db_url.split('@')[-1]}")
        
        _engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=settings.database_pool_size,
            max_overflow=10,
            pool_pre_ping=True,
            echo=echo,
        )

    # Create session factory
    _session_factory = scoped_session(
        sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
        )
    )

    logger.success(f"Database connection initialized ({'SQLite' if is_sqlite else 'PostgreSQL'})")


def create_all_tables() -> None:
    """
    Create all tables in the database

    Uses SQLAlchemy metadata to create tables that don't exist.
    Safe to call multiple times (won't recreate existing tables).
    """
    global _engine

    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=_engine)
    logger.success("All tables created successfully")


def drop_all_tables() -> None:
    """
    Drop all tables from the database

    WARNING: This will delete all data! Use only for testing or resets.
    """
    global _engine

    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=_engine)
    logger.info("All tables dropped")


def get_db_session() -> Session:
    """
    Get a new database session

    Returns:
        SQLAlchemy Session instance

    Example:
        session = get_db_session()
        try:
            trade = session.query(Trade).filter_by(id=123).first()
        finally:
            session.close()
    """
    global _session_factory

    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    return _session_factory()


@contextmanager
def db_session_scope() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback

    Yields:
        SQLAlchemy Session

    Example:
        with db_session_scope() as session:
            trade = Trade(strategy='iron_condor', ...)
            session.add(trade)
            # Automatically commits on success, rolls back on error
    """
    session = get_db_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def close_database() -> None:
    """
    Close all database connections and clean up resources

    Call this when shutting down the application
    """
    global _engine, _session_factory

    if _session_factory:
        _session_factory.remove()
        _session_factory = None
        logger.info("Session factory closed")

    if _engine:
        _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")


def health_check() -> bool:
    """
    Check database connectivity

    Returns:
        True if database is accessible, False otherwise
    """
    try:
        with db_session_scope() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def get_engine():
    """
    Get the database engine

    Returns:
        SQLAlchemy Engine instance

    Raises:
        RuntimeError: If database not initialized
    """
    global _engine
    
    if _engine is None:
        # Auto-initialize if not done
        init_database()
    
    return _engine


def get_session():
    """
    Alias for db_session_scope for compatibility
    
    Usage:
        with get_session() as db:
            user = db.query(User).filter_by(id=1).first()
    """
    return db_session_scope()


# Expose engine property for migrations
@property
def engine():
    """Property to get the database engine"""
    return get_engine()
