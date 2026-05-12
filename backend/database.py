"""
Cloud Anomaly Detection - Database Configuration

Configures SQLAlchemy ORM for cloud anomaly detection database.
Provides database engine, session factory, and base class for models.

Module Components:
    - Declarative Base: Base SQL model class for all ORM tables
    - Engine: SQLAlchemy database engine with connection pooling
    - SessionLocal: Session factory for creating database sessions
    - get_db_session(): Context manager for safe session handling

Database Connection:
    - Engine: Configured with SQLite database (cloud.db)
    - Pool: QueuePool with connection pooling and recycling
    - Checks: Same-thread checks disabled for async compatibility
    - Echo: SQL statement logging configurable

Session Management:
    - SessionLocal: Main session factory for application use
    - get_db_session(): Generator for use in dependency injection
    - Autoflush: Enabled to prevent stale session issues
    - Expire: Automatic expiration on commit to ensure fresh data

Returns:
    Database engine, session factory, and declarative base for ORM models
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy import text
# ============================================================================
# Configuration Section
# ============================================================================

# Database connection
DATABASE_URL: str = "sqlite:///./cloud.db"
DB_ECHO: bool = False  # Set to True for SQL statement logging

# Connection pool configuration
POOL_SIZE: int = 5
MAX_OVERFLOW: int = 10
POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
POOL_PRE_PING: bool = True  # Test connections before use
CONNECT_TIMEOUT: int = 10  # Connection timeout in seconds
DB_ISOLATION_LEVEL: str = "READ_COMMITTED"  # Transaction isolation level

# Retry configuration
CONNECTION_RETRY_ATTEMPTS: int = 3
CONNECTION_RETRY_DELAY: float = 1.0  # Initial delay in seconds (exponential backoff)

# Session configuration
SESSION_AUTOFLUSH: bool = True
SESSION_EXPIRE_ON_COMMIT: bool = True
SESSION_OPERATION_TIMEOUT: int = 30  # Timeout for session operations

# ============================================================================
# Logging Setup
# ============================================================================

logger = logging.getLogger(__name__)
import os
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler()
    ]
)


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_database_url(url: str) -> None:
    """
    Validate database URL format before creating engine.

    Args:
        url: Database connection URL string

    Raises:
        ValueError: If URL format is invalid

    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"DATABASE_URL must be non-empty string, got '{url}'")

    # Check for valid SQLAlchemy URL format
    valid_prefixes = ("sqlite://", "postgresql://", "mysql://", "oracle://")
    if not any(url.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(
            f"DATABASE_URL must start with valid dialect "
            f"({', '.join(valid_prefixes)}), got '{url}'"
        )

    logger.debug(f"Database URL validated: {url}")


def _configure_sqlite_optimizations(engine: Engine) -> None:
    """
    Apply SQLite-specific optimizations and pragmas.

    Args:
        engine: SQLAlchemy engine instance

    """
    if "sqlite" in engine.url.drivername:
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            # Enable foreign keys
            cursor.execute("PRAGMA foreign_keys=ON")
            # Improve concurrent access
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        logger.info("SQLite optimizations applied (Foreign Keys, WAL mode)")


def _validate_engine_connection(engine: Engine, attempt: int = 1) -> None:
    """
    Test database engine connection before returning.

    Args:
        engine: SQLAlchemy engine instance
        attempt: Current retry attempt number

    Raises:
        RuntimeError: If database connection cannot be established after retries
        OperationalError: If connection fails

    """
    try:
        logger.debug(f"Testing database connection (attempt {attempt}/{CONNECTION_RETRY_ATTEMPTS})")
        with engine.connect() as connection:
            # Verify connection is actually working
            with engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.close()
            logger.info("Database connection test successful")
    except OperationalError as e:
        error_msg = f"Database connection failed (attempt {attempt}): {e}"
        logger.warning(error_msg)
        
        if attempt < CONNECTION_RETRY_ATTEMPTS:
            delay = CONNECTION_RETRY_DELAY * (2 ** (attempt - 1))  # Exponential backoff
            logger.info(f"Retrying connection in {delay:.1f} seconds...")
            time.sleep(delay)
            _validate_engine_connection(engine, attempt + 1)
        else:
            error_msg = f"Failed to connect to database after {CONNECTION_RETRY_ATTEMPTS} attempts: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error during connection test: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


# ============================================================================
# Database Engine Creation
# ============================================================================

def _create_engine() -> Engine:
    """
    Create and configure SQLAlchemy database engine.

    Applies connection pooling, logging, and database optimizations.

    Returns:
        Configured SQLAlchemy Engine instance

    Raises:
        ValueError: If database URL is invalid
        RuntimeError: If engine creation or connection fails

    """
    try:
        logger.info("=" * 60)
        logger.info("Creating database engine")

        # Validate database URL
        _validate_database_url(DATABASE_URL)

        # Create engine with connection pooling
        logger.debug(f"Engine configuration: pool_size={POOL_SIZE}, "
                    f"max_overflow={MAX_OVERFLOW}, echo={DB_ECHO}")

        engine = create_engine(
            DATABASE_URL,
            echo=DB_ECHO,
            poolclass=QueuePool,
            pool_size=POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_recycle=POOL_RECYCLE,
            pool_pre_ping=POOL_PRE_PING,
            connect_args={
                "check_same_thread": False,  # SQLite compatibility
                "timeout": CONNECT_TIMEOUT,  # Connection timeout
            },
        )

        logger.debug("Engine created successfully")

        # Apply database-specific optimizations
        _configure_sqlite_optimizations(engine)

        # Test database connection
        _validate_engine_connection(engine)

        logger.info("=" * 60)
        return engine

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.critical(f"Failed to create database engine: {e}")
        raise RuntimeError(f"Database initialization failed: {e}") from e


def _create_session_factory(engine: Engine) -> sessionmaker:
    """
    Create SQLAlchemy session factory with proper configuration.

    Args:
        engine: SQLAlchemy Engine instance

    Returns:
        Configured sessionmaker instance

    """
    logger.debug("Creating session factory")

    session_factory = sessionmaker(
        bind=engine,
        autoflush=SESSION_AUTOFLUSH,
        expire_on_commit=SESSION_EXPIRE_ON_COMMIT,
    )

    logger.debug(f"Session factory configured: "
                f"autoflush={SESSION_AUTOFLUSH}, "
                f"expire_on_commit={SESSION_EXPIRE_ON_COMMIT}")

    return session_factory


# ============================================================================
# Module-Level Initialization
# ============================================================================

try:
    # Create database engine
    engine: Engine = _create_engine()

    # Create session factory
    SessionLocal: sessionmaker = _create_session_factory(engine)

    # Create declarative base for ORM models
    Base = declarative_base()

    logger.info("Database module initialized successfully")

except Exception as e:
    logger.critical(f"Failed to initialize database module: {e}", exc_info=True)
    raise


# ============================================================================
# Database Session Management
# ============================================================================

def get_db_session() -> Generator[Session, None, None]:
    """
    Provide database session as dependency for application.

    Yields database session with automatic cleanup on completion.
    Handles rollback on error and proper resource cleanup.
    Prevents session leaks with multi-stage cleanup.

    Yields:
        SQLAlchemy Session instance with active database connection

    Raises:
        RuntimeError: If session creation fails

    Example:
        # For use in FastAPI dependency injection:
        @app.get("/predictions")
        async def get_predictions(db: Session = Depends(get_db_session)):
            return db.query(Prediction).all()

    """
    session: Optional[Session] = None
    session_active = False
    
    try:
        logger.debug("Creating new database session")
        session = SessionLocal()
        session_active = True
        logger.debug(f"Session created: {id(session)}")
        yield session
        
        # Commit any pending changes on successful completion
        if session.is_active:
            try:
                session.commit()
                logger.debug(f"Session {id(session)} committed successfully")
            except Exception as commit_error:
                logger.warning(f"Failed to commit session {id(session)}: {commit_error}")
                session.rollback()
                raise

    except OperationalError as e:
        logger.error(f"Operational database error in session {id(session) if session else 'unknown'}: {e}")
        if session and session_active:
            try:
                session.rollback()
                logger.debug(f"Transaction rolled back for session {id(session)}")
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {rollback_error}", exc_info=True)
        raise RuntimeError(f"Database operational error: {str(e)}") from e

    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error in session {id(session) if session else 'unknown'}: {e}")
        if session and session_active:
            try:
                session.rollback()
                logger.debug(f"Transaction rolled back for session {id(session)}")
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {rollback_error}", exc_info=True)
        raise RuntimeError(f"Database error: {str(e)}") from e

    except Exception as e:
        # Re-raise HTTPException so FastAPI can return the correct status code (e.g., 403 Forbidden)
        if "HTTPException" in type(e).__name__:
            raise e
            
        logger.error(f"Unexpected error in database session {id(session) if session else 'unknown'}: {e}", exc_info=True)
        if session and session_active:
            try:
                session.rollback()
                logger.debug(f"Transaction rolled back for session {id(session)}")
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {rollback_error}", exc_info=True)
        raise RuntimeError(f"Database session error: {str(e)}") from e

    finally:
        # Multi-stage cleanup to prevent session leaks
        if session:
            session_id = id(session)
            try:
                # Stage 1: Rollback any uncommitted transaction
                if session.is_active:
                    logger.debug(f"Rolling back active transaction in session {session_id}")
                    session.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback in session {session_id}: {rollback_error}")
            
            try:
                # Stage 2: Close the session
                logger.debug(f"Closing database session {session_id}")
                session.close()
                logger.debug(f"Session {session_id} closed successfully")
            except Exception as close_error:
                logger.error(f"Error closing session {session_id}: {close_error}", exc_info=True)
            
            try:
                # Stage 3: Verify session is properly closed
                if session.is_active:
                    logger.warning(f"Session {session_id} still active after close attempt")
                    session.expunge_all()  # Clear session cache
                    logger.debug(f"Session {session_id} expunged")
            except Exception as verify_error:
                logger.error(f"Error verifying session closure {session_id}: {verify_error}")


@contextmanager
def session_scope():
    """
    Context manager for database session with guaranteed cleanup.

    Provides automatic session cleanup on both success and error.
    Preferred over get_db_session() for synchronous code.

    Yields:
        SQLAlchemy Session instance

    Example:
        with session_scope() as session:
            result = session.query(Prediction).all()

    """
    session: Optional[Session] = None
    try:
        session = SessionLocal()
        logger.debug(f"Session scope opened: {id(session)}")
        yield session
        session.commit()
        logger.debug(f"Session scope committed: {id(session)}")
    except Exception as e:
        if session:
            try:
                session.rollback()
                logger.debug(f"Session scope rolled back on error: {id(session)}")
            except Exception as rollback_error:
                logger.error(f"Error rolling back in session scope: {rollback_error}")
        logger.error(f"Error in session scope: {e}", exc_info=True)
        raise
    finally:
        if session:
            try:
                session.close()
                logger.debug(f"Session scope closed: {id(session)}")
            except Exception as close_error:
                logger.error(f"Error closing session scope: {close_error}")


def get_db_health() -> dict:
    """
    Check database health and connection pool status.

    Returns:
        Dictionary with health status and pool statistics

    Example:
        health = get_db_health()
        if health['status'] == 'healthy':
            logger.info(f"Pool size: {health['pool_size']}")

    """
    try:
        logger.debug("Checking database health")
        
        # Test connection
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            result.close()
        
        # Get pool statistics
        pool = engine.pool
        health_status = {
            "status": "healthy",
            "pool_size": pool.size(),
            "pool_checked_in": pool.checkedin(),
            "pool_checked_out": pool.checkedout(),
            "database_url": str(engine.url),
        }
        
        logger.debug(f"Database health: {health_status}")
        return health_status
        
    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "error": str(e),
        }
        logger.error(f"Database health check failed: {e}")
        return error_status



def init_db() -> None:
    """
    Initialize database tables by creating all Base metadata.

    Creates all ORM tables defined in Base.metadata.
    Safe to call multiple times (idempotent).

    Raises:
        RuntimeError: If table creation fails

    """
    try:
        logger.info("Initializing database tables")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")

    except OperationalError as e:
        error_msg = f"Operational error initializing database tables: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except SQLAlchemyError as e:
        error_msg = f"SQLAlchemy error initializing database tables: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to initialize database tables: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
