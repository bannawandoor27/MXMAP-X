"""Database session management."""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings

# Convert postgresql:// to postgresql+asyncpg://
database_url = settings.database_url_str.replace("postgresql://", "postgresql+asyncpg://")

# SQLite doesn't support connection pool settings (pool_size, max_overflow)
# Only apply them when using PostgreSQL
_is_sqlite = database_url.startswith("sqlite")

if _is_sqlite:
    # SQLite: use StaticPool for async compatibility
    from sqlalchemy.pool import StaticPool
    engine = create_async_engine(
        database_url,
        echo=settings.DATABASE_ECHO,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    # PostgreSQL: full connection pool
    engine = create_async_engine(
        database_url,
        echo=settings.DATABASE_ECHO,
        future=True,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for SQLAlchemy models
Base = declarative_base()
