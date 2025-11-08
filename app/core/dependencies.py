"""FastAPI dependency injection functions."""

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import async_session_maker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session.
    
    Yields:
        AsyncSession: SQLAlchemy async session
        
    Example:
        ```python
        @router.get("/devices")
        async def get_devices(db: AsyncSession = Depends(get_db)):
            ...
        ```
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
