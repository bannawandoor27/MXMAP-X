import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import async_session_maker
from app.api.v1.endpoints.advanced import explore_chemistry_space

async def main():
    async with async_session_maker() as session:
        try:
            res = await explore_chemistry_space(db=session)
            print("SUCCESS")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
