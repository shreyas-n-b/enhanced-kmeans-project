import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://postgres:postgres123@localhost/enhanced_kmeans_db"

async def migrate():
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS silhouette_score FLOAT;"))
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS davies_bouldin_score FLOAT;"))
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS num_outliers INTEGER;"))
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS variance FLOAT;"))
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS iterations INTEGER;"))
        await conn.execute(text("ALTER TABLE results ADD COLUMN IF NOT EXISTS history JSONB;"))
    print("Migration successful")

if __name__ == "__main__":
    asyncio.run(migrate())
