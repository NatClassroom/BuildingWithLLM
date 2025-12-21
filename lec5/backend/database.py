"""Database connection and session management"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from sqlalchemy import event
from sqlalchemy.engine import Engine

# Database URL - matches the one in alembic/env.py
DATABASE_URL = "postgresql+asyncpg://username:pass123@localhost:5434/llm"

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    pool_pre_ping=True,
    connect_args={
        "server_settings": {
            "application_name": "vector_db_demo",
        }
    },
)


# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


async def init_db():
    """Initialize database with pgvector extension"""
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            # Register pgvector for this connection if not already registered
            try:
                from pgvector.asyncpg import register_vector

                # Get the underlying asyncpg connection from the session
                conn = await session.connection()
                # The connection object should have the asyncpg connection
                # Try to register if we can access it
                if hasattr(conn, "dbapi_connection") and conn.dbapi_connection:
                    try:
                        await register_vector(conn.dbapi_connection)
                    except Exception:
                        # Might already be registered, that's okay
                        pass
            except Exception as e:
                print(f"Warning: Could not register pgvector for session: {e}")
            yield session
        finally:
            await session.close()
