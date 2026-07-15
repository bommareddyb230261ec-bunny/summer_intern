from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)

from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ==========================
# Database Engine
# ==========================

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,          # Set to False in production
    future=True
)


# ==========================
# Session Factory
# ==========================

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ==========================
# Base Model
# ==========================

class Base(DeclarativeBase):
    pass


# ==========================
# Database Dependency
# ==========================

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()