"""
Async database layer using SQLAlchemy 2.0 + asyncpg for PostgreSQL.

Falls back to aiosqlite for local development.
Manages the predictions log table with full lifecycle management.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    String,
    Text,
    select,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

# ── ORM Base ────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Logged prediction record."""

    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    filename = Column(String(255), nullable=True)
    analysis_type = Column(String(20), nullable=False)  # "roast" or "defect"
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(JSON, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    image_url = Column(Text, nullable=True)


# ── Engine & Session Factory ────────────────────────────────────────────

_engine = None
_session_factory = None


def _get_connect_args(url: str) -> dict:
    """Get driver-specific connection arguments."""
    if "asyncpg" in url:
        # Supabase pooler doesn't support prepared statements
        return {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
        }
    return {}


async def init_db() -> None:
    """Initialize the database engine and create tables."""
    global _engine, _session_factory

    settings = get_settings()
    url = settings.DATABASE_URL

    _engine = create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
        connect_args=_get_connect_args(url),
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables (safe for existing tables)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("  ✅ Database initialized")


async def close_db() -> None:
    """Dispose of the database engine."""
    global _engine
    if _engine:
        await _engine.dispose()
        print("  ✅ Database connection closed")


def get_session() -> AsyncSession:
    """Get a new async session."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


# ── CRUD Operations ─────────────────────────────────────────────────────


async def log_prediction(
    filename: str | None,
    analysis_type: str,
    predicted_class: str,
    confidence: float,
    probabilities: dict | None = None,
    inference_time_ms: float | None = None,
    image_url: str | None = None,
) -> str:
    """
    Log a prediction to the database.

    Returns:
        The prediction ID.
    """
    prediction = Prediction(
        filename=filename,
        analysis_type=analysis_type,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities_json=probabilities,
        inference_time_ms=inference_time_ms,
        image_url=image_url,
    )

    async with get_session() as session:
        session.add(prediction)
        await session.commit()
        return prediction.id


async def get_history(limit: int = 20, offset: int = 0) -> list[dict]:
    """Get recent predictions, newest first."""
    async with get_session() as session:
        result = await session.execute(
            select(Prediction)
            .order_by(Prediction.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = result.scalars().all()

        return [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "filename": row.filename,
                "analysis_type": row.analysis_type,
                "predicted_class": row.predicted_class,
                "confidence": row.confidence,
                "probabilities": row.probabilities_json,
                "inference_time_ms": row.inference_time_ms,
                "image_url": row.image_url,
            }
            for row in rows
        ]


async def get_stats() -> dict:
    """Get aggregate prediction statistics."""
    async with get_session() as session:
        # Total predictions
        total_result = await session.execute(
            select(func.count(Prediction.id))
        )
        total = total_result.scalar() or 0

        # Average inference time
        avg_time_result = await session.execute(
            select(func.avg(Prediction.inference_time_ms))
        )
        avg_time = avg_time_result.scalar()

        # Predictions by analysis type
        type_result = await session.execute(
            select(
                Prediction.analysis_type,
                func.count(Prediction.id),
            ).group_by(Prediction.analysis_type)
        )
        by_type = {row[0]: row[1] for row in type_result.all()}

        # Most common predictions
        class_result = await session.execute(
            select(
                Prediction.predicted_class,
                func.count(Prediction.id),
            )
            .group_by(Prediction.predicted_class)
            .order_by(func.count(Prediction.id).desc())
            .limit(10)
        )
        top_classes = {row[0]: row[1] for row in class_result.all()}

        return {
            "total_predictions": total,
            "avg_inference_time_ms": round(avg_time, 2) if avg_time else 0,
            "predictions_by_type": by_type,
            "top_predicted_classes": top_classes,
        }
