"""
Async database layer using SQLAlchemy 2.0 + asyncpg for PostgreSQL.

Falls back to aiosqlite for local development.
Manages the predictions log table with full lifecycle management.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

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
from sqlalchemy.exc import InterfaceError, OperationalError
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
    analysis_type = Column(String(20), nullable=False)  # "roast" | "defect" (green) | "roasted_defect"
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(JSON, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    image_url = Column(Text, nullable=True)


# ── Engine & Session Factory ────────────────────────────────────────────

_engine = None
_session_factory = None

# Connection health bookkeeping so a failed startup (e.g. a paused Supabase
# project) can be retried lazily instead of leaving the app broken until the
# next deploy.
_last_error: str | None = None
_last_attempt_monotonic: float = 0.0
_RETRY_INTERVAL_SECONDS = 30.0
_init_lock: asyncio.Lock | None = None


class DatabaseUnavailable(RuntimeError):
    """Raised when the database cannot be reached."""


from sqlalchemy.pool import NullPool

def _get_connect_args(url: str) -> dict:
    """Get driver-specific connection arguments."""
    if "asyncpg" in url:
        # Supabase's transaction-mode PgBouncer can reuse a server connection
        # whose prepared-statement names collide with another client. Disable
        # both asyncpg caches and generate globally unique names (the pattern
        # recommended by SQLAlchemy for PgBouncer).
        return {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
            "timeout": 10,
        }
    return {}


def _normalize_url(url: str) -> str:
    """Supabase gives postgresql:// URLs; SQLAlchemy async needs postgresql+asyncpg://."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _describe_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip().splitlines()[0]
    # /health is public: never leak the connection string, credentials or host.
    settings = get_settings()
    secrets = [settings.DATABASE_URL, settings.SUPABASE_SERVICE_KEY]
    try:
        parsed = urlsplit(_normalize_url(settings.DATABASE_URL))
        secrets += [parsed.password, parsed.username, parsed.hostname]
    except ValueError:
        pass
    for secret in secrets:
        if secret and secret in text:
            text = text.replace(secret, "[redacted]")
    return text[:300]


async def init_db() -> bool:
    """
    Initialize the database engine and create tables.

    Returns True on success. Never raises: failures are recorded in
    ``_last_error`` and retried lazily by ``ensure_db``.
    """
    global _engine, _session_factory, _last_error, _last_attempt_monotonic, _init_lock

    if _init_lock is None:
        _init_lock = asyncio.Lock()

    async with _init_lock:
        if _session_factory is not None:
            return True

        _last_attempt_monotonic = time.monotonic()
        settings = get_settings()
        url = _normalize_url(settings.DATABASE_URL)
        is_postgres = "asyncpg" in url

        engine_kwargs = {"echo": False, "connect_args": _get_connect_args(url)}
        if is_postgres:
            engine_kwargs["poolclass"] = NullPool
        else:
            engine_kwargs["pool_pre_ping"] = True
            # Make sure the parent directory exists for file-based SQLite.
            db_path = url.split("///")[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine = None
        try:
            engine = create_async_engine(url, **engine_kwargs)
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        except Exception as exc:  # noqa: BLE001 — any driver/network error
            _last_error = _describe_error(exc)
            print(f"  ⚠️  Database unavailable: {_last_error}")
            if engine is not None:
                await engine.dispose()
            return False

        _engine = engine
        _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        _last_error = None
        print("  ✅ Database initialized")
        return True


async def ensure_db() -> bool:
    """Return True if the DB is usable, retrying initialisation at most every 30s."""
    if _session_factory is not None:
        return True
    if time.monotonic() - _last_attempt_monotonic < _RETRY_INTERVAL_SECONDS:
        return False
    return await init_db()


def db_status() -> dict:
    """Snapshot of connection state for /health."""
    return {
        "connected": _session_factory is not None,
        "error": _last_error,
    }


async def close_db() -> None:
    """Dispose of the database engine."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        print("  ✅ Database connection closed")


def get_session() -> AsyncSession:
    """Get a new async session."""
    if _session_factory is None:
        raise DatabaseUnavailable(_last_error or "Database not initialized")
    return _session_factory()


def _mark_connection_error(exc: BaseException) -> None:
    """Drop the engine after a connection-level failure so the next call re-initialises."""
    global _engine, _session_factory, _last_error, _last_attempt_monotonic
    _last_error = _describe_error(exc)
    _last_attempt_monotonic = time.monotonic()
    _engine = None
    _session_factory = None
    print(f"  ⚠️  Database connection lost: {_last_error}")


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
    if not await ensure_db():
        raise DatabaseUnavailable(_last_error or "Database not initialized")

    prediction = Prediction(
        filename=filename,
        analysis_type=analysis_type,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities_json=probabilities,
        inference_time_ms=inference_time_ms,
        image_url=image_url,
    )

    try:
        async with get_session() as session:
            session.add(prediction)
            await session.commit()
            return prediction.id
    except (OperationalError, InterfaceError, OSError) as exc:
        _mark_connection_error(exc)
        raise DatabaseUnavailable(_last_error) from exc


async def get_history(limit: int = 20, offset: int = 0) -> list[dict]:
    """Get recent predictions, newest first."""
    if not await ensure_db():
        raise DatabaseUnavailable(_last_error or "Database not initialized")

    try:
        return await _get_history(limit, offset)
    except (OperationalError, InterfaceError, OSError) as exc:
        _mark_connection_error(exc)
        raise DatabaseUnavailable(_last_error) from exc


async def _get_history(limit: int, offset: int) -> list[dict]:
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
    if not await ensure_db():
        raise DatabaseUnavailable(_last_error or "Database not initialized")

    try:
        return await _get_stats()
    except (OperationalError, InterfaceError, OSError) as exc:
        _mark_connection_error(exc)
        raise DatabaseUnavailable(_last_error) from exc


async def _get_stats() -> dict:
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

        # Roast-level distribution (one "roast" row is logged per analysed image)
        roast_result = await session.execute(
            select(Prediction.predicted_class, func.count(Prediction.id))
            .where(Prediction.analysis_type == "roast")
            .group_by(Prediction.predicted_class)
        )
        roast_distribution = {row[0]: row[1] for row in roast_result.all()}

        # Defect distribution across both defect models, excluding "clean" outcomes
        defect_result = await session.execute(
            select(Prediction.predicted_class, func.count(Prediction.id))
            .where(Prediction.analysis_type.in_(["defect", "roasted_defect"]))
            .where(Prediction.predicted_class.notin_(["Normal", "No Defect", "None (Clean)"]))
            .group_by(Prediction.predicted_class)
            .order_by(func.count(Prediction.id).desc())
            .limit(10)
        )
        defect_distribution = {row[0]: row[1] for row in defect_result.all()}

        return {
            "total_predictions": total,
            "total_samples": by_type.get("roast", 0),
            "avg_inference_time_ms": round(avg_time, 2) if avg_time else 0,
            "predictions_by_type": by_type,
            "top_predicted_classes": top_classes,
            "roast_distribution": roast_distribution,
            "defect_distribution": defect_distribution,
        }
