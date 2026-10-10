"""
Supabase Storage client for uploading user images.

Uploads images to a Supabase Storage bucket and returns public URLs.
Falls back gracefully if Supabase is not configured (local development).
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.config import get_settings

_client = None
_bucket_name = None
_configured = False
_last_error: str | None = None


def storage_status() -> dict:
    """Snapshot of storage state for /health."""
    return {
        "configured": _configured,
        "connected": _client is not None,
        "error": _last_error,
    }


async def init_storage() -> None:
    """Initialize the Supabase Storage client. Never raises."""
    global _client, _bucket_name, _configured, _last_error

    settings = get_settings()

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        _configured = False
        print("  ⚠️  Supabase not configured — image uploads will be skipped")
        return
    _configured = True

    try:
        from supabase import create_client

        import asyncio

        client = await asyncio.to_thread(create_client, settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
        bucket = settings.SUPABASE_BUCKET or "coffee-uploads"

        # Ensure bucket exists (network call → thread)
        def _ensure_bucket():
            try:
                client.storage.get_bucket(bucket)
            except Exception:
                client.storage.create_bucket(bucket, options={"public": True})

        await asyncio.to_thread(_ensure_bucket)

        _client = client
        _bucket_name = bucket
        _last_error = None
        print(f"  ✅ Supabase Storage initialized (bucket: {_bucket_name})")
    except ImportError:
        _last_error = "supabase package not installed"
        print("  ⚠️  supabase package not installed — image uploads disabled")
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"[:300]
        print(f"  ⚠️  Supabase Storage init failed: {_last_error}")


async def upload_image(
    image: Image.Image,
    original_filename: str | None = None,
) -> str | None:
    """
    Upload an image to Supabase Storage.

    Args:
        image: PIL Image to upload.
        original_filename: Original filename for extension detection.

    Returns:
        Public URL of the uploaded image, or None if upload failed.
    """
    global _last_error

    if _client is None:
        # Startup init may have failed (e.g. project was paused); retry once per call.
        if _configured:
            await init_storage()
        if _client is None:
            return None

    try:
        import asyncio

        # Generate unique filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = Path(original_filename).suffix.lower() if original_filename else ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        unique_name = f"{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
        storage_path = f"predictions/{unique_name}"

        # Convert to bytes
        buffer = io.BytesIO()
        img_format = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}[ext[1:]]
        image.save(buffer, format=img_format, quality=90)
        content_type = f"image/{img_format.lower()}"

        # The supabase client is synchronous; keep the event loop free.
        await asyncio.to_thread(
            _client.storage.from_(_bucket_name).upload,
            storage_path,
            buffer.getvalue(),
            {"content-type": content_type},
        )

        settings = get_settings()
        _last_error = None
        return f"{settings.SUPABASE_URL}/storage/v1/object/public/{_bucket_name}/{storage_path}"

    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"[:300]
        print(f"  ⚠️  Image upload failed: {_last_error}")
        return None
