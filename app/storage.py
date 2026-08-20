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


async def init_storage() -> None:
    """Initialize the Supabase Storage client."""
    global _client, _bucket_name

    settings = get_settings()

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        print("  ⚠️  Supabase not configured — image uploads will be skipped")
        return

    try:
        from supabase import create_client

        _client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY,
        )
        _bucket_name = settings.SUPABASE_BUCKET

        # Ensure bucket exists
        try:
            _client.storage.get_bucket(_bucket_name)
        except Exception:
            _client.storage.create_bucket(
                _bucket_name,
                options={"public": True},
            )

        print(f"  ✅ Supabase Storage initialized (bucket: {_bucket_name})")
    except ImportError:
        print("  ⚠️  supabase package not installed — image uploads disabled")
    except Exception as e:
        print(f"  ⚠️  Supabase Storage init failed: {e}")


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
    if _client is None or _bucket_name is None:
        return None

    try:
        # Generate unique filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = Path(original_filename).suffix if original_filename else ".jpg"
        unique_name = f"{timestamp}_{uuid.uuid4().hex[:8]}{ext}"
        storage_path = f"predictions/{unique_name}"

        # Convert to bytes
        buffer = io.BytesIO()
        img_format = "JPEG" if ext.lower() in (".jpg", ".jpeg") else "PNG"
        image.save(buffer, format=img_format, quality=90)
        buffer.seek(0)

        # Upload
        content_type = f"image/{img_format.lower()}"
        _client.storage.from_(_bucket_name).upload(
            path=storage_path,
            file=buffer.getvalue(),
            file_options={"content-type": content_type},
        )

        # Get public URL
        settings = get_settings()
        public_url = (
            f"{settings.SUPABASE_URL}/storage/v1/object/public/"
            f"{_bucket_name}/{storage_path}"
        )
        return public_url

    except Exception as e:
        print(f"  ⚠️  Image upload failed: {e}")
        return None
