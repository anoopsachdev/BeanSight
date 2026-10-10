"""Connection-level regression tests for Supabase PostgreSQL and Storage."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from PIL import Image

import app.database as database
import app.storage as storage


def test_pg_bouncer_connect_args_disable_caches_and_use_unique_names():
    args = database._get_connect_args(
        "postgresql+asyncpg://user:pass@host:6543/postgres"
    )

    assert args["statement_cache_size"] == 0
    assert args["prepared_statement_cache_size"] == 0
    assert args["prepared_statement_name_func"]() != args[
        "prepared_statement_name_func"
    ]()
    assert args["timeout"] == 10


def test_non_postgres_connect_args_are_empty():
    assert database._get_connect_args("sqlite+aiosqlite:///data/test.db") == {}


def test_storage_strips_trailing_slash_from_supabase_url():
    class Settings:
        SUPABASE_URL = "https://example.supabase.co///"
        SUPABASE_SERVICE_KEY = "service-key"
        SUPABASE_BUCKET = "coffee-uploads"

    client = MagicMock()
    client.storage.get_bucket.return_value = {"name": "coffee-uploads"}

    storage._client = None
    storage._bucket_name = None
    storage._configured = False
    storage._last_error = None

    with (
        patch.object(storage, "get_settings", return_value=Settings()),
        patch("supabase.create_client", return_value=client) as create_client,
    ):
        asyncio.run(storage.init_storage())

    create_client.assert_called_once_with(
        "https://example.supabase.co", "service-key"
    )
    client.storage.get_bucket.assert_called_once_with("coffee-uploads")
    assert storage.storage_status()["connected"] is True


def test_upload_public_url_has_one_separator():
    class Settings:
        SUPABASE_URL = "https://example.supabase.co/"
        SUPABASE_SERVICE_KEY = "service-key"
        SUPABASE_BUCKET = "coffee-uploads"

    bucket = MagicMock()
    client = MagicMock()
    client.storage.from_.return_value = bucket
    storage._client = client
    storage._bucket_name = "coffee-uploads"
    storage._configured = True

    with patch.object(storage, "get_settings", return_value=Settings()):
        url = asyncio.run(
            storage.upload_image(
                Image.new("RGB", (8, 8), "brown"),
                "bean.jpg",
            )
        )

    assert url.startswith(
        "https://example.supabase.co/storage/v1/object/public/"
        "coffee-uploads/predictions/"
    )
    assert ".co//storage" not in url
    bucket.upload.assert_called_once()
