"""Optional Cloudinary upload. Local disk remains the default."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def cloudinary_configured() -> bool:
    return bool(settings.cloudinary_url)


def upload_file(path: Path, public_id: str) -> str | None:
    if not settings.cloudinary_url:
        return None
    parsed = urlparse(settings.cloudinary_url)
    if parsed.scheme != "cloudinary" or not parsed.hostname or not parsed.username or parsed.password is None:
        logger.error("CLOUDINARY_URL is set but is not a cloudinary:// URL")
        return None
    cloud_name = parsed.hostname
    api_key = parsed.username
    api_secret = parsed.password
    timestamp = int(time.time())
    signature_base = f"public_id={public_id}&timestamp={timestamp}{api_secret}"
    signature = hashlib.sha1(signature_base.encode()).hexdigest()
    data = {"public_id": public_id, "timestamp": timestamp, "api_key": api_key, "signature": signature}
    url = f"https://api.cloudinary.com/v1_1/{cloud_name}/video/upload"
    try:
        with path.open("rb") as handle, httpx.Client(timeout=120) as client:
            response = client.post(url, data=data, files={"file": handle})
        response.raise_for_status()
        payload = response.json()
        return payload.get("secure_url")
    except Exception as exc:
        logger.error("cloudinary upload failed: %s", exc)
        return None
