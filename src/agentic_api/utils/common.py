from datetime import datetime, timezone

from uuid_utils import uuid7 as _uuid7


def uuid7_str(prefix: str = "") -> str:
    return f"{prefix}{_uuid7()}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
