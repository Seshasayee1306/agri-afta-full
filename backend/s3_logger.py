import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _enabled() -> bool:
    return bool(os.getenv("S3_BUCKET"))


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _s3_client():
    import boto3  # lazy import so local runs without boto3 still work

    # Use default AWS credential chain (env vars / IAM role / etc.)
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"))


def put_json(
    *,
    key: str,
    payload: Dict[str, Any],
    bucket: Optional[str] = None,
) -> bool:
    if not _enabled():
        return False

    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        return False

    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        return True
    except Exception as e:
        print("⚠️ S3 put_json failed:", e)
        return False


def put_text(
    *,
    key: str,
    text: str,
    bucket: Optional[str] = None,
    content_type: str = "text/plain",
) -> bool:
    if not _enabled():
        return False

    bucket = bucket or os.getenv("S3_BUCKET")
    if not bucket:
        return False

    try:
        _s3_client().put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType=content_type,
        )
        return True
    except Exception as e:
        print("⚠️ S3 put_text failed:", e)
        return False


def make_key(prefix: str, suffix: str) -> str:
    prefix = (prefix or "").strip().strip("/")
    base = os.getenv("S3_PREFIX", "agri").strip().strip("/")
    ts = _utc_now_compact()
    rid = uuid.uuid4().hex
    if prefix:
        return f"{base}/{prefix}/{ts}_{rid}.{suffix.lstrip('.')}"
    return f"{base}/{ts}_{rid}.{suffix.lstrip('.')}"

