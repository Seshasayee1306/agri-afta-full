import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SyncConfig:
    bucket: str
    region: str | None
    prefix: str
    key: str
    dest_path: str
    strict: bool


def _s3_client(region: str | None):
    import boto3
    if region:
        return boto3.client("s3", region_name=region)
    return boto3.client("s3")


def _default_key(prefix: str) -> str:
    prefix = (prefix or "agri").strip().strip("/")
    return f"{prefix}/models/final_model_latest.pkl"


def sync_latest_model() -> None:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("S3_BUCKET is not set")

    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "").strip() or None
    prefix = (os.getenv("S3_PREFIX") or "agri").strip().strip("/")
    key = (os.getenv("S3_MODEL_KEY") or _default_key(prefix)).strip().lstrip("/")

    dest_path = os.getenv("MODEL_DEST_PATH", "/models/final_model.pkl").strip()
    strict = (os.getenv("MODEL_SYNC_STRICT", "0").strip() == "1")

    cfg = SyncConfig(
        bucket=bucket,
        region=region,
        prefix=prefix,
        key=key,
        dest_path=dest_path,
        strict=strict,
    )

    os.makedirs(os.path.dirname(cfg.dest_path), exist_ok=True)

    s3 = _s3_client(cfg.region)
    try:
        s3.download_file(cfg.bucket, cfg.key, cfg.dest_path)
        print(f"✅ Downloaded model from s3://{cfg.bucket}/{cfg.key} -> {cfg.dest_path}")
    except Exception as e:
        msg = f"⚠️ Model download failed for s3://{cfg.bucket}/{cfg.key}: {e}"
        if cfg.strict:
            raise RuntimeError(msg)
        print(msg)


def main():
    sync_latest_model()


if __name__ == "__main__":
    main()

