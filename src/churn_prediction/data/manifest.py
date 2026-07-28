"""Source data manifest handling and checksum tracking."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class SourceManifest(BaseModel):
    """Manifest for tracking dataset provenance and checksums."""

    source_path: str = Field(description="Path to raw source data file.")
    received_timestamp: str = Field(
        description="ISO 8601 UTC timestamp when dataset was processed."
    )
    row_count: int = Field(
        ge=0, description="Total number of data rows (excluding header)."
    )
    file_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the source data file.",
    )
    schema_version: str = Field(
        default="1.0.0", description="Data contract schema version applied."
    )
    quality_report_path: str | None = Field(
        default=None, description="Path to generated data quality report."
    )


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def count_csv_rows(file_path: Path) -> int:
    """Count non-empty data rows in a CSV file excluding header.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Number of data rows.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return max(0, len(lines) - 1) if lines else 0


def create_source_manifest(
    file_path: str | Path,
    schema_version: str = "1.0.0",
    quality_report_path: str | Path | None = None,
) -> SourceManifest:
    """Create a source data manifest for a data file.

    Args:
        file_path: Path to the source CSV file.
        schema_version: Schema version for the contract.
        quality_report_path: Optional path to quality report JSON.

    Returns:
        SourceManifest object containing checksum, row count, and metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found at: {path}")

    sha256 = compute_sha256(path)
    row_count = count_csv_rows(path)
    timestamp = datetime.now(timezone.utc).isoformat()

    return SourceManifest(
        source_path=str(path.resolve()),
        received_timestamp=timestamp,
        row_count=row_count,
        file_sha256=sha256,
        schema_version=schema_version,
        quality_report_path=str(Path(quality_report_path).resolve())
        if quality_report_path
        else None,
    )
