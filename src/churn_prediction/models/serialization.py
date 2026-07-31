"""Model and pipeline serialization/deserialization utilities."""

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    output_dir: str | Path,
    pipeline_filename: str = "baseline_pipeline.joblib",
    metadata_filename: str = "baseline_metadata.json",
) -> tuple[Path, Path]:
    """Save trained pipeline and model metadata to specified directory.

    Args:
        pipeline: Trained scikit-learn Pipeline instance.
        metadata: Dictionary containing model metadata and training parameters.
        output_dir: Target directory path for artifacts.
        pipeline_filename: Filename for serialized pipeline.
        metadata_filename: Filename for metadata JSON.

    Returns:
        Tuple of (pipeline_path, metadata_path).
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = target_dir / pipeline_filename
    metadata_path = target_dir / metadata_filename

    joblib.dump(pipeline, pipeline_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return pipeline_path, metadata_path


def load_pipeline(pipeline_path: str | Path) -> Pipeline:
    """Load a serialized scikit-learn pipeline from file.

    Args:
        pipeline_path: Path to serialized pipeline joblib file.

    Returns:
        Loaded scikit-learn Pipeline instance.
    """
    path = Path(pipeline_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline artifact not found at: {path}")
    pipeline: Pipeline = joblib.load(path)
    return pipeline


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load model metadata from JSON file.

    Args:
        metadata_path: Path to metadata JSON file.

    Returns:
        Metadata dictionary.
    """
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata artifact not found at: {path}")
    with open(path, encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)
    return metadata


def load_artifacts(
    output_dir: str | Path,
    pipeline_filename: str = "baseline_pipeline.joblib",
    metadata_filename: str = "baseline_metadata.json",
) -> tuple[Pipeline, dict[str, Any]]:
    """Load both pipeline and metadata from an artifact directory.

    Args:
        output_dir: Directory containing artifacts.
        pipeline_filename: Filename of pipeline artifact.
        metadata_filename: Filename of metadata artifact.

    Returns:
        Tuple of (loaded_pipeline, metadata_dict).
    """
    target_dir = Path(output_dir)
    pipeline_path = target_dir / pipeline_filename
    metadata_path = target_dir / metadata_filename

    pipeline = load_pipeline(pipeline_path)
    metadata = load_metadata(metadata_path)

    return pipeline, metadata
