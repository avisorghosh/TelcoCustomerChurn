"""Unit tests for model restoration and rollback functionality."""

import pytest

from churn_prediction.models.restore import restore_model_from_dir
from churn_prediction.models.serialization import save_artifacts
from churn_prediction.models.trainer import train_baseline


def test_restore_model_from_dir_success(tmp_path, synthetic_dataset):
    """Verify restoring model artifacts from a source backup directory."""
    raw_csv = tmp_path / "Telco-Customer-Churn.csv"
    synthetic_dataset.to_csv(raw_csv, index=False)

    source_dir = tmp_path / "source_backup"
    target_dir = tmp_path / "target_active"

    pipeline, metadata, _ = train_baseline(
        data_path_override=raw_csv,
        log_to_mlflow=False,
        output_dir_override=tmp_path / "train_artifacts",
    )
    save_artifacts(pipeline, metadata, output_dir=source_dir)

    tgt_pipe_p, tgt_meta_p, restored_meta = restore_model_from_dir(
        source_dir=source_dir,
        target_dir=target_dir,
    )

    assert tgt_pipe_p.exists()
    assert tgt_meta_p.exists()
    assert restored_meta["model_name"] == metadata["model_name"]


def test_restore_model_from_dir_missing_source_fails(tmp_path):
    """Verify failure when source backup directory is missing artifact files."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    target_dir = tmp_path / "target_active"

    with pytest.raises(FileNotFoundError, match="Source directory"):
        restore_model_from_dir(source_dir=empty_dir, target_dir=target_dir)
