"""
Tests for Hugflow configuration.
"""

import pytest
from pydantic import ValidationError

from hugflow.config import DatasetSpec, RemoveSpec, Config


def test_dataset_spec_minimal():
    """Test creating a minimal dataset specification."""
    spec = DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Malay ASR training data",
    )

    assert spec.hf_id == "mozilla-foundation/common_voice_11_0"
    assert spec.description == "Malay ASR training data"
    assert spec.revision == "main"  # default
    assert spec.subset is None  # default
    assert spec.split == "train"  # default
    assert spec.audio_column == "audio"  # default
    assert spec.text_column == "text"  # default


def test_dataset_spec_full():
    """Test creating a full dataset specification with all options."""
    spec = DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Malay ASR training data",
        revision="v1.0",
        subset="ms",
        split="test",
        audio_column="audio_data",
        text_column="transcript",
    )

    assert spec.hf_id == "mozilla-foundation/common_voice_11_0"
    assert spec.description == "Malay ASR training data"
    assert spec.revision == "v1.0"
    assert spec.subset == "ms"
    assert spec.split == "test"
    assert spec.audio_column == "audio_data"
    assert spec.text_column == "transcript"


def test_dataset_spec_validation_empty_hf_id():
    """Test that empty HF ID raises validation error."""
    with pytest.raises(ValidationError):
        DatasetSpec(
            hf_id="",
            description="Test",
        )


def test_dataset_spec_validation_invalid_hf_id_format():
    """Test that invalid HF ID format raises validation error."""
    with pytest.raises(ValidationError):
        DatasetSpec(
            hf_id="org/too/many/parts",
            description="Test",
        )


def test_dataset_spec_download_mode():
    """Test download_mode property."""
    # Full download (no subset)
    spec_full = DatasetSpec(
        hf_id="org/dataset",
        description="Test",
    )
    assert spec_full.download_mode == "full"

    # Specific download (with subset)
    spec_specific = DatasetSpec(
        hf_id="org/dataset",
        description="Test",
        subset="ms",
    )
    assert spec_specific.download_mode == "specific"


def test_dataset_spec_storage_name():
    """Test storage_name property."""
    # Full download
    spec_full = DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Test",
        revision="main",
    )
    assert spec_full.storage_name == "mozilla-foundation__common_voice_11_0__rev_main"

    # Specific download
    spec_specific = DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Test",
        revision="main",
        subset="ms",
        split="train",
    )
    assert spec_specific.storage_name == "mozilla-foundation__common_voice_11_0__subset_ms__split_train__rev_main"


def test_remove_spec():
    """Test creating a removal specification."""
    spec = RemoveSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        reason="No longer needed",
    )

    assert spec.hf_id == "mozilla-foundation/common_voice_11_0"
    assert spec.reason == "No longer needed"


def test_remove_spec_validation():
    """Test that empty fields raise validation errors."""
    with pytest.raises(ValidationError):
        RemoveSpec(
            hf_id="",
            reason="Test",
        )

    with pytest.raises(ValidationError):
        RemoveSpec(
            hf_id="org/dataset",
            reason="",
        )


def test_config_defaults():
    """Test that Config has sensible defaults."""
    config = Config()

    assert config.storage.root == "./datasets"
    assert config.storage.state_dir == ".hugflow-state"
    assert config.download.default_revision == "main"
    assert config.download.default_split == "train"
    assert config.download.progress_interval == 10000
    assert config.network.max_concurrent_downloads == 5
    assert config.validation.validate_hf_id is True
    assert config.behavior.auto_merge is False  # Changed default to False
    assert config.logging.level == "INFO"


def test_config_from_env(monkeypatch):
    """Test loading config from environment variables."""
    monkeypatch.setenv("STORAGE_ROOT", "/data/datasets")
    monkeypatch.setenv("MAX_CONCURRENT_DOWNLOADS", "10")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = Config()

    assert config.storage.root == "/data/datasets"
    assert config.network.max_concurrent_downloads == 10
    assert config.logging.level == "DEBUG"


def test_slack_config_enabled():
    """Test that Slack config is enabled when webhook URL is set."""
    from hugflow.config import SlackConfig

    # No webhook URL
    config_no_url = SlackConfig(webhook_url="")
    assert config_no_url.enabled is False

    # With webhook URL
    config_with_url = SlackConfig(webhook_url="https://hooks.slack.com/services/T00/B00/XXX")
    assert config_with_url.enabled is True
