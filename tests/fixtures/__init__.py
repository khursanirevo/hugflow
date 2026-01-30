"""
Test fixtures for Hugflow.
"""

import pytest
from pathlib import Path

from hugflow.config import Config, DatasetSpec, RemoveSpec


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def sample_add_yaml(tmp_path):
    """Create a sample add dataset YAML file."""
    import yaml

    yaml_path = tmp_path / "add_dataset.yaml"
    data = {
        "hf_id": "mozilla-foundation/common_voice_11_0",
        "description": "Malay ASR training data",
        "subset": "ms",
        "split": "train",
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    return yaml_path


@pytest.fixture
def sample_remove_yaml(tmp_path):
    """Create a sample remove dataset YAML file."""
    import yaml

    yaml_path = tmp_path / "remove_dataset.yaml"
    data = {
        "hf_id": "mozilla-foundation/common_voice_11_0",
        "reason": "No longer needed",
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    return yaml_path


@pytest.fixture
def sample_dataset_spec():
    """Create a sample dataset specification."""
    return DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Malay ASR training data",
        subset="ms",
        split="train",
    )


@pytest.fixture
def sample_remove_spec():
    """Create a sample removal specification."""
    return RemoveSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        reason="No longer needed",
    )
