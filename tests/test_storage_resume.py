"""
Tests for storage resume functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hugflow.config import DatasetSpec
from hugflow.storage import ResumeState, StorageManager, StorageError


@pytest.fixture
def temp_storage_dir():
    """Create a temporary storage directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_storage_dir):
    """Create a mock config with temporary storage."""
    config = MagicMock()
    config.storage.root = str(temp_storage_dir / "datasets")
    config.storage.state_dir = str(temp_storage_dir / ".hugflow-state")
    return config


@pytest.fixture
def storage_manager(mock_config):
    """Create a StorageManager instance."""
    return StorageManager(mock_config)


@pytest.fixture
def sample_spec():
    """Create a sample DatasetSpec."""
    return DatasetSpec(
        hf_id="test-org/test-dataset",
        description="Test dataset",
        revision="main",
        subset=None,
        split="train",
    )


class TestResumeState:
    """Tests for ResumeState dataclass."""

    def test_resume_state_initialization(self):
        """Test ResumeState initialization with default values."""
        state = ResumeState()
        assert state.completed_rows == set()
        assert state.completed_audio_files == set()
        assert state.completed_json_files == set()
        assert state.last_successful_row == -1
        assert state.expected_total_rows == 0
        assert state.incomplete_row is None
        assert state.version == "2.0"

    def test_resume_state_to_dict(self):
        """Test ResumeState to_dict conversion."""
        state = ResumeState(
            completed_rows={0, 1, 2},
            completed_audio_files={"file_0.mp3", "file_1.mp3"},
            completed_json_files={"0.json", "1.json"},
            last_successful_row=2,
            expected_total_rows=100,
            incomplete_row=None,
            version="2.0",
        )
        data = state.to_dict()

        assert data["completed_rows"] == [0, 1, 2]
        assert sorted(data["completed_audio_files"]) == sorted(["file_0.mp3", "file_1.mp3"])
        assert sorted(data["completed_json_files"]) == sorted(["0.json", "1.json"])
        assert data["last_successful_row"] == 2
        assert data["expected_total_rows"] == 100
        assert data["incomplete_row"] is None
        assert data["version"] == "2.0"

    def test_resume_state_from_dict(self):
        """Test ResumeState from_dict conversion."""
        data = {
            "completed_rows": [0, 1, 2],
            "completed_audio_files": ["file_0.mp3", "file_1.mp3"],
            "completed_json_files": ["0.json", "1.json"],
            "last_successful_row": 2,
            "expected_total_rows": 100,
            "incomplete_row": None,
            "version": "2.0",
        }
        state = ResumeState.from_dict(data)

        assert state.completed_rows == {0, 1, 2}
        assert state.completed_audio_files == {"file_0.mp3", "file_1.mp3"}
        assert state.completed_json_files == {"0.json", "1.json"}
        assert state.last_successful_row == 2
        assert state.expected_total_rows == 100
        assert state.incomplete_row is None
        assert state.version == "2.0"

    def test_resume_state_from_dict_legacy(self):
        """Test ResumeState from_dict with legacy format (version 1.0)."""
        data = {
            "completed_rows": [0, 1, 2],
            "last_successful_row": 2,
        }
        state = ResumeState.from_dict(data)

        assert state.completed_rows == {0, 1, 2}
        assert state.last_successful_row == 2
        assert state.version == "1.0"


class TestStorageResumeMethods:
    """Tests for StorageManager resume methods."""

    def test_save_and_load_resume_state(self, storage_manager, sample_spec, mock_config):
        """Test saving and loading resume state."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Create a resume state
        state = ResumeState(
            completed_rows={0, 1, 2},
            completed_audio_files={"file_0.mp3", "file_1.mp3"},
            completed_json_files={"0.json", "1.json"},
            last_successful_row=2,
            expected_total_rows=100,
        )

        # Save the state
        storage_manager.save_resume_state(sample_spec, state)

        # Load the state
        loaded_state = storage_manager.load_resume_state(sample_spec)

        assert loaded_state is not None
        assert loaded_state.completed_rows == {0, 1, 2}
        assert loaded_state.last_successful_row == 2
        assert loaded_state.expected_total_rows == 100

    def test_load_resume_state_not_found(self, storage_manager, sample_spec, mock_config):
        """Test loading resume state when it doesn't exist."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Try to load non-existent state
        loaded_state = storage_manager.load_resume_state(sample_spec)

        assert loaded_state is None

    def test_mark_row_complete(self, storage_manager, sample_spec, mock_config):
        """Test marking a row as complete."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Mark rows as complete
        storage_manager.mark_row_complete(
            sample_spec,
            row_idx=0,
            audio_filename="file_0.mp3",
            json_filename="0.json",
            expected_total=100,
        )

        storage_manager.mark_row_complete(
            sample_spec,
            row_idx=1,
            audio_filename="file_1.mp3",
            json_filename="1.json",
            expected_total=100,
        )

        # Load and verify
        state = storage_manager.load_resume_state(sample_spec)
        assert state is not None
        assert state.completed_rows == {0, 1}
        assert state.completed_audio_files == {"file_0.mp3", "file_1.mp3"}
        assert state.completed_json_files == {"0.json", "1.json"}
        assert state.last_successful_row == 1
        assert state.expected_total_rows == 100
        assert state.incomplete_row is None

    def test_mark_row_incomplete(self, storage_manager, sample_spec, mock_config):
        """Test marking a row as incomplete."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Mark row as incomplete
        storage_manager.mark_row_incomplete(sample_spec, row_idx=5)

        # Load and verify
        state = storage_manager.load_resume_state(sample_spec)
        assert state is not None
        assert state.incomplete_row == 5

    def test_get_download_resume_point(self, storage_manager, sample_spec, mock_config):
        """Test getting download resume point."""
        # Initialize directories
        storage_manager.initialize_directories()

        # No resume state - should return 0
        assert storage_manager.get_download_resume_point(sample_spec) == 0

        # Mark some rows as complete
        storage_manager.mark_row_complete(sample_spec, row_idx=0, expected_total=100)
        storage_manager.mark_row_complete(sample_spec, row_idx=1, expected_total=100)
        storage_manager.mark_row_complete(sample_spec, row_idx=2, expected_total=100)

        # Should resume from row 3 (next after last successful)
        assert storage_manager.get_download_resume_point(sample_spec) == 3

    def test_should_skip_row(self, storage_manager, sample_spec, mock_config):
        """Test checking if a row should be skipped."""
        # Initialize directories
        storage_manager.initialize_directories()

        # No resume state - no rows should be skipped
        assert not storage_manager.should_skip_row(sample_spec, 0)
        assert not storage_manager.should_skip_row(sample_spec, 1)

        # Mark some rows as complete
        storage_manager.mark_row_complete(sample_spec, row_idx=0, expected_total=100)
        storage_manager.mark_row_complete(sample_spec, row_idx=2, expected_total=100)

        # Only completed rows should be skipped
        assert storage_manager.should_skip_row(sample_spec, 0)
        assert not storage_manager.should_skip_row(sample_spec, 1)
        assert storage_manager.should_skip_row(sample_spec, 2)
        assert not storage_manager.should_skip_row(sample_spec, 3)

    def test_save_resume_state_atomic_write(self, storage_manager, sample_spec, mock_config):
        """Test that resume state is saved atomically (temp file + rename)."""
        # Initialize directories
        storage_manager.initialize_directories()

        state = ResumeState(
            completed_rows={0, 1, 2},
            last_successful_row=2,
            expected_total_rows=100,
        )

        # Save the state
        storage_manager.save_resume_state(sample_spec, state)

        # Check that .tmp file was not left behind
        progress_file = storage_manager.progress_dir / f"{sample_spec.storage_name}.json"
        tmp_file = storage_manager.progress_dir / f"{sample_spec.storage_name}.json.tmp"

        assert progress_file.exists()
        assert not tmp_file.exists()

    def test_is_download_complete_with_v2_progress(self, storage_manager, sample_spec, mock_config):
        """Test is_download_complete with v2 progress format."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Create resume state showing incomplete download
        state = ResumeState(
            completed_rows={0, 1, 2},
            last_successful_row=2,
            expected_total_rows=100,
        )
        storage_manager.save_resume_state(sample_spec, state)

        # Should not be complete
        assert not storage_manager.is_download_complete(sample_spec, audio_count=3, json_count=3)

        # Mark all rows as complete
        state.completed_rows = set(range(100))
        state.last_successful_row = 99
        storage_manager.save_resume_state(sample_spec, state)

        # Should be complete
        assert storage_manager.is_download_complete(sample_spec, audio_count=100, json_count=100)

    def test_is_download_complete_with_incomplete_marker(self, storage_manager, sample_spec, mock_config):
        """Test is_download_complete with incomplete row marker."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Create resume state with incomplete marker
        state = ResumeState(
            completed_rows={0, 1, 2},
            last_successful_row=2,
            expected_total_rows=100,
            incomplete_row=3,  # Row 3 failed
        )
        storage_manager.save_resume_state(sample_spec, state)

        # Should not be complete even with many files
        assert not storage_manager.is_download_complete(sample_spec, audio_count=50, json_count=50)

    def test_is_download_complete_legacy_format(self, storage_manager, sample_spec, mock_config):
        """Test is_download_complete with legacy progress format."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Create legacy progress file (no resume_data)
        progress = {
            "dataset_name": sample_spec.hf_id,
            "status": "downloading",
            "downloaded_files": 100,
            "total_files": 100,
            "percent_complete": 100.0,
        }
        storage_manager.save_progress(sample_spec, progress)

        # Should be complete (matching counts)
        assert storage_manager.is_download_complete(sample_spec, audio_count=100, json_count=100)

        # Should not be complete (mismatched counts)
        assert not storage_manager.is_download_complete(sample_spec, audio_count=50, json_count=100)

    def test_dataset_exists_with_check_complete(self, storage_manager, sample_spec, mock_config, temp_storage_dir):
        """Test dataset_exists with check_complete parameter."""
        # Initialize directories and create dataset directory
        storage_manager.initialize_directories()

        dataset_path = storage_manager.storage_root / sample_spec.storage_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "audio").mkdir(exist_ok=True)
        (dataset_path / "json").mkdir(exist_ok=True)

        # Create some files
        (dataset_path / "audio" / "file_0.mp3").touch()
        (dataset_path / "json" / "0.json").touch()

        # Without check_complete, should return True (has files)
        assert storage_manager.dataset_exists(sample_spec, check_complete=False)

        # With check_complete and no progress, should return True (backward compatible)
        assert storage_manager.dataset_exists(sample_spec, check_complete=True)

        # With progress showing incomplete, should return False
        state = ResumeState(
            completed_rows={0},
            last_successful_row=0,
            expected_total_rows=100,
        )
        storage_manager.save_resume_state(sample_spec, state)

        assert not storage_manager.dataset_exists(sample_spec, check_complete=True)

    def test_resume_state_preserves_existing_progress_metadata(self, storage_manager, sample_spec, mock_config):
        """Test that saving resume state preserves existing progress metadata."""
        # Initialize directories
        storage_manager.initialize_directories()

        # Create initial progress file with metadata
        progress = {
            "dataset_name": sample_spec.hf_id,
            "status": "downloading",
            "downloaded_files": 10,
            "total_files": 100,
            "percent_complete": 10.0,
            "start_time": "2026-02-09T10:00:00",
            "custom_field": "should_be_preserved",
        }
        storage_manager.save_progress(sample_spec, progress)

        # Save resume state
        state = ResumeState(
            completed_rows={0, 1, 2},
            last_successful_row=2,
            expected_total_rows=100,
        )
        storage_manager.save_resume_state(sample_spec, state)

        # Load progress and verify metadata is preserved
        loaded_progress = storage_manager.load_progress(sample_spec)
        assert loaded_progress is not None
        assert loaded_progress["dataset_name"] == sample_spec.hf_id
        assert loaded_progress["status"] == "downloading"
        assert loaded_progress["custom_field"] == "should_be_preserved"
        assert "resume_data" in loaded_progress
        assert loaded_progress["resume_data"]["version"] == "2.0"
