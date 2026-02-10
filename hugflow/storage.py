"""
Storage and file system operations for Hugflow.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from structlog import get_logger

from hugflow.config import Config, DatasetSpec
from hugflow.constants import (
    ACTIVE_MANIFEST,
    ARCHIVED_MANIFEST,
    AUDIO_DIR,
    JSON_DIR,
    PROGRESS_DIR,
    STATE_DIR,
    STORAGE_ROOT,
    PROCESSED_ROOT,
    DOWNLOAD_MODE_FULL,
    DOWNLOAD_MODE_SPECIFIC,
)

logger = get_logger(__name__)


@dataclass
class ResumeState:
    """Track download resume state for incremental downloads."""

    completed_rows: Set[int] = field(default_factory=set)
    completed_audio_files: Set[str] = field(default_factory=set)
    completed_json_files: Set[str] = field(default_factory=set)
    last_successful_row: int = -1
    expected_total_rows: int = 0
    incomplete_row: Optional[int] = None
    version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "completed_rows": sorted(self.completed_rows),
            "completed_audio_files": sorted(self.completed_audio_files),
            "completed_json_files": sorted(self.completed_json_files),
            "last_successful_row": self.last_successful_row,
            "expected_total_rows": self.expected_total_rows,
            "incomplete_row": self.incomplete_row,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeState":
        """Create from dictionary."""
        return cls(
            completed_rows=set(data.get("completed_rows", [])),
            completed_audio_files=set(data.get("completed_audio_files", [])),
            completed_json_files=set(data.get("completed_json_files", [])),
            last_successful_row=data.get("last_successful_row", -1),
            expected_total_rows=data.get("expected_total_rows", 0),
            incomplete_row=data.get("incomplete_row"),
            version=data.get("version", "1.0"),
        )


class StorageError(Exception):
    """Base exception for storage errors."""

    pass


class DatasetExistsError(StorageError):
    """Dataset already exists in storage."""

    pass


class StorageManager:
    """Manager for dataset storage operations."""

    def __init__(self, config: Config):
        """
        Initialize storage manager.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.storage_root = Path(config.storage.root)
        self.state_dir = Path(config.storage.state_dir)
        self.progress_dir = self.state_dir / PROGRESS_DIR

    def initialize_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        log = logger.bind(
            storage_root=str(self.storage_root),
            state_dir=str(self.state_dir),
        )
        log.info("Initializing storage directories")

        try:
            # Create storage root
            self.storage_root.mkdir(parents=True, exist_ok=True)

            # Create state directory
            self.state_dir.mkdir(parents=True, exist_ok=True)

            # Create progress directory
            self.progress_dir.mkdir(parents=True, exist_ok=True)

            # Initialize manifests if they don't exist
            self._initialize_manifests()

            log.info("Storage directories initialized successfully")

        except Exception as e:
            log.error("Failed to initialize directories", error=str(e))
            raise StorageError(f"Failed to initialize directories: {e}") from e

    def _initialize_manifests(self) -> None:
        """Initialize manifest files if they don't exist."""
        active_manifest_path = self.state_dir / "active.json"
        archived_manifest_path = self.state_dir / "archived.json"

        if not active_manifest_path.exists():
            active_manifest_path.write_text(json.dumps({"datasets": [], "last_updated": None}, indent=2))

        if not archived_manifest_path.exists():
            archived_manifest_path.write_text(json.dumps({"operations": [], "last_updated": None}, indent=2))

    def organize_dataset(self, spec: DatasetSpec) -> Path:
        """
        Create storage path for a dataset.

        Does NOT raise an error if dataset already exists - this allows
        resume functionality to work. The caller is responsible for
        checking if download should be skipped or resumed.

        Args:
            spec: Dataset specification

        Returns:
            Path to dataset directory
        """
        storage_name = spec.storage_name
        dataset_path = self.storage_root / storage_name

        log = logger.bind(
            hf_id=spec.hf_id,
            storage_name=storage_name,
            path=str(dataset_path),
        )

        # Ensure dataset directory structure exists
        log.info("Ensuring dataset directory structure exists")
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / AUDIO_DIR).mkdir(exist_ok=True)
        (dataset_path / JSON_DIR).mkdir(exist_ok=True)

        return dataset_path

    def create_processed_symlink(self, spec: DatasetSpec) -> Path:
        """
        Create a symlink from the processed directory to the dataset.

        Creates a symlink at /mnt/data/processed/{storage_name} pointing to
        the actual dataset in STORAGE_ROOT/{storage_name}.

        Args:
            spec: Dataset specification

        Returns:
            Path to the symlink

        Raises:
            StorageError: If symlink creation fails
        """
        storage_name = spec.storage_name
        dataset_path = self.storage_root / storage_name
        processed_root = Path(PROCESSED_ROOT)
        symlink_path = processed_root / storage_name

        log = logger.bind(
            hf_id=spec.hf_id,
            storage_name=storage_name,
            dataset_path=str(dataset_path),
            symlink_path=str(symlink_path),
        )

        try:
            # Ensure processed directory exists
            processed_root.mkdir(parents=True, exist_ok=True)

            # Remove existing symlink if it exists
            if symlink_path.is_symlink() or symlink_path.exists():
                log.info("Removing existing symlink")
                symlink_path.unlink()

            # Create absolute symlink
            absolute_target = dataset_path.resolve()
            log.info("Creating absolute symlink", absolute_target=str(absolute_target))
            symlink_path.symlink_to(absolute_target)

            log.info("Processed symlink created successfully")
            return symlink_path

        except Exception as e:
            log.error("Failed to create processed symlink", error=str(e))
            raise StorageError(f"Failed to create processed symlink: {e}") from e

    def delete_dataset(self, spec: DatasetSpec) -> None:
        """
        Delete a dataset from storage.

        Args:
            spec: Dataset specification

        Raises:
            StorageError: If deletion fails
        """
        storage_name = spec.storage_name
        dataset_path = self.storage_root / storage_name

        log = logger.bind(
            hf_id=spec.hf_id,
            storage_name=storage_name,
            path=str(dataset_path),
        )

        try:
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
                log.info("Dataset deleted successfully")
            else:
                log.warning("Dataset path does not exist, nothing to delete")

        except Exception as e:
            log.error("Failed to delete dataset", error=str(e))
            raise StorageError(f"Failed to delete dataset: {e}") from e

    def dataset_exists(self, spec: DatasetSpec, check_complete: bool = False) -> bool:
        """
        Check if a dataset already exists in storage.

        Args:
            spec: Dataset specification
            check_complete: If True, verify download is complete (not just partial)

        Returns:
            True if dataset exists. If check_complete=True, only returns True if download is complete.
        """
        storage_name = spec.storage_name
        dataset_path = self.storage_root / storage_name

        if not dataset_path.exists():
            return False

        # Check if the dataset has actual files (not just empty directories)
        audio_dir = dataset_path / AUDIO_DIR
        json_dir = dataset_path / JSON_DIR

        # Count files in both directories
        audio_files = list(audio_dir.glob("*")) if audio_dir.exists() else []
        json_files = list(json_dir.glob("*")) if json_dir.exists() else []

        # No files at all
        if len(audio_files) == 0 and len(json_files) == 0:
            return False

        # If checking for completeness, verify against progress
        if check_complete:
            return self.is_download_complete(spec, len(audio_files), len(json_files))

        # Backward compatible: any files = exists
        return True

    def is_download_complete(self, spec: DatasetSpec, audio_count: int, json_count: int) -> bool:
        """
        Check if a download is complete by verifying against expected counts.

        Args:
            spec: Dataset specification
            audio_count: Number of audio files found
            json_count: Number of JSON files found

        Returns:
            True if download appears complete
        """
        # Load progress to get expected counts
        progress = self.load_progress(spec)

        if progress is None:
            # No progress file - can't determine completeness
            # Assume complete if files exist (backward compatible)
            return True

        resume_data = progress.get("resume_data", {})

        # If we have resume data with version 2.0+, use it
        if resume_data.get("version") == "2.0":
            expected_rows = resume_data.get("expected_total_rows", 0)
            completed_rows = len(resume_data.get("completed_rows", []))

            # Check if we have all expected rows
            if expected_rows > 0 and completed_rows >= expected_rows:
                return True

            # Check for incomplete row marker
            if resume_data.get("incomplete_row") is not None:
                return False

            return False

        # Legacy progress file (version 1.0) - use simple heuristic
        total_files = progress.get("total_files", 0)
        downloaded_files = progress.get("downloaded_files", 0)

        if total_files > 0 and downloaded_files >= total_files:
            return True

        # If we have matching audio and json counts, likely complete
        if audio_count == json_count and audio_count > 0:
            return True

        return False

    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.

        Returns:
            Dictionary with storage stats:
            - total_bytes: Total storage used
            - total_datasets: Number of datasets
            - datasets: List of dataset info
        """
        log = logger.bind(storage_root=str(self.storage_root))
        log.info("Calculating storage usage")

        total_bytes = 0
        datasets_info = []

        try:
            for dataset_dir in self.storage_root.iterdir():
                if dataset_dir.is_dir():
                    # Calculate size of this dataset
                    dataset_size = sum(
                        f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()
                    )

                    total_bytes += dataset_size
                    datasets_info.append(
                        {
                            "name": dataset_dir.name,
                            "path": str(dataset_dir),
                            "size_bytes": dataset_size,
                        }
                    )

            result = {
                "total_bytes": total_bytes,
                "total_datasets": len(datasets_info),
                "datasets": datasets_info,
            }

            log.info(
                "Storage usage calculated",
                total_bytes=total_bytes,
                total_datasets=len(datasets_info),
            )

            return result

        except Exception as e:
            log.error("Failed to calculate storage usage", error=str(e))
            raise StorageError(f"Failed to calculate storage usage: {e}") from e

    def update_manifest(
        self,
        spec: DatasetSpec,
        action: str,
        **metadata,
    ) -> None:
        """
        Update active and archived manifests.

        Args:
            spec: Dataset specification
            action: Action performed ("add" or "remove")
            **metadata: Additional metadata to store
        """
        log = logger.bind(
            hf_id=spec.hf_id,
            action=action,
        )
        log.info("Updating manifests")

        try:
            # Load active manifest
            active_manifest_path = self.state_dir / "active.json"
            with open(active_manifest_path) as f:
                active_manifest = json.load(f)

            # Load archived manifest
            archived_manifest_path = self.state_dir / "archived.json"
            with open(archived_manifest_path) as f:
                archived_manifest = json.load(f)

            timestamp = datetime.utcnow().isoformat()

            if action == "add":
                # Add to active manifest
                dataset_entry = {
                    "hf_id": spec.hf_id,
                    "description": spec.description,
                    "revision": spec.revision,
                    "subset": spec.subset,
                    "split": spec.split,
                    "download_mode": spec.download_mode,
                    "storage_name": spec.storage_name,
                    "added_at": timestamp,
                    **metadata,
                }

                # Check for duplicates
                active_manifest["datasets"] = [
                    d for d in active_manifest["datasets"] if d.get("storage_name") != spec.storage_name
                ]
                active_manifest["datasets"].append(dataset_entry)

                # Add to archived manifest
                archived_manifest["operations"].append(
                    {
                        "action": "add",
                        "dataset": dataset_entry,
                        "timestamp": timestamp,
                    }
                )

            elif action == "remove":
                # Remove from active manifest
                active_manifest["datasets"] = [
                    d for d in active_manifest["datasets"] if d.get("storage_name") != spec.storage_name
                ]

                # Add to archived manifest
                archived_manifest["operations"].append(
                    {
                        "action": "remove",
                        "hf_id": spec.hf_id,
                        "storage_name": spec.storage_name,
                        "reason": metadata.get("reason", "Unknown"),
                        "timestamp": timestamp,
                    }
                )

            # Update timestamps
            active_manifest["last_updated"] = timestamp
            archived_manifest["last_updated"] = timestamp

            # Write back
            with open(active_manifest_path, "w") as f:
                json.dump(active_manifest, f, indent=2)

            with open(archived_manifest_path, "w") as f:
                json.dump(archived_manifest, f, indent=2)

            log.info("Manifests updated successfully")

        except Exception as e:
            log.error("Failed to update manifests", error=str(e))
            raise StorageError(f"Failed to update manifests: {e}") from e

    def get_active_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of active datasets from manifest.

        Returns:
            List of dataset entries
        """
        active_manifest_path = self.state_dir / "active.json"

        try:
            with open(active_manifest_path) as f:
                active_manifest = json.load(f)

            return active_manifest.get("datasets", [])

        except Exception as e:
            logger.error("Failed to read active manifest", error=str(e))
            raise StorageError(f"Failed to read active manifest: {e}") from e

    def save_progress(
        self,
        spec: DatasetSpec,
        progress: Dict[str, Any],
    ) -> None:
        """
        Save download progress for a dataset.

        Args:
            spec: Dataset specification
            progress: Progress dictionary
        """
        progress_file = self.progress_dir / f"{spec.storage_name}.json"

        try:
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)

            logger.debug("Progress saved", progress_file=str(progress_file))

        except Exception as e:
            logger.warning("Failed to save progress", error=str(e))

    def load_progress(
        self,
        spec: DatasetSpec,
    ) -> Optional[Dict[str, Any]]:
        """
        Load download progress for a dataset.

        Args:
            spec: Dataset specification

        Returns:
            Progress dictionary or None if not found
        """
        progress_file = self.progress_dir / f"{spec.storage_name}.json"

        try:
            if progress_file.exists():
                with open(progress_file) as f:
                    return json.load(f)

            return None

        except Exception as e:
            logger.warning("Failed to load progress", error=str(e))
            return None

    def delete_progress(self, spec: DatasetSpec) -> None:
        """
        Delete progress file for a dataset.

        Args:
            spec: Dataset specification
        """
        progress_file = self.progress_dir / f"{spec.storage_name}.json"

        try:
            if progress_file.exists():
                progress_file.unlink()

        except Exception as e:
            logger.warning("Failed to delete progress file", error=str(e))

    def save_resume_state(
        self,
        spec: DatasetSpec,
        resume_state: ResumeState,
    ) -> None:
        """
        Save detailed resume state for a dataset.

        Args:
            spec: Dataset specification
            resume_state: Resume state to save
        """
        progress_file = self.progress_dir / f"{spec.storage_name}.json"

        try:
            # Load existing progress to preserve metadata
            existing_progress = {}
            if progress_file.exists():
                with open(progress_file) as f:
                    existing_progress = json.load(f)

            # Update with resume data
            existing_progress["resume_data"] = resume_state.to_dict()
            existing_progress["last_update"] = datetime.utcnow().isoformat()

            # Atomic write using temp file
            temp_file = progress_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(existing_progress, f, indent=2)

            # Atomic rename
            temp_file.replace(progress_file)

            logger.debug("Resume state saved", progress_file=str(progress_file))

        except Exception as e:
            logger.warning("Failed to save resume state", error=str(e))

    def load_resume_state(self, spec: DatasetSpec) -> Optional[ResumeState]:
        """
        Load detailed resume state for a dataset.

        Args:
            spec: Dataset specification

        Returns:
            ResumeState or None if not found
        """
        progress_file = self.progress_dir / f"{spec.storage_name}.json"

        try:
            if progress_file.exists():
                with open(progress_file) as f:
                    progress = json.load(f)

                resume_data = progress.get("resume_data")
                if resume_data:
                    return ResumeState.from_dict(resume_data)

            return None

        except Exception as e:
            logger.warning("Failed to load resume state", error=str(e))
            return None

    def mark_row_complete(
        self,
        spec: DatasetSpec,
        row_idx: int,
        audio_filename: Optional[str] = None,
        json_filename: Optional[str] = None,
        expected_total: int = 0,
    ) -> None:
        """
        Mark a row as completed in the resume state.

        Args:
            spec: Dataset specification
            row_idx: Row index that was completed
            audio_filename: Audio file name (if any)
            json_filename: JSON file name
            expected_total: Expected total number of rows
        """
        # Load existing state
        resume_state = self.load_resume_state(spec)

        if resume_state is None:
            resume_state = ResumeState(expected_total_rows=expected_total)

        # Update state
        resume_state.completed_rows.add(row_idx)
        resume_state.last_successful_row = row_idx

        if audio_filename:
            resume_state.completed_audio_files.add(audio_filename)

        if json_filename:
            resume_state.completed_json_files.add(json_filename)

        resume_state.expected_total_rows = max(resume_state.expected_total_rows, expected_total)
        resume_state.incomplete_row = None  # Clear incomplete marker

        # Save updated state
        self.save_resume_state(spec, resume_state)

    def mark_row_incomplete(self, spec: DatasetSpec, row_idx: int) -> None:
        """
        Mark a row as incomplete (for recovery).

        Args:
            spec: Dataset specification
            row_idx: Row index that failed
        """
        resume_state = self.load_resume_state(spec)

        if resume_state is None:
            resume_state = ResumeState()

        resume_state.incomplete_row = row_idx
        self.save_resume_state(spec, resume_state)

    def get_download_resume_point(self, spec: DatasetSpec) -> int:
        """
        Get the row index to resume downloading from.

        Args:
            spec: Dataset specification

        Returns:
            Row index to start from (0 if no resume state)
        """
        resume_state = self.load_resume_state(spec)

        if resume_state is None:
            return 0

        # Resume from the next row after last successful
        return resume_state.last_successful_row + 1

    def should_skip_row(self, spec: DatasetSpec, row_idx: int) -> bool:
        """
        Check if a row should be skipped (already downloaded).

        Args:
            spec: Dataset specification
            row_idx: Row index to check

        Returns:
            True if row should be skipped
        """
        resume_state = self.load_resume_state(spec)

        if resume_state is None:
            return False

        return row_idx in resume_state.completed_rows
