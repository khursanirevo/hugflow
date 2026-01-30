"""
Storage and file system operations for Hugflow.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

        Args:
            spec: Dataset specification

        Returns:
            Path to dataset directory

        Raises:
            DatasetExistsError: If dataset already exists
        """
        storage_name = spec.storage_name
        dataset_path = self.storage_root / storage_name

        log = logger.bind(
            hf_id=spec.hf_id,
            storage_name=storage_name,
            path=str(dataset_path),
        )

        if dataset_path.exists():
            log.warning("Dataset already exists in storage")
            raise DatasetExistsError(f"Dataset already exists: {dataset_path}")

        # Create dataset directory structure
        log.info("Creating dataset directory structure")
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

    def dataset_exists(self, spec: DatasetSpec) -> bool:
        """
        Check if a dataset already exists in storage.

        Args:
            spec: Dataset specification

        Returns:
            True if dataset exists and has files
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

        # Only consider the dataset as existing if it has files
        return len(audio_files) > 0 or len(json_files) > 0

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
