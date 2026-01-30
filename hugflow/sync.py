"""
Main sync logic for Hugflow.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from structlog import get_logger

from hugflow.audit import AuditLogger
from hugflow.config import Config, DatasetSpec, RemoveSpec, get_dataset_spec_from_yaml, get_remove_spec_from_yaml
from hugflow.gitops import GitOpsManager
from hugflow.hf_client import HFClient, HFDownloadError, HFNotFoundError
from hugflow.slack import SlackNotifier
from hugflow.storage import StorageManager
from hugflow.validator import AssetValidator, ValidationError

logger = get_logger(__name__)


class SyncError(Exception):
    """Base exception for sync errors."""

    pass


class SyncManager:
    """Manager for dataset sync operations."""

    def __init__(self, config: Config):
        """
        Initialize sync manager.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.storage = StorageManager(config)
        self.hf_client = HFClient(config)
        self.validator = AssetValidator(config, self.hf_client)
        self.gitops = GitOpsManager(config)
        self.slack = SlackNotifier(config)
        self.audit = AuditLogger(config)

        # Initialize storage directories
        self.storage.initialize_directories()

    def sync_add(self, yaml_path: Path, ci_mode: bool = False) -> dict:
        """
        Sync a dataset add request.

        Args:
            yaml_path: Path to YAML file
            ci_mode: Running in CI mode (GitHub Actions)

        Returns:
            Results dictionary with status and metadata
        """
        log = logger.bind(yaml_path=str(yaml_path))
        log.info("Starting add sync")

        try:
            # Load and parse YAML
            spec = get_dataset_spec_from_yaml(yaml_path)

            log = log.bind(hf_id=spec.hf_id)
            log.info("Parsed dataset specification")

            # Get PR info if in CI mode
            pr_url = self.gitops.get_pr_url() if ci_mode else ""
            requested_by = self.gitops.get_pr_author() if ci_mode else ""

            # Send start notifications
            if ci_mode:
                self.gitops.comment_start(dataset_name=spec.hf_id)
            self.slack.send_download_started(spec, pr_url=pr_url, requested_by=requested_by)
            self.audit.log_download_started(spec.hf_id, requested_by=requested_by)

            # Validate specification
            log.info("Validating dataset specification")
            self.validator.validate_dataset_spec(spec)

            # Check for duplicates
            active_datasets = self.storage.get_active_datasets()
            self.validator.check_duplicates(spec, active_datasets)

            # Check storage quota
            storage_usage = self.storage.get_storage_usage()
            self.validator.check_storage_quota(spec, storage_usage["total_bytes"])

            # Check if dataset already exists on disk
            if self.storage.dataset_exists(spec):
                log.info("Dataset already exists, skipping download")
                existing_path = self.storage.storage_root / spec.storage_name

                # Update manifest anyway
                self.storage.update_manifest(spec, "add", skipped=True, reason="Already exists")

                result = {
                    "status": "success",
                    "dataset_name": spec.hf_id,
                    "file_count": 0,
                    "storage_path": str(existing_path),
                    "skipped": True,
                    "reason": "Dataset already exists",
                }

                if ci_mode:
                    self.gitops.write_results(result)
                    self.gitops.comment_success(
                        dataset_name=spec.hf_id,
                        file_count=0,
                        storage_path=str(existing_path),
                    )

                return result

            # Create storage directory
            log.info("Creating storage directory")
            dataset_path = self.storage.organize_dataset(spec)

            # Download dataset with progress tracking
            log.info("Starting dataset download")

            start_time = time.time()
            last_pr_update = 0
            last_slack_update = 0

            def progress_callback(downloaded: int, total: int, current_file: str = ""):
                nonlocal last_pr_update, last_slack_update

                elapsed = time.time() - start_time
                percent = (downloaded / total * 100) if total > 0 else 0

                # Calculate ETA
                if downloaded > 0:
                    rate = downloaded / elapsed
                    remaining = total - downloaded
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"
                else:
                    eta = "Unknown"

                # Save progress
                progress = {
                    "dataset_name": spec.hf_id,
                    "status": "downloading",
                    "downloaded_files": downloaded,
                    "total_files": total,
                    "percent_complete": percent,
                    "start_time": datetime.utcnow().isoformat(),
                    "last_update": datetime.utcnow().isoformat(),
                    "current_file": current_file,
                    "eta": eta,
                }
                self.storage.save_progress(spec, progress)

                # PR comment update
                if (
                    ci_mode
                    and downloaded - last_pr_update >= self.config.download.pr_comment_interval
                ):
                    self.gitops.comment_progress(
                        dataset_name=spec.hf_id,
                        downloaded=downloaded,
                        total=total,
                        percent=percent,
                        eta=eta,
                    )
                    last_pr_update = downloaded

                # Slack notification update
                if downloaded - last_slack_update >= self.config.download.slack_notification_interval:
                    self.slack.send_progress(
                        dataset_name=spec.hf_id,
                        downloaded=downloaded,
                        total=total,
                        percent=percent,
                        eta=eta,
                        pr_url=pr_url,
                    )
                    last_slack_update = downloaded

            # Download
            download_result = self.hf_client.download_dataset(
                hf_id=spec.hf_id,
                dest=dataset_path,
                revision=spec.revision,
                subset=spec.subset,
                split=spec.split,
                audio_column=spec.audio_column,
                text_column=spec.text_column,
                progress_callback=progress_callback,
            )

            elapsed = time.time() - start_time

            log.info(
                "Download completed",
                file_count=download_result["downloaded_files"],
                size_bytes=download_result["total_size"],
                elapsed_seconds=elapsed,
            )

            # Update manifest
            self.storage.update_manifest(
                spec,
                "add",
                file_count=download_result["downloaded_files"],
                size_bytes=download_result["total_size"],
                download_time_seconds=elapsed,
            )

            # Create symlink in processed directory
            log.info("Creating processed symlink")
            try:
                symlink_path = self.storage.create_processed_symlink(spec)
                log.info("Processed symlink created", symlink_path=str(symlink_path))
            except Exception as e:
                log.warning("Failed to create processed symlink, continuing anyway", error=str(e))

            # Delete progress file if configured
            if self.config.behavior.cleanup_progress_on_success:
                self.storage.delete_progress(spec)

            # Send success notifications
            size_gb = download_result["total_size"] / (1024**3)
            if ci_mode:
                self.gitops.comment_success(
                    dataset_name=spec.hf_id,
                    file_count=download_result["downloaded_files"],
                    storage_path=str(dataset_path),
                )

            self.slack.send_download_complete(
                spec,
                file_count=download_result["downloaded_files"],
                storage_path=str(dataset_path),
                size_gb=size_gb,
                pr_url=pr_url,
                requested_by=requested_by,
            )

            self.audit.log_download_completed(
                spec.hf_id,
                storage_path=str(dataset_path),
                file_count=download_result["downloaded_files"],
                size_bytes=download_result["total_size"],
                requested_by=requested_by,
            )

            result = {
                "status": "success",
                "dataset_name": spec.hf_id,
                "file_count": download_result["downloaded_files"],
                "storage_path": str(dataset_path),
                "size_bytes": download_result["total_size"],
                "size_gb": size_gb,
                "download_time_seconds": elapsed,
            }

            if ci_mode:
                self.gitops.write_results(result)

            return result

        except ValidationError as e:
            log.error("Validation failed", error=str(e))

            # Send failure notifications
            if ci_mode:
                self.gitops.comment_failure(
                    dataset_name=spec.hf_id if spec else "Unknown",
                    error=str(e),
                    error_type="ValidationError",
                    suggestion="Please check your YAML file and try again.",
                )

            self.slack.send_download_failed(spec, error=str(e), pr_url=pr_url, requested_by=requested_by)
            self.audit.log_validation_error(spec.hf_id if spec else "Unknown", error=str(e), error_type="ValidationError")

            result = {
                "status": "failed",
                "dataset_name": spec.hf_id if spec else "Unknown",
                "error": str(e),
                "error_type": "ValidationError",
            }

            if ci_mode:
                self.gitops.write_results(result)

            raise SyncError(f"Validation failed: {e}") from e

        except (HFDownloadError, HFNotFoundError) as e:
            log.error("Download failed", error=str(e))

            # Send failure notifications
            if ci_mode:
                self.gitops.comment_failure(
                    dataset_name=spec.hf_id if spec else "Unknown",
                    error=str(e),
                    error_type=type(e).__name__,
                    suggestion="Please check the Hugging Face dataset ID and try again.",
                )

            self.slack.send_download_failed(spec, error=str(e), pr_url=pr_url, requested_by=requested_by)
            self.audit.log_download_failed(
                spec.hf_id if spec else "Unknown",
                error=str(e),
                error_type=type(e).__name__,
                requested_by=requested_by,
            )

            result = {
                "status": "failed",
                "dataset_name": spec.hf_id if spec else "Unknown",
                "error": str(e),
                "error_type": type(e).__name__,
            }

            if ci_mode:
                self.gitops.write_results(result)

            raise SyncError(f"Download failed: {e}") from e

        except Exception as e:
            log.error("Unexpected error during sync", error=str(e))

            # Send failure notifications
            if ci_mode:
                self.gitops.comment_failure(
                    dataset_name=spec.hf_id if spec else "Unknown",
                    error=str(e),
                    error_type="UnexpectedError",
                    suggestion="An unexpected error occurred. Please check the logs.",
                )

            if spec:
                self.slack.send_download_failed(spec, error=str(e), pr_url=pr_url, requested_by=requested_by)
                self.audit.log_download_failed(
                    spec.hf_id,
                    error=str(e),
                    error_type="UnexpectedError",
                    requested_by=requested_by,
                )

            result = {
                "status": "failed",
                "dataset_name": spec.hf_id if spec else "Unknown",
                "error": str(e),
                "error_type": "UnexpectedError",
            }

            if ci_mode:
                self.gitops.write_results(result)

            raise SyncError(f"Unexpected error: {e}") from e

    def sync_remove(self, yaml_path: Path, ci_mode: bool = False) -> dict:
        """
        Sync a dataset remove request.

        Args:
            yaml_path: Path to YAML file
            ci_mode: Running in CI mode

        Returns:
            Results dictionary
        """
        log = logger.bind(yaml_path=str(yaml_path))
        log.info("Starting remove sync")

        try:
            # Load and parse YAML
            spec = get_remove_spec_from_yaml(yaml_path)

            log = log.bind(hf_id=spec.hf_id)
            log.info("Parsed removal specification")

            # Get PR info if in CI mode
            pr_url = self.gitops.get_pr_url() if ci_mode else ""
            requested_by = self.gitops.get_pr_author() if ci_mode else ""

            # Validate
            self.validator.validate_remove_spec(spec)

            # Get current storage usage
            storage_before = self.storage.get_storage_usage()

            # Delete from storage
            # Note: We need to find the dataset in the manifest first to get storage_name
            active_datasets = self.storage.get_active_datasets()
            dataset_entry = None

            for ds in active_datasets:
                if ds["hf_id"] == spec.hf_id:
                    dataset_entry = ds
                    break

            if dataset_entry:
                # Create a mock DatasetSpec for deletion
                from hugflow.config import DatasetSpec

                delete_spec = DatasetSpec(
                    hf_id=dataset_entry["hf_id"],
                    description="",
                    revision=dataset_entry.get("revision", "main"),
                    subset=dataset_entry.get("subset"),
                    split=dataset_entry.get("split", "train"),
                )

                self.storage.delete_dataset(delete_spec)

            # Update manifest
            self.storage.update_manifest(
                spec,  # RemoveSpec
                "remove",
                reason=spec.reason,
            )

            # Get new storage usage
            storage_after = self.storage.get_storage_usage()
            space_freed = storage_before["total_bytes"] - storage_after["total_bytes"]

            log.info("Dataset removed successfully", space_freed_bytes=space_freed)

            # Send notifications
            self.slack.send_removal_complete(
                spec,
                space_freed_gb=space_freed / (1024**3),
                pr_url=pr_url,
                requested_by=requested_by,
            )

            self.audit.log_dataset_removed(
                spec.hf_id,
                reason=spec.reason,
                space_freed_bytes=space_freed,
                requested_by=requested_by,
            )

            result = {
                "status": "success",
                "dataset_name": spec.hf_id,
                "space_freed_bytes": space_freed,
                "space_freed_gb": space_freed / (1024**3),
            }

            if ci_mode:
                self.gitops.write_results(result)

            return result

        except Exception as e:
            log.error("Error during remove sync", error=str(e))

            result = {
                "status": "failed",
                "dataset_name": spec.hf_id if spec else "Unknown",
                "error": str(e),
            }

            if ci_mode:
                self.gitops.write_results(result)

            raise SyncError(f"Remove failed: {e}") from e
