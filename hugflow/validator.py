"""
Validation logic for Hugflow.
"""

from pathlib import Path
from typing import Dict, List

from structlog import get_logger

from hugflow.config import Config, DatasetSpec, RemoveSpec
from hugflow.hf_client import HFClient, HFNotFoundError

logger = get_logger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class DuplicateDatasetError(ValidationError):
    """Dataset already exists."""

    pass


class StorageQuotaExceededError(ValidationError):
    """Storage quota would be exceeded."""

    pass


class AssetValidator:
    """Validator for dataset specifications."""

    def __init__(self, config: Config, hf_client: HFClient):
        """
        Initialize validator.

        Args:
            config: Hugflow configuration
            hf_client: HuggingFace client
        """
        self.config = config
        self.hf_client = hf_client

    def validate_dataset_spec(self, spec: DatasetSpec) -> None:
        """
        Validate a dataset specification.

        Args:
            spec: Dataset specification to validate

        Raises:
            ValidationError: If validation fails
        """
        log = logger.bind(hf_id=spec.hf_id)

        # 1. Validate HF ID exists
        if self.config.validation.validate_hf_id:
            log.info("Validating HF ID exists on HuggingFace Hub")
            try:
                self.hf_client.validate_dataset_id(spec.hf_id)
            except HFNotFoundError as e:
                log.error("HF ID validation failed")
                raise ValidationError(f"Dataset '{spec.hf_id}' not found on HuggingFace Hub") from e

        # 2. Validate subset if specified
        if spec.subset:
            log.info("Validating subset exists")
            try:
                available_subsets = self.hf_client.get_available_subsets(spec.hf_id, spec.revision)

                if available_subsets and spec.subset not in available_subsets:
                    log.error("Subset not found", subset=spec.subset, available=available_subsets)
                    raise ValidationError(
                        f"Subset '{spec.subset}' not found in dataset '{spec.hf_id}'. "
                        f"Available subsets: {', '.join(available_subsets)}"
                    )
            except Exception as e:
                log.warning("Could not validate subset", error=str(e))
                # Don't fail on subset validation errors, as it might not be critical
                pass

        log.info("Dataset specification validated successfully")

    def validate_remove_spec(self, spec: RemoveSpec) -> None:
        """
        Validate a removal specification.

        Args:
            spec: Removal specification to validate

        Raises:
            ValidationError: If validation fails
        """
        log = logger.bind(hf_id=spec.hf_id)

        # Basic validation - just ensure HF ID is provided
        if not spec.hf_id or not spec.hf_id.strip():
            log.error("HF ID is empty")
            raise ValidationError("HF ID cannot be empty")

        if not spec.reason or not spec.reason.strip():
            log.error("Reason is empty")
            raise ValidationError("Reason cannot be empty")

        log.info("Removal specification validated successfully")

    def check_duplicates(
        self,
        spec: DatasetSpec,
        active_specs: List[Dict],
    ) -> None:
        """
        Check for duplicate datasets.

        Args:
            spec: Dataset specification to check
            active_specs: List of active dataset specifications from manifest

        Raises:
            DuplicateDatasetError: If duplicate found
        """
        log = logger.bind(hf_id=spec.hf_id, storage_name=spec.storage_name)

        if not self.config.validation.check_duplicates:
            log.debug("Duplicate checking disabled")
            return

        # Check if storage name already exists
        for active_spec in active_specs:
            if active_spec.get("storage_name") == spec.storage_name:
                # Verify the dataset actually exists on disk (not just in manifest)
                from hugflow.storage import StorageManager
                storage = StorageManager(self.config)

                if not storage.dataset_exists(spec, check_complete=True):
                    # Dataset is in manifest but download is incomplete or missing - allow resume
                    log.info("Dataset in manifest but download is incomplete - allowing resume", existing=active_spec)
                    return

                log.error("Duplicate dataset found", existing=active_spec)
                raise DuplicateDatasetError(
                    f"Dataset '{spec.hf_id}' (subset={spec.subset}, split={spec.split}, "
                    f"revision={spec.revision}) already exists in storage"
                )

        log.info("No duplicates found")

    def check_storage_quota(self, spec: DatasetSpec, current_usage_bytes: int) -> None:
        """
        Check if adding a dataset would exceed storage quota.

        Args:
            spec: Dataset specification
            current_usage_bytes: Current storage usage in bytes

        Raises:
            StorageQuotaExceededError: If quota would be exceeded
        """
        log = logger.bind(
            hf_id=spec.hf_id,
            current_usage_bytes=current_usage_bytes,
            quota_gb=self.config.validation.storage_quota_gb,
        )

        quota_gb = self.config.validation.storage_quota_gb

        if quota_gb <= 0:
            log.debug("No storage quota configured")
            return

        # Get dataset info to check size
        try:
            dataset_info = self.hf_client.get_dataset_info(spec.hf_id, spec.revision)
            dataset_size_bytes = dataset_info.get("size", 0)

            quota_bytes = quota_gb * 1024 * 1024 * 1024
            projected_usage = current_usage_bytes + dataset_size_bytes

            if projected_usage > quota_bytes:
                log.error(
                    "Storage quota would be exceeded",
                    dataset_size_bytes=dataset_size_bytes,
                    projected_usage_bytes=projected_usage,
                    quota_bytes=quota_bytes,
                )
                raise StorageQuotaExceededError(
                    f"Adding dataset '{spec.hf_id}' ({dataset_size_bytes / (1024**3):.2f} GB) "
                    f"would exceed storage quota ({quota_gb} GB). "
                    f"Current usage: {current_usage_bytes / (1024**3):.2f} GB"
                )

            log.info("Storage quota check passed")

        except Exception as e:
            log.warning("Could not check storage quota", error=str(e))
            # Don't fail on quota check errors, as size estimation might be inaccurate

    def check_yaml_file(self, yaml_path: Path) -> None:
        """
        Validate YAML file exists and is readable.

        Args:
            yaml_path: Path to YAML file

        Raises:
            ValidationError: If file is invalid
        """
        log = logger.bind(yaml_path=str(yaml_path))

        if not yaml_path.exists():
            log.error("YAML file does not exist")
            raise ValidationError(f"YAML file does not exist: {yaml_path}")

        if not yaml_path.is_file():
            log.error("Path is not a file")
            raise ValidationError(f"Path is not a file: {yaml_path}")

        # Try to read the file
        try:
            with open(yaml_path) as f:
                content = f.read()
                if not content.strip():
                    log.error("YAML file is empty")
                    raise ValidationError(f"YAML file is empty: {yaml_path}")

        except Exception as e:
            log.error("Failed to read YAML file", error=str(e))
            raise ValidationError(f"Failed to read YAML file: {e}") from e
