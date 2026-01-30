"""
Audit and logging for Hugflow.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

from hugflow.config import Config


def setup_logging(config: Config) -> None:
    """
    Configure structured logging for Hugflow.

    Args:
        config: Hugflow configuration
    """
    # Configure processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add format-specific processor
    if config.logging.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Text format for console
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging for third-party libraries
    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.logging.level),
    )


class AuditLogger:
    """Audit logger for tracking operations."""

    def __init__(self, config: Config):
        """
        Initialize audit logger.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.state_dir = Path(config.storage.state_dir)
        self.audit_log_path = self.state_dir / "audit.log"

    def _ensure_audit_log(self) -> None:
        """Ensure audit log file exists."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.audit_log_path.exists():
            self.audit_log_path.write_text("")

    def log_event(
        self,
        event_type: str,
        dataset_name: str = "",
        **kwargs,
    ) -> None:
        """
        Log an audit event.

        Args:
            event_type: Type of event (e.g., "dataset_requested", "download_started")
            dataset_name: Name of dataset
            **kwargs: Additional event metadata
        """
        self._ensure_audit_log()

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "dataset": dataset_name,
            **kwargs,
        }

        # Log to file
        try:
            with open(self.audit_log_path, "a") as f:
                import json

                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.warning("Failed to write to audit log", error=str(e))

        # Also log to stdout
        self.logger.info(event_type, dataset=dataset_name, **kwargs)

    def log_dataset_requested(
        self,
        hf_id: str,
        description: str,
        requested_by: str = "",
        pr_url: str = "",
    ) -> None:
        """
        Log dataset request.

        Args:
            hf_id: Hugging Face dataset ID
            description: Dataset description
            requested_by: User who requested (e.g., "@username")
            pr_url: URL to PR
        """
        self.log_event(
            "dataset_requested",
            dataset_name=hf_id,
            description=description,
            requested_by=requested_by,
            pr_url=pr_url,
        )

    def log_download_started(
        self,
        hf_id: str,
        storage_path: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Log download start.

        Args:
            hf_id: Hugging Face dataset ID
            storage_path: Path where dataset will be stored
            requested_by: User who requested
        """
        self.log_event(
            "download_started",
            dataset_name=hf_id,
            storage_path=storage_path,
            requested_by=requested_by,
        )

    def log_download_completed(
        self,
        hf_id: str,
        storage_path: str = "",
        file_count: int = 0,
        size_bytes: int = 0,
        requested_by: str = "",
    ) -> None:
        """
        Log successful download completion.

        Args:
            hf_id: Hugging Face dataset ID
            storage_path: Path where dataset was stored
            file_count: Number of files downloaded
            size_bytes: Total size in bytes
            requested_by: User who requested
        """
        self.log_event(
            "download_completed",
            dataset_name=hf_id,
            storage_path=storage_path,
            file_count=file_count,
            size_bytes=size_bytes,
            size_gb=size_bytes / (1024**3),
            requested_by=requested_by,
        )

    def log_download_failed(
        self,
        hf_id: str,
        error: str = "",
        error_type: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Log failed download.

        Args:
            hf_id: Hugging Face dataset ID
            error: Error message
            error_type: Type of error
            requested_by: User who requested
        """
        self.log_event(
            "download_failed",
            dataset_name=hf_id,
            error=error,
            error_type=error_type,
            requested_by=requested_by,
        )

    def log_dataset_removed(
        self,
        hf_id: str,
        reason: str = "",
        space_freed_bytes: int = 0,
        requested_by: str = "",
    ) -> None:
        """
        Log dataset removal.

        Args:
            hf_id: Hugging Face dataset ID
            reason: Reason for removal
            space_freed_bytes: Space freed in bytes
            requested_by: User who requested
        """
        self.log_event(
            "dataset_removed",
            dataset_name=hf_id,
            reason=reason,
            space_freed_bytes=space_freed_bytes,
            space_freed_gb=space_freed_bytes / (1024**3),
            requested_by=requested_by,
        )

    def log_validation_error(
        self,
        hf_id: str,
        error: str = "",
        error_type: str = "",
    ) -> None:
        """
        Log validation error.

        Args:
            hf_id: Hugging Face dataset ID
            error: Error message
            error_type: Type of error
        """
        self.log_event(
            "validation_error",
            dataset_name=hf_id,
            error=error,
            error_type=error_type,
        )

    def log_storage_change(
        self,
        previous_bytes: int = 0,
        new_bytes: int = 0,
        change_bytes: int = 0,
    ) -> None:
        """
        Log storage usage change.

        Args:
            previous_bytes: Previous storage usage
            new_bytes: New storage usage
            change_bytes: Change in bytes
        """
        self.log_event(
            "storage_changed",
            dataset_name="",
            previous_bytes=previous_bytes,
            previous_gb=previous_bytes / (1024**3),
            new_bytes=new_bytes,
            new_gb=new_bytes / (1024**3),
            change_bytes=change_bytes,
            change_gb=change_bytes / (1024**3),
        )
