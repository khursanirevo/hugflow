"""
Configuration and Pydantic schemas for Hugflow.
"""

from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file if it exists
load_dotenv()


class DatasetSpec(BaseModel):
    """Schema for adding a dataset."""

    hf_id: str = Field(..., description="Hugging Face dataset ID (e.g., 'org/dataset')")
    description: str = Field(..., description="Human-readable description of the dataset")
    revision: str = Field(default="main", description="Dataset revision/branch/tag")
    subset: Optional[str] = Field(default=None, description="Dataset subset/config name")
    split: str = Field(default="train", description="Dataset split name")
    audio_column: str = Field(default="audio", description="Name of audio column")
    text_column: str = Field(default="text", description="Name of text column")
    update: bool = Field(default=False, description="Allow updating to new commit if SHA changed")

    @field_validator("hf_id")
    @classmethod
    def validate_hf_id(cls, v: str) -> str:
        """Validate HF ID format (org/dataset or dataset)."""
        if not v or not v.strip():
            raise ValueError("HF ID cannot be empty")
        # Basic format check: should contain at least one slash or be a valid dataset name
        parts = v.strip().split("/")
        if len(parts) > 2:
            raise ValueError(f"Invalid HF ID format: '{v}'. Expected 'org/dataset' or 'dataset'")
        return v.strip()

    @property
    def download_mode(self) -> str:
        """Determine if this is a full or specific download."""
        if self.subset is None:
            return "full"
        return "specific"

    @property
    def storage_name(self) -> str:
        """Generate storage directory name for this dataset."""
        from hugflow.constants import DOWNLOAD_MODE_FULL

        # Sanitize HF ID for filesystem
        safe_hf_id = self.hf_id.replace("/", "__")

        if self.download_mode == DOWNLOAD_MODE_FULL:
            return f"{safe_hf_id}__rev_{self.revision}"
        else:
            return f"{safe_hf_id}__subset_{self.subset}__split_{self.split}__rev_{self.revision}"


class RemoveSpec(BaseModel):
    """Schema for removing a dataset."""

    hf_id: str = Field(..., description="Hugging Face dataset ID to remove")
    reason: str = Field(..., description="Reason for removal")


class HFConfig(BaseModel):
    """Hugging Face API configuration."""

    token: str = Field(default="")


class GitHubConfig(BaseModel):
    """GitHub API configuration."""

    token: str = Field(default="", env="GITHUB_TOKEN")
    repo: str = Field(default="", env="GITHUB_REPO")
    server_url: str = Field(default="https://github.com", env="GITHUB_SERVER_URL")

    @property
    def enabled(self) -> bool:
        """Check if GitHub integration is enabled."""
        return bool(self.token and self.repo)


class SlackConfig(BaseModel):
    """Slack notification configuration."""

    webhook_url: str = Field(default="", env="SLACK_WEBHOOK_URL")
    channel: str = Field(default="", env="SLACK_CHANNEL")
    enabled: bool = Field(default=True)

    def model_post_init(self, __context: object) -> None:
        """Post-init to set enabled based on webhook_url."""
        self.enabled = bool(self.webhook_url)


class StorageConfig(BaseModel):
    """Storage configuration."""

    root: str
    state_dir: str

    @classmethod
    def from_constants(cls) -> "StorageConfig":
        """Create StorageConfig from constants."""
        from hugflow.constants import STORAGE_ROOT, STATE_DIR
        return cls(root=STORAGE_ROOT, state_dir=STATE_DIR)


class DownloadConfig(BaseModel):
    """Download behavior configuration."""

    default_revision: str = Field(default="main", env="DEFAULT_REVISION")
    default_split: str = Field(default="train", env="DEFAULT_SPLIT")
    default_audio_column: str = Field(default="audio", env="DEFAULT_AUDIO_COLUMN")
    default_text_column: str = Field(default="text", env="DEFAULT_TEXT_COLUMN")
    progress_interval: int = Field(default=10000, env="PROGRESS_INTERVAL")
    pr_comment_interval: int = Field(default=10000, env="PR_COMMENT_INTERVAL")
    slack_notification_interval: int = Field(default=10000, env="SLACK_NOTIFICATION_INTERVAL")


class NetworkConfig(BaseModel):
    """Network configuration."""

    max_concurrent_downloads: int = Field(default=5, env="MAX_CONCURRENT_DOWNLOADS")
    connection_timeout: int = Field(default=30, env="CONNECTION_TIMEOUT")
    read_timeout: int = Field(default=300, env="READ_TIMEOUT")
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_backoff: int = Field(default=2, env="RETRY_BACKOFF")


class ValidationConfig(BaseModel):
    """Validation configuration."""

    check_duplicates: bool = Field(default=True, env="CHECK_DUPLICATES")
    storage_quota_gb: int = Field(default=0, env="STORAGE_QUOTA_GB")
    validate_hf_id: bool = Field(default=True, env="VALIDATE_HF_ID")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    file: str = Field(default="", env="LOG_FILE")
    format: Literal["json", "text"] = Field(default="json", env="LOG_FORMAT")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class BehaviorConfig(BaseModel):
    """System behavior configuration."""

    auto_merge: bool = Field(default=False, env="AUTO_MERGE")
    auto_cleanup_requests: bool = Field(default=True, env="AUTO_CLEANUP_REQUESTS")
    cleanup_progress_on_success: bool = Field(default=True, env="CLEANUP_PROGRESS_ON_SUCCESS")
    cleanup_hf_cache: bool = Field(default=False, env="CLEANUP_HF_CACHE")


class CacheConfig(BaseModel):
    """HuggingFace cache cleanup configuration."""

    enabled: bool = Field(default=False, env="CACHE_CLEANUP_ENABLED")
    cleanup_on_update: bool = Field(default=True, env="CACHE_CLEANUP_ON_UPDATE")
    cleanup_on_add: bool = Field(default=False, env="CACHE_CLEANUP_ON_ADD")
    preserve_days: int = Field(default=7, env="CACHE_PRESERVE_DAYS")


class Config(BaseSettings):
    """Main configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    hf: HFConfig = Field(default_factory=HFConfig)
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig.from_constants)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def validate_tokens(self) -> "Config":
        """Validate that required tokens are available for their respective features."""
        # HF token is required for downloads
        # Support both HF_TOKEN (legacy) and HF__TOKEN (nested format)
        if not self.hf.token:
            import os
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF__TOKEN", "")
            if hf_token:
                object.__setattr__(self.hf, "token", hf_token)
        return self


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Reads from .env file if present, otherwise uses defaults.
    """
    return Config()


def get_dataset_spec_from_yaml(yaml_path: Path) -> DatasetSpec:
    """
    Load and validate a dataset specification from a YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        DatasetSpec object

    Raises:
        ValidationError: If YAML is invalid
    """
    import yaml

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        raise ValueError(f"Failed to read YAML file {yaml_path}: {e}")

    if not data:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    try:
        return DatasetSpec(**data)
    except Exception as e:
        raise ValueError(f"Invalid dataset specification in {yaml_path}: {e}")


def get_remove_spec_from_yaml(yaml_path: Path) -> RemoveSpec:
    """
    Load and validate a removal specification from a YAML file.

    Args:
        yaml_path: Path to YAML file

    Returns:
        RemoveSpec object

    Raises:
        ValidationError: If YAML is invalid
    """
    import yaml

    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        raise ValueError(f"Failed to read YAML file {yaml_path}: {e}")

    if not data:
        raise ValueError(f"YAML file is empty: {yaml_path}")

    try:
        return RemoveSpec(**data)
    except Exception as e:
        raise ValueError(f"Invalid removal specification in {yaml_path}: {e}")
