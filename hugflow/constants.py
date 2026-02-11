"""
Constants and path definitions for Hugflow.
"""

# Directory paths
# Note: STORAGE_ROOT and STATE_DIR use absolute paths to work correctly with
# GitHub Actions runner which clones repo into a nested work directory
ADD_DIR = "requests/add"
REMOVE_DIR = "requests/remove"
STORAGE_ROOT = ".datasets"  # Absolute path for consistent location
STATE_DIR = "/mnt/data/raw/hugflow/.hugflow-state"  # Absolute path for shared state
PROCESSED_ROOT = "/mnt/data/processed"  # Symlink target for processed datasets

# Manifest files
ACTIVE_MANIFEST = ".hugflow-state/active.json"
ARCHIVED_MANIFEST = ".hugflow-state/archived.json"

# Progress tracking
PROGRESS_DIR = ".hugflow-state/progress"

# Storage layout
AUDIO_DIR = "audio"
JSON_DIR = "json"

# Filename patterns
RESULTS_FILE = "results.json"

# Download modes
DOWNLOAD_MODE_FULL = "full"
DOWNLOAD_MODE_SPECIFIC = "specific"

# Progress update thresholds
DEFAULT_PROGRESS_INTERVAL = 10000
DEFAULT_PR_COMMENT_INTERVAL = 10000
DEFAULT_SLACK_NOTIFICATION_INTERVAL = 10000

# Network defaults
DEFAULT_MAX_CONCURRENT_DOWNLOADS = 5
DEFAULT_CONNECTION_TIMEOUT = 30
DEFAULT_READ_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2

# Validation defaults
DEFAULT_STORAGE_QUOTA_GB = 0  # 0 = unlimited
