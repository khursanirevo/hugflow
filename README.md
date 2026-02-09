# Hugflow

**Hugflow** is a GitOps-based declarative automation system for managing Hugging Face datasets. It automates dataset downloads via pull requests with progress tracking, validation, and manual merge on success.

## Features

- **Ultra-simple YAML** - Just 2 required fields (HF ID + description)
- **Smart Audio Extraction** - Automatically extracts audio files to MP3 format, metadata to JSON
  - Supports dict audio (path/array) and AudioDecoder objects (torchcodec)
  - Handles float32 tensors, converts to int16 PCM for MP3 encoding
- **Optimized Storage** - Audio converted to MP3 with original sample rate preserved, mono channel for safety
- **No Data Duplication** - Audio bytes removed from JSON, keeping only path references
- **Processed Symlinks** - Auto-creates symlinks in `/mnt/data/processed/{dataset_name}` for easy access
- **GitOps Workflow** - Create PR → Auto-download → Manual review & merge
- **Slack Notifications** - Optional real-time progress updates
- **Resume Capability** - Interrupted downloads automatically resume from the last successful row
- **Self-Hosted Runner** - Runs on your own infrastructure with unlimited timeout

---

## Quick Start

### 1. Clone & Install

```bash
# Install ffmpeg (required for MP3 conversion)
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS

# Install hugflow
git clone https://github.com/your-org/hugflow.git
cd hugflow
pip install -e .
```

### 2. Configure

Create a `.env` file (copy from `.env.example`):

```bash
# REQUIRED: Hugging Face token
HF_TOKEN=hf_your_token_here

# OPTIONAL: For GitHub features (PR comments, manual merge required)
gh auth login  # Easiest method - no token needed!

# OPTIONAL: Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/XXX
SLACK_CHANNEL=#ml-datasets

# OPTIONAL: Custom storage location
STORAGE_ROOT=/path/to/datasets  # Default: ./datasets
```

### 3. Initialize

```bash
hugflow init
```

This creates necessary directories:
- `datasets/` - Downloaded datasets
- `.hugflow-state/` - State tracking

---

## Server Setup Guide

### For Your Own Server/Infrastructure

#### Step 1: Install GitHub Actions Self-Hosted Runner

On your server (e.g., HGX machine):

```bash
# Create directory for runner
sudo mkdir -p /opt/actions-runner
cd /opt/actions-runner

# Download latest runner
sudo curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
sudo tar xzf ./actions-runner-linux-x64.tar.gz

# Configure runner
sudo ./config.sh --url https://github.com/your-org/hugflow --token YOUR_RUNNER_TOKEN
# Get token from: https://github.com/your-org/hugflow/settings/actions/runners/new

# Install as service (auto-start on boot)
sudo ./svc.sh install
sudo ./svc.sh start
```

#### Step 2: Configure GitHub Secrets

Go to your repo settings: `https://github.com/your-org/hugflow/settings/secrets/actions`

Add these secrets:

- **`HF_TOKEN`** (Required) - Get from: https://huggingface.co/settings/tokens
- **`SLACK_WEBHOOK_URL`** (Optional) - Slack notifications

**Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions, no need to add it.

#### Step 3: Verify Runner

Check runner status at: `https://github.com/your-org/hugflow/actions/runners`

Should show:
- ✅ Status: `online`
- ✅ Labels: `self-hosted`, `Linux`, `X64`

---

## How to Add Datasets

### Method 1: Via GitOps PR (Recommended)

#### Step 1: Create YAML File

```bash
cd hugflow

# Create YAML file
cat > requests/add/my_dataset.yaml << EOF
hf_id: "org/dataset-name"
description: "Brief description of what this dataset is for"
EOF
```

#### Step 2: Commit & Push

```bash
git add requests/add/my_dataset.yaml
git commit -m "Add my-dataset for training"
git push origin add-my-dataset
```

#### Step 3: Create Pull Request

```bash
gh pr create --title "Add my-dataset" --body "Adding dataset for training"
```

#### Step 4: Automation Takes Over

- 🤖 Bot comments "Starting download..." on your PR
- 📥 Dataset downloads to `datasets/` directory
  - Audio files → `audio/` folder
  - Metadata → `json/` folder
- ✅ On success: Bot comments with location + PR ready for manual review
- ❌ On failure: Bot comments with error details + keeps PR open

---

### Method 2: Local Testing

```bash
# Validate YAML
hugflow validate requests/add/my_dataset.yaml

# Download locally (for testing)
hugflow local-sync requests/add/my_dataset.yaml

# Check status
hugflow status
```

---

## YAML Schema

### Minimal (Recommended)

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data"
```

### With All Options

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data"
revision: "main"           # Git revision (default: "main")
subset: "ms"               # Dataset subset/config (default: None)
split: "train"             # Dataset split (default: "train")
audio_column: "audio"      # Column containing audio (default: "audio")
text_column: "text"        # Column containing text (default: "text")
```

### Remove Dataset

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
reason: "No longer needed, replaced with v12"
```

---

## Dataset Storage Structure

After download, datasets are organized as:

```
datasets/
└── org__dataset__subset_X__split_Y__rev_Z/     # Sanitized name
    ├── audio/                                     # Audio files (MP3 format)
    │   ├── original_filename.mp3                  # Named from audio_filename field
    │   └── ...
    └── json/                                      # Metadata only (no audio bytes!)
        ├── 0.json
        │   ├── audio: "datasets/.../audio/original_filename.mp3"  # Path reference
        │   ├── text: "Transcript here"
        │   └── ...other metadata
        └── ...
```

**Key Points:**
- Audio files automatically converted to MP3 format (192k bitrate)
- **Original sample rate preserved** - no quality loss from resampling
- **Mono channel** for safety (prevents multiprocess crashes with high sample rates)
- Files named using `audio_filename` field from dataset (or row number if unavailable)
- JSON contains metadata + path to audio file (NOT embedded bytes)
- Significant space savings: ~900MB → ~330MB per 1801 files
- **Symlink created** at `/mnt/data/processed/{dataset_name}` pointing to dataset location

### Supported Audio Formats

Hugflow automatically detects and handles multiple audio formats:

| Format Type | Detection | Processing |
|-------------|-----------|------------|
| **Dict with 'path'** | `isinstance(audio, dict) and 'path' in audio` | Extracts external file, converts to MP3 |
| **Dict with 'array'** | `isinstance(audio, dict) and 'array' in audio` | Converts numpy array to MP3 |
| **AudioDecoder** | `hasattr(audio, 'get_all_samples')` | Extracts torch tensor, converts to MP3 |
| **Bytes** | `isinstance(audio, bytes)` | Decodes bytes, converts to MP3 |

---

## Configuration

### Required Settings

| Setting | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | Required |
| `STORAGE_ROOT` | Where datasets are stored | `./datasets` |
| `PROCESSED_ROOT` | Where symlinks to processed datasets are created | `/mnt/data/processed` |

### Optional Settings

| Setting | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub token (or use `gh auth login`) | Auto-detect |
| `SLACK_WEBHOOK_URL` | Slack notifications | Disabled |
| `PROGRESS_INTERVAL` | Progress update frequency | 10000 files |
| `MAX_CONCURRENT_DOWNLOADS` | Parallel downloads | 5 |
| `AUTO_MERGE` | Auto-merge successful PRs (requires manual merge by default) | `false` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

See `.env.example` for all available settings.

---

## CLI Commands

```bash
# Initialize directories
hugflow init

# Validate YAML without applying
hugflow validate <yaml-file>

# Show status of all datasets
hugflow status

# Download dataset locally (testing)
hugflow local-sync <yaml-file>

# Sync in CI mode (called by GitHub Actions)
hugflow sync --ci-mode

# Check version
hugflow --version
```

---

## Troubleshooting

### Dataset Already Exists

If you get "Dataset already exists in storage":

```bash
# Option 1: Remove it and re-download
hugflow remove <yaml-file>

# Option 2: Use different subset/split
# Edit your YAML to add:
subset: "different_subset"
split: "validation"
```

### GitHub Actions Timeout

Self-hosted runners have unlimited timeout, but if using GitHub-hosted runners, update workflow:

```yaml
# In .github/workflows/dataset-sync.yml
timeout-minutes: 360  # 6 hours
```

### Runner Not Triggering

Check:
1. Runner is online: https://github.com/your-org/hugflow/actions/runners
2. Workflow file exists: `.github/workflows/dataset-sync.yml`
3. PR modifies files in `requests/add/` or `requests/remove/`

### Audio Files Not Extracted

Check that:
1. Dataset actually has audio data column
2. `audio_column` in YAML matches the column name in dataset
3. Check logs for "Audio column exists but no audio files extracted" warning

### AudioDecoder Objects

Some HuggingFace datasets use `AudioDecoder` objects (from torchcodec):
- **Supported**: Automatically detected and extracted via `get_all_samples()`
- **Format**: Converts float32 tensors → int16 PCM → WAV → MP3
- **Sample Rate**: Preserved from original audio
- **Channels**: Converts to mono for safety

Logs will show:
- `"Found AudioDecoder object, extracting audio"`
- `"Successfully extracted and converted AudioDecoder audio"`

### MP3 Conversion Issues

If audio conversion fails:
1. **Ensure ffmpeg is installed**: `ffmpeg -version`
2. Install ffmpeg: `sudo apt-get install ffmpeg` (Ubuntu/Debian) or `brew install ffmpeg` (macOS)
3. Check logs for specific conversion errors

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hugflow --cov-report=html

# Type checking
mypy hugflow
```

---

## Example: Adding Your First Dataset

```bash
# 1. Create YAML
cat > requests/add/common_voice.yaml << EOF
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data for speech recognition"
subset: "ms"
split: "train"
EOF

# 2. Validate
hugflow validate requests/add/common_voice.yaml

# 3. Commit & Push
git add requests/add/common_voice.yaml
git commit -m "Add Common Voice Malay dataset"
git push origin add-common-voice

# 4. Create PR
gh pr create --title "Add Common Voice Malay" \
  --body "Adding Malay subset of Common Voice v11 for ASR training"

# 5. Watch progress
# - Bot will comment on PR
# - If Slack configured: get notifications in #ml-datasets
# - Dataset downloads in background
# - On success: Review PR and merge manually
```

---

## How It Works

### Data Flow

```
User PR → GitHub Actions → Self-Hosted Runner → Download Dataset
                                              ↓
                                         Extract audio bytes
                                              ↓
                                         Convert to MP3 (preserves sample rate, mono channel)
                                              ↓
                                         Audio files → audio/*.mp3
                                         Metadata → json/ (with path refs)
                                              ↓
                                         Update manifests
                                              ↓
                                         Comment success + manual merge
```

### Idempotency

- **Duplicate Detection**: Won't re-download if dataset exists and is complete
- **Resume Capability**: If interrupted, automatically resumes from last successful row
  - Tracks which specific rows/files have been downloaded
  - Skips already-completed rows on resume
  - Uses atomic file operations (temp file + rename) to prevent corruption
- **Clean Removal**: `remove` command cleans up both files and manifests

### Progress Tracking

For large datasets (1000+ files), progress updates are posted every 10,000 files:
- PR comment (if GitHub features enabled)
- Slack notification (if configured)
- Progress file: `.hugflow-state/progress/{dataset}.json`

#### Resume Capability Details

When a download is interrupted (network failure, runner restart, etc.), the system automatically resumes:

**How it works:**
1. **Row-level tracking**: Each successfully downloaded row is tracked in the progress file
2. **Atomic writes**: JSON files are written to `.tmp` first, then renamed atomically
3. **Auto-resume**: On restart, completed rows are skipped, downloading continues from the last successful row
4. **Completion detection**: System verifies if a download is truly complete before cleanup

**Progress file format** (`.hugflow-state/progress/{dataset}.json`):
```json
{
  "dataset_name": "org/dataset",
  "status": "downloading",
  "downloaded_files": 1500,
  "total_files": 10000,
  "percent_complete": 15.0,
  "resumed_from": 1000,
  "newly_downloaded": 500,
  "resume_data": {
    "completed_rows": [0, 1, 2, ..., 1499],
    "last_successful_row": 1499,
    "expected_total_rows": 10000,
    "version": "2.0"
  }
}
```

**Interruption scenarios handled:**
- Network failure during download
- GitHub Actions runner restart
- Process killed (SIGTERM/SIGINT)
- Disk full (after freeing space, resume continues)

---

## Architecture

```
hugflow/
├── cli.py              # Typer CLI interface
├── config.py           # Pydantic schemas + env loading
├── gitops.py           # GitHub PR handling
├── slack.py            # Slack notifications
├── hf_client.py        # HuggingFace API wrapper
├── storage.py          # File system operations
├── validator.py        # Validation logic
├── audit.py            # Logging/audit trail
└── sync.py             # Main sync orchestration
```

---

## License

MIT
