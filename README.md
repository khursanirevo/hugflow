# Hugflow

**Hugflow** is a GitOps-based declarative automation system for managing Hugging Face datasets on HGX infrastructure. It's a cleaner, simpler rewrite of hgx_hfdownloader with a focus on ease of use and maintainability.

## Features

- **Ultra-simple YAML** - Just 2 required fields (HF ID + description)
- **No conversion** - Downloads datasets as-is, faster and simpler
- **Clean code** - Modular architecture with clear separation of concerns
- **Better state tracking** - JSON manifests vs log files
- **Proven GitOps workflow** - Same pattern as hgx_hfdownloader
- **Slack notifications** - Optional real-time progress updates
- **Progress tracking** - Resume capability for long downloads

## Quick Start

### 1. Installation

```bash
pip install -e .
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
# Edit .env with your HF token, GitHub token, and Slack webhook (optional)
```

### 3. Add a Dataset

Create a YAML file in `requests/add/`:

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data"
```

### 4. Create a PR

```bash
git add requests/add/common_voice.yaml
git commit -m "Add Common Voice dataset"
git push origin add-common-voice
gh pr create --title "Add Common Voice" --body "Adding Malay ASR dataset"
```

### 5. GitHub Actions Takes Over

- Bot acknowledges PR with "Starting download..."
- Dataset downloads to `./datasets/`
- On success: Comments with path + auto-merges PR
- On failure: Comments with error + keeps PR open

## YAML Schema

### Add Dataset (Minimal)

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data"
```

### Add Dataset (With Options)

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
description: "Malay ASR training data"
revision: "main"           # default: "main"
subset: "ms"               # default: None (downloads all)
split: "train"             # default: "train"
audio_column: "audio"      # default: "audio"
text_column: "text"        # default: "text"
```

### Remove Dataset

```yaml
hf_id: "mozilla-foundation/common_voice_11_0"
reason: "No longer needed"
```

## CLI Commands

```bash
# Validate YAML without applying
hugflow validate <yaml-file>

# Show status of all datasets
hugflow status

# Sync single dataset (local testing)
hugflow local-sync <yaml-file>

# Initialize directories
hugflow init

# Sync in CI mode (called by GitHub Actions)
hugflow sync --ci-mode
```

## Storage Layout

```
datasets/
├── mozilla-foundation__common_voice_11_0__rev_main/
│   ├── audio/          # Audio files
│   └── json/           # All other data as JSON
├── mozilla-foundation__common_voice_11_0__subset_ms__split_train__rev_main/
│   ├── audio/
│   └── json/

.hugflow-state/  # Gitignored
├── active.json      # Currently managed datasets
├── archived.json    # Historical operations
└── progress/        # Download progress tracking
```

## Configuration

All settings are configurable via environment variables in `.env`:

- `HF_TOKEN` - Hugging Face API token (required)
- `GITHUB_TOKEN` - GitHub token for PR operations (optional, see below)
- `SLACK_WEBHOOK_URL` - Optional Slack webhook for notifications
- `STORAGE_ROOT` - Dataset storage directory (default: `./datasets`)
- `PROGRESS_INTERVAL` - Progress update interval (default: 10000 files)
- `MAX_CONCURRENT_DOWNLOADS` - Concurrent download limit (default: 5)
- `AUTO_MERGE` - Auto-merge successful PRs (default: true)

See `.env.example` for all available settings.

### GitHub Authentication (Two Options)

Hugflow supports **two authentication methods** for GitHub operations:

#### Option 1: GitHub Personal Access Token (Recommended for CI)

Set `GITHUB_TOKEN` in your `.env` file:

```bash
# Create a token at: https://github.com/settings/tokens
# Required scopes: repo (full control)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Use this for:**
- CI/CD (GitHub Actions) - token is automatically provided
- Testing PR operations locally
- Automated scripts

#### Option 2: GitHub CLI (`gh`) - Easiest for Local Dev

Use `gh` CLI without setting any token:

```bash
# Install gh CLI: https://cli.github.com/
# Authenticate once:
gh auth login

# Hugflow will auto-detect and use it!
# No need to set GITHUB_TOKEN in .env
```

**Use this for:**
- Local development and testing
- Interactive workflows
- When you don't want to manage tokens

**How it works:**
- Hugflow first tries to use `GITHUB_TOKEN` if set
- Falls back to `gh CLI` if token not available
- Disables GitHub features if neither is available (with warning)

## GitOps Workflow

1. User creates PR with YAML file in `requests/` directory
2. GitHub Actions workflow triggers on PR (self-hosted runner on HGX)
3. Bot acknowledges PR with "Starting download..." comment
4. Downloads dataset and validates
5. On success: Comments with path location and auto-merges PR
6. On failure: Comments with error stack trace and keeps PR open

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=hugflow

# Type checking
mypy hugflow
```

## License

MIT
