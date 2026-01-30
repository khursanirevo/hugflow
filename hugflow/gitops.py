"""
GitOps integration for GitHub PR handling.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from github import Github, GithubException
from structlog import get_logger

from hugflow.config import Config

logger = get_logger(__name__)


class GitOpsError(Exception):
    """Base exception for GitOps errors."""

    pass


class GitOpsManager:
    """Manager for GitHub PR operations."""

    def __init__(self, config: Config):
        """
        Initialize GitOps manager.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.pr_number = None
        self.repo = None
        self.use_gh_cli = False

        # Try to initialize with GitHub token (for CI)
        if config.github.enabled and config.github.token:
            self.github = Github(config.github.token)
            try:
                self.repo = self.github.get_repo(config.github.repo)
                logger.info("Using GitHub token for authentication")
            except Exception as e:
                logger.warning("Failed to initialize GitHub client with token", error=str(e))
                self.github = None
                self.repo = None
        else:
            self.github = None
            self.repo = None

        # Fall back to gh CLI if token not available
        if not self.repo:
            if self._check_gh_cli():
                self.use_gh_cli = True
                logger.info("Using gh CLI for authentication")
            else:
                logger.warning("No GitHub authentication available (token or gh CLI)")

        # Try to get PR number from environment
        self.pr_number = os.getenv("PR_NUMBER")
        if self.pr_number:
            self.pr_number = int(self.pr_number)

    def _check_gh_cli(self) -> bool:
        """Check if gh CLI is available and authenticated."""
        try:
            # Check if gh is installed
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return False

            # Check if authenticated
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_gh(self, args: List[str]) -> str:
        """
        Run a gh CLI command and return output.

        Args:
            args: Command arguments (without 'gh' prefix)

        Returns:
            Command output as string

        Raises:
            GitOpsError: If command fails
        """
        try:
            cmd = ["gh"] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return result.stdout.strip()

        except subprocess.CalledProcessError as e:
            raise GitOpsError(f"gh command failed: {e.stderr}") from e
        except subprocess.TimeoutExpired:
            raise GitOpsError("gh command timed out") from None
        except Exception as e:
            raise GitOpsError(f"Unexpected error running gh: {e}") from e

    def get_pr_files(self, pr_number: Optional[int] = None) -> List[Path]:
        """
        Get list of YAML files changed in a PR.

        Args:
            pr_number: PR number (defaults to environment variable)

        Returns:
            List of paths to YAML files

        Raises:
            GitOpsError: If unable to fetch PR files
        """
        pr_number = pr_number or self.pr_number

        if not pr_number:
            raise GitOpsError("PR number not provided and not found in environment")

        log = logger.bind(pr_number=pr_number)
        log.info("Fetching PR files")

        try:
            pull_request = self.repo.get_pull(pr_number)
            files = pull_request.get_files()

            yaml_files = []
            for file in files:
                if file.filename.startswith("requests/add/") or file.filename.startswith("requests/remove/"):
                    if file.filename.endswith((".yaml", ".yml")):
                        yaml_files.append(Path(file.filename))

            log.info("Found YAML files in PR", count=len(yaml_files))
            return yaml_files

        except GithubException as e:
            log.error("GitHub API error", error=str(e))
            raise GitOpsError(f"GitHub API error: {e}") from e
        except Exception as e:
            log.error("Unexpected error fetching PR files", error=str(e))
            raise GitOpsError(f"Unexpected error: {e}") from e

    def comment_start(self, pr_number: Optional[int] = None, dataset_name: str = "") -> None:
        """
        Post "Starting download..." comment on PR.

        Args:
            pr_number: PR number
            dataset_name: Name of dataset being downloaded
        """
        pr_number = pr_number or self.pr_number

        if not pr_number or not self.repo:
            logger.warning("Cannot comment: PR number or repo not available")
            return

        log = logger.bind(pr_number=pr_number, dataset_name=dataset_name)
        log.info("Posting start comment")

        try:
            pull_request = self.repo.get_pull(pr_number)
            comment = f"""### 🚀 Starting Download

Downloading dataset: **{dataset_name}**

This may take hours or days depending on size.

I'll post progress updates every {self.config.download.pr_comment_interval} files.
"""
            pull_request.create_comment(comment)
            log.info("Start comment posted successfully")

        except Exception as e:
            log.warning("Failed to post start comment", error=str(e))

    def comment_progress(
        self,
        pr_number: Optional[int] = None,
        dataset_name: str = "",
        downloaded: int = 0,
        total: int = 0,
        percent: float = 0.0,
        eta: str = "",
    ) -> None:
        """
        Post progress update comment on PR.

        Args:
            pr_number: PR number
            dataset_name: Name of dataset
            downloaded: Number of files downloaded
            total: Total number of files
            percent: Percentage complete
            eta: Estimated time remaining
        """
        pr_number = pr_number or self.pr_number

        if not pr_number or not self.repo:
            logger.warning("Cannot comment: PR number or repo not available")
            return

        log = logger.bind(pr_number=pr_number, dataset_name=dataset_name, downloaded=downloaded, total=total)
        log.debug("Posting progress comment")

        try:
            pull_request = self.repo.get_pull(pr_number)
            comment = f"""### 📥 Download Progress

**Dataset:** {dataset_name}
**Progress:** {downloaded:,} / {total:,} files ({percent:.1f}%)
**ETA:** {eta}

Currently downloading...
"""
            pull_request.create_comment(comment)
            log.info("Progress comment posted successfully")

        except Exception as e:
            log.warning("Failed to post progress comment", error=str(e))

    def comment_success(
        self,
        pr_number: Optional[int] = None,
        dataset_name: str = "",
        file_count: int = 0,
        storage_path: str = "",
    ) -> None:
        """
        Post success comment and auto-merge PR.

        Args:
            pr_number: PR number
            dataset_name: Name of dataset
            file_count: Number of files downloaded
            storage_path: Path to downloaded dataset
        """
        pr_number = pr_number or self.pr_number

        if not pr_number or not self.repo:
            logger.warning("Cannot comment: PR number or repo not available")
            return

        log = logger.bind(pr_number=pr_number, dataset_name=dataset_name, file_count=file_count)
        log.info("Posting success comment")

        try:
            pull_request = self.repo.get_pull(pr_number)
            comment = f"""### ✅ Download Complete!

**Dataset:** {dataset_name}
**Files downloaded:** {file_count:,}
**Location:** `{storage_path}`

Auto-merging PR...
"""
            pull_request.create_comment(comment)

            # Auto-merge if configured
            if self.config.behavior.auto_merge:
                self.merge_pr(pr_number)

            log.info("Success comment posted, PR merged")

        except Exception as e:
            log.warning("Failed to post success comment", error=str(e))

    def comment_failure(
        self,
        pr_number: Optional[int] = None,
        dataset_name: str = "",
        error: str = "",
        error_type: str = "",
        suggestion: str = "",
    ) -> None:
        """
        Post failure comment with error details.

        Args:
            pr_number: PR number
            dataset_name: Name of dataset
            error: Error message
            error_type: Type of error
            suggestion: Helpful suggestion for fixing the error
        """
        pr_number = pr_number or self.pr_number

        if not pr_number or not self.repo:
            logger.warning("Cannot comment: PR number or repo not available")
            return

        log = logger.bind(pr_number=pr_number, dataset_name=dataset_name, error_type=error_type)
        log.info("Posting failure comment")

        try:
            pull_request = self.repo.get_pull(pr_number)

            comment = f"""### ❌ Failed to download dataset

**Dataset:** {dataset_name}
**Error:** {error_type or "Unknown error"}

<details>
<summary>🔍 Detailed Error Information</summary>

**Error:** {error}

"""
            if suggestion:
                comment += f"**💡 Suggestion:** {suggestion}\n\n"

            comment += """Please fix the YAML and push a new commit.
</details>
"""

            pull_request.create_comment(comment)
            log.info("Failure comment posted successfully")

        except Exception as e:
            log.warning("Failed to post failure comment", error=str(e))

    def merge_pr(self, pr_number: Optional[int] = None) -> None:
        """
        Merge a PR.

        Args:
            pr_number: PR number

        Raises:
            GitOpsError: If merge fails
        """
        pr_number = pr_number or self.pr_number

        if not pr_number or not self.repo:
            logger.warning("Cannot merge: PR number or repo not available")
            return

        log = logger.bind(pr_number=pr_number)
        log.info("Merging PR")

        try:
            pull_request = self.repo.get_pull(pr_number)
            pull_request.merge(
                commit_title="Auto-merge: Dataset download successful",
                commit_message="Automatically merged after successful dataset download.",
                merge_method="merge",
            )
            log.info("PR merged successfully")

        except GithubException as e:
            log.error("Failed to merge PR", error=str(e))
            raise GitOpsError(f"Failed to merge PR: {e}") from e
        except Exception as e:
            log.error("Unexpected error merging PR", error=str(e))
            raise GitOpsError(f"Unexpected error: {e}") from e

    def write_results(self, results: Dict[str, Any]) -> None:
        """
        Write results.json for GitHub Actions to read.

        Args:
            results: Results dictionary
        """
        results_path = Path("results.json")

        log = logger.bind(results_path=str(results_path))
        log.info("Writing results.json")

        try:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            log.info("results.json written successfully")

        except Exception as e:
            log.error("Failed to write results.json", error=str(e))
            raise GitOpsError(f"Failed to write results.json: {e}") from e

    def get_pr_url(self, pr_number: Optional[int] = None) -> str:
        """
        Get URL for a PR.

        Args:
            pr_number: PR number

        Returns:
            PR URL
        """
        pr_number = pr_number or self.pr_number

        if not pr_number:
            return ""

        # Try PyGithub first
        if self.repo:
            try:
                pull_request = self.repo.get_pull(pr_number)
                return pull_request.html_url
            except Exception:
                pass

        # Fall back to gh CLI
        if self.use_gh_cli:
            try:
                return self._run_gh([
                    "pr", "view", str(pr_number),
                    "--json", "url",
                    "-q", ".url"
                ])
            except Exception:
                pass

        return ""

    def get_pr_author(self, pr_number: Optional[int] = None) -> str:
        """
        Get PR author username.

        Args:
            pr_number: PR number

        Returns:
            Author username (e.g., "@username")
        """
        pr_number = pr_number or self.pr_number

        if not pr_number:
            return ""

        # Try PyGithub first
        if self.repo:
            try:
                pull_request = self.repo.get_pull(pr_number)
                return f"@{pull_request.user.login}"
            except Exception:
                pass

        # Fall back to gh CLI
        if self.use_gh_cli:
            try:
                author = self._run_gh([
                    "pr", "view", str(pr_number),
                    "--json", "author",
                    "-q", ".author.login"
                ])
                return f"@{author}"
            except Exception:
                pass

        return ""
