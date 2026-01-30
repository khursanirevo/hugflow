"""
Tests for GitOps integration.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hugflow.config import Config
from hugflow.gitops import GitOpsManager


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def gitops_manager(config):
    """Create a GitOps manager for testing."""
    return GitOpsManager(config)


def test_gitops_manager_init_no_github(config):
    """Test GitOpsManager initialization without GitHub."""
    # Config without GitHub settings
    config.github.token = ""
    config.github.repo = ""

    manager = GitOpsManager(config)

    assert manager.github is None
    assert manager.repo is None
    assert manager.pr_number is None


def test_gitops_manager_init_with_github(config):
    """Test GitOpsManager initialization with GitHub."""
    config.github.token = "test_token"
    config.github.repo = "org/repo"

    with patch("hugflow.gitops.Github") as mock_github:
        mock_repo = MagicMock()
        mock_github.return_value.get_repo.return_value = mock_repo

        manager = GitOpsManager(config)

        assert manager.github is not None
        assert manager.repo is not None
        mock_github.assert_called_once_with("test_token")


def test_write_results(gitops_manager, tmp_path):
    """Test writing results.json."""
    import os
    os.chdir(tmp_path)

    results = {
        "status": "success",
        "dataset_name": "test/dataset",
        "file_count": 1000,
    }

    gitops_manager.write_results(results)

    results_file = tmp_path / "results.json"
    assert results_file.exists()

    with open(results_file) as f:
        loaded = json.load(f)

    assert loaded == results


def test_write_results_error(gitops_manager):
    """Test error handling when writing results fails."""
    results = {"status": "success"}

    # Should not raise exception even if write fails
    with patch("builtins.open", side_effect=IOError("Permission denied")):
        with pytest.raises(Exception):  # Should raise GitOpsError
            gitops_manager.write_results(results)


def test_get_pr_url(gitops_manager):
    """Test getting PR URL."""
    # No repo configured
    assert gitops_manager.get_pr_url() == ""

    # With repo mock
    gitops_manager.repo = MagicMock()
    mock_pr = MagicMock()
    mock_pr.html_url = "https://github.com/org/repo/pull/42"
    gitops_manager.repo.get_pull.return_value = mock_pr

    url = gitops_manager.get_pr_url(42)
    assert url == "https://github.com/org/repo/pull/42"


def test_get_pr_author(gitops_manager):
    """Test getting PR author."""
    # No repo configured
    assert gitops_manager.get_pr_author() == ""

    # With repo mock
    gitops_manager.repo = MagicMock()
    mock_pr = MagicMock()
    mock_user = MagicMock()
    mock_user.login = "testuser"
    mock_pr.user = mock_user
    gitops_manager.repo.get_pull.return_value = mock_pr

    author = gitops_manager.get_pr_author(42)
    assert author == "@testuser"


def test_comment_start_no_repo(gitops_manager):
    """Test that comment_start gracefully handles no repo."""
    # Should not raise exception
    gitops_manager.comment_start(pr_number=42, dataset_name="test/dataset")


def test_comment_success_no_repo(gitops_manager):
    """Test that comment_success gracefully handles no repo."""
    # Should not raise exception
    gitops_manager.comment_success(
        pr_number=42,
        dataset_name="test/dataset",
        file_count=1000,
        storage_path="/data/datasets/test",
    )


def test_comment_failure_no_repo(gitops_manager):
    """Test that comment_failure gracefully handles no repo."""
    # Should not raise exception
    gitops_manager.comment_failure(
        pr_number=42,
        dataset_name="test/dataset",
        error="Test error",
        error_type="TestError",
    )


def test_merge_pr_no_repo(gitops_manager):
    """Test that merge_pr gracefully handles no repo."""
    # Should not raise exception
    gitops_manager.merge_pr(pr_number=42)
