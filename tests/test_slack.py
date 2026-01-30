"""
Tests for Slack notifications.
"""

from unittest.mock import MagicMock, patch

import pytest

from hugflow.config import Config, DatasetSpec
from hugflow.slack import SlackNotifier


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def slack_notifier(config):
    """Create a Slack notifier for testing."""
    return SlackNotifier(config)


@pytest.fixture
def dataset_spec():
    """Create a test dataset specification."""
    return DatasetSpec(
        hf_id="mozilla-foundation/common_voice_11_0",
        description="Malay ASR training data",
        subset="ms",
        split="train",
    )


def test_slack_notifier_disabled(config):
    """Test that SlackNotifier is disabled without webhook URL."""
    config.slack.webhook_url = ""
    config.slack.channel = ""

    notifier = SlackNotifier(config)

    assert notifier.enabled is False


def test_slack_notifier_enabled(config):
    """Test that SlackNotifier is enabled with webhook URL."""
    config.slack.webhook_url = "https://hooks.slack.com/services/T00/B00/XXX"
    config.slack.channel = "#ml-datasets"

    notifier = SlackNotifier(config)

    assert notifier.enabled is True
    assert notifier.webhook_url == "https://hooks.slack.com/services/T00/B00/XXX"
    assert notifier.channel == "#ml-datasets"


def test_send_download_started(slack_notifier, dataset_spec):
    """Test sending download started notification."""
    with patch.object(slack_notifier, "_send_message") as mock_send:
        slack_notifier.send_download_started(
            dataset_spec,
            pr_url="https://github.com/org/repo/pull/42",
            requested_by="@testuser",
        )

        # Should have called _send_message with blocks
        assert mock_send.called
        blocks = mock_send.call_args[0][0]
        assert isinstance(blocks, list)


def test_send_progress(slack_notifier):
    """Test sending progress notification."""
    with patch.object(slack_notifier, "_send_message") as mock_send:
        slack_notifier.send_progress(
            dataset_name="test/dataset",
            downloaded=10000,
            total=45000,
            percent=22.2,
            eta="5h",
            pr_url="https://github.com/org/repo/pull/42",
        )

        assert mock_send.called


def test_send_download_complete(slack_notifier, dataset_spec):
    """Test sending download complete notification."""
    with patch.object(slack_notifier, "_send_message") as mock_send:
        slack_notifier.send_download_complete(
            dataset_spec,
            file_count=45230,
            storage_path="/data/datasets/test",
            size_gb=12.3,
            pr_url="https://github.com/org/repo/pull/42",
            requested_by="@testuser",
        )

        assert mock_send.called


def test_send_download_failed(slack_notifier, dataset_spec):
    """Test sending download failed notification."""
    with patch.object(slack_notifier, "_send_message") as mock_send:
        slack_notifier.send_download_failed(
            dataset_spec,
            error="Dataset not found",
            pr_url="https://github.com/org/repo/pull/42",
            requested_by="@testuser",
        )

        assert mock_send.called


def test_send_removal_complete(slack_notifier, dataset_spec):
    """Test sending removal complete notification."""
    with patch.object(slack_notifier, "_send_message") as mock_send:
        slack_notifier.send_removal_complete(
            dataset_spec,
            space_freed_gb=5.2,
            pr_url="https://github.com/org/repo/pull/43",
            requested_by="@testuser",
        )

        assert mock_send.called


def test_send_message_disabled(slack_notifier, caplog):
    """Test that _send_message does nothing when disabled."""
    slack_notifier.enabled = False

    # Should not raise exception or try to send
    slack_notifier._send_message([])

    # No errors should occur


def test_send_message_http_error(slack_notifier, dataset_spec):
    """Test handling of HTTP errors when sending to Slack."""
    import aiohttp

    slack_notifier.enabled = True
    slack_notifier.webhook_url = "https://hooks.slack.com/services/T00/B00/XXX"

    # Mock aiohttp to raise an error
    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = "Internal Server Error"

        mock_post = MagicMock()
        mock_post.__aenter__.return_value = mock_response
        mock_post.__aexit__.return_value = None

        mock_session.return_value.__aenter__.return_value.post.return_value = mock_post

        # Should not raise exception even if Slack fails
        slack_notifier.send_download_started(dataset_spec)
