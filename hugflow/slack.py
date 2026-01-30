"""
Slack notifications for Hugflow.
"""

from typing import Optional

import aiohttp
from structlog import get_logger

from hugflow.config import Config, DatasetSpec

logger = get_logger(__name__)


class SlackNotifier:
    """Notifier for Slack integration."""

    def __init__(self, config: Config):
        """
        Initialize Slack notifier.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.webhook_url = config.slack.webhook_url
        self.channel = config.slack.channel
        self.enabled = config.slack.enabled and bool(self.webhook_url)

    def _send_message(self, blocks: list) -> None:
        """
        Send message to Slack webhook.

        Args:
            blocks: Slack Block Kit blocks
        """
        if not self.enabled:
            logger.debug("Slack notifications disabled, skipping")
            return

        log = logger.bind(webhook_url=self.webhook_url[:20] + "...")
        log.info("Sending Slack notification")

        try:
            import asyncio

            async def send():
                async with aiohttp.ClientSession() as session:
                    payload = {"blocks": blocks}
                    if self.channel:
                        payload["channel"] = self.channel

                    async with session.post(self.webhook_url, json=payload) as response:
                        if response.status != 200:
                            text = await response.text()
                            log.warning(
                                "Slack webhook returned non-200 status",
                                status=response.status,
                                response=text,
                            )

            asyncio.run(send())

        except Exception as e:
            log.warning("Failed to send Slack notification", error=str(e))
            # Don't raise - Slack failures should not block downloads

    def send_download_started(
        self,
        spec: DatasetSpec,
        pr_url: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Send notification when download starts.

        Args:
            spec: Dataset specification
            pr_url: URL to PR
            requested_by: User who requested download (e.g., "@username")
        """
        subset_info = ""
        if spec.subset:
            subset_info = f"\n*Subset:* {spec.subset}"
        if spec.split:
            subset_info += f"\n*Split:* {spec.split}"

        pr_info = f"\n*PR:* {pr_url}" if pr_url else ""
        requested_by_info = f"\n*Requested by:* {requested_by}" if requested_by else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📥 Dataset Download Started",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {spec.hf_id}"},
                ],
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Description:* {spec.description}"},
                ],
            },
        ]

        if subset_info or pr_info or requested_by_info:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": subset_info or " "},
                        {"type": "mrkdwn", "text": requested_by_info or " "},
                    ],
                }
            )

        if pr_info:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": pr_info},
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"I'll post progress updates every {self.config.download.slack_notification_interval} files.",
                    }
                ],
            }
        )

        self._send_message(blocks)

    def send_progress(
        self,
        dataset_name: str,
        downloaded: int,
        total: int,
        percent: float,
        eta: str = "",
        pr_url: str = "",
    ) -> None:
        """
        Send progress update notification.

        Args:
            dataset_name: Name of dataset
            downloaded: Number of files downloaded
            total: Total number of files
            percent: Percentage complete
            eta: Estimated time remaining
            pr_url: URL to PR
        """
        pr_info = f"\n*PR:* {pr_url}" if pr_url else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 Download Progress: {percent:.1f}%",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {dataset_name}"},
                    {"type": "mrkdwn", "text": f"*Files:* {downloaded:,} / {total:,} ({percent:.1f}%)"},
                    {"type": "mrkdwn", "text": f"*ETA:* {eta}"},
                    {"type": "mrkdwn", "text": " "},
                ],
            },
        ]

        if pr_info:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": pr_info},
                }
            )

        self._send_message(blocks)

    def send_download_complete(
        self,
        spec: DatasetSpec,
        file_count: int,
        storage_path: str,
        size_gb: float = 0.0,
        pr_url: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Send notification when download completes successfully.

        Args:
            spec: Dataset specification
            file_count: Number of files downloaded
            storage_path: Path to downloaded dataset
            size_gb: Size in GB
            pr_url: URL to PR
            requested_by: User who requested download
        """
        subset_info = ""
        if spec.subset:
            subset_info = f" ({spec.subset}, {spec.split})"

        requested_by_info = f"\n*Requested by:* {requested_by}" if requested_by else ""
        pr_info = f"\n*PR:* {pr_url} ✓ Merged" if pr_url else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "✅ Download Complete!",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {spec.hf_id}{subset_info}"},
                    {"type": "mrkdwn", "text": f"*Files:* {file_count:,}"},
                ],
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Size:* {size_gb:.2f} GB"},
                    {"type": "mrkdwn", "text": " "},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Location:* `{storage_path}`",
                },
            },
        ]

        if requested_by_info or pr_info:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": requested_by_info or " "},
                        {"type": "mrkdwn", "text": pr_info or " "},
                    ],
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "🎉 Ready to use!",
                    }
                ],
            }
        )

        self._send_message(blocks)

    def send_download_failed(
        self,
        spec: DatasetSpec,
        error: str,
        pr_url: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Send notification when download fails.

        Args:
            spec: Dataset specification
            error: Error message
            pr_url: URL to PR
            requested_by: User who requested download
        """
        subset_info = ""
        if spec.subset:
            subset_info = f" ({spec.subset}, {spec.split})"

        requested_by_info = f"\n*Requested by:* {requested_by}" if requested_by else ""
        pr_info = f"\n*PR:* {pr_url}" if pr_url else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "❌ Download Failed",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {spec.hf_id}{subset_info}"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:* {error}",
                },
            },
        ]

        if requested_by_info or pr_info:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": requested_by_info or " "},
                        {"type": "mrkdwn", "text": pr_info or " "},
                    ],
                }
            )

        self._send_message(blocks)

    def send_removal_complete(
        self,
        spec: DatasetSpec,
        space_freed_gb: float = 0.0,
        pr_url: str = "",
        requested_by: str = "",
    ) -> None:
        """
        Send notification when dataset removal completes.

        Args:
            spec: Dataset specification (being removed)
            space_freed_gb: Space freed in GB
            pr_url: URL to PR
            requested_by: User who requested removal
        """
        requested_by_info = f"\n*Requested by:* {requested_by}" if requested_by else ""
        pr_info = f"\n*PR:* {pr_url}" if pr_url else ""

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🗑️ Dataset Removed",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Dataset:* {spec.hf_id}"},
                ],
            },
        ]

        if space_freed_gb > 0:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Space freed:* {space_freed_gb:.2f} GB"},
                        {"type": "mrkdwn", "text": " "},
                    ],
                }
            )

        if requested_by_info or pr_info:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": requested_by_info or " "},
                        {"type": "mrkdwn", "text": pr_info or " "},
                    ],
                }
            )

        self._send_message(blocks)
