"""
Hugging Face API client for dataset operations.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from structlog import get_logger

from hugflow.config import Config

logger = get_logger(__name__)


def convert_to_mp3(input_path: Path, output_path: Path, bitrate: str = "192k") -> Path:
    """
    Convert audio file to MP3 format preserving original sample rate.

    Args:
        input_path: Path to input audio file (any format)
        output_path: Desired output path (will use .mp3 extension)
        bitrate: MP3 bitrate (default: 192k for high quality)

    Returns:
        Path to the converted MP3 file
    """
    from pydub import AudioSegment

    log = logger.bind(input=str(input_path), output=str(output_path))

    try:
        # Load audio file (pydub auto-detects format)
        audio = AudioSegment.from_file(input_path)

        # Keep original sample rate, convert to mono to avoid multiprocess issues
        original_frame_rate = audio.frame_rate
        audio = audio.set_channels(1)  # Mono to avoid issues

        # Ensure output path has .mp3 extension
        if not output_path.suffix.lower() == ".mp3":
            output_path = output_path.with_suffix(".mp3")

        # Export to MP3 with original sample rate
        audio.export(str(output_path), format="mp3", bitrate=bitrate, parameters=["-ar", str(original_frame_rate)])

        log.info(
            "Audio converted to MP3",
            sample_rate=original_frame_rate,
            bitrate=bitrate,
            size_mb=output_path.stat().st_size / (1024 * 1024),
        )

        return output_path

    except Exception as e:
        log.error("Failed to convert audio to MP3", error=str(e))
        raise


class HFClientError(Exception):
    """Base exception for HF client errors."""

    pass


class HFNotFoundError(HFClientError):
    """Dataset not found on HuggingFace Hub."""

    pass


class HFDownloadError(HFClientError):
    """Error downloading dataset."""

    pass


class HFClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self, config: Config):
        """
        Initialize HF client.

        Args:
            config: Hugflow configuration
        """
        self.config = config
        self.api = HfApi(token=config.hf.token if config.hf.token else None)

    def validate_dataset_id(self, hf_id: str) -> bool:
        """
        Check if a dataset exists on HuggingFace Hub.

        Args:
            hf_id: Dataset ID (e.g., 'org/dataset')

        Returns:
            True if dataset exists

        Raises:
            HFNotFoundError: If dataset doesn't exist
            HFClientError: For other API errors
        """
        log = logger.bind(hf_id=hf_id)
        log.info("Validating dataset ID")

        try:
            # Try to get dataset info
            self.api.dataset_info(repo_id=hf_id)
            log.info("Dataset ID validated successfully")
            return True
        except RepositoryNotFoundError as e:
            log.error("Dataset not found on HuggingFace Hub")
            raise HFNotFoundError(f"Dataset '{hf_id}' not found on HuggingFace Hub") from e
        except HfHubHTTPError as e:
            log.error("HTTP error validating dataset", error=str(e))
            raise HFClientError(f"HTTP error validating dataset: {e}") from e
        except Exception as e:
            log.error("Unexpected error validating dataset", error=str(e))
            raise HFClientError(f"Unexpected error: {e}") from e

    def get_dataset_info(self, hf_id: str, revision: str = "main") -> Dict[str, Any]:
        """
        Get detailed information about a dataset.

        Args:
            hf_id: Dataset ID
            revision: Dataset revision/branch/tag

        Returns:
            Dictionary with dataset metadata including:
            - size: Total size in bytes
            - subsets: List of available subsets/configs
            - siblings: List of files
            - metadata: Other metadata

        Raises:
            HFNotFoundError: If dataset doesn't exist
            HFClientError: For other errors
        """
        log = logger.bind(hf_id=hf_id, revision=revision)
        log.info("Fetching dataset info")

        try:
            info = self.api.dataset_info(repo_id=hf_id, revision=revision)

            # Extract useful information
            result = {
                "hf_id": hf_id,
                "revision": revision,
                "size": sum(s.size for s in info.siblings if s.size) if info.siblings else 0,
                "siblings": [s.rfilename for s in info.siblings] if info.siblings else [],
                "card_data": info.cardData if hasattr(info, "cardData") else None,
                "sha": info.sha if hasattr(info, "sha") else None,
                "last_modified": str(info.last_modified) if hasattr(info, "last_modified") else None,
            }

            log.info("Dataset info fetched successfully", size=result["size"])
            return result

        except RepositoryNotFoundError as e:
            log.error("Dataset not found")
            raise HFNotFoundError(f"Dataset '{hf_id}' not found") from e
        except Exception as e:
            log.error("Error fetching dataset info", error=str(e))
            raise HFClientError(f"Error fetching dataset info: {e}") from e

    def get_available_subsets(self, hf_id: str, revision: str = "main") -> List[str]:
        """
        Get list of available subsets/configs for a dataset.

        Args:
            hf_id: Dataset ID
            revision: Dataset revision

        Returns:
            List of subset/config names

        Raises:
            HFClientError: If unable to fetch subsets
        """
        log = logger.bind(hf_id=hf_id, revision=revision)
        log.info("Fetching available subsets")

        try:
            # For many datasets, subsets are stored as configs
            # We need to try to list them or infer from the dataset structure
            info = self.api.dataset_info(repo_id=hf_id, revision=revision)

            # Try to get card data which might have config info
            if hasattr(info, "cardData") and info.cardData:
                configs = info.cardData.get("configs", [])
                if configs:
                    subset_names = [c.get("config_name") for c in configs if c.get("config_name")]
                    log.info("Found subsets in card data", count=len(subset_names))
                    return subset_names

            # If no configs in card data, check if it's a single-config dataset
            # Return empty list to indicate no subsets (full download)
            log.info("No subsets found, treating as single-config dataset")
            return []

        except Exception as e:
            log.warning("Could not fetch subsets, assuming single-config", error=str(e))
            # Don't fail, just assume single-config
            return []

    def download_dataset(
        self,
        hf_id: str,
        dest: Path,
        revision: str = "main",
        subset: Optional[str] = None,
        split: str = "train",
        audio_column: str = "audio",
        text_column: str = "text",
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Download a dataset from HuggingFace Hub.

        Downloads audio files to audio/ directory and everything else to json/.

        Args:
            hf_id: Dataset ID
            dest: Destination directory (will create audio/ and json/ subdirs)
            revision: Dataset revision
            subset: Dataset subset/config (None for full download)
            split: Dataset split
            audio_column: Name of column containing audio data
            text_column: Name of column containing text data
            progress_callback: Optional callback function(downloaded, total, current_file)

        Returns:
            Dictionary with download statistics:
            - downloaded_files: Number of files downloaded
            - total_files: Total number of files
            - audio_files: Number of audio files
            - json_files: Number of JSON files
            - total_size: Total size in bytes

        Raises:
            HFDownloadError: If download fails
        """
        from datasets import load_dataset

        log = logger.bind(
            hf_id=hf_id,
            revision=revision,
            subset=subset,
            split=split,
            dest=str(dest),
        )
        log.info("Starting dataset download")

        # Create destination directories
        audio_dir = dest / "audio"
        json_dir = dest / "json"
        audio_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset with streaming to handle large datasets
        try:
            log.info("Loading dataset from HuggingFace Hub")
            if subset:
                dataset = load_dataset(
                    hf_id,
                    name=subset,
                    split=split,
                    revision=revision,
                    token=self.config.hf.token or True,
                )
            else:
                # Load without subset (default config)
                dataset = load_dataset(
                    hf_id,
                    split=split,
                    revision=revision,
                    token=self.config.hf.token or True,
                )

        except Exception as e:
            log.error("Failed to load dataset", error=str(e))
            raise HFDownloadError(f"Failed to load dataset: {e}") from e

        # Validate audio column exists and contains audio-like data
        if len(dataset) > 0:
            first_row = dataset[0]
            if audio_column not in first_row:
                # Try to auto-detect audio column
                log.warning(
                    f"Audio column '{audio_column}' not found in dataset",
                    available_columns=list(first_row.keys()),
                    audio_column=audio_column,
                )
                # Suggest common audio column names
                suggestions = [col for col in first_row.keys() if 'audio' in col.lower()]
                if suggestions:
                    raise HFDownloadError(
                        f"Audio column '{audio_column}' not found in dataset. "
                        f"Available columns: {list(first_row.keys())}. "
                        f"Did you mean one of these? {suggestions}. "
                        f"Please specify 'audio_column' in your YAML file."
                    )
                else:
                    raise HFDownloadError(
                        f"Audio column '{audio_column}' not found in dataset. "
                        f"Available columns: {list(first_row.keys())}. "
                        f"Please specify the correct 'audio_column' in your YAML file."
                    )
            else:
                # Column exists, check if it looks like audio data
                audio_data = first_row[audio_column]
                log.debug(
                    "Validating audio column",
                    audio_column=audio_column,
                    audio_type=type(audio_data).__name__,
                )

                # Check if it looks like audio (bytes, dict with bytes/path, AudioDecoder, etc.)
                is_audio = False
                if isinstance(audio_data, dict):
                    if "bytes" in audio_data or "path" in audio_data or "array" in audio_data:
                        is_audio = True
                elif isinstance(audio_data, bytes):
                    is_audio = True
                elif hasattr(audio_data, 'decode') and 'AudioDecoder' in type(audio_data).__name__:
                    # AudioDecoder or similar decodable objects
                    is_audio = True
                    log.info("Detected AudioDecoder object in audio column", audio_type=type(audio_data).__name__)

                if not is_audio:
                    log.warning(
                        "Audio column may not contain audio data",
                        audio_column=audio_column,
                        audio_type=type(audio_data).__name__,
                        suggestion="Check if this is the correct column"
                    )

        # Check if it's a streaming dataset or in-memory
        is_streaming = hasattr(dataset, "_excelecutor") and hasattr(dataset, "_streaming")

        if is_streaming:
            log.info("Dataset is in streaming mode, processing...")
            return self._process_streaming_dataset(
                dataset,
                audio_dir,
                json_dir,
                audio_column,
                text_column,
                progress_callback,
            )
        else:
            log.info("Dataset is loaded in memory", num_rows=len(dataset))
            return self._process_in_memory_dataset(
                dataset,
                audio_dir,
                json_dir,
                audio_column,
                text_column,
                progress_callback,
            )

    def _process_streaming_dataset(
        self,
        dataset,
        audio_dir: Path,
        json_dir: Path,
        audio_column: str,
        text_column: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Process a streaming dataset."""
        import shutil

        log = logger.bind()
        log.info("Processing streaming dataset")

        # For streaming datasets, we need to iterate and download
        downloaded_files = 0
        audio_files = 0
        json_files = 0
        total_size = 0

        # Get total count if available (may not be for streaming)
        try:
            total_files = len(list(dataset.take(100000)))  # This might be slow/inaccurate
        except:
            total_files = 0  # Unknown for streaming

        for idx, example in enumerate(dataset):
            try:
                # Process audio if present
                audio_file_path = None  # Track extracted audio file path

                if audio_column in example:
                    audio_data = example[audio_column]

                    # Handle AudioDecoder objects from torchcodec (not dict, not bytes)
                    if hasattr(audio_data, 'get_all_samples'):
                        log.info("Found AudioDecoder object, extracting audio", row=idx)
                        try:
                            import torch
                            import numpy as np
                            import wave

                            # Get samples - this returns AudioSamples object
                            samples = audio_data.get_all_samples()
                            audio_tensor = samples.data  # torch.Tensor
                            sample_rate = samples.sample_rate

                            # Convert to numpy
                            audio_array = audio_tensor.numpy()
                            # Remove channel dimension if present: (1, samples) -> (samples,)
                            if len(audio_array.shape) == 2 and audio_array.shape[0] == 1:
                                audio_array = audio_array.squeeze(0)

                            # Get filename
                            filename_field = f"{audio_column}_filename"
                            if filename_field in example and example[filename_field]:
                                filename = Path(example[filename_field]).stem
                            else:
                                filename = f"{idx}"
                            mp3_filename = f"{filename}.mp3"
                            dest_path = audio_dir / mp3_filename

                            # Save as WAV first (for pydub compatibility)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp_path = Path(tmp.name)
                                # Write WAV file
                                with wave.open(str(tmp_path), 'wb') as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)  # 16-bit PCM
                                    wav_file.setframerate(sample_rate)
                                    # Convert float32 to int16
                                    audio_int16 = (audio_array * 32767).astype(np.int16)
                                    wav_file.writeframes(audio_int16.tobytes())

                            # Convert to MP3
                            try:
                                dest_path = convert_to_mp3(tmp_path, dest_path, bitrate="192k")
                                audio_files += 1
                                total_size += dest_path.stat().st_size
                                downloaded_files += 1
                                log.info("Successfully extracted and converted AudioDecoder audio", row=idx, file=str(dest_path))
                            finally:
                                tmp_path.unlink(missing_ok=True)

                            # Replace with path
                            example[audio_column] = str(dest_path)
                        except Exception as e:
                            log.warning("Failed to extract audio from AudioDecoder", row=idx, error=str(e))
                            example[audio_column] = None

                    elif isinstance(audio_data, dict):
                        # Handle dict with bytes field (embedded audio data)
                        if "bytes" in audio_data:
                            # Extract bytes and save to temp file first
                            audio_bytes = audio_data["bytes"]
                            # Determine filename: check for {column}_filename field, then audio_data fields, then row number
                            filename = None
                            # First check if there's a separate field like {audio_column}_filename in the example
                            filename_field = f"{audio_column}_filename"
                            if filename_field in example and example[filename_field]:
                                filename = Path(example[filename_field]).stem  # Remove extension, will add .mp3
                            elif "filename" in audio_data:
                                filename = Path(audio_data["filename"]).stem
                            elif "path" in audio_data:
                                filename = Path(audio_data["path"]).stem
                            else:
                                filename = f"{idx}"
                            # Use .mp3 extension
                            mp3_filename = f"{filename}.mp3"
                            dest_path = audio_dir / mp3_filename
                            # Write to temp file first, then convert to MP3
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp.write(audio_bytes)
                                tmp_path = Path(tmp.name)
                            # Convert to MP3 with safe settings
                            try:
                                dest_path = convert_to_mp3(tmp_path, dest_path, bitrate="192k")
                                audio_files += 1
                                total_size += dest_path.stat().st_size
                                downloaded_files += 1
                            finally:
                                # Clean up temp file
                                tmp_path.unlink(missing_ok=True)
                            # Replace audio field with just the path string
                            example[audio_column] = str(dest_path)
                        # Handle dict with path field (file reference only, no bytes)
                        elif "path" in audio_data:
                            src_path = Path(audio_data["path"])
                            if src_path.exists():
                                # Use the filename stem from path, add .mp3 extension
                                filename = src_path.stem
                                mp3_filename = f"{filename}.mp3"
                                dest_path = audio_dir / mp3_filename
                                # Convert to MP3 with safe settings
                                try:
                                    dest_path = convert_to_mp3(src_path, dest_path, bitrate="192k")
                                    audio_files += 1
                                    total_size += dest_path.stat().st_size
                                    downloaded_files += 1
                                except Exception as e:
                                    log.warning("Failed to convert audio file to MP3", path=str(src_path), error=str(e))
                                    raise
                                # Replace audio field with just the path string
                                example[audio_column] = str(dest_path)
                        else:
                            # Unknown audio format in dict
                            log.warning(
                                "Unknown audio format in dict, skipping audio extraction",
                                audio_keys=list(audio_data.keys()),
                                row=idx,
                            )
                            example[audio_column] = None
                    elif isinstance(audio_data, bytes):
                        # Audio data is raw bytes, save to temp file and convert to MP3
                        filename = f"{idx}"
                        mp3_filename = f"{filename}.mp3"
                        dest_path = audio_dir / mp3_filename
                        # Write to temp file first, then convert to MP3
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_data)
                            tmp_path = Path(tmp.name)
                        # Convert to MP3 with safe settings
                        try:
                            dest_path = convert_to_mp3(tmp_path, dest_path, sample_rate=16000, bitrate="192k")
                            audio_files += 1
                            total_size += dest_path.stat().st_size
                            downloaded_files += 1
                        finally:
                            # Clean up temp file
                            tmp_path.unlink(missing_ok=True)
                        # Replace bytes with path string
                        example[audio_column] = str(dest_path)

                # Save metadata as JSON (without audio bytes, just path reference)
                json_path = json_dir / f"{idx}.json"
                with open(json_path, "w") as f:
                    # Convert to JSON-serializable format (bytes already removed above)
                    json.dump(self._make_json_serializable(example), f, indent=2)
                json_files += 1
                downloaded_files += 1

                # Progress callback
                if progress_callback and idx % self.config.download.progress_interval == 0:
                    progress_callback(downloaded_files, total_files, f"Processing row {idx}")

            except Exception as e:
                log.warning("Failed to process row", row=idx, error=str(e))
                continue

        log.info(
            "Streaming dataset download complete",
            downloaded_files=downloaded_files,
            audio_files=audio_files,
            json_files=json_files,
            total_size=total_size,
        )

        return {
            "downloaded_files": downloaded_files,
            "total_files": total_files,
            "audio_files": audio_files,
            "json_files": json_files,
            "total_size": total_size,
        }

    def _process_in_memory_dataset(
        self,
        dataset,
        audio_dir: Path,
        json_dir: Path,
        audio_column: str,
        text_column: str,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Process an in-memory dataset."""
        import shutil

        log = logger.bind(num_rows=len(dataset))
        log.info("Processing in-memory dataset")

        downloaded_files = 0
        audio_files = 0
        json_files = 0
        total_size = 0
        total_files = len(dataset)

        for idx, example in enumerate(dataset):
            try:
                # Process audio if present
                audio_file_path = None  # Track extracted audio file path

                if audio_column in example:
                    audio_data = example[audio_column]

                    # Handle AudioDecoder objects from torchcodec (not dict, not bytes)
                    if hasattr(audio_data, 'get_all_samples'):
                        log.info("Found AudioDecoder object, extracting audio", row=idx)
                        try:
                            import torch
                            import numpy as np
                            import wave

                            # Get samples - this returns AudioSamples object
                            samples = audio_data.get_all_samples()
                            audio_tensor = samples.data  # torch.Tensor
                            sample_rate = samples.sample_rate

                            # Convert to numpy
                            audio_array = audio_tensor.numpy()
                            # Remove channel dimension if present: (1, samples) -> (samples,)
                            if len(audio_array.shape) == 2 and audio_array.shape[0] == 1:
                                audio_array = audio_array.squeeze(0)

                            # Get filename
                            filename_field = f"{audio_column}_filename"
                            if filename_field in example and example[filename_field]:
                                filename = Path(example[filename_field]).stem
                            else:
                                filename = f"{idx}"
                            mp3_filename = f"{filename}.mp3"
                            dest_path = audio_dir / mp3_filename

                            # Save as WAV first (for pydub compatibility)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp_path = Path(tmp.name)
                                # Write WAV file
                                with wave.open(str(tmp_path), 'wb') as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)  # 16-bit PCM
                                    wav_file.setframerate(sample_rate)
                                    # Convert float32 to int16
                                    audio_int16 = (audio_array * 32767).astype(np.int16)
                                    wav_file.writeframes(audio_int16.tobytes())

                            # Convert to MP3
                            try:
                                dest_path = convert_to_mp3(tmp_path, dest_path, bitrate="192k")
                                audio_files += 1
                                total_size += dest_path.stat().st_size
                                downloaded_files += 1
                                log.info("Successfully extracted and converted AudioDecoder audio", row=idx, file=str(dest_path))
                            finally:
                                tmp_path.unlink(missing_ok=True)

                            # Replace with path
                            example[audio_column] = str(dest_path)
                        except Exception as e:
                            log.warning("Failed to extract audio from AudioDecoder", row=idx, error=str(e))
                            example[audio_column] = None

                    elif isinstance(audio_data, dict):
                        # Handle dict with bytes field (embedded audio data)
                        if "bytes" in audio_data:
                            # Extract bytes and save to temp file first
                            audio_bytes = audio_data["bytes"]
                            # Determine filename: check for {column}_filename field, then audio_data fields, then row number
                            filename = None
                            # First check if there's a separate field like {audio_column}_filename in the example
                            filename_field = f"{audio_column}_filename"
                            if filename_field in example and example[filename_field]:
                                filename = Path(example[filename_field]).stem  # Remove extension, will add .mp3
                            elif "filename" in audio_data:
                                filename = Path(audio_data["filename"]).stem
                            elif "path" in audio_data:
                                filename = Path(audio_data["path"]).stem
                            else:
                                filename = f"{idx}"
                            # Use .mp3 extension
                            mp3_filename = f"{filename}.mp3"
                            dest_path = audio_dir / mp3_filename
                            # Write to temp file first, then convert to MP3
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                                tmp.write(audio_bytes)
                                tmp_path = Path(tmp.name)
                            # Convert to MP3 with safe settings
                            try:
                                dest_path = convert_to_mp3(tmp_path, dest_path, bitrate="192k")
                                audio_files += 1
                                total_size += dest_path.stat().st_size
                                downloaded_files += 1
                            finally:
                                # Clean up temp file
                                tmp_path.unlink(missing_ok=True)
                            # Replace audio field with just the path string
                            example[audio_column] = str(dest_path)
                        # Handle dict with path field (file reference only, no bytes)
                        elif "path" in audio_data:
                            src_path = Path(audio_data["path"])
                            if src_path.exists():
                                # Use the filename stem from path, add .mp3 extension
                                filename = src_path.stem
                                mp3_filename = f"{filename}.mp3"
                                dest_path = audio_dir / mp3_filename
                                # Convert to MP3 with safe settings
                                try:
                                    dest_path = convert_to_mp3(src_path, dest_path, bitrate="192k")
                                    audio_files += 1
                                    total_size += dest_path.stat().st_size
                                    downloaded_files += 1
                                except Exception as e:
                                    log.warning("Failed to convert audio file to MP3", path=str(src_path), error=str(e))
                                    raise
                                # Replace audio field with just the path string
                                example[audio_column] = str(dest_path)
                        else:
                            # Unknown audio format in dict
                            log.warning(
                                "Unknown audio format in dict, skipping audio extraction",
                                audio_keys=list(audio_data.keys()),
                                row=idx,
                            )
                            example[audio_column] = None
                    elif isinstance(audio_data, bytes):
                        # Audio data is raw bytes, save to temp file and convert to MP3
                        filename = f"{idx}"
                        mp3_filename = f"{filename}.mp3"
                        dest_path = audio_dir / mp3_filename
                        # Write to temp file first, then convert to MP3
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_data)
                            tmp_path = Path(tmp.name)
                        # Convert to MP3 with safe settings
                        try:
                            dest_path = convert_to_mp3(tmp_path, dest_path, sample_rate=16000, bitrate="192k")
                            audio_files += 1
                            total_size += dest_path.stat().st_size
                            downloaded_files += 1
                        finally:
                            # Clean up temp file
                            tmp_path.unlink(missing_ok=True)
                        # Replace bytes with path string
                        example[audio_column] = str(dest_path)

                # Save metadata as JSON (without audio bytes, just path reference)
                json_path = json_dir / f"{idx}.json"
                with open(json_path, "w") as f:
                    # Convert to JSON-serializable format (bytes already removed above)
                    json.dump(self._make_json_serializable(example), f, indent=2)
                json_files += 1
                downloaded_files += 1

                # Progress callback
                if progress_callback and idx % self.config.download.progress_interval == 0:
                    progress_callback(downloaded_files, total_files, f"Processing row {idx}")

            except Exception as e:
                log.warning("Failed to process row", row=idx, error=str(e))
                continue

        log.info(
            "In-memory dataset download complete",
            downloaded_files=downloaded_files,
            audio_files=audio_files,
            json_files=json_files,
            total_size=total_size,
        )

        # Validation: Warn if audio column exists but no audio files were downloaded
        if audio_files == 0 and total_files > 0:
            # Check if first row has audio column
            try:
                first_row = dataset[0]
                if audio_column in first_row:
                    log.warning(
                        "Audio column exists but no audio files were extracted!",
                        audio_column=audio_column,
                        total_rows=total_files,
                        message="This might indicate a problem with audio extraction",
                    )
            except:
                pass

        return {
            "downloaded_files": downloaded_files,
            "total_files": total_files,
            "audio_files": audio_files,
            "json_files": json_files,
            "total_size": total_size,
        }

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert an object to JSON-serializable format.

        Handles common non-serializable types from datasets.
        Skips large bytes fields to avoid bloating JSON files.
        """
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, bytes):
            # Skip large byte fields (likely audio/video data) to avoid bloating JSON
            # Only decode small byte fields (metadata, etc.)
            if len(obj) > 1024:  # 1KB threshold
                return f"<{len(obj)} bytes of binary data (skipped)>"
            return obj.decode("utf-8", errors="replace")
        elif hasattr(obj, '__class__') and 'AudioDecoder' in obj.__class__.__name__:
            # AudioDecoder or similar audio objects from torchcodec
            return f"<{obj.__class__.__name__} object (skipped)>"
        else:
            return obj

    def delete_local_copy(self, path: Path) -> None:
        """
        Delete a locally downloaded dataset.

        Args:
            path: Path to dataset directory

        Raises:
            HFClientError: If deletion fails
        """
        import shutil

        log = logger.bind(path=str(path))
        log.info("Deleting local dataset copy")

        try:
            if path.exists():
                shutil.rmtree(path)
                log.info("Local dataset copy deleted successfully")
            else:
                log.warning("Path does not exist, nothing to delete")
        except Exception as e:
            log.error("Failed to delete local copy", error=str(e))
            raise HFClientError(f"Failed to delete local copy: {e}") from e
