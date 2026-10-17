"""
Tiện ích xử lý video cơ bản bằng FFmpeg.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Các thao tác video/audio dùng chung trong pipeline."""

    def __init__(self, ffmpeg_bin: str = "ffmpeg", ffprobe_bin: str = "ffprobe"):
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin

    def get_duration(self, file_path: str) -> float:
        """Trả về thời lượng (giây)."""
        cmd = [
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())

    def extract_audio(
        self,
        video_path: str,
        output_audio_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> str:
        """
        Tách audio thành WAV mono 16kHz (chuẩn tốt cho Whisper).
        """
        Path(output_audio_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            output_audio_path,
        ]
        logger.info("Extracting audio: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        return output_audio_path

    def get_media_info(self, file_path: str) -> dict:
        """Lấy thông tin cơ bản (duration, resolution, codec...)."""
        import json

        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def ensure_dir(self, path: str) -> str:
        Path(path).mkdir(parents=True, exist_ok=True)
        return path
