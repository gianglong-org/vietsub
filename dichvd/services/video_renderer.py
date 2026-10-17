"""
Ghép audio TTS + video gốc → MP4 đầu ra.
Dùng FFmpeg.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class VideoRenderer:
    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg = ffmpeg_bin

    def render(
        self,
        video_path: str,
        segments: List[dict],
        output_path: str,
        keep_original_audio: bool = False,
        audio_volume: float = 1.0,
    ) -> str:
        """
        Tạo file audio tổng từ các đoạn TTS (theo timeline),
        sau đó mix với video.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        work_dir = Path(output_path).parent / "render_tmp"
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Tạo file audio timeline (silence + chèn TTS)
        timeline_audio = str(work_dir / "timeline.wav")
        self._build_timeline_audio(segments, timeline_audio, video_path)

        # 2. Mix vào video
        if keep_original_audio:
            # Giữ audio gốc + overlay TTS (có thể chỉnh volume)
            cmd = [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-i", timeline_audio,
                "-filter_complex",
                f"[1:a]volume={audio_volume}[a1];[0:a][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]
        else:
            # Thay thế hoàn toàn audio gốc bằng TTS
            cmd = [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-i", timeline_audio,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]

        logger.info("Rendering: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def _build_timeline_audio(
        self,
        segments: List[dict],
        output_wav: str,
        video_path: str,
    ):
        """
        Tạo 1 file audio dài bằng video, chèn từng đoạn TTS đúng vị trí start_ms.
        Cách đơn giản: dùng filter complex adelay + amix.
        """
        # Lấy duration video
        duration = self._get_duration(video_path)
        sample_rate = 24000

        # Tạo silence nền
        silence = str(Path(output_wav).parent / "silence.wav")
        subprocess.run(
            [
                self.ffmpeg, "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=mono",
                "-t", str(duration),
                silence,
            ],
            check=True,
            capture_output=True,
        )

        # Nếu không có đoạn nào có audio → dùng silence
        valid = [s for s in segments if s.get("audio_path") and Path(s["audio_path"]).exists()]
        if not valid:
            Path(silence).rename(output_wav)
            return

        # Xây filter: mỗi TTS được delay rồi amix với silence
        inputs = ["-i", silence]
        filter_parts = []
        mix_inputs = ["[0:a]"]

        for i, seg in enumerate(valid):
            inputs.extend(["-i", seg["audio_path"]])
            delay_ms = int(seg.get("start_ms", 0))
            # adelay tính bằng ms cho mỗi channel
            filter_parts.append(
                f"[{i+1}:a]adelay={delay_ms}|{delay_ms},volume=1.0[a{i}]"
            )
            mix_inputs.append(f"[a{i}]")

        filter_complex = ";".join(filter_parts)
        filter_complex += f";{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=0[aout]"

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-ar", str(sample_rate),
            "-ac", "1",
            output_wav,
        ]
        logger.info("Building timeline audio with %d segments", len(valid))
        subprocess.run(cmd, check=True, capture_output=True)

    def _get_duration(self, path: str) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
