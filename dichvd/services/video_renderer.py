"""
Ghép audio TTS + video gốc → MP4 đầu ra.
Dùng FFmpeg.

- Kiểm tra video và từng file audio đầu vào bằng ffprobe trước khi chạy FFmpeg.
- FFmpeg ghi vào thư mục/file tạm; chỉ khi chạy thành công và file đầu ra hợp lệ
  mới đổi tên thành output_path (file cũ, nếu có, không bị đụng tới khi lỗi).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """Lỗi trong quá trình render (đầu vào không hợp lệ hoặc FFmpeg thất bại)."""


class VideoRenderer:
    def __init__(
        self,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        timeout: Optional[int] = None,
        fit_audio: bool = True,
        max_speedup: float = 1.3,
        gap_ms: int = 50,
    ):
        self.ffmpeg = ffmpeg_bin
        self.ffprobe = ffprobe_bin
        self.timeout = timeout  # giây; None = không giới hạn
        # Khớp độ dài giọng đọc với thời gian trống trong video:
        self.fit_audio = fit_audio
        self.max_speedup = min(max(max_speedup, 1.0), 2.0)  # atempo ổn định trong khoảng này
        self.gap_ms = max(gap_ms, 0)  # khoảng lặng tối thiểu trước câu kế tiếp
        # Báo cáo của lần render gần nhất: {"sped_up": int, "overflow": [index, ...]}
        self.last_report: dict = {"sped_up": 0, "overflow": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(
        self,
        video_path: str,
        segments: List[dict],
        output_path: str,
        keep_original_audio: bool = False,
        audio_volume: float = 1.0,
        strict: bool = True,
    ) -> str:
        """
        Tạo file audio tổng từ các đoạn TTS (theo timeline), sau đó mix với video.

        strict=True : segment có audio_path nhưng file thiếu/hỏng -> raise RenderError.
        strict=False: bỏ qua segment đó và ghi cảnh báo.
        Chỉ công bố output_path sau khi FFmpeg thành công và file đầu ra hợp lệ.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        video_info = self._probe(video_path)
        if not video_info["has_video"]:
            raise RenderError(f"File đầu vào không có luồng video: {video_path}")
        duration = video_info["duration"]
        if duration <= 0:
            raise RenderError(f"Không xác định được độ dài video: {video_path}")

        valid = self._validate_segments(segments, duration, strict)
        self._plan_fit(valid, duration)

        # Video gốc không có audio thì không thể giữ audio gốc.
        if keep_original_audio and not video_info["has_audio"]:
            logger.warning("Video không có audio gốc, bỏ qua keep_original_audio")
            keep_original_audio = False

        with tempfile.TemporaryDirectory(dir=output.parent, prefix="render_tmp_") as tmp:
            work_dir = Path(tmp)

            # 1. Audio timeline
            timeline_audio = str(work_dir / "timeline.wav")
            self._build_timeline_audio(valid, timeline_audio, duration, work_dir)

            # 2. Mix vào video, ghi ra file tạm
            tmp_output = str(work_dir / f"output{output.suffix or '.mp4'}")
            cmd = self._build_mix_cmd(
                video_path, timeline_audio, tmp_output, keep_original_audio, audio_volume
            )
            logger.info("Rendering: %s", " ".join(cmd))
            self._run(cmd, "render video")

            # 3. Kiểm tra file đầu ra trước khi công bố
            self._verify_output(tmp_output, duration)

            # 4. Công bố nguyên tử (tmp nằm cùng thư mục với output nên cùng filesystem)
            staged = output.with_name(f".{output.name}.partial")
            try:
                os.replace(tmp_output, staged)
                os.replace(staged, output)
            finally:
                staged.unlink(missing_ok=True)

        return output_path

    # ------------------------------------------------------------------
    # Kiểm tra đầu vào
    # ------------------------------------------------------------------
    def _validate_segments(
        self, segments: List[dict], video_duration: float, strict: bool
    ) -> List[dict]:
        valid: List[dict] = []
        for seg in segments:
            path = seg.get("audio_path")
            if not path:
                continue  # segment không có lời (không có translated_text)

            idx = seg.get("index")
            try:
                p = Path(path)
                if not p.is_file():
                    raise RenderError(f"Segment {idx}: không tìm thấy file audio {path}")
                if p.stat().st_size == 0:
                    raise RenderError(f"Segment {idx}: file audio rỗng {path}")
                info = self._probe(path)
                if not info["has_audio"]:
                    raise RenderError(f"Segment {idx}: file không có luồng audio {path}")
                if info["duration"] <= 0:
                    raise RenderError(f"Segment {idx}: audio có độ dài 0 {path}")
            except RenderError as e:
                if strict:
                    raise
                logger.warning("Bỏ qua audio không hợp lệ: %s", e)
                continue

            start_ms = max(0, int(seg.get("start_ms", 0) or 0))
            if start_ms / 1000 >= video_duration:
                logger.warning(
                    "Segment %s bắt đầu (%.1fs) sau khi video kết thúc (%.1fs), bỏ qua",
                    idx, start_ms / 1000, video_duration,
                )
                continue
            valid.append({**seg, "start_ms": start_ms, "audio_duration": info["duration"]})
        return valid

    def _plan_fit(self, valid: List[dict], video_duration: float) -> None:
        """
        Với mỗi đoạn TTS, tính khoảng thời gian tối đa được phép đọc: từ start_ms
        tới lúc câu kế tiếp bắt đầu (hoặc hết video) trừ gap_ms. Nếu audio dài hơn,
        tăng tốc bằng atempo, nhưng không quá max_speedup để giọng không bị méo.
        Đoạn vẫn dài hơn mức cho phép được ghi vào self.last_report["overflow"].
        Kết quả đặt vào seg["tempo"] (1.0 = giữ nguyên).
        """
        report = {"sped_up": 0, "overflow": []}
        self.last_report = report
        for seg in valid:
            seg["tempo"] = 1.0
        if not self.fit_audio:
            return

        ordered = sorted(valid, key=lambda s: s["start_ms"])
        video_ms = int(video_duration * 1000)
        for i, seg in enumerate(ordered):
            next_start = ordered[i + 1]["start_ms"] if i + 1 < len(ordered) else video_ms
            available_ms = next_start - seg["start_ms"] - self.gap_ms
            audio_ms = seg["audio_duration"] * 1000
            idx = seg.get("index")

            if available_ms <= 0:
                report["overflow"].append(idx)
                logger.warning("Segment %s: không còn khoảng trống để đọc", idx)
                continue
            if audio_ms <= available_ms:
                continue

            needed = audio_ms / available_ms
            seg["tempo"] = min(needed, self.max_speedup)
            report["sped_up"] += 1
            if needed > self.max_speedup + 1e-6:
                report["overflow"].append(idx)
                logger.warning(
                    "Segment %s: audio %.1fs dài hơn khung %.1fs (cần x%.2f, tối đa x%.2f)"
                    " - vẫn có thể chồng lên câu sau",
                    idx, audio_ms / 1000, available_ms / 1000, needed, self.max_speedup,
                )

        logger.info(
            "Khớp độ dài: tăng tốc %d đoạn, %d đoạn vẫn quá dài",
            report["sped_up"], len(report["overflow"]),
        )

    def _verify_output(self, path: str, expected_duration: float) -> None:
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            raise RenderError("FFmpeg chạy xong nhưng không tạo được file đầu ra.")
        info = self._probe(path)
        if not info["has_video"] or not info["has_audio"]:
            raise RenderError("File đầu ra thiếu luồng video hoặc audio.")
        # Cho phép lệch nhẹ do làm tròn/-shortest, nhưng không được ngắn bất thường.
        if info["duration"] < expected_duration * 0.9:
            raise RenderError(
                f"File đầu ra quá ngắn ({info['duration']:.1f}s so với {expected_duration:.1f}s)."
            )

    # ------------------------------------------------------------------
    # FFmpeg
    # ------------------------------------------------------------------
    def _build_mix_cmd(
        self,
        video_path: str,
        timeline_audio: str,
        output_path: str,
        keep_original_audio: bool,
        audio_volume: float,
    ) -> List[str]:
        if keep_original_audio:
            # Giữ audio gốc + overlay TTS (có thể chỉnh volume)
            return [
                self.ffmpeg, "-y",
                "-i", video_path,
                "-i", timeline_audio,
                "-filter_complex",
                f"[1:a]volume={audio_volume}[a1];"
                f"[0:a][a1]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path,
            ]
        # Thay thế hoàn toàn audio gốc bằng TTS
        return [
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

    def _build_timeline_audio(
        self,
        valid: List[dict],
        output_wav: str,
        duration: float,
        work_dir: Path,
    ) -> None:
        """
        Tạo 1 file audio dài bằng video, chèn từng đoạn TTS đúng vị trí start_ms.
        Dùng filter complex adelay + amix.
        """
        sample_rate = 24000

        # Silence nền
        silence = str(work_dir / "silence.wav")
        self._run(
            [
                self.ffmpeg, "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r={sample_rate}:cl=mono",
                "-t", str(duration),
                silence,
            ],
            "tạo silence",
        )

        # Không có đoạn nào có audio → dùng silence
        if not valid:
            os.replace(silence, output_wav)
            return

        inputs = ["-i", silence]
        filter_parts = []
        mix_inputs = ["[0:a]"]

        for i, seg in enumerate(valid):
            inputs.extend(["-i", seg["audio_path"]])
            delay_ms = seg["start_ms"]
            tempo = seg.get("tempo", 1.0)
            tempo_filter = f"atempo={tempo:.4f}," if tempo > 1.005 else ""
            filter_parts.append(
                f"[{i+1}:a]{tempo_filter}adelay={delay_ms}|{delay_ms},volume=1.0[a{i}]"
            )
            mix_inputs.append(f"[a{i}]")

        filter_complex = ";".join(filter_parts)
        filter_complex += (
            f";{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}"
            f":duration=first:dropout_transition=0[aout]"
        )

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
        self._run(cmd, "ghép audio timeline")

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------
    def _run(self, cmd: List[str], what: str) -> subprocess.CompletedProcess:
        """Chạy lệnh; lỗi thì ném RenderError kèm phần cuối stderr để dễ debug."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as e:
            raise RenderError(f"Không tìm thấy chương trình '{cmd[0]}': {e}") from e
        except subprocess.TimeoutExpired as e:
            raise RenderError(f"Quá thời gian khi {what} ({self.timeout}s).") from e
        except subprocess.CalledProcessError as e:
            tail = (e.stderr or "").strip()[-2000:]
            raise RenderError(f"Lỗi khi {what} (mã {e.returncode}):\n{tail}") from e

    def _probe(self, path: str) -> dict:
        """Trả về {'has_video', 'has_audio', 'duration'} của một file media."""
        if not Path(path).is_file():
            raise RenderError(f"Không tìm thấy file: {path}")
        result = self._run(
            [
                self.ffprobe, "-v", "error",
                "-show_entries", "stream=codec_type:format=duration",
                "-of", "json",
                path,
            ],
            f"đọc thông tin {Path(path).name}",
        )
        try:
            data = json.loads(result.stdout or "{}")
        except ValueError as e:
            raise RenderError(f"ffprobe trả về dữ liệu không hợp lệ cho {path}") from e

        types = {s.get("codec_type") for s in data.get("streams", [])}
        try:
            duration = float(data.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "has_video": "video" in types,
            "has_audio": "audio" in types,
            "duration": duration,
        }

    def _get_duration(self, path: str) -> float:
        return self._probe(path)["duration"]