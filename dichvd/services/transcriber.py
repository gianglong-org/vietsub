"""
Nhận diện lời nói (ASR).
Hỗ trợ Whisper local hoặc API tương đương.

Với Gemini, audio được cắt thành các đoạn (chunk) rồi gửi lần lượt. Mốc thời gian
Gemini trả về đôi khi lệch (vượt độ dài đoạn, end <= start, dạng "1:05", đơn vị
mili giây...). Thay vì hủy cả quá trình, code tự chuẩn hoá và chỉ bỏ những đoạn
không thể sửa; chỉ báo lỗi khi không còn đoạn hợp lệ nào.
"""
from __future__ import annotations

import json
import logging
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .gemini import create_gemini_client

logger = logging.getLogger(__name__)

# Dung sai khi Gemini làm tròn mốc cuối đoạn (giây).
END_TOLERANCE = 1.0
# Độ dài tối thiểu của một đoạn phụ đề sau khi sửa (giây).
MIN_SEGMENT_SECONDS = 0.3
# Ước lượng thời gian đọc mỗi ký tự khi phải tự suy ra mốc kết thúc (giây).
SECONDS_PER_CHAR = 0.07
# Số lần gọi Gemini tối đa cho mỗi đoạn audio nếu kết quả hỏng/không dùng được.
MAX_ATTEMPTS = 2


@dataclass
class TranscriptSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None


def _parse_seconds(value) -> Optional[float]:
    """
    Đổi giá trị thời gian Gemini trả về thành số giây.
    Hỗ trợ: số, "12.5", "1:05", "01:02:03.5", "00:01:05,500". Không đọc được -> None.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None

    text = value.strip().lower().replace(",", ".")
    text = re.sub(r"\s*(s|sec|seconds?)$", "", text)
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if len(parts) > 3 or any(not math.isfinite(p) for p in parts):
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _clean_chunk_segments(raw_segments: list, chunk_length: float) -> tuple[list, int]:
    """
    Chuẩn hoá các đoạn của một chunk. Trả về ([(start, end, text), ...], số đoạn bị bỏ).
    Thời gian tính bằng giây, tương đối so với đầu chunk.
    """
    parsed = []
    dropped = 0
    for item in raw_segments:
        if not isinstance(item, dict):
            dropped += 1
            continue
        text = str(item.get("text") or "").strip()
        start = _parse_seconds(item.get("start_seconds"))
        end = _parse_seconds(item.get("end_seconds"))
        if not text or start is None or end is None:
            dropped += 1
            continue
        parsed.append([start, end, text])

    if not parsed:
        return [], dropped

    # Gemini đôi khi trả mili giây dù yêu cầu giây: nếu mọi giá trị đều "quá lớn" nhưng
    # chia 1000 thì vừa khít độ dài chunk, coi như đơn vị mili giây.
    biggest = max(max(s, e) for s, e, _ in parsed)
    if biggest > chunk_length * 1.5 and biggest / 1000 <= chunk_length + END_TOLERANCE:
        logger.warning("Gemini có vẻ trả mốc thời gian bằng mili giây, tự đổi sang giây")
        for seg in parsed:
            seg[0] /= 1000
            seg[1] /= 1000

    cleaned = []
    for start, end, text in parsed:
        start = max(0.0, start)
        if start >= chunk_length - 0.05:
            dropped += 1  # bắt đầu sau khi đoạn audio đã hết
            continue
        if end < start:
            start, end = end, start  # Gemini đảo ngược start/end
            start = max(0.0, start)
        if end <= start:
            end = start + max(MIN_SEGMENT_SECONDS, len(text) * SECONDS_PER_CHAR)
        end = min(end, chunk_length)  # chỉ lệch vượt cuối đoạn thì kẹp lại
        if end - start < 0.05:
            dropped += 1
            continue
        cleaned.append([start, end, text])

    # Sắp theo thời gian và gỡ chồng lấn để mốc luôn tăng dần.
    cleaned.sort(key=lambda s: (s[0], s[1]))
    result = []
    for start, end, text in cleaned:
        if result:
            prev = result[-1]
            if start < prev[1]:
                if start - prev[0] >= MIN_SEGMENT_SECONDS:
                    prev[1] = start  # cắt đoạn trước cho khớp
                else:
                    start = prev[1]  # đoạn trước quá ngắn, dời đoạn này lên sau
                    if end - start < 0.05:
                        dropped += 1
                        continue
        result.append([start, end, text])
    return [(s, e, t) for s, e, t in result], dropped


class Transcriber:
    """
    Nhận diện lời nói qua Gemini hoặc Whisper local.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        language: str = "auto",
        provider: str = "gemini",
        api_model: str = "gemini-3.8-flash",
        chunk_seconds: int = 480,
    ):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.provider = provider
        self.api_model = api_model
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds phải lớn hơn 0.")
        self.chunk_seconds = chunk_seconds
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Loaded Whisper model: %s on %s", self.model_size, self.device)
        except ImportError:
            raise RuntimeError(
                "openai-whisper chưa được cài. Chạy: pip install openai-whisper"
            )

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> List[TranscriptSegment]:
        """
        Trả về danh sách đoạn có timestamp.
        """
        if self.provider == "gemini":
            return self._transcribe_gemini(audio_path, language)
        if self.provider != "local":
            raise ValueError(f"Transcription provider không hỗ trợ: {self.provider}")

        self._load_model()
        lang = language or (None if self.language == "auto" else self.language)

        result = self._model.transcribe(
            audio_path,
            language=lang,
            word_timestamps=False,
            verbose=False,
        )

        segments: List[TranscriptSegment] = []
        for i, seg in enumerate(result.get("segments", [])):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            segments.append(
                TranscriptSegment(
                    index=i,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=seg.get("avg_logprob"),  # Whisper không có confidence chuẩn
                )
            )

        logger.info("Transcribed %d segments from %s", len(segments), audio_path)
        return segments

    def _request_chunk(self, client, types, chunk_path: Path, language: Optional[str]) -> list:
        """Gửi một đoạn audio lên Gemini, trả về danh sách segment thô (chưa kiểm tra)."""
        language_instruction = (
            f"Ngôn ngữ lời thoại: {language}."
            if language
            else "Tự nhận diện ngôn ngữ lời thoại."
        )
        prompt = f"""Chép chính xác lời thoại trong audio thành các đoạn phụ đề.
{language_instruction}
Trả về JSON theo schema, mỗi đoạn gồm start_seconds và end_seconds là SỐ GIÂY
(số thập phân, không phải mm:ss, không phải mili giây) tính từ đầu file audio này,
cùng text là nguyên văn lời thoại. end_seconds phải lớn hơn start_seconds, các đoạn
xếp theo thứ tự thời gian và không chồng lấn.
Không dịch, không tóm tắt, không tạo nội dung không nghe thấy.
Chỉ đưa vào segments các đoạn có lời nói."""
        with chunk_path.open("rb") as audio_file:
            audio_part = types.Part.from_bytes(
                data=audio_file.read(),
                mime_type="audio/wav",
            )
        response = client.models.generate_content(
            model=self.api_model,
            contents=[prompt, audio_part],
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start_seconds": {"type": "number"},
                                    "end_seconds": {"type": "number"},
                                    "text": {"type": "string"},
                                },
                                "required": ["start_seconds", "end_seconds", "text"],
                            },
                        }
                    },
                    "required": ["segments"],
                },
            },
        )
        result = json.loads(response.text or "")
        segments = result.get("segments", []) if isinstance(result, dict) else []
        return segments if isinstance(segments, list) else []

    def _transcribe_gemini(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        client = create_gemini_client()
        from google.genai import types

        ffprobe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(ffprobe.stdout.strip())
        if duration <= 0:
            raise RuntimeError("Không tìm thấy âm thanh có thời lượng hợp lệ trong video.")

        segments: List[TranscriptSegment] = []
        total_dropped = 0
        with tempfile.TemporaryDirectory(prefix="vietsub-transcription-") as temp_dir:
            for start_seconds in range(0, math.ceil(duration), self.chunk_seconds):
                chunk_path = Path(temp_dir) / f"chunk_{start_seconds}.wav"
                chunk_length = min(self.chunk_seconds, duration - start_seconds)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(start_seconds),
                        "-i",
                        audio_path,
                        "-t",
                        str(chunk_length),
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        str(chunk_path),
                    ],
                    capture_output=True,
                    check=True,
                )
                if chunk_path.stat().st_size > 20_000_000:
                    raise RuntimeError(
                        "Đoạn âm thanh vượt giới hạn tải trực tiếp lên Gemini; "
                        "hãy giảm chunk_seconds."
                    )

                cleaned: list = []
                last_error: Optional[Exception] = None
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        raw_segments = self._request_chunk(client, types, chunk_path, language)
                    except (ValueError, KeyError, TypeError) as e:  # JSON hỏng/bị cắt
                        last_error = e
                        logger.warning(
                            "Chunk %ss: Gemini trả dữ liệu không đọc được (lần %d/%d): %s",
                            start_seconds, attempt, MAX_ATTEMPTS, e,
                        )
                        continue

                    cleaned, dropped = _clean_chunk_segments(raw_segments, chunk_length)
                    if dropped:
                        logger.warning(
                            "Chunk %ss: bỏ %d/%d đoạn có timestamp không sửa được (lần %d/%d)",
                            start_seconds, dropped, len(raw_segments), attempt, MAX_ATTEMPTS,
                        )
                    # Có kết quả dùng được, hoặc chunk thực sự không có lời -> dừng thử lại.
                    if cleaned or not raw_segments:
                        total_dropped += dropped
                        break
                    last_error = RuntimeError("mọi đoạn đều có timestamp không hợp lệ")
                else:
                    if not cleaned and last_error is not None:
                        raise RuntimeError(
                            f"Gemini trả về timestamp không hợp lệ khi nhận diện audio "
                            f"(đoạn từ giây {start_seconds}): {last_error}"
                        )

                for start, end, text in cleaned:
                    segments.append(
                        TranscriptSegment(
                            index=len(segments),
                            start_ms=int(round((start_seconds + start) * 1000)),
                            end_ms=int(round((start_seconds + end) * 1000)),
                            text=text,
                        )
                    )

        if not segments:
            raise RuntimeError("Không nhận diện được lời thoại nào trong video.")
        logger.info(
            "Transcribed %d segments from %s (bỏ %d đoạn timestamp không hợp lệ)",
            len(segments), audio_path, total_dropped,
        )
        return segments

    def transcribe_to_dicts(self, audio_path: str, language: Optional[str] = None) -> List[dict]:
        segs = self.transcribe(audio_path, language=language)
        return [
            {
                "index": s.index,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "text": s.text,
                "confidence": s.confidence,
                "speaker": s.speaker,
            }
            for s in segs
        ]