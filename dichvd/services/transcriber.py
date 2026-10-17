"""
Nhận diện lời nói (ASR).
Hỗ trợ Whisper local hoặc API tương đương.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .gemini import create_gemini_client

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None


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
        api_model: str = "gemini-2.5-flash",
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

                language_instruction = (
                    f"Ngôn ngữ lời thoại: {language}."
                    if language
                    else "Tự nhận diện ngôn ngữ lời thoại."
                )
                prompt = f"""Chép chính xác lời thoại trong audio thành các đoạn phụ đề.
{language_instruction}
Trả về JSON theo schema, mỗi đoạn gồm start_seconds và end_seconds tính bằng giây
tương đối với đầu file audio này, cùng text là nguyên văn lời thoại.
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
                                        "required": [
                                            "start_seconds",
                                            "end_seconds",
                                            "text",
                                        ],
                                    },
                                }
                            },
                            "required": ["segments"],
                        },
                    },
                )
                result = json.loads(response.text or "")
                for segment in result.get("segments", []):
                    text = segment["text"].strip()
                    if not text:
                        continue
                    start = float(segment["start_seconds"])
                    end = float(segment["end_seconds"])
                    chunk_length = min(
                        self.chunk_seconds,
                        duration - start_seconds,
                    )
                    if (
                        not math.isfinite(start)
                        or not math.isfinite(end)
                        or start < 0
                        or end <= start
                        or end > chunk_length
                    ):
                        raise RuntimeError(
                            "Gemini trả về timestamp không hợp lệ khi nhận diện audio."
                        )
                    segments.append(
                        TranscriptSegment(
                            index=len(segments),
                            start_ms=int((start_seconds + start) * 1000),
                            end_ms=int((start_seconds + end) * 1000),
                            text=text,
                        )
                    )

        if not segments:
            raise RuntimeError("Không nhận diện được lời thoại nào trong video.")
        logger.info("Transcribed %d segments from %s", len(segments), audio_path)
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
