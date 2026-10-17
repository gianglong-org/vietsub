"""
Tạo giọng đọc tiếng Việt (TTS).
Mặc định dùng Gemini TTS, có thể chọn edge-tts.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import List, Optional

from .gemini import create_gemini_client

logger = logging.getLogger(__name__)


class SpeechGenerator:
    """
    Sinh audio cho từng segment hoặc toàn bộ.
    Mặc định dùng Gemini TTS.
    """

    def __init__(
        self,
        provider: str = "gemini",
        voice: str = "Kore",
        output_dir: str = "",
        model: str = "gemini-3.8-flash-tts",
        llm_client=None,
    ):
        self.provider = provider
        self.voice = voice
        self.output_dir = output_dir
        self.model = model
        self.llm = llm_client

    def generate_for_segments(
        self,
        segments: List[dict],
        job_id: str,
        base_dir: str,
    ) -> List[dict]:
        """
        Sinh file wav/mp3 cho từng segment có translated_text.
        Trả về segments với thêm key "audio_path".
        """
        out_dir = Path(base_dir) / str(job_id) / "tts"
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for seg in segments:
            text = (seg.get("translated_text") or "").strip()
            if not text:
                results.append({**seg, "audio_path": ""})
                continue

            extension = "wav" if self.provider == "gemini" else "mp3"
            audio_path = str(out_dir / f"seg_{seg['index']:04d}.{extension}")
            try:
                if self.provider == "edge":
                    self._edge_tts(text, audio_path, self.voice)
                elif self.provider == "gemini":
                    self._gemini_tts(text, audio_path)
                else:
                    raise ValueError(f"Provider không hỗ trợ: {self.provider}")
                if not Path(audio_path).is_file() or Path(audio_path).stat().st_size == 0:
                    raise RuntimeError(
                        f"TTS không tạo được tệp audio cho segment {seg['index']}."
                    )
                results.append({**seg, "audio_path": audio_path})
            except Exception:
                logger.exception("TTS lỗi segment %s", seg.get("index"))
                raise

        return results

    def _edge_tts(self, text: str, output_path: str, voice: str):
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("Cài edge-tts: pip install edge-tts")

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        asyncio.run(_run())

    def _gemini_tts(self, text: str, output_path: str):
        if self.llm is None:
            self.llm = create_gemini_client()
        response = self.llm.interactions.create(
            model=self.model,
            input=[
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": self.voice}]},
        )
        audio = response.output_audio
        if audio is None or not audio.data:
            raise RuntimeError("Gemini không trả về audio cho yêu cầu TTS.")
        Path(output_path).write_bytes(base64.b64decode(audio.data, validate=True))

    def set_voice(self, voice: str):
        self.voice = voice
