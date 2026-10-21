"""
Tạo giọng đọc tiếng Việt (TTS).
Mặc định dùng Gemini TTS, có thể chọn edge-tts.

Cache: mỗi file audio có một file metadata đi kèm (seg_0001.meta.json) chứa
"dấu vân tay" của yêu cầu TTS (provider, model, voice, nội dung). Nếu dấu vân
tay khớp và file audio còn nguyên vẹn thì tái sử dụng, không gọi lại API.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from .gemini import create_gemini_client

logger = logging.getLogger(__name__)

# Tăng số này khi đổi cách sinh audio (ví dụ đổi định dạng) để vô hiệu cache cũ.
CACHE_VERSION = 1

# Số lần gọi edge-tts tối đa cho mỗi câu (dịch vụ đôi khi trả "No audio was received").
EDGE_MAX_ATTEMPTS = 3

# Ký tự điều khiển và ký tự độ rộng bằng 0 làm edge-tts không đọc được.
_JUNK_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁠﻿]")


def clean_tts_text(text: str) -> str:
    """Bỏ ký tự rác và gom khoảng trắng trước khi đưa cho dịch vụ đọc."""
    text = _JUNK_CHARS.sub("", text or "")
    return re.sub(r"\s+", " ", text).strip()


def is_speakable(text: str) -> bool:
    """Có ít nhất một chữ hoặc chữ số để đọc (chỉ có dấu câu, ký hiệu, emoji thì không)."""
    return bool(re.search(r"[^\W_]", text, re.UNICODE))


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
        reuse_cache: bool = True,
    ):
        self.provider = provider
        self.voice = voice
        self.output_dir = output_dir
        self.model = model
        self.llm = llm_client
        self.reuse_cache = reuse_cache

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _fingerprint(self, text: str) -> str:
        """Hash của mọi thứ ảnh hưởng tới audio đầu ra."""
        payload = {
            "v": CACHE_VERSION,
            "provider": self.provider,
            # edge-tts không dùng self.model, nên bỏ qua để đổi model không làm mất cache vô ích.
            "model": self.model if self.provider == "gemini" else "",
            "voice": self.voice,
            "text": text,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _meta_path(audio_path: str) -> Path:
        return Path(audio_path).with_suffix(".meta.json")

    def _is_cached(self, audio_path: str, fingerprint: str) -> bool:
        audio = Path(audio_path)
        meta_file = self._meta_path(audio_path)
        if not audio.is_file() or not meta_file.is_file():
            return False
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            size = audio.stat().st_size
        except (OSError, ValueError):
            return False
        return (
            size > 0
            and meta.get("fingerprint") == fingerprint
            and meta.get("size") == size  # phát hiện file bị cắt/ghi dở
        )

    def _write_meta(self, audio_path: str, fingerprint: str) -> None:
        meta_file = self._meta_path(audio_path)
        tmp = meta_file.with_name(meta_file.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "size": Path(audio_path).stat().st_size,
                    "provider": self.provider,
                    "model": self.model,
                    "voice": self.voice,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, meta_file)

    def _synthesize(self, text: str, audio_path: str) -> None:
        """Sinh audio vào file tạm rồi đổi tên, để lỗi giữa chừng không để lại file hỏng."""
        final = Path(audio_path)
        tmp_path = final.with_name(f"{final.stem}.tmp{final.suffix}")
        try:
            if self.provider == "edge":
                self._edge_tts(text, str(tmp_path), self.voice)
            elif self.provider == "gemini":
                self._gemini_tts(text, str(tmp_path))
            else:
                raise ValueError(f"Provider không hỗ trợ: {self.provider}")

            if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
                raise RuntimeError(f"TTS không tạo được tệp audio: {final.name}")

            # Xoá meta cũ TRƯỚC khi thay audio: nếu bị ngắt giữa chừng, audio mới
            # không bao giờ bị ghép nhầm với meta của nội dung cũ.
            self._meta_path(audio_path).unlink(missing_ok=True)
            os.replace(tmp_path, final)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_for_segments(
        self,
        segments: List[dict],
        job_id: str,
        base_dir: str,
    ) -> List[dict]:
        """
        Sinh file wav/mp3 cho từng segment có translated_text.
        Tái sử dụng audio cũ nếu nội dung, giọng đọc và cấu hình TTS không đổi.
        Trả về segments với thêm key "audio_path".
        """
        out_dir = Path(base_dir) / str(job_id) / "tts"
        out_dir.mkdir(parents=True, exist_ok=True)

        extension = "wav" if self.provider == "gemini" else "mp3"
        results = []
        reused = generated = 0

        for seg in segments:
            text = clean_tts_text(seg.get("translated_text") or "")
            if not text:
                results.append({**seg, "audio_path": ""})
                continue
            if not is_speakable(text):
                # Ví dụ "...", "♪", chỉ có emoji: dịch vụ TTS không trả audio, bỏ qua câu này.
                logger.warning("Segment %s không có nội dung đọc được (%r), bỏ qua",
                               seg.get("index"), text[:40])
                results.append({**seg, "audio_path": ""})
                continue

            audio_path = str(out_dir / f"seg_{seg['index']:04d}.{extension}")
            fingerprint = self._fingerprint(text)

            if self.reuse_cache and self._is_cached(audio_path, fingerprint):
                reused += 1
                results.append({**seg, "audio_path": audio_path})
                continue

            try:
                self._synthesize(text, audio_path)
                self._write_meta(audio_path, fingerprint)
            except Exception as e:
                logger.exception("TTS lỗi segment %s", seg.get("index"))
                # Báo rõ câu nào lỗi để người dùng sửa được ngay trong trình chỉnh sửa.
                raise RuntimeError(
                    f"TTS lỗi ở segment {seg.get('index')} ({text[:60]!r}): {e}"
                ) from e
            generated += 1
            results.append({**seg, "audio_path": audio_path})

        logger.info(
            "TTS job %s: tái sử dụng %d, tạo mới %d segment", job_id, reused, generated
        )
        return results

    def _edge_tts(self, text: str, output_path: str, voice: str):
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("Cài edge-tts: pip install edge-tts")

        async def _run():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        last_error: Exception | None = None
        for attempt in range(1, EDGE_MAX_ATTEMPTS + 1):
            try:
                asyncio.run(_run())
                return
            except (ValueError, ImportError):
                raise  # sai tên giọng/tham số: thử lại cũng vô ích
            except Exception as e:  # NoAudioReceived, lỗi mạng, ...
                last_error = e
                logger.warning("edge-tts lần %d/%d lỗi: %s", attempt, EDGE_MAX_ATTEMPTS, e)
                if attempt < EDGE_MAX_ATTEMPTS:
                    time.sleep(attempt)  # chờ 1s, 2s rồi thử lại
        raise RuntimeError(
            f"edge-tts không trả về audio sau {EDGE_MAX_ATTEMPTS} lần thử "
            f"(giọng '{voice}'): {last_error}"
        ) from last_error

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