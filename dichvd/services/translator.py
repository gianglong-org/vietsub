"""
Dịch phụ đề có ngữ cảnh (context-aware).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .gemini import create_gemini_client, generate_json

logger = logging.getLogger(__name__)


class Translator:
    """
    Dịch từng batch segment, truyền context + glossary + characters.
    """

    def __init__(
        self,
        llm_client=None,
        model: str = "gemini-2.5-flash",
        batch_size: int = 20,
    ):
        self.model = model
        self.batch_size = batch_size
        self.llm = llm_client or create_gemini_client()

    def translate(
        self,
        segments: List[dict],
        context: Dict[str, Any],
        target_language: str = "vi",
    ) -> List[dict]:
        """
        Input segments: [{index, start_ms, end_ms, text, ...}]
        context: output từ ContextBuilder
        Return: list segment đã có "translated_text"
        """
        if not self.llm:
            raise RuntimeError("Không có Gemini client để dịch phụ đề.")

        results = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i : i + self.batch_size]
            translated_batch = self._translate_batch(batch, context, target_language)
            results.extend(translated_batch)

        return results

    def _translate_batch(
        self,
        batch: List[dict],
        context: Dict[str, Any],
        target_language: str,
    ) -> List[dict]:
        summary = context.get("summary", "")
        characters = context.get("characters", [])
        glossary = context.get("glossary", [])

        system_prompt = f"""Bạn là dịch giả chuyên nghiệp, dịch phụ đề sang tiếng {target_language}.
Yêu cầu:
- Giữ đúng ý, tự nhiên, phù hợp văn phong phim (tu tiên / cổ trang nếu có).
- Tôn trọng từ điển thuật ngữ đã cho (glossary).
- Giữ xưng hô nhất quán theo nhân vật.
- Không thêm chú thích, không giải thích.
- Độ dài câu dịch không lệch quá nhiều so với gốc (để khớp timing).

Trả về ĐÚNG JSON:
{{
  "translations": [
    {{"index": 0, "translated_text": "..."}}
  ]
}}
Chỉ trả JSON."""

        user_payload = {
            "summary": summary,
            "characters": characters,
            "glossary": glossary,
            "segments": [
                {"index": s["index"], "text": s.get("text") or s.get("original_text", "")}
                for s in batch
            ],
        }

        try:
            raw = self._call_llm(system_prompt, user_payload)
            data = self._parse_json(raw)
            mapping = {
                item["index"]: item["translated_text"]
                for item in data.get("translations", [])
            }
            out = []
            for segment in batch:
                translated_text = mapping.get(segment["index"])
                if not isinstance(translated_text, str) or not translated_text.strip():
                    raise ValueError(
                        f"Không có bản dịch hợp lệ cho đoạn {segment['index']}."
                    )
                out.append({**segment, "translated_text": translated_text.strip()})
            return out
        except Exception as exc:
            logger.exception("Lỗi dịch batch")
            raise RuntimeError("Không dịch được một nhóm phụ đề.") from exc

    def _call_llm(self, system_prompt: str, user_payload: Dict[str, Any]) -> str:
        return generate_json(
            self.llm,
            self.model,
            system_prompt,
            user_payload,
            temperature=0.3,
        )

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
