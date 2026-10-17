from __future__ import annotations

import json
import logging
from typing import Any

from .gemini import create_gemini_client, generate_json

logger = logging.getLogger(__name__)


class ContextBuilder:
    def __init__(self, llm_client=None, model: str = "gemini-2.5-flash"):
        self.model = model
        self.llm = llm_client or create_gemini_client()

    def build(self, segments: list[dict[str, Any]], title: str = "") -> dict[str, Any]:
        if not self.llm:
            raise RuntimeError("Không có Gemini client để xây dựng ngữ cảnh.")

        system_prompt = (
            "Phân tích phụ đề video và trả về JSON gồm summary (tóm tắt ngắn), "
            "characters (danh sách object name, aliases, gender, description) "
            "và glossary (danh sách object source_term, target_term, category, notes). "
            "Giữ nguyên thuật ngữ nguồn; nếu không xác định được thì trả danh sách rỗng."
        )
        try:
            content = generate_json(
                self.llm,
                self.model,
                system_prompt,
                {"title": title, "segments": segments},
            )
            result = json.loads(content)
            return {
                "summary": result.get("summary", ""),
                "characters": result.get("characters", []),
                "glossary": result.get("glossary", []),
            }
        except Exception as exc:
            logger.exception("Không tạo được ngữ cảnh video")
            raise RuntimeError("Không tạo được ngữ cảnh video.") from exc
