"""
Kiểm định bản dịch.
- Code rule đơn giản trước
- AI review sâu cho câu đáng ngờ
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .gemini import create_gemini_client, generate_json

logger = logging.getLogger(__name__)


class TranslationReviewer:
    """
    Review từng segment, trả về issues + suggestion + score.
    """

    def __init__(
        self,
        llm_client=None,
        model: str = "gemini-2.5-flash",
        suspicious_threshold: float = 0.6,
    ):
        self.model = model
        self.suspicious_threshold = suspicious_threshold
        self.llm = llm_client or create_gemini_client()

    def review_segments(
        self,
        segments: List[dict],
        context: Dict[str, Any],
        only_suspicious: bool = True,
    ) -> List[dict]:
        """
        Trả về list review:
        {
          "index": ...,
          "issues": [...],
          "suggestion": "...",
          "score": 0.0-1.0,
          "needs_ai": bool
        }
        """
        results = []
        for seg in segments:
            rule_result = self._rule_based_check(seg, context)
            needs_ai = rule_result["score"] < self.suspicious_threshold or rule_result["issues"]

            if only_suspicious and not needs_ai:
                results.append({
                    "index": seg["index"],
                    "issues": [],
                    "suggestion": "",
                    "score": rule_result["score"],
                    "needs_ai": False,
                })
                continue

            if self.llm and needs_ai:
                ai_result = self._ai_review(seg, context)
                results.append({
                    "index": seg["index"],
                    "issues": ai_result.get("issues", rule_result["issues"]),
                    "suggestion": ai_result.get("suggestion", ""),
                    "score": ai_result.get("score", rule_result["score"]),
                    "needs_ai": True,
                })
            else:
                results.append({
                    "index": seg["index"],
                    "issues": rule_result["issues"],
                    "suggestion": "",
                    "score": rule_result["score"],
                    "needs_ai": needs_ai,
                })

        return results

    def _rule_based_check(self, seg: dict, context: Dict[str, Any]) -> dict:
        original = (seg.get("original_text") or seg.get("text") or "").strip()
        translated = (seg.get("translated_text") or "").strip()
        issues = []
        score = 1.0

        if not translated:
            issues.append({"type": "empty", "detail": "Bản dịch trống"})
            score = 0.0
            return {"issues": issues, "score": score}

        # Độ dài lệch quá nhiều
        if len(original) > 0:
            ratio = len(translated) / len(original)
            if ratio > 2.5 or ratio < 0.3:
                issues.append({
                    "type": "length_mismatch",
                    "detail": f"Tỷ lệ độ dài {ratio:.2f}",
                })
                score -= 0.25

        # Glossary bị bỏ qua
        glossary = context.get("glossary") or []
        for term in glossary:
            src = term.get("source_term", "")
            tgt = term.get("target_term", "")
            if src and src.lower() in original.lower():
                if tgt and tgt not in translated:
                    issues.append({
                        "type": "glossary_miss",
                        "detail": f"Thiếu thuật ngữ '{tgt}' (từ '{src}')",
                    })
                    score -= 0.3

        # Ký tự lạ / tiếng Anh còn sót nhiều
        if re.search(r"[A-Za-z]{4,}", translated) and not re.search(r"[A-Za-z]{4,}", original):
            # Có thể là tên riêng, chỉ trừ nhẹ
            score -= 0.1

        score = max(0.0, min(1.0, score))
        return {"issues": issues, "score": score}

    def _ai_review(self, seg: dict, context: Dict[str, Any]) -> dict:
        system_prompt = """Bạn là biên tập viên phụ đề chuyên nghiệp.
Hãy kiểm tra bản dịch tiếng Việt:
- Có sai nghĩa / mất ngữ cảnh không?
- Có dùng sai thuật ngữ trong glossary không?
- Có xưng hô không nhất quán không?
- Có quá dài / quá ngắn so với timing không?

Trả JSON:
{
  "issues": [{"type": "meaning|glossary|honorific|length|other", "detail": "..."}],
  "suggestion": "câu dịch đề xuất nếu cần sửa, để trống nếu OK",
  "score": 0.0-1.0
}
Chỉ trả JSON."""

        payload = {
            "summary": context.get("summary", ""),
            "glossary": context.get("glossary", []),
            "characters": context.get("characters", []),
            "original": seg.get("original_text") or seg.get("text", ""),
            "translated": seg.get("translated_text", ""),
        }

        try:
            raw = self._call_llm(system_prompt, payload)
            return self._parse_json(raw)
        except Exception as exc:
            logger.exception("AI review lỗi")
            raise RuntimeError("Không kiểm định được bản dịch.") from exc

    def _call_llm(self, system_prompt: str, payload: Dict[str, Any]) -> str:
        return generate_json(
            self.llm,
            self.model,
            system_prompt,
            payload,
        )

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)
