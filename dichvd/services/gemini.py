from __future__ import annotations

import json
import os
from typing import Any


def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa cấu hình GEMINI_API_KEY.")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện Google GenAI. Chạy: pip install -r requirements.txt"
        ) from exc
    return genai.Client(api_key=api_key)


def generate_json(
    client,
    model: str,
    system_instruction: str,
    payload: dict[str, Any],
    temperature: float = 0.2,
) -> str:
    if not hasattr(client, "models"):
        raise RuntimeError("Gemini client không hỗ trợ models.generate_content.")

    response = client.models.generate_content(
        model=model,
        contents=json.dumps(payload, ensure_ascii=False),
        config={
            "system_instruction": system_instruction,
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    content = response.text
    if not content:
        raise RuntimeError("Gemini không trả về nội dung.")
    return content
