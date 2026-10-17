"""
Đoạn cấu hình cần thêm vào config/settings.py của bạn.
Không phải file settings đầy đủ — chỉ phần liên quan pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Apps ---
INSTALLED_APPS = [
    # ... django default ...
    "apps.accounts",
    "apps.videos",
]

# --- Media ---
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# --- Celery ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_TIMEZONE = "Asia/Ho_Chi_Minh"

# --- AI / Processing ---
# Configure GEMINI_API_KEY in the environment.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gemini-2.5-flash")
TRANSCRIPTION_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "gemini")
TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gemini-2.5-flash")
TRANSLATE_BATCH_SIZE = int(os.getenv("TRANSLATE_BATCH_SIZE", "20"))
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "gemini")
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
KEEP_ORIGINAL_AUDIO = os.getenv("KEEP_ORIGINAL_AUDIO", "False").lower() == "true"

# File upload limit (bytes) — ví dụ 500MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
