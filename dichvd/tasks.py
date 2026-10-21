"""
Celery tasks cho pipeline xử lý video.

Mỗi bước là một hàm độc lập (step_*), chỉ đọc/ghi dữ liệu qua DB nên có thể
chạy lại riêng lẻ. Khi một bước lỗi, tên bước được lưu vào job.failed_step
để retry_from_step / process_video_pipeline chạy tiếp từ đúng chỗ đó.

YÊU CẦU model (đã có trong models.py, cần makemigrations + migrate):
    VideoJob.failed_step / locked_by / heartbeat_at / checkpoint,
    SubtitleSegment.manually_edited, TranslationReview.round

Điểm kiểm tra: translating và reviewing chạy theo lô, mỗi lô xong được lưu vào
job.checkpoint. Retry sau lỗi chỉ làm tiếp các lô còn lại (nếu đầu vào không đổi).

Bảo vệ bản dịch thủ công: bước translating và reviewing không ghi đè
SubtitleSegment.manually_edited=True; reviewing giữ lịch sử theo "round".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.db.models import Max, Q
from django.utils import timezone

logger = logging.getLogger(__name__)

# Model Gemini mặc định cho phiên âm, ngữ cảnh, dịch và kiểm định. Có thể ghi đè bằng
# settings: TRANSCRIPTION_MODEL, GEMINI_MODEL, TRANSLATION_MODEL. Gemini 2.5 Flash đã
# ngừng cấp cho người dùng mới nên đổi sang 3.8 Flash.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


# ======================================================================
# Helpers
# ======================================================================
def _get_job(job_id: str):
    from dichvd.models import VideoJob
    return VideoJob.objects.get(id=job_id)


def _job_dir(job_id: str) -> Path:
    base = Path(getattr(settings, "MEDIA_ROOT", "media")) / "jobs" / str(job_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _set_step(job, step: str, status, progress: int) -> None:
    """Đánh dấu bắt đầu một bước."""
    job.status = status
    job.current_step = step
    job.progress = progress
    job.save(update_fields=["status", "current_step", "progress", "updated_at"])


def _set_progress(job, progress: int) -> None:
    job.progress = progress
    job.save(update_fields=["progress", "updated_at"])


def _load_segments_for_service(job) -> list[dict]:
    """Segment gốc dưới dạng dict {index, start_ms, end_ms, text}."""
    from dichvd.models import SubtitleSegment

    rows = list(
        SubtitleSegment.objects.filter(job=job)
        .order_by("index")
        .values("index", "start_ms", "end_ms", "original_text")
    )
    for s in rows:
        s["text"] = s.pop("original_text")
    return rows


def _load_context(job) -> dict:
    """Lấy context mới nhất từ DB (thay cho biến cục bộ ở bản cũ)."""
    snap = job.context_snapshots.order_by("-version").first()
    if not snap:
        raise RuntimeError("Chưa có context. Hãy chạy lại từ bước building_context.")
    return {
        "summary": snap.summary,
        "characters": snap.characters_json or [],
        "glossary": snap.glossary_json or [],
    }


# ----------------------------------------------------------------------
# Điểm kiểm tra (checkpoint) cho các bước chạy theo lô
# ----------------------------------------------------------------------
def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _fingerprint(*parts) -> str:
    """Hash của toàn bộ đầu vào của một bước; đổi đầu vào thì checkpoint cũ bị bỏ."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cp_get(job, step: str, fingerprint: str) -> dict:
    """Checkpoint còn hợp lệ (cùng đầu vào) hoặc một checkpoint trống mới."""
    cp = (job.checkpoint or {}).get(step)
    if isinstance(cp, dict) and cp.get("fingerprint") == fingerprint:
        return cp
    return {"fingerprint": fingerprint, "done": []}


def _cp_save(job, step: str, cp: dict) -> None:
    data = dict(job.checkpoint or {})
    data[step] = cp
    job.checkpoint = data
    job.save(update_fields=["checkpoint", "updated_at"])


def _cp_clear(job, *steps: str) -> None:
    data = dict(job.checkpoint or {})
    changed = False
    for step in steps:
        if step in data:
            del data[step]
            changed = True
    if changed:
        job.checkpoint = data
        job.save(update_fields=["checkpoint", "updated_at"])


# ----------------------------------------------------------------------
# Tự chờ và thử lại khi Gemini báo hết quota (429 RESOURCE_EXHAUSTED)
# ----------------------------------------------------------------------
def _quota_info(exc: BaseException):
    """
    Nếu exc (hoặc nguyên nhân của nó) là lỗi 429 của Gemini, trả về
    (số giây nên chờ, có phải quota theo ngày không). Ngược lại trả về None.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            daily = "PerDay" in text
            match = (
                re.search(r"retry in ([\d.]+)\s*s", text, re.I)
                or re.search(r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", text)
            )
            wait = float(match.group(1)) + 2 if match else 60.0
            return min(wait, 300.0), daily
        exc = exc.__cause__ or exc.__context__
    return None


def _call_with_quota_retry(func, *args, **kwargs):
    """
    Gọi func; nếu Gemini báo vượt hạn mức theo phút thì chờ đúng thời gian nó yêu cầu
    rồi thử lại (tối đa settings.GEMINI_QUOTA_MAX_RETRIES lần). Hết quota theo ngày thì
    báo lỗi rõ ràng vì chờ cũng không giúp được. Heartbeat vẫn chạy nên job không bị coi là treo.
    """
    max_retries = int(getattr(settings, "GEMINI_QUOTA_MAX_RETRIES", 8))
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            info = _quota_info(exc)
            if info is None:
                raise
            wait, daily = info
            if daily:
                raise RuntimeError(
                    "Đã hết hạn mức Gemini theo NGÀY. Hãy bật thanh toán cho project "
                    "trong Google AI Studio, đổi API key/model, hoặc thử lại vào ngày mai."
                ) from exc
            if attempt >= max_retries:
                raise
            logger.warning(
                "Gemini báo vượt hạn mức theo phút, chờ %.0fs rồi thử lại (lần %d/%d)",
                wait, attempt + 1, max_retries,
            )
            time.sleep(wait)


def _review_enabled() -> bool:
    """
    settings.REVIEW_ENABLED; chấp nhận cả chuỗi từ biến môi trường ('false', '0', 'no', 'off')
    để os.getenv('REVIEW_ENABLED', ...) không bị coi là True vì chuỗi không rỗng.
    """
    value = getattr(settings, "REVIEW_ENABLED", True)
    return str(value).strip().lower() not in ("false", "0", "no", "off", "")


# ======================================================================
# CÁC BƯỚC ĐỘC LẬP
# ======================================================================
def step_transcribe(job, work: Path) -> None:
    from dichvd.models import SubtitleSegment, VideoJob
    from dichvd.services import Transcriber, VideoProcessor

    _set_step(job, "transcribing", VideoJob.Status.TRANSCRIBING, 5)

    processor = VideoProcessor()
    audio_path = str(work / "audio.wav")
    processor.extract_audio(job.original_file.path, audio_path)

    job.duration_seconds = processor.get_duration(job.original_file.path)
    job.save(update_fields=["duration_seconds", "updated_at"])

    transcriber = Transcriber(
        model_size=getattr(settings, "WHISPER_MODEL_SIZE", "medium"),
        device=getattr(settings, "WHISPER_DEVICE", "cpu"),
        provider=getattr(settings, "TRANSCRIPTION_PROVIDER", "gemini"),
        api_model=getattr(settings, "TRANSCRIPTION_MODEL", DEFAULT_GEMINI_MODEL),
    )
    lang = None if job.source_language == "auto" else job.source_language
    raw_segments = _call_with_quota_retry(
        transcriber.transcribe_to_dicts, audio_path, language=lang
    )
    if not raw_segments:
        raise RuntimeError("Không nhận diện được lời thoại nào trong video.")

    SubtitleSegment.objects.filter(job=job).delete()
    for s in raw_segments:
        SubtitleSegment.objects.create(
            job=job,
            index=s["index"],
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            original_text=s["text"],
            confidence=s.get("confidence"),
        )
    _set_progress(job, 20)


def step_build_context(job, work: Path) -> None:
    from dichvd.models import (
        Character,
        ContextSnapshot,
        GlossaryTerm,
        SubtitleSegment,
        VideoJob,
    )
    from dichvd.services import ContextBuilder

    _set_step(job, "building_context", VideoJob.Status.BUILDING_CONTEXT, 25)

    if not SubtitleSegment.objects.filter(job=job).exists():
        raise RuntimeError("Chưa có segment. Hãy chạy lại từ bước transcribing.")

    builder = ContextBuilder(model=getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    context = _call_with_quota_retry(
        builder.build,
        segments=_load_segments_for_service(job),
        title=job.title or "",
    )

    version = (job.context_snapshots.count() or 0) + 1
    ContextSnapshot.objects.create(
        job=job,
        version=version,
        summary=context.get("summary", ""),
        characters_json=context.get("characters", []),
        glossary_json=context.get("glossary", []),
    )

    Character.objects.filter(job=job).delete()
    for c in context.get("characters", []):
        Character.objects.create(
            job=job,
            name=c.get("name", "Unknown"),
            aliases=c.get("aliases", []),
            gender=c.get("gender", ""),
            description=c.get("description", ""),
        )

    GlossaryTerm.objects.filter(job=job).delete()
    for g in context.get("glossary", []):
        GlossaryTerm.objects.create(
            job=job,
            source_term=g.get("source_term", ""),
            target_term=g.get("target_term", ""),
            category=g.get("category", ""),
            notes=g.get("notes", ""),
        )
    _set_progress(job, 35)


def step_translate(job, work: Path) -> None:
    from django.db import transaction

    from dichvd.models import SubtitleSegment, VideoJob
    from dichvd.services import Translator

    _set_step(job, "translating", VideoJob.Status.TRANSLATING, 40)

    model = getattr(settings, "TRANSLATION_MODEL", DEFAULT_GEMINI_MODEL)
    batch_size = max(1, int(getattr(settings, "TRANSLATE_BATCH_SIZE", 20)))
    segments = _load_segments_for_service(job)
    context = _load_context(job)

    # Checkpoint chỉ dùng lại khi mọi đầu vào y hệt lần trước (model, ngôn ngữ, context
    # gồm glossary, nội dung gốc). Sửa glossary hay phiên âm lại thì tự dịch lại từ đầu.
    fp = _fingerprint(
        model, batch_size, job.target_language, context,
        [(s["index"], s["text"]) for s in segments],
    )
    cp = _cp_get(job, "translating", fp)
    done = set(cp["done"])
    if done:
        logger.info("Job %s: tiếp tục dịch, đã xong %d/%d câu", job.id, len(done), len(segments))

    translator = Translator(model=model, batch_size=batch_size)
    kept = 0
    for batch in _chunks(segments, batch_size):
        pending = [s for s in batch if s["index"] not in done]
        if not pending:
            continue

        translated = _call_with_quota_retry(
            translator.translate,
            segments=pending,
            context=context,
            target_language=job.target_language,
        )
        returned = {item["index"] for item in translated}
        blank = {s["index"] for s in pending if not s["text"].strip()}

        # Lưu bản dịch của lô và đánh dấu lô xong trong cùng một giao dịch.
        with transaction.atomic():
            # Không ghi đè câu người dùng đã sửa tay. Điều kiện manually_edited=False nằm
            # ngay trong câu UPDATE nên an toàn cả khi người dùng vừa lưu lúc đang dịch.
            for item in translated:
                updated = SubtitleSegment.objects.filter(
                    job=job, index=item["index"], manually_edited=False
                ).update(
                    translated_text=item.get("translated_text", ""),
                    updated_at=timezone.now(),
                )
                if not updated:
                    kept += 1
            done.update((returned | blank) & {s["index"] for s in pending})
            cp["done"] = sorted(done)
            _cp_save(job, "translating", cp)

        _set_progress(job, 40 + int(15 * len(done) / max(len(segments), 1)))

    # Translator bỏ sót câu có nội dung: báo lỗi để retry chỉ dịch lại các câu này,
    # thay vì lặng lẽ để trống rồi TTS bỏ qua câu đó.
    missing = sorted(s["index"] for s in segments if s["index"] not in done)
    if missing:
        raise RuntimeError(
            f"Translator không trả về bản dịch cho {len(missing)} câu (segment {missing[:20]}). "
            "Retry sẽ chỉ dịch lại các câu này."
        )

    if kept:
        logger.info("Job %s: giữ nguyên %d câu đã sửa thủ công", job.id, kept)
    # Bản dịch mới -> điểm kiểm tra kiểm định cũ không còn đúng.
    _cp_clear(job, "translating", "reviewing")
    _set_progress(job, 55)


def _apply_reviews(job, reviews: list, review_round: int, model_name: str) -> None:
    """Lưu kết quả kiểm định của một lô và áp dụng gợi ý (không đè câu sửa tay)."""
    from dichvd.models import SubtitleSegment, TranslationReview

    for r in reviews:
        if not (r.get("issues") or r.get("suggestion")):
            continue
        seg = SubtitleSegment.objects.get(job=job, index=r["index"])
        score = r.get("score")
        good = (score if score is not None else 1) >= 0.7  # score = 0.0 là điểm thấp, không phải "tốt"
        TranslationReview.objects.create(
            segment=seg,
            round=review_round,
            issues=r.get("issues", []),
            suggestion=r.get("suggestion", ""),
            score=score,
            model_used=model_name,
        )
        # Mọi thay đổi trên segment đều có điều kiện manually_edited=False trong chính câu
        # UPDATE, để không đè lên câu người dùng vừa sửa trong lúc AI đang kiểm định.
        protected = SubtitleSegment.objects.filter(pk=seg.pk, manually_edited=False)
        if r.get("suggestion"):
            # Chỉ áp dụng suggestion khi score thấp
            if not good:
                protected.update(
                    translated_text=r["suggestion"],
                    review_status="fixed",
                    is_reviewed=True,
                    updated_at=timezone.now(),
                )
        else:
            protected.update(
                review_status="ok" if good else "needs_fix",
                is_reviewed=True,
                updated_at=timezone.now(),
            )


def step_review(job, work: Path) -> None:
    from django.db import transaction

    from dichvd.models import SubtitleSegment, TranslationReview, VideoJob
    from dichvd.services import TranslationReviewer

    _set_step(job, "reviewing", VideoJob.Status.REVIEWING, 60)

    # Cho phép bỏ qua kiểm định AI (settings.REVIEW_ENABLED = False), ví dụ khi dùng gói
    # miễn phí hạn mức thấp. Bản dịch vẫn được dùng nguyên như bước dịch tạo ra.
    if not _review_enabled():
        logger.info("Job %s: REVIEW_ENABLED=False, bỏ qua bước kiểm định", job.id)
        _cp_clear(job, "reviewing")
        _set_progress(job, 70)
        return

    model_name = getattr(settings, "GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    # Reviewer gọi Gemini một lần cho mỗi câu, nên lô nhỏ (mặc định 4) để không vượt
    # 5 yêu cầu/phút của gói miễn phí và mỗi lô hoàn thành kịp được lưu checkpoint.
    batch_size = max(1, int(getattr(settings, "REVIEW_BATCH_SIZE", 4)))
    context = _load_context(job)
    # Câu người dùng đã sửa tay coi như đã được con người duyệt: không gửi đi kiểm định.
    segs_for_review = list(
        SubtitleSegment.objects.filter(job=job, manually_edited=False)
        .order_by("index")
        .values("index", "original_text", "translated_text")
    )
    if not segs_for_review:
        _cp_clear(job, "reviewing")
        _set_progress(job, 70)
        return

    # Không đưa translated_text vào fingerprint: nó thay đổi khi AI sửa câu, sẽ làm
    # checkpoint mất hiệu lực giữa chừng. Bước dịch xong sẽ tự xoá checkpoint cũ.
    fp = _fingerprint(
        model_name, batch_size, context,
        [(s["index"], s["original_text"]) for s in segs_for_review],
    )
    cp = _cp_get(job, "reviewing", fp)
    done = set(cp["done"])

    # Giữ lịch sử: mỗi lần chạy là một "round" mới. Khi tiếp tục sau lỗi thì dùng lại
    # round đang dở để kết quả của cùng một lần kiểm định không bị tách ra.
    if not cp.get("round"):
        last_round = (
            TranslationReview.objects.filter(segment__job=job).aggregate(m=Max("round"))["m"] or 0
        )
        cp["round"] = last_round + 1
    review_round = cp["round"]
    if done:
        logger.info("Job %s: tiếp tục kiểm định, đã xong %d/%d câu",
                    job.id, len(done), len(segs_for_review))

    reviewer = TranslationReviewer(model=model_name)
    for batch in _chunks(segs_for_review, batch_size):
        pending = [s for s in batch if s["index"] not in done]
        if not pending:
            continue
        reviews = _call_with_quota_retry(
            reviewer.review_segments,
            pending,
            context=context,
            only_suspicious=True,
        )
        # Lưu kết quả lô và đánh dấu lô xong trong cùng một giao dịch.
        with transaction.atomic():
            _apply_reviews(job, reviews, review_round, model_name)
            done.update(s["index"] for s in pending)
            cp["done"] = sorted(done)
            _cp_save(job, "reviewing", cp)
        _set_progress(job, 60 + int(10 * len(done) / len(segs_for_review)))

    _cp_clear(job, "reviewing")
    _set_progress(job, 70)


def step_generate_audio(job, work: Path) -> None:
    from dichvd.models import SubtitleSegment, VideoJob
    from dichvd.services import SpeechGenerator

    _set_step(job, "generating_audio", VideoJob.Status.GENERATING_AUDIO, 75)

    segs_tts = list(
        SubtitleSegment.objects.filter(job=job)
        .order_by("index")
        .values("index", "start_ms", "end_ms", "translated_text")
    )
    tts = SpeechGenerator(
        provider=getattr(settings, "TTS_PROVIDER", "gemini"),
        voice=getattr(settings, "TTS_VOICE", "Kore"),
        model=getattr(settings, "TTS_MODEL", "gemini-3.8-flash-tts"),
    )
    segs_with_audio = tts.generate_for_segments(
        segs_tts,
        job_id=str(job.id),
        base_dir=str(Path(settings.MEDIA_ROOT) / "jobs"),
    )
    for item in segs_with_audio:
        if item.get("audio_path"):
            SubtitleSegment.objects.filter(job=job, index=item["index"]).update(
                audio_path=item["audio_path"],
            )
    _set_progress(job, 88)


def step_render(job, work: Path) -> None:
    from django.core.files import File

    from dichvd.models import SubtitleSegment, VideoJob
    from dichvd.services import VideoRenderer

    _set_step(job, "rendering", VideoJob.Status.RENDERING, 90)

    output_path = str(work / "rendered.mp4")
    segs_render = list(
        SubtitleSegment.objects.filter(job=job)
        .order_by("index")
        .values("index", "start_ms", "end_ms", "audio_path")
    )
    renderer = VideoRenderer(
        max_speedup=getattr(settings, "DUB_MAX_SPEEDUP", 1.3),
    )
    renderer.render(
        video_path=job.original_file.path,
        segments=segs_render,
        output_path=output_path,
        keep_original_audio=getattr(settings, "KEEP_ORIGINAL_AUDIO", False),
    )
    report = renderer.last_report
    if report["overflow"]:
        logger.warning(
            "Job %s: %d câu vẫn dài hơn khung thời gian (segment %s). "
            "Nên rút gọn bản dịch của các câu này.",
            job.id, len(report["overflow"]), report["overflow"],
        )

    with open(output_path, "rb") as f:
        job.output_file.save("dubbed.mp4", File(f), save=False)


# Thứ tự chạy của pipeline
PIPELINE_STEPS: list[tuple[str, callable]] = [
    ("transcribing", step_transcribe),
    ("building_context", step_build_context),
    ("translating", step_translate),
    ("reviewing", step_review),
    ("generating_audio", step_generate_audio),
    ("rendering", step_render),
]
STEP_NAMES = [name for name, _ in PIPELINE_STEPS]


# ======================================================================
# Runner
# ======================================================================
class LockLost(RuntimeError):
    """Khóa job đã bị worker khác chiếm hoặc bị thu hồi vì treo quá lâu."""


def _lock_timeout() -> int:
    """Số giây không có heartbeat thì coi worker đã chết (settings.JOB_LOCK_TIMEOUT)."""
    return int(getattr(settings, "JOB_LOCK_TIMEOUT", 600))


def _heartbeat_interval() -> int:
    return int(getattr(settings, "JOB_HEARTBEAT_INTERVAL", 30))


def _acquire_lock(job_id: str, token: str) -> bool:
    """
    Giành khóa job bằng một câu UPDATE có điều kiện (nguyên tử).
    Thành công khi job chưa có ai giữ, hoặc người giữ đã quá hạn heartbeat.
    """
    from dichvd.models import VideoJob

    now = timezone.now()
    stale_before = now - timedelta(seconds=_lock_timeout())
    got = VideoJob.objects.filter(id=job_id).filter(
        Q(locked_by="") | Q(heartbeat_at__isnull=True) | Q(heartbeat_at__lt=stale_before)
    ).update(locked_by=token, heartbeat_at=now)
    if not got and not VideoJob.objects.filter(id=job_id).exists():
        raise VideoJob.DoesNotExist(f"Không có job {job_id}")
    return bool(got)


def _release_lock(job_id: str, token: str) -> None:
    from dichvd.models import VideoJob

    VideoJob.objects.filter(id=job_id, locked_by=token).update(locked_by="", heartbeat_at=None)


class _Heartbeat:
    """
    Luồng nền cập nhật heartbeat_at định kỳ trong lúc các bước dài (TTS, FFmpeg...)
    đang chặn luồng chính. Nếu khóa không còn thuộc worker này thì đặt cờ `lost`.
    """

    def __init__(self, job_id: str, token: str):
        self.job_id = job_id
        self.token = token
        # Phải đủ dày so với timeout để vài lần lỡ nhịp không bị coi là chết.
        self.interval = max(1, min(_heartbeat_interval(), _lock_timeout() // 3))
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"heartbeat-{job_id}"
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def raise_if_lost(self) -> None:
        if self.lost.is_set():
            raise LockLost(f"Job {self.job_id} không còn thuộc worker này")

    def _run(self) -> None:
        from django.db import connection

        from dichvd.models import VideoJob

        try:
            while not self._stop.wait(self.interval):
                try:
                    updated = VideoJob.objects.filter(
                        id=self.job_id, locked_by=self.token
                    ).update(heartbeat_at=timezone.now())
                except Exception:
                    logger.exception("Heartbeat job %s: lỗi DB, sẽ thử lại", self.job_id)
                    continue
                if not updated:
                    self.lost.set()
                    return
        finally:
            connection.close()  # luồng này có kết nối DB riêng


def fail_stale_jobs(job_id: str | None = None) -> int:
    """
    Đánh dấu failed các job đang ở trạng thái xử lý nhưng quá lâu không có heartbeat
    (worker bị tắt/chết giữa chừng). Giữ lại failed_step để Retry chạy đúng bước.
    Trả về số job đã xử lý. Gọi được từ view (lazy) hoặc từ Celery beat.
    """
    from dichvd.models import VideoJob

    S = VideoJob.Status
    now = timezone.now()
    timeout = _lock_timeout()
    cutoff = now - timedelta(seconds=timeout)
    active = [
        S.TRANSCRIBING, S.BUILDING_CONTEXT, S.TRANSLATING,
        S.REVIEWING, S.GENERATING_AUDIO, S.RENDERING,
    ]
    # Job cũ chưa từng có heartbeat (trước khi nâng cấp) được xét theo updated_at.
    stale = Q(heartbeat_at__lt=cutoff) | Q(heartbeat_at__isnull=True, updated_at__lt=cutoff)

    qs = VideoJob.objects.filter(status__in=active).filter(stale)
    if job_id:
        qs = qs.filter(id=job_id)

    count = 0
    for job in list(qs):
        step = job.current_step if job.current_step in STEP_NAMES else ""
        count += VideoJob.objects.filter(id=job.id, status=job.status).filter(stale).update(
            status=S.FAILED,
            failed_step=step,
            error_message=(
                f"Job bị treo ở bước '{job.current_step}': worker không phản hồi "
                f"quá {timeout}s. Hãy bấm Retry."
            ),
            locked_by="",
            heartbeat_at=None,
            updated_at=now,
        )
        logger.warning("Job %s treo ở bước %s, đã đánh dấu failed", job.id, job.current_step)
    return count


@shared_task(ignore_result=True)
def reap_stale_jobs():
    """Tuỳ chọn: lên lịch bằng Celery beat để dọn job treo định kỳ."""
    return fail_stale_jobs()


def _run_steps(job_id: str, start_from: str | None = None, only: str | None = None) -> dict:
    """
    Vỏ bọc: giành khóa job, chạy heartbeat, rồi chạy các bước.
    Nếu job đang được worker khác xử lý thì bỏ qua, KHÔNG đụng tới trạng thái job.
    """
    token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    if not _acquire_lock(job_id, token):
        logger.warning("Job %s đang được worker khác xử lý, bỏ qua task này", job_id)
        return {"job_id": str(job_id), "status": "skipped_locked"}

    heartbeat = _Heartbeat(job_id, token)
    heartbeat.start()
    try:
        return _run_steps_locked(job_id, start_from, only, heartbeat)
    except LockLost:
        logger.warning("Job %s: mất khóa, worker này dừng lại", job_id)
        return {"job_id": str(job_id), "status": "lock_lost"}
    finally:
        heartbeat.stop()
        _release_lock(job_id, token)


def _run_steps_locked(
    job_id: str,
    start_from: str | None,
    only: str | None,
    heartbeat: _Heartbeat,
) -> dict:
    """
    Chạy các bước từ `start_from` (mặc định: bước đã fail trước đó, hoặc bước đầu).
    `only`: chỉ chạy đúng một bước rồi dừng.
    Khi lỗi: lưu job.failed_step, đánh dấu failed, ném lại exception.
    """
    job = _get_job(job_id)
    work = _job_dir(job_id)

    # Task bị giao lặp lại sau khi job đã xong: không chạy lại từ đầu.
    if job.status == job.Status.DONE and not only:
        logger.info("Job %s đã hoàn thành, bỏ qua", job_id)
        return {"job_id": str(job_id), "status": "already_done"}

    start_from = start_from or (getattr(job, "failed_step", "") or STEP_NAMES[0])
    if start_from not in STEP_NAMES:
        raise ValueError(f"Bước không hợp lệ: {start_from!r}. Chọn một trong {STEP_NAMES}")

    steps = PIPELINE_STEPS[STEP_NAMES.index(start_from):]
    if only:
        steps = [s for s in steps if s[0] == only][:1]

    # Bắt đầu chạy lại -> xoá dấu lỗi cũ
    if getattr(job, "failed_step", ""):
        job.failed_step = ""
        job.save(update_fields=["failed_step", "updated_at"])

    for name, func in steps:
        heartbeat.raise_if_lost()
        # Tắt kiểm định: bỏ hẳn bước, không đổi trạng thái sang "reviewing" và không gọi Gemini.
        if name == "reviewing" and not _review_enabled():
            logger.info("Job %s: REVIEW_ENABLED tắt, bỏ qua hẳn bước kiểm định", job_id)
            _cp_clear(job, "reviewing")
            _set_progress(job, 70)
            continue
        try:
            logger.info("Job %s: bắt đầu bước %s", job_id, name)
            func(job, work)
        except Exception as e:
            if heartbeat.lost.is_set():
                # Job đã thuộc về worker khác: không ghi đè trạng thái của họ.
                logger.warning("Job %s lỗi ở bước %s nhưng đã mất khóa: %s", job_id, name, e)
                raise LockLost(f"Job {job_id} mất khóa") from e
            logger.exception("Job %s lỗi ở bước %s: %s", job_id, name, e)
            job = _get_job(job_id)
            job.mark_failed(str(e))
            job.failed_step = name
            job.save(update_fields=["failed_step", "updated_at"])
            raise

    if not only and steps and steps[-1][0] == STEP_NAMES[-1]:
        heartbeat.raise_if_lost()
        job.mark_done()
        logger.info("Job %s hoàn thành", job_id)
    return {"job_id": str(job_id), "status": "done" if not only else "step_done"}


# ======================================================================
# Celery tasks
# ======================================================================
@shared_task(bind=True, ignore_result=True)
def process_video_pipeline(self, job_id: str, start_from: str | None = None):
    """
    Pipeline đầy đủ:
    transcribing → building_context → translating → reviewing
    → generating_audio → rendering → done

    start_from: tên bước để chạy tiếp (None = từ bước đã fail, hoặc từ đầu).
    """
    return _run_steps(job_id, start_from=start_from)


@shared_task(ignore_result=True)
def retry_from_step(job_id: str, step: str | None = None):
    """
    Chạy lại từ một bước cụ thể (mặc định: bước đã fail).
    step: transcribing | building_context | translating | reviewing | generating_audio | rendering
    """
    return _run_steps(job_id, start_from=step)


@shared_task(ignore_result=True)
def run_single_step(job_id: str, step: str):
    """Chỉ chạy đúng một bước (hữu ích khi debug / chạy lại bước review...)."""
    return _run_steps(job_id, start_from=step, only=step)