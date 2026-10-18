"""
Celery tasks cho pipeline xử lý video.

Mỗi bước là một hàm độc lập (step_*), chỉ đọc/ghi dữ liệu qua DB nên có thể
chạy lại riêng lẻ. Khi một bước lỗi, tên bước được lưu vào job.failed_step
để retry_from_step / process_video_pipeline chạy tiếp từ đúng chỗ đó.

YÊU CẦU model (đã có trong models.py, cần makemigrations + migrate):
    VideoJob.failed_step, SubtitleSegment.manually_edited, TranslationReview.round

Bảo vệ bản dịch thủ công: bước translating và reviewing không ghi đè
SubtitleSegment.manually_edited=True; reviewing giữ lịch sử theo "round".
"""
from __future__ import annotations

import logging
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.db.models import Max
from django.utils import timezone

logger = logging.getLogger(__name__)


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
        api_model=getattr(settings, "TRANSCRIPTION_MODEL", "gemini-2.5-flash"),
    )
    lang = None if job.source_language == "auto" else job.source_language
    raw_segments = transcriber.transcribe_to_dicts(audio_path, language=lang)
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

    builder = ContextBuilder(model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"))
    context = builder.build(
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
    from dichvd.models import SubtitleSegment, VideoJob
    from dichvd.services import Translator

    _set_step(job, "translating", VideoJob.Status.TRANSLATING, 40)

    translator = Translator(
        model=getattr(settings, "TRANSLATION_MODEL", "gemini-2.5-flash"),
        batch_size=getattr(settings, "TRANSLATE_BATCH_SIZE", 20),
    )
    translated = translator.translate(
        segments=_load_segments_for_service(job),
        context=_load_context(job),
        target_language=job.target_language,
    )
    # Không ghi đè câu người dùng đã sửa tay. Điều kiện manually_edited=False nằm
    # ngay trong câu UPDATE nên an toàn cả khi người dùng vừa lưu trong lúc đang dịch.
    kept = 0
    for item in translated:
        updated = SubtitleSegment.objects.filter(
            job=job, index=item["index"], manually_edited=False
        ).update(
            translated_text=item.get("translated_text", ""),
            updated_at=timezone.now(),
        )
        if not updated:
            kept += 1
    if kept:
        logger.info("Job %s: giữ nguyên %d câu đã sửa thủ công", job.id, kept)
    _set_progress(job, 55)


def step_review(job, work: Path) -> None:
    from dichvd.models import SubtitleSegment, TranslationReview, VideoJob
    from dichvd.services import TranslationReviewer

    _set_step(job, "reviewing", VideoJob.Status.REVIEWING, 60)

    model_name = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
    # Câu người dùng đã sửa tay coi như đã được con người duyệt: không gửi đi kiểm định.
    segs_for_review = list(
        SubtitleSegment.objects.filter(job=job, manually_edited=False)
        .order_by("index")
        .values("index", "original_text", "translated_text")
    )
    if not segs_for_review:
        _set_progress(job, 70)
        return
    reviewer = TranslationReviewer(model=model_name)
    reviews = reviewer.review_segments(
        segs_for_review,
        context=_load_context(job),
        only_suspicious=True,
    )

    # Giữ lịch sử: mỗi lần chạy là một "round" mới, không xoá kết quả cũ.
    last_round = (
        TranslationReview.objects.filter(segment__job=job).aggregate(m=Max("round"))["m"] or 0
    )
    review_round = last_round + 1
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
    VideoRenderer().render(
        video_path=job.original_file.path,
        segments=segs_render,
        output_path=output_path,
        keep_original_audio=getattr(settings, "KEEP_ORIGINAL_AUDIO", False),
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
def _run_steps(job_id: str, start_from: str | None = None, only: str | None = None) -> dict:
    """
    Chạy các bước từ `start_from` (mặc định: bước đã fail trước đó, hoặc bước đầu).
    `only`: chỉ chạy đúng một bước rồi dừng.
    Khi lỗi: lưu job.failed_step, đánh dấu failed, ném lại exception.
    """
    job = _get_job(job_id)
    work = _job_dir(job_id)

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
        try:
            logger.info("Job %s: bắt đầu bước %s", job_id, name)
            func(job, work)
        except Exception as e:
            logger.exception("Job %s lỗi ở bước %s: %s", job_id, name, e)
            job = _get_job(job_id)
            job.mark_failed(str(e))
            job.failed_step = name
            job.save(update_fields=["failed_step", "updated_at"])
            raise

    if not only and steps and steps[-1][0] == STEP_NAMES[-1]:
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