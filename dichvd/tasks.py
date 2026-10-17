"""
Celery tasks cho pipeline xử lý video.
Mỗi bước cập nhật status + progress, có thể retry độc lập.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


def _get_job(job_id: str):
    from dichvd.models import VideoJob
    return VideoJob.objects.get(id=job_id)


def _job_dir(job_id: str) -> Path:
    base = Path(getattr(settings, "MEDIA_ROOT", "media")) / "jobs" / str(job_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


@shared_task(bind=True, max_retries=2, default_retry_delay=30)
def process_video_pipeline(self, job_id: str):
    """
    Pipeline đầy đủ MVP:
    uploaded → transcribing → building_context → translating
    → reviewing → generating_audio → rendering → done
    """
    from dichvd.models import (
        VideoJob,
        SubtitleSegment,
        Character,
        GlossaryTerm,
        ContextSnapshot,
        TranslationReview,
    )
    from dichvd.services import (
        VideoProcessor,
        Transcriber,
        ContextBuilder,
        Translator,
        TranslationReviewer,
        SpeechGenerator,
        VideoRenderer,
    )

    job = _get_job(job_id)
    work = _job_dir(job_id)

    try:
        # ---------- 1. TRANSCRIBE ----------
        job.status = VideoJob.Status.TRANSCRIBING
        job.current_step = "transcribing"
        job.progress = 5
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

        processor = VideoProcessor()
        audio_path = str(work / "audio.wav")
        processor.extract_audio(job.original_file.path, audio_path)

        duration = processor.get_duration(job.original_file.path)
        job.duration_seconds = duration
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

        # Lưu segment gốc
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

        job.progress = 20
        job.save(update_fields=["progress", "updated_at"])

        # ---------- 2. BUILD CONTEXT ----------
        job.status = VideoJob.Status.BUILDING_CONTEXT
        job.current_step = "building_context"
        job.progress = 25
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

        segments_data = list(
            SubtitleSegment.objects.filter(job=job)
            .order_by("index")
            .values("index", "start_ms", "end_ms", "original_text")
        )
        # đổi key cho service
        for s in segments_data:
            s["text"] = s.pop("original_text")

        builder = ContextBuilder(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
        )
        context = builder.build(
            segments=segments_data,
            title=job.title or "",
        )

        # Lưu snapshot
        version = (job.context_snapshots.count() or 0) + 1
        ContextSnapshot.objects.create(
            job=job,
            version=version,
            summary=context.get("summary", ""),
            characters_json=context.get("characters", []),
            glossary_json=context.get("glossary", []),
        )

        # Lưu Character
        Character.objects.filter(job=job).delete()
        name_to_char = {}
        for c in context.get("characters", []):
            char = Character.objects.create(
                job=job,
                name=c.get("name", "Unknown"),
                aliases=c.get("aliases", []),
                gender=c.get("gender", ""),
                description=c.get("description", ""),
            )
            name_to_char[char.name] = char

        # Lưu Glossary
        GlossaryTerm.objects.filter(job=job).delete()
        for g in context.get("glossary", []):
            GlossaryTerm.objects.create(
                job=job,
                source_term=g.get("source_term", ""),
                target_term=g.get("target_term", ""),
                category=g.get("category", ""),
                notes=g.get("notes", ""),
            )

        job.progress = 35
        job.save(update_fields=["progress", "updated_at"])

        # ---------- 3. TRANSLATE ----------
        job.status = VideoJob.Status.TRANSLATING
        job.current_step = "translating"
        job.progress = 40
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

        translator = Translator(
            model=getattr(settings, "TRANSLATION_MODEL", "gemini-2.5-flash"),
            batch_size=getattr(settings, "TRANSLATE_BATCH_SIZE", 20),
        )
        translated = translator.translate(
            segments=segments_data,
            context=context,
            target_language=job.target_language,
        )

        for item in translated:
            SubtitleSegment.objects.filter(job=job, index=item["index"]).update(
                translated_text=item.get("translated_text", ""),
            )

        job.progress = 55
        job.save(update_fields=["progress", "updated_at"])

        # ---------- 4. REVIEW ----------
        job.status = VideoJob.Status.REVIEWING
        job.current_step = "reviewing"
        job.progress = 60
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

        segs_for_review = list(
            SubtitleSegment.objects.filter(job=job)
            .order_by("index")
            .values("index", "original_text", "translated_text")
        )
        reviewer = TranslationReviewer(
            model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
        )
        reviews = reviewer.review_segments(
            segs_for_review,
            context=context,
            only_suspicious=True,
        )

        TranslationReview.objects.filter(segment__job=job).delete()
        for r in reviews:
            seg = SubtitleSegment.objects.get(job=job, index=r["index"])
            if r.get("issues") or r.get("suggestion"):
                TranslationReview.objects.create(
                    segment=seg,
                    issues=r.get("issues", []),
                    suggestion=r.get("suggestion", ""),
                    score=r.get("score"),
                    model_used=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
                )
                if r.get("suggestion"):
                    # Áp dụng suggestion nếu score thấp
                    if (r.get("score") or 1) < 0.7:
                        seg.translated_text = r["suggestion"]
                        seg.review_status = "fixed"
                        seg.is_reviewed = True
                        seg.save(update_fields=["translated_text", "review_status", "is_reviewed", "updated_at"])
                else:
                    seg.review_status = "ok" if (r.get("score") or 1) >= 0.7 else "needs_fix"
                    seg.is_reviewed = True
                    seg.save(update_fields=["review_status", "is_reviewed", "updated_at"])

        job.progress = 70
        job.save(update_fields=["progress", "updated_at"])

        # ---------- 5. GENERATE AUDIO (TTS) ----------
        job.status = VideoJob.Status.GENERATING_AUDIO
        job.current_step = "generating_audio"
        job.progress = 75
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

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

        job.progress = 88
        job.save(update_fields=["progress", "updated_at"])

        # ---------- 6. RENDER VIDEO ----------
        job.status = VideoJob.Status.RENDERING
        job.current_step = "rendering"
        job.progress = 90
        job.save(update_fields=["status", "current_step", "progress", "updated_at"])

        output_path = str(work / "rendered.mp4")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        segs_render = list(
            SubtitleSegment.objects.filter(job=job)
            .order_by("index")
            .values("index", "start_ms", "end_ms", "audio_path")
        )
        renderer = VideoRenderer()
        renderer.render(
            video_path=job.original_file.path,
            segments=segs_render,
            output_path=output_path,
            keep_original_audio=getattr(settings, "KEEP_ORIGINAL_AUDIO", False),
        )

        # Gán file output vào model
        from django.core.files import File
        with open(output_path, "rb") as f:
            job.output_file.save("dubbed.mp4", File(f), save=False)

        job.mark_done()
        logger.info("Job %s hoàn thành", job_id)
        return {"job_id": str(job_id), "status": "done"}

    except Exception as e:
        logger.exception("Pipeline lỗi job %s: %s", job_id, e)
        job = _get_job(job_id)
        job.mark_failed(str(e))
        raise


@shared_task
def retry_from_step(job_id: str, step: str):
    """
    Chạy lại từ một bước cụ thể (dùng khi failed).
    step: transcribing | building_context | translating | reviewing | generating_audio | rendering
    """
    # Đơn giản: gọi lại toàn pipeline (có thể tối ưu sau)
    # Hoặc implement từng bước riêng nếu cần.
    return process_video_pipeline.delay(job_id)
