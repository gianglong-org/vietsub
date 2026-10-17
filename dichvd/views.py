import os
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST, require_GET
from kombu.exceptions import OperationalError

from .forms import VideoUploadForm
from .models import VideoJob, SubtitleSegment
from .tasks import process_video_pipeline


def serve_media(request, path):
    jobs_root = (Path(settings.MEDIA_ROOT) / "jobs").resolve()
    file_path = (jobs_root / path).resolve()
    if not file_path.is_relative_to(jobs_root) or not file_path.is_file():
        raise Http404
    return FileResponse(file_path.open("rb"))


def dashboard(request):
    jobs = VideoJob.objects.order_by("-created_at")[:50]
    return render(request, "videos/dashboard.html", {"jobs": jobs})


def upload(request):
    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            if not os.environ.get("GEMINI_API_KEY"):
                form.add_error(
                    None,
                    "Cần cấu hình GEMINI_API_KEY trước khi xử lý video.",
                )
                return render(request, "videos/upload.html", {"form": form})
            job = form.save(commit=False)
            job.owner = None
            job.status = VideoJob.Status.UPLOADED
            job.save()
            try:
                process_video_pipeline.delay(str(job.id))
            except OperationalError:
                job.mark_failed(
                    "Không kết nối được Redis. Hãy khởi động Redis rồi thử xử lý lại."
                )
                messages.error(
                    request,
                    "Video đã lưu nhưng chưa đưa vào hàng đợi. Hãy khởi động Redis/Celery rồi thử lại.",
                )
            return redirect("videos:detail", job_id=job.id)
    else:
        form = VideoUploadForm()
    return render(request, "videos/upload.html", {"form": form})


def detail(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    segments = job.segments.all()[:200]
    return render(request, "videos/detail.html", {"job": job, "segments": segments})


def editor(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    segments = job.segments.order_by("index")
    return render(request, "videos/editor.html", {"job": job, "segments": segments})


@require_POST
def update_segment(request, job_id, segment_id):
    job = get_object_or_404(VideoJob, id=job_id)
    seg = get_object_or_404(SubtitleSegment, id=segment_id, job=job)
    text = request.POST.get("translated_text", "").strip()
    if text:
        seg.translated_text = text
        seg.is_reviewed = True
        seg.review_status = "fixed"
        seg.save(update_fields=["translated_text", "is_reviewed", "review_status", "updated_at"])
    return JsonResponse({"ok": True, "translated_text": seg.translated_text})


@require_GET
def job_status(request, job_id):
    """API polling trạng thái."""
    job = get_object_or_404(VideoJob, id=job_id)
    return JsonResponse({
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_message": job.error_message,
        "output_url": job.output_file.url if job.output_file else None,
    })


@require_POST
def retry_job(request, job_id):
    job = get_object_or_404(VideoJob, id=job_id)
    if job.status != VideoJob.Status.FAILED:
        return JsonResponse({"ok": False, "error": "Chỉ retry job failed"}, status=400)
    job.status = VideoJob.Status.UPLOADED
    job.error_message = ""
    job.progress = 0
    job.save(update_fields=["status", "error_message", "progress", "updated_at"])
    try:
        process_video_pipeline.delay(str(job.id))
    except OperationalError:
        job.mark_failed(
            "Không kết nối được Redis. Hãy khởi động Redis rồi thử xử lý lại."
        )
        return JsonResponse(
            {"ok": False, "error": job.error_message},
            status=503,
        )
    return JsonResponse({"ok": True})
