import os
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST, require_GET
from kombu.exceptions import OperationalError

from .forms import VideoUploadForm
from .models import VideoJob, SubtitleSegment
from .tasks import STEP_NAMES, fail_stale_jobs, process_video_pipeline

# Các lỗi có thể xảy ra khi đưa task vào hàng đợi mà Redis/Celery không sẵn sàng
# (OperationalError: broker; RuntimeError: result backend; ConnectionError: mạng).
QUEUE_ERRORS = (OperationalError, RuntimeError, ConnectionError)


def serve_media(request, path):
    jobs_root = (Path(settings.MEDIA_ROOT) / "jobs").resolve()
    file_path = (jobs_root / path).resolve()
    if not file_path.is_relative_to(jobs_root) or not file_path.is_file():
        raise Http404
    return FileResponse(file_path.open("rb"))


def dashboard(request):
    fail_stale_jobs()  # dọn job treo trước khi hiển thị
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
            except QUEUE_ERRORS:
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
        seg.manually_edited = True  # pipeline không được ghi đè câu này
        seg.save(
            update_fields=[
                "translated_text",
                "is_reviewed",
                "review_status",
                "manually_edited",
                "updated_at",
            ]
        )
    return JsonResponse({"ok": True, "translated_text": seg.translated_text})


@require_GET
def job_status(request, job_id):
    """API polling trạng thái."""
    fail_stale_jobs(job_id)  # job treo sẽ thành failed để giao diện hiện nút Retry
    job = get_object_or_404(VideoJob, id=job_id)
    return JsonResponse({
        "id": str(job.id),
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "failed_step": getattr(job, "failed_step", ""),
        "error_message": job.error_message,
        "output_url": job.output_file.url if job.output_file else None,
    })


@require_POST
def retry_job(request, job_id):
    """
    Retry job failed từ đúng bước lỗi.

    POST param tuỳ chọn `step`: chạy lại từ một bước sớm hơn (hoặc bằng) bước lỗi,
    ví dụ muốn dịch lại sau khi sửa glossary. Không cho nhảy tới bước sau bước
    lỗi vì dữ liệu của các bước trước đó chưa có.
    """
    fail_stale_jobs(job_id)  # cho phép retry ngay cả khi job vừa bị phát hiện là treo
    job = get_object_or_404(VideoJob, id=job_id)

    if job.status != VideoJob.Status.FAILED:
        return JsonResponse(
            {"ok": False, "error": "Chỉ retry job failed"},
            status=400,
        )

    # Bước lỗi: ưu tiên failed_step, sau đó current_step. Job chưa từng chạy
    # (ví dụ lỗi Redis lúc upload) hoặc giá trị không hợp lệ -> chạy từ đầu.
    failed_step = getattr(job, "failed_step", "") or job.current_step
    if failed_step not in STEP_NAMES:
        failed_step = STEP_NAMES[0]

    step = request.POST.get("step", "").strip() or failed_step
    if step not in STEP_NAMES:
        return JsonResponse(
            {"ok": False, "error": f"Bước không hợp lệ. Chọn một trong: {', '.join(STEP_NAMES)}"},
            status=400,
        )
    if STEP_NAMES.index(step) > STEP_NAMES.index(failed_step):
        return JsonResponse(
            {
                "ok": False,
                "error": f"Không thể bỏ qua bước lỗi '{failed_step}'. "
                         f"Hãy chạy từ '{failed_step}' hoặc bước trước đó.",
            },
            status=400,
        )

    # Chặn chạy đồng thời: chuyển FAILED -> UPLOADED bằng một câu UPDATE có điều kiện.
    # Chỉ đúng một request thắng (rows == 1); request thứ hai (double click,
    # hai tab, hai người dùng) thấy status đã đổi nên bị từ chối.
    claimed = VideoJob.objects.filter(
        id=job.id, status=VideoJob.Status.FAILED
    ).update(
        status=VideoJob.Status.UPLOADED,
        error_message="",
        updated_at=timezone.now(),
    )
    if not claimed:
        return JsonResponse(
            {"ok": False, "error": "Job đang được xử lý hoặc đã được retry."},
            status=409,
        )

    try:
        process_video_pipeline.delay(str(job.id), start_from=step)
    except QUEUE_ERRORS:
        job.refresh_from_db()
        job.mark_failed(
            "Không kết nối được Redis. "
            "Hãy khởi động Redis/Celery rồi thử lại."
        )
        # Giữ lại bước lỗi để lần retry sau vẫn chạy đúng chỗ.
        if hasattr(job, "failed_step"):
            job.failed_step = failed_step
            job.save(update_fields=["failed_step", "updated_at"])
        return JsonResponse(
            {"ok": False, "error": job.error_message},
            status=503,
        )

    return JsonResponse({
        "ok": True,
        "retry_step": step,
        "message": f"Đã đưa job vào hàng đợi, chạy lại từ bước '{step}'.",
    })