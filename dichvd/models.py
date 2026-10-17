from django.db import models
from django.conf import settings
from django.utils import timezone
import uuid
import os


def video_upload_path(instance, filename):
    return f"jobs/{instance.id}/original/{filename}"


def output_upload_path(instance, filename):
    return f"jobs/{instance.id}/output/{filename}"


class VideoJob(models.Model):
    """Một lần xử lý dịch + lồng tiếng video."""

    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Đã tải lên"
        TRANSCRIBING = "transcribing", "Đang nhận diện lời thoại"
        BUILDING_CONTEXT = "building_context", "Đang xây dựng ngữ cảnh"
        TRANSLATING = "translating", "Đang dịch phụ đề"
        REVIEWING = "reviewing", "Đang kiểm định bản dịch"
        GENERATING_AUDIO = "generating_audio", "Đang tạo giọng đọc"
        RENDERING = "rendering", "Đang ghép video"
        DONE = "done", "Hoàn thành"
        FAILED = "failed", "Có lỗi"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="video_jobs",
        null=True,
        blank=True,
    )
    title = models.CharField(max_length=255, blank=True)
    original_file = models.FileField(upload_to=video_upload_path)
    output_file = models.FileField(upload_to=output_upload_path, blank=True, null=True)

    source_language = models.CharField(max_length=16, default="auto")
    target_language = models.CharField(max_length=16, default="vi")

    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.UPLOADED,
        db_index=True,
    )
    progress = models.PositiveSmallIntegerField(default=0)  # 0-100
    current_step = models.CharField(max_length=64, blank=True)
    error_message = models.TextField(blank=True)

    # Chi phí & metadata
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    actual_cost = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Video Job"
        verbose_name_plural = "Video Jobs"

    def __str__(self):
        return f"{self.title or self.id} [{self.status}]"

    def mark_failed(self, message: str):
        self.status = self.Status.FAILED
        self.error_message = message
        self.save(update_fields=["status", "error_message", "updated_at"])

    def mark_done(self):
        self.status = self.Status.DONE
        self.progress = 100
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "progress", "completed_at", "updated_at"])


class Character(models.Model):
    """Nhân vật trong video / series."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        VideoJob,
        on_delete=models.CASCADE,
        related_name="characters",
        null=True,
        blank=True,
    )
    # Có thể mở rộng thành Series sau
    name = models.CharField(max_length=128)
    aliases = models.JSONField(default=list, blank=True)  # ["A", "Anh A", ...]
    gender = models.CharField(max_length=16, blank=True)  # male / female / unknown
    voice_id = models.CharField(max_length=64, blank=True)  # ID giọng TTS
    description = models.TextField(blank=True)
    relationships = models.JSONField(default=dict, blank=True)  # {"B": "sư đệ", ...}

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["job", "name"]]

    def __str__(self):
        return self.name


class GlossaryTerm(models.Model):
    """Từ điển thuật ngữ (tu tiên, chiêu thức, môn phái...)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        VideoJob,
        on_delete=models.CASCADE,
        related_name="glossary_terms",
        null=True,
        blank=True,
    )
    source_term = models.CharField(max_length=255)
    target_term = models.CharField(max_length=255)
    category = models.CharField(max_length=64, blank=True)  # technique, realm, sect...
    notes = models.TextField(blank=True)
    is_locked = models.BooleanField(default=False)  # không cho AI sửa

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["job", "source_term"]]

    def __str__(self):
        return f"{self.source_term} → {self.target_term}"


class ContextSnapshot(models.Model):
    """Tóm tắt ngữ cảnh theo phiên bản (để tái sử dụng / debug)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        VideoJob,
        on_delete=models.CASCADE,
        related_name="context_snapshots",
    )
    version = models.PositiveIntegerField(default=1)
    summary = models.TextField()  # tóm tắt cốt truyện / cảnh
    characters_json = models.JSONField(default=list, blank=True)
    glossary_json = models.JSONField(default=list, blank=True)
    extra = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version"]
        unique_together = [["job", "version"]]

    def __str__(self):
        return f"Context v{self.version} for {self.job_id}"


class SubtitleSegment(models.Model):
    """Một đoạn phụ đề có mốc thời gian."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        VideoJob,
        on_delete=models.CASCADE,
        related_name="segments",
    )
    index = models.PositiveIntegerField()  # thứ tự trong video
    start_ms = models.PositiveIntegerField()
    end_ms = models.PositiveIntegerField()

    original_text = models.TextField()
    translated_text = models.TextField(blank=True)

    speaker = models.ForeignKey(
        Character,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="segments",
    )
    speaker_name = models.CharField(max_length=128, blank=True)  # fallback nếu chưa có Character

    confidence = models.FloatField(null=True, blank=True)  # từ ASR
    is_reviewed = models.BooleanField(default=False)
    review_status = models.CharField(
        max_length=32,
        blank=True,
        choices=[
            ("ok", "OK"),
            ("needs_fix", "Cần sửa"),
            ("fixed", "Đã sửa"),
        ],
    )

    # File audio TTS của đoạn này (nếu tách riêng)
    audio_path = models.CharField(max_length=512, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["index"]
        unique_together = [["job", "index"]]

    def __str__(self):
        return f"[{self.index}] {self.original_text[:40]}..."

    @property
    def start_sec(self):
        return self.start_ms / 1000.0

    @property
    def end_sec(self):
        return self.end_ms / 1000.0

    @property
    def duration_sec(self):
        return (self.end_ms - self.start_ms) / 1000.0


class TranslationReview(models.Model):
    """Kết quả kiểm định AI cho một đoạn phụ đề."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    segment = models.ForeignKey(
        SubtitleSegment,
        on_delete=models.CASCADE,
        related_name="reviews",
    )
    issues = models.JSONField(default=list, blank=True)  # [{"type": "...", "detail": "..."}]
    suggestion = models.TextField(blank=True)
    score = models.FloatField(null=True, blank=True)  # 0-1
    model_used = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for segment {self.segment_id}"
