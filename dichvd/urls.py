from django.urls import path
from . import views

app_name = "videos"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload, name="upload"),
    path("media/jobs/<path:path>", views.serve_media, name="media"),
    path("<uuid:job_id>/", views.detail, name="detail"),
    path("<uuid:job_id>/editor/", views.editor, name="editor"),
    path("<uuid:job_id>/status/", views.job_status, name="status"),
    path("<uuid:job_id>/retry/", views.retry_job, name="retry"),
    path("<uuid:job_id>/segment/<uuid:segment_id>/", views.update_segment, name="update_segment"),
]
