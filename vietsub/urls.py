"""
URL configuration for vietsub project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", include("dichvd.urls")),
]

# Phục vụ file upload và video kết quả (chỉ hoạt động khi DEBUG = True)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)