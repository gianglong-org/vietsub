from django import forms
from .models import VideoJob


class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoJob
        fields = ["title", "original_file", "source_language", "target_language"]
        widgets = {
            "title": forms.TextInput(attrs={"class": "form-control", "placeholder": "Tên video (tuỳ chọn)"}),
            "original_file": forms.FileInput(attrs={"class": "form-control", "accept": "video/*"}),
            "source_language": forms.Select(
                attrs={"class": "form-select"},
                choices=[
                    ("auto", "Tự nhận diện"),
                    ("zh", "Tiếng Trung"),
                    ("en", "Tiếng Anh"),
                    ("ja", "Tiếng Nhật"),
                    ("ko", "Tiếng Hàn"),
                ],
            ),
            "target_language": forms.HiddenInput(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["target_language"].initial = "vi"
        self.fields["source_language"].initial = "auto"
