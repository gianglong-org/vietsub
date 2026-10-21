import base64
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, TestCase

from .models import VideoJob
from .services.context_builder import ContextBuilder
from .services.speech_genarator import SpeechGenerator
from .services.transcriber import Transcriber
from .services.translator import Translator
from .services.translation_reviewer import TranslationReviewer


class VideoJobTests(TestCase):
    def test_mark_done_persists_output_file(self):
        job = VideoJob.objects.create(
            original_file="jobs/job-1/original/source.mp4",
            output_file="jobs/job-1/output/dubbed.mp4",
        )

        job.mark_done()
        job.refresh_from_db()

        self.assertEqual(job.status, VideoJob.Status.DONE)
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.output_file.name, "jobs/job-1/output/dubbed.mp4")


class TranscriberTests(SimpleTestCase):
    def test_gemini_transcription_splits_audio_and_offsets_timestamps(self):
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(
                text='{"segments": [{"start_seconds": 1.5, "end_seconds": 2, "text": " first "}]}'
            ),
            SimpleNamespace(
                text='{"segments": [{"start_seconds": 3, "end_seconds": 4, "text": "second"}]}'
            ),
        ]

        def fake_ffmpeg(command, **kwargs):
            if command[0] == "ffprobe":
                return SimpleNamespace(stdout="500")
            Path(command[-1]).write_bytes(b"wav")
            return SimpleNamespace(stdout="")

        with patch(
            "dichvd.services.transcriber.create_gemini_client",
            return_value=client,
        ):
            with patch(
                "dichvd.services.transcriber.subprocess.run",
                side_effect=fake_ffmpeg,
            ):
                segments = Transcriber(chunk_seconds=480).transcribe(
                    "audio.wav",
                    language="zh",
                )

        self.assertEqual(
            [(segment.index, segment.start_ms, segment.end_ms, segment.text) for segment in segments],
            [(0, 1500, 2000, "first"), (1, 483000, 484000, "second")],
        )
        self.assertEqual(client.models.generate_content.call_count, 2)
        call_args = client.models.generate_content.call_args.kwargs
        self.assertEqual(call_args["model"], "gemini-2.5-flash")
        self.assertIn("Ngôn ngữ lời thoại: zh", call_args["contents"][0])
        self.assertEqual(call_args["contents"][1].inline_data.data, b"wav")

    def test_gemini_transcription_requires_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                Transcriber().transcribe("audio.wav")


class GeminiServiceTests(SimpleTestCase):
    def test_context_builder_uses_gemini_json_generation(self):
        client = Mock()
        client.models.generate_content.return_value.text = (
            '{"summary": "A story", "characters": [], "glossary": []}'
        )

        context = ContextBuilder(llm_client=client).build(
            [{"index": 0, "text": "Hello"}],
            title="Sample",
        )

        self.assertEqual(context["summary"], "A story")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")

    def test_translation_reviewer_uses_gemini_json_generation(self):
        client = Mock()
        client.models.generate_content.return_value.text = (
            '{"issues": [], "suggestion": "", "score": 0.9}'
        )
        reviewer = TranslationReviewer(llm_client=client)

        result = reviewer._ai_review(
            {"index": 0, "original_text": "Hello", "translated_text": "Xin chào"},
            context={},
        )

        self.assertEqual(result["score"], 0.9)
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")

    def test_gemini_tts_writes_wav_audio(self):
        audio_data = b"RIFF-test-wav"
        client = Mock()
        client.interactions.create.return_value.output_audio.data = (
            base64.b64encode(audio_data).decode("ascii")
        )
        generator = SpeechGenerator(llm_client=client)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generator.generate_for_segments(
                [{"index": 4, "translated_text": "Xin chào"}],
                job_id="job-1",
                base_dir=temp_dir,
            )
            output = Path(result[0]["audio_path"])
            self.assertEqual(output.suffix, ".wav")
            self.assertEqual(output.read_bytes(), audio_data)

        kwargs = client.interactions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.8-flash-tts")
        self.assertEqual(
            kwargs["generation_config"]["speech_config"],
            [{"voice": "Kore"}],
        )


class TranslatorTests(SimpleTestCase):
    def test_gemini_translation_uses_generate_content_json_configuration(self):
        client = Mock()
        client.models.generate_content.return_value.text = (
            '{"translations": [{"index": 1, "translated_text": "Xin chào"}]}'
        )
        translator = Translator(llm_client=client)

        translated = translator._translate_batch(
            [{"index": 1, "text": "Hello"}],
            context={"summary": "A greeting"},
            target_language="vi",
        )

        self.assertEqual(translated[0]["translated_text"], "Xin chào")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(
            kwargs["contents"],
            (
                '{"summary": "A greeting", "characters": [], "glossary": [], '
                '"segments": [{"index": 1, "text": "Hello"}]}'
            ),
        )
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")
        self.assertIn("tiếng vi", kwargs["config"]["system_instruction"])

    def test_translation_requires_gemini_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                Translator()

    def test_missing_translation_fails_instead_of_copying_source_text(self):
        translator = Translator(llm_client=object())
        translator._call_llm = Mock(return_value='{"translations": []}')

        with self.assertRaisesRegex(RuntimeError, "Không dịch được"):
            translator._translate_batch(
                [{"index": 0, "text": "source text"}],
                context={},
                target_language="vi",
            )
