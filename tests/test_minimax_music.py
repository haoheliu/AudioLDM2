"""Unit tests for the MiniMax music-generation client.

These tests exercise the request building, regional handling, output-format
validation and response parsing of :mod:`audioldm2.minimax_music` without making
any real network calls. The HTTP layer is stubbed via ``requests.post``.
"""

import json
import unittest
from unittest import mock

from audioldm2 import minimax_music
from audioldm2.minimax_music import (
    ENDPOINTS,
    generate_music,
    is_music_model,
    MiniMaxMusicError,
)


class _FakeResponse(object):
    def __init__(self, payload, status_code=200, text=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


class GenerateMusicTests(unittest.TestCase):
    def _common_kwargs(self, **overrides):
        kwargs = {
            "api_key": "test-key",
            "model": "music-3.0",
            "prompt": "upbeat electronic dance",
        }
        kwargs.update(overrides)
        return kwargs

    def test_default_model_used_when_none(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "https://x/a.mp3"}}
            )
            result = generate_music(api_key="test-key", prompt="calm piano")
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["model"], "music-3.0")
        self.assertTrue(result["completed"])

    def test_global_endpoint_and_headers(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "https://x/a.mp3"}}
            )
            generate_music(**self._common_kwargs(region="global_en"))
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], ENDPOINTS["global_en"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        # Global region must not add the CN-only watermark field.
        self.assertNotIn("aigc_watermark", kwargs["json"])

    def test_cn_region_adds_watermark_field(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "https://x/a.mp3"}}
            )
            generate_music(**self._common_kwargs(region="cn_zh"))
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(mock_post.call_args.args[0], ENDPOINTS["cn_zh"])
        self.assertIn("aigc_watermark", sent_payload)

    def test_request_fields_passed_through(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "hexdata"}}
            )
            generate_music(
                api_key="test-key",
                model="music-2.6",
                prompt="lo-fi beats",
                lyrics="la la la",
                stream=True,
                output_format="hex",
                is_instrumental=False,
                region="global_en",
            )
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "music-2.6")
        self.assertEqual(payload["lyrics"], "la la la")
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["output_format"], "hex")

    def test_stream_requires_hex_output(self):
        with self.assertRaises(ValueError):
            generate_music(**self._common_kwargs(stream=True, output_format="url"))

    def test_invalid_output_format_rejected(self):
        with self.assertRaises(ValueError):
            generate_music(**self._common_kwargs(output_format="ogg"))

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            generate_music(**self._common_kwargs(model="not-a-model"))

    def test_invalid_region_rejected(self):
        with self.assertRaises(ValueError):
            generate_music(**self._common_kwargs(region="mars"))

    def test_api_key_required(self):
        with self.assertRaises(ValueError):
            generate_music(api_key="", model="music-3.0", prompt="x")

    def test_non_zero_status_raises(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 1004, "status_msg": "bad model"}}
            )
            with self.assertRaises(MiniMaxMusicError):
                generate_music(**self._common_kwargs())

    def test_http_error_raises(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 1001, "status_msg": "unauthorized"}},
                status_code=401,
            )
            with self.assertRaises(MiniMaxMusicError):
                generate_music(**self._common_kwargs())

    def test_cover_requires_one_audio_input(self):
        with self.assertRaises(ValueError):
            generate_music(api_key="k", cover=True, model="music-cover")
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 2, "audio": "hex"}}
            )
            generate_music(api_key="k", cover=True, model="music-cover", audio_url="https://x/a.mp3")
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "music-cover")

    def test_in_progress_status(self):
        with mock.patch("requests.post") as mock_post:
            mock_post.return_value = _FakeResponse(
                {"base_resp": {"status_code": 0}, "data": {"status": 1}}
            )
            result = generate_music(**self._common_kwargs())
        self.assertFalse(result["completed"])
        self.assertEqual(result["status"], 1)


class IsMusicModelTests(unittest.TestCase):
    def test_generation_models(self):
        for m in ["music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free"]:
            self.assertTrue(is_music_model(m), m)

    def test_cover_models(self):
        for m in ["music-cover", "music-cover-free"]:
            self.assertTrue(is_music_model(m), m)

    def test_non_music_model(self):
        self.assertFalse(is_music_model("audioldm2-full"))


class ConfigTests(unittest.TestCase):
    def test_both_regions_present(self):
        self.assertIn("global_en", ENDPOINTS)
        self.assertIn("cn_zh", ENDPOINTS)
        self.assertEqual(ENDPOINTS["global_en"], "https://api.minimax.io/v1/music_generation")
        self.assertEqual(ENDPOINTS["cn_zh"], "https://api.minimaxi.com/v1/music_generation")

    def test_request_fields_match_reference(self):
        self.assertEqual(minimax_music.REQUIRED_FIELDS, ["model"])
        for field in [
            "model", "prompt", "lyrics", "stream", "output_format", "audio_setting",
            "lyrics_optimizer", "is_instrumental", "audio_url", "audio_base64",
            "cover_feature_id",
        ]:
            self.assertIn(field, minimax_music.REQUEST_FIELDS, field)

    def test_output_and_audio_formats(self):
        self.assertEqual(minimax_music.OUTPUT_FORMATS, ["url", "hex"])
        self.assertEqual(minimax_music.STREAM_OUTPUT_FORMATS, ["hex"])
        self.assertEqual(minimax_music.AUDIO_FORMATS, ["mp3", "wav", "pcm"])

    def test_response_status_values(self):
        self.assertEqual(minimax_music.RESPONSE_STATUS_VALUES, {"in_progress": 1, "completed": 2})
        self.assertEqual(minimax_music.RESPONSE_SUCCESS_CODE, 0)


if __name__ == "__main__":
    unittest.main()
