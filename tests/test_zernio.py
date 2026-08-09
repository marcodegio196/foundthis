"""Zernio protocol tests.

The payload shape is asserted here rather than discovered from a 400 halfway
through a posting run. Shapes come from the working integration in `new-visu`
(`new_visu/customer/social/late_service.py`).
"""

import json
import unittest

from pipeline import zernio
from pipeline.zernio import ZernioError


class TestApiKey(unittest.TestCase):
    def test_reads_key(self):
        self.assertEqual(zernio.read_api_key({"ZERNIO_API_KEY": "sk_" + "a" * 64}), "sk_" + "a" * 64)

    def test_strips_quotes_and_bom(self):
        """A .env edited on Windows picks these up; the symptom is a bare 401."""
        key = "sk_" + "a" * 64
        self.assertEqual(zernio.read_api_key({"ZERNIO_API_KEY": f'﻿"{key}" '}), key)

    def test_missing_key_raises(self):
        with self.assertRaises(ZernioError):
            zernio.read_api_key({})

    def test_blank_key_raises(self):
        with self.assertRaises(ZernioError):
            zernio.read_api_key({"ZERNIO_API_KEY": "   "})

    def test_odd_key_warns_but_is_returned(self):
        with self.assertLogs(zernio.log, level="WARNING"):
            self.assertEqual(zernio.read_api_key({"ZERNIO_API_KEY": "nope"}), "nope")


class TestContentTypes(unittest.TestCase):
    def test_video_types(self):
        self.assertEqual(zernio.content_type_for("clip.mp4"), "video/mp4")
        self.assertEqual(zernio.content_type_for("clip.MOV"), "video/quicktime")

    def test_media_item_type_split(self):
        self.assertEqual(zernio.media_item_type("clip.mp4"), "video")
        self.assertEqual(zernio.media_item_type("thumb.jpg"), "image")

    def test_unknown_extension_raises(self):
        with self.assertRaises(ZernioError):
            zernio.content_type_for("clip.raw")


class TestPostBody(unittest.TestCase):
    def body(self, **kwargs):
        params = dict(
            caption="Found this.",
            accounts={"instagram": "acct_ig", "tiktok": "acct_tt"},
            media_url="https://cdn.zernio.com/x.mp4",
        )
        params.update(kwargs)
        return zernio.build_post_body(**params)

    def test_platforms_carry_account_ids(self):
        entries = {p["platform"]: p["accountId"] for p in self.body()["platforms"]}
        self.assertEqual(entries, {"instagram": "acct_ig", "tiktok": "acct_tt"})

    def test_instagram_posts_as_a_reel(self):
        """Vertical video only lands as a reel if platformSpecificData says so."""
        entry = next(p for p in self.body()["platforms"] if p["platform"] == "instagram")
        self.assertEqual(entry["platformSpecificData"]["contentType"], "reels")
        self.assertTrue(entry["platformSpecificData"]["shareToFeed"])

    def test_tiktok_has_no_instagram_specific_data(self):
        entry = next(p for p in self.body()["platforms"] if p["platform"] == "tiktok")
        self.assertNotIn("platformSpecificData", entry)

    def test_cover_frame_converted_to_microseconds(self):
        body = self.body(cover_frame_ms=1500)
        entry = next(p for p in body["platforms"] if p["platform"] == "instagram")
        self.assertEqual(entry["platformSpecificData"]["thumbOffset"], 1_500_000)

    def test_reel_can_be_disabled(self):
        body = self.body(as_reel=False)
        entry = next(p for p in body["platforms"] if p["platform"] == "instagram")
        self.assertNotIn("platformSpecificData", entry)

    def test_media_item_attached(self):
        self.assertEqual(
            self.body()["mediaItems"],
            [{"url": "https://cdn.zernio.com/x.mp4", "type": "video"}],
        )

    def test_publishes_now(self):
        self.assertTrue(self.body()["publishNow"])

    def test_profile_id_included_when_set(self):
        self.assertEqual(self.body(profile_id="prof_1")["profileId"], "prof_1")

    def test_profile_id_omitted_when_unset(self):
        self.assertNotIn("profileId", self.body())

    def test_empty_caption_becomes_a_space(self):
        """Zernio rejects empty content."""
        self.assertEqual(self.body(caption="   ")["content"], " ")

    def test_instagram_without_media_is_refused_locally(self):
        """Better a clear local error than an opaque API rejection."""
        with self.assertRaises(ZernioError) as ctx:
            self.body(media_url=None)
        self.assertIn("instagram", str(ctx.exception))

    def test_no_accounts_is_refused(self):
        with self.assertRaises(ZernioError):
            self.body(accounts={})


class TestErrorParsing(unittest.TestCase):
    def test_unpacks_validation_issues(self):
        """Without this the message is just 'Bad Request'."""
        body = json.dumps({
            "message": "Validation failed",
            "issues": [{"path": ["platforms", 0, "accountId"], "message": "Required"}],
        }).encode()
        error = zernio.parse_error(400, body)
        self.assertIn("platforms.0.accountId: Required", str(error))

    def test_401_explains_the_usual_cause(self):
        error = zernio.parse_error(401, b'{"message": "Unauthorized"}')
        self.assertIn("ZERNIO_API_KEY", str(error))

    def test_non_json_body_is_reported_not_swallowed(self):
        error = zernio.parse_error(502, b"<html>bad gateway</html>")
        self.assertEqual(error.status_code, 502)
        self.assertIn("502", str(error))

    def test_retryable_classification(self):
        self.assertTrue(zernio.parse_error(429, b"{}").retryable)
        self.assertTrue(zernio.parse_error(503, b"{}").retryable)
        self.assertFalse(zernio.parse_error(400, b"{}").retryable)


class TestDeliverySummary(unittest.TestCase):
    def test_flattens_platform_rows(self):
        post = {"platforms": [
            {"platform": "instagram", "status": "published"},
            {"platform": "tiktok", "status": "pending"},
        ]}
        self.assertEqual(
            zernio.summarise_delivery(post),
            {"instagram": "published", "tiktok": "pending"},
        )

    def test_keeps_the_error_text(self):
        post = {"platforms": [
            {"platform": "instagram", "status": "failed", "error": "media too long"}
        ]}
        self.assertEqual(
            zernio.summarise_delivery(post), {"instagram": "failed: media too long"}
        )

    def test_missing_status_is_unknown(self):
        self.assertEqual(
            zernio.summarise_delivery({"platforms": [{"platform": "tiktok"}]}),
            {"tiktok": "unknown"},
        )

    def test_no_platforms(self):
        self.assertEqual(zernio.summarise_delivery({}), {})


class TestClientConfig(unittest.TestCase):
    def client(self, **kwargs):
        params = {"api_key": "sk_" + "a" * 64}
        params.update(kwargs)
        return zernio.ZernioClient(**params)

    def test_default_base_url_matches_the_working_integration(self):
        self.assertEqual(self.client().base_url, "https://zernio.com/api/v1")

    def test_trailing_slash_stripped(self):
        """A trailing slash in .env otherwise produces //posts."""
        self.assertEqual(
            self.client(base_url="https://zernio.com/api/v1/").base_url,
            "https://zernio.com/api/v1",
        )

    def test_accounts_are_mapped_by_platform(self):
        client = self.client()
        client._call = lambda *a, **k: {"accounts": [
            {"platform": "Instagram", "_id": "acct_ig"},
            {"platform": "tiktok", "id": "acct_tt"},
            {"platform": "", "_id": "ignored"},
        ]}
        self.assertEqual(
            client.accounts(), {"instagram": "acct_ig", "tiktok": "acct_tt"}
        )

    def test_accounts_are_cached(self):
        client = self.client()
        calls = []
        client._call = lambda *a, **k: calls.append(1) or {"accounts": []}
        client.accounts()
        client.accounts()
        self.assertEqual(len(calls), 1)

    def test_publish_refuses_unconnected_platform(self):
        client = self.client()
        client._call = lambda *a, **k: {"accounts": [
            {"platform": "instagram", "_id": "acct_ig"}
        ]}
        with self.assertRaises(ZernioError):
            client.publish("/tmp/x.mp4", "caption", platforms=["tiktok"])


if __name__ == "__main__":
    unittest.main()
