import io
import json
import os
import tempfile
import unittest
from pathlib import Path

_TEMP_DIR = tempfile.TemporaryDirectory(prefix="socialmediaposts-tests-")
os.environ["DATA_DIR"] = str(Path(_TEMP_DIR.name) / "data")
os.environ["APP_PASSWORD"] = "test-only-password"
os.environ["SECRET_KEY"] = "test-only-secret-key"
os.environ["COOKIE_SECURE"] = "false"

import app as application  # noqa: E402


class SocialMediaPostsApiTests(unittest.TestCase):
    def setUp(self):
        application.app.config.update(TESTING=True)
        self.client = application.app.test_client()

    def login(self):
        response = self.client.post("/api/login", json={"password": "test-only-password"})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()["csrf_token"]

    def test_homepage_is_available(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SocialMedia", response.data)

    def test_private_api_requires_login(self):
        self.assertEqual(self.client.get("/api/posts").status_code, 401)

    def test_login_rejects_wrong_password(self):
        response = self.client.post("/api/login", json={"password": "wrong"})
        self.assertEqual(response.status_code, 401)

    def test_write_requires_csrf(self):
        self.login()
        response = self.client.post(
            "/api/posts",
            json={"title": "Draft", "body": "Hello", "networks": ["facebook"]},
        )
        self.assertEqual(response.status_code, 403)

    def test_save_and_list_timezone_aware_schedule(self):
        csrf = self.login()
        payload = {
            "title": "Education announcement",
            "body": "A test post",
            "networks": ["facebook"],
            "targets": [],
            "schedule": [{"network": "facebook", "when": "2030-01-02T09:30:00+02:00"}],
            "media": [],
        }
        response = self.client.post(
            "/api/posts",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        rows = self.client.get("/api/posts").get_json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Education announcement")
        self.assertEqual(len(rows[0]["schedule"]), 1)

    def test_schedule_rejects_timezone_naive_datetime(self):
        csrf = self.login()
        payload = {
            "title": "Bad schedule",
            "body": "Test",
            "networks": ["facebook"],
            "targets": [],
            "schedule": [{"network": "facebook", "when": "2030-01-02T09:30:00"}],
            "media": [],
        }
        response = self.client.post(
            "/api/posts",
            data=json.dumps(payload),
            content_type="application/json",
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(response.status_code, 400)

    def test_target_requires_https_platform_url(self):
        csrf = self.login()
        invalid = self.client.post(
            "/api/targets",
            json={"network": "facebook", "target_type": "page", "name": "Example", "url": "http://www.facebook.com/example"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(invalid.status_code, 400)
        valid = self.client.post(
            "/api/targets",
            json={"network": "facebook", "target_type": "page", "name": "Example", "url": "https://www.facebook.com/example"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(valid.status_code, 201)

    def test_upload_returns_private_media_reference(self):
        csrf = self.login()
        response = self.client.post(
            "/api/upload",
            data={"files": (io.BytesIO(b"unit-test-image"), "test.jpg")},
            headers={"X-CSRF-Token": csrf},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        item = response.get_json()["files"][0]
        self.assertTrue(item["filename"].endswith(".jpg"))
        self.assertEqual(self.client.get(item["url"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
