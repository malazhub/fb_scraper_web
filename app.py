import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

from flask import Flask, jsonify, render_template, request, session, send_from_directory
from werkzeug.utils import secure_filename

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR / "instance"))
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "socialmediaposts.sqlite3"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "webm", "m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "300")) * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE", "true").lower() == "true",
)


def db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
            media_json TEXT NOT NULL DEFAULT '[]', networks_json TEXT NOT NULL DEFAULT '[]',
            targets_json TEXT NOT NULL DEFAULT '[]', schedule_json TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'draft', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS targets (
            id TEXT PRIMARY KEY, network TEXT NOT NULL, target_type TEXT NOT NULL,
            name TEXT NOT NULL, url TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)
        conn.commit()


init_db()


def require_login(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("authenticated"):
            return jsonify({"error": "Sign in to continue."}), 401
        return fn(*args, **kwargs)
    return wrapped


def require_csrf(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        expected = session.get("csrf_token")
        supplied = request.headers.get("X-CSRF-Token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            return jsonify({"error": "Session validation failed. Refresh and try again."}), 403
        return fn(*args, **kwargs)
    return wrapped


def row_to_post(row):
    item = dict(row)
    for key, output in (("media_json", "media"), ("networks_json", "networks"),
                        ("targets_json", "targets"), ("schedule_json", "schedule")):
        item[output] = json.loads(item.pop(key) or "[]")
    return item


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/me")
def me():
    if not session.get("authenticated"):
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True, "csrf_token": session.get("csrf_token")})


@app.post("/api/login")
def login():
    configured_password = os.environ.get("APP_PASSWORD")
    if not configured_password:
        return jsonify({"error": "Administrator must set APP_PASSWORD in the hosting environment."}), 503
    payload = request.get_json(silent=True) or {}
    if not secrets.compare_digest(str(payload.get("password", "")), configured_password):
        return jsonify({"error": "Incorrect password."}), 401
    session.clear()
    session["authenticated"] = True
    session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify({"ok": True, "csrf_token": session["csrf_token"]})


@app.post("/api/logout")
@require_login
@require_csrf
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/posts")
@require_login
def list_posts():
    with db() as conn:
        rows = conn.execute("SELECT * FROM posts ORDER BY updated_at DESC").fetchall()
    return jsonify([row_to_post(row) for row in rows])


@app.post("/api/posts")
@require_login
@require_csrf
def create_post():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title", "")).strip()[:160]
    body = str(payload.get("body", "")).strip()[:10000]
    media, networks = payload.get("media", []), payload.get("networks", [])
    targets, schedule = payload.get("targets", []), payload.get("schedule", [])
    if not title and not body and not media:
        return jsonify({"error": "Add text or at least one media file."}), 400
    if not all(isinstance(value, list) for value in (media, networks, targets, schedule)):
        return jsonify({"error": "Invalid post data."}), 400
    networks = [x for x in networks if x in {"facebook", "instagram", "tiktok"}]
    if not networks:
        return jsonify({"error": "Select at least one social network."}), 400
    clean_schedule = []
    for item in schedule[:1000]:
        if not isinstance(item, dict) or item.get("network") not in networks:
            continue
        try:
            parsed = datetime.fromisoformat(str(item.get("when", "")).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError()
        except ValueError:
            return jsonify({"error": "Schedule times must include a timezone and be valid."}), 400
        clean_schedule.append({"network": item["network"], "when": parsed.isoformat()})
    post_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute("""INSERT INTO posts
            (id,title,body,media_json,networks_json,targets_json,schedule_json,status,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (post_id, title or "Untitled post", body, json.dumps(media), json.dumps(networks),
             json.dumps(targets), json.dumps(clean_schedule), "scheduled" if clean_schedule else "draft", now, now))
        conn.commit()
    return jsonify({"ok": True, "id": post_id}), 201


@app.delete("/api/posts/<post_id>")
@require_login
@require_csrf
def delete_post(post_id):
    with db() as conn:
        result = conn.execute("DELETE FROM posts WHERE id=?", (post_id,))
        conn.commit()
    if not result.rowcount:
        return jsonify({"error": "Post not found."}), 404
    return jsonify({"ok": True})


@app.get("/api/targets")
@require_login
def list_targets():
    with db() as conn:
        rows = conn.execute("SELECT * FROM targets ORDER BY created_at DESC").fetchall()
    return jsonify([dict(row) for row in rows])


@app.post("/api/targets")
@require_login
@require_csrf
def add_target():
    payload = request.get_json(silent=True) or {}
    network, target_type = str(payload.get("network", "")), str(payload.get("target_type", ""))
    name, url = str(payload.get("name", "")).strip()[:160], str(payload.get("url", "")).strip()[:1000]
    if network not in {"facebook", "instagram", "tiktok"}:
        return jsonify({"error": "Choose a supported platform."}), 400
    if target_type not in {"page", "group", "profile"} or not name:
        return jsonify({"error": "Enter a destination name and type."}), 400
    valid_hosts = ("https://www.facebook.com/", "https://facebook.com/",
                   "https://www.instagram.com/", "https://instagram.com/",
                   "https://www.tiktok.com/", "https://tiktok.com/")
    if not url.startswith(valid_hosts):
        return jsonify({"error": "Use a valid HTTPS link from Facebook, Instagram or TikTok."}), 400
    target_id = str(uuid.uuid4())
    with db() as conn:
        conn.execute("INSERT INTO targets VALUES (?,?,?,?,?,?)",
                     (target_id, network, target_type, name, url, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    return jsonify({"ok": True, "id": target_id}), 201


@app.delete("/api/targets/<target_id>")
@require_login
@require_csrf
def delete_target(target_id):
    with db() as conn:
        conn.execute("DELETE FROM targets WHERE id=?", (target_id,))
        conn.commit()
    return jsonify({"ok": True})


@app.post("/api/upload")
@require_login
@require_csrf
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Choose files to upload."}), 400
    results = []
    for file in files:
        if not file or not file.filename:
            continue
        safe_name = secure_filename(file.filename)
        if "." not in safe_name or safe_name.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type: {safe_name}"}), 400
        extension = safe_name.rsplit(".", 1)[1].lower()
        stored = f"{uuid.uuid4().hex}.{extension}"
        path = UPLOAD_DIR / stored
        file.save(path)
        results.append({"name": safe_name, "filename": stored, "url": f"/uploads/{stored}",
                        "kind": "video" if extension in {"mp4", "mov", "webm", "m4v"} else "image",
                        "bytes": path.stat().st_size})
    return jsonify({"files": results}), 201


@app.get("/uploads/<path:filename>")
@require_login
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Upload too large. Increase MAX_UPLOAD_MB or use smaller media."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
