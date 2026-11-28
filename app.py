import json
import os
import secrets
import sqlite3
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps
from html.parser import HTMLParser

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
    return send_from_directory(APP_DIR, "index.html")


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




@app.get("/api/status")
def service_status():
    return jsonify({
        "ok": True,
        "service": "smpost-search-api",
        "search_configured": True,
        "search_provider": "Brave Search API with DuckDuckGo public-web fallback and result completion",
        "brave_configured": bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
    })


class DuckDuckGoResultParser(HTMLParser):
    """Extract ordinary public-web results from DuckDuckGo's HTML endpoint."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self.current = None
        self.last_result = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if self.current is not None:
            self.current["depth"] += 1
            return
        if tag == "a" and "result__a" in classes:
            self.current = {
                "kind": "title", "depth": 1,
                "url": attrs.get("href", ""), "text": []
            }
        elif "result__snippet" in classes:
            self.current = {"kind": "snippet", "depth": 1, "text": []}

    def handle_startendtag(self, tag, attrs):
        return

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"].append(data)

    def handle_endtag(self, tag):
        if self.current is None:
            return
        self.current["depth"] -= 1
        if self.current["depth"] > 0:
            return
        item = self.current
        self.current = None
        text = " ".join("".join(item["text"]).split())
        if item["kind"] == "title":
            if item["url"] and text:
                self.last_result = {"url": item["url"], "title": text, "snippet": ""}
                self.results.append(self.last_result)
        elif item["kind"] == "snippet" and self.last_result is not None:
            self.last_result["snippet"] = text


def _unwrap_search_url(value):
    """Resolve DuckDuckGo redirect links, including its relative /l/?uddg= form."""
    value = str(value or "").strip()
    try:
        parsed = urllib.parse.urlparse(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        is_duckduckgo_host = (
            host == "duckduckgo.com" or host.endswith(".duckduckgo.com")
        )
        is_relative_redirect = not host and parsed.path.startswith("/l/")
        if is_duckduckgo_host or is_relative_redirect:
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [])
            if target:
                value = target[0].strip()
    except (TypeError, ValueError):
        return ""
    return value


def _allowed_destination_url(value, platform, target_type):
    value = _unwrap_search_url(value)
    try:
        parsed = urllib.parse.urlparse(value)
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        return ""
    domains = {
        "facebook": ("facebook.com",),
        "instagram": ("instagram.com",),
        "tiktok": ("tiktok.com",),
        "telegram": ("t.me",),
        "x": ("x.com", "twitter.com"),
        "linkedin": ("linkedin.com",),
        "youtube": ("youtube.com", "youtu.be"),
        "pinterest": ("pinterest.com",),
    }
    if platform == "other":
        return value
    if not any(host == domain or host.endswith("." + domain)
               for domain in domains.get(platform, ())):
        return ""
    if platform == "facebook" and target_type == "group" and "/groups/" not in parsed.path:
        return ""
    if platform == "facebook" and target_type == "page" and "/groups/" in parsed.path:
        return ""
    return value.split("#", 1)[0]


def _duckduckgo_search(search_query, limit):
    results, seen = [], set()
    for offset in range(0, limit, 30):
        params = urllib.parse.urlencode({"q": search_query, "s": offset})
        url = "https://html.duckduckgo.com/html/?" + params
        req = urllib.request.Request(url, headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        })
        with urllib.request.urlopen(req, timeout=20) as response:
            html = response.read().decode("utf-8", errors="replace")
        parser = DuckDuckGoResultParser()
        parser.feed(html)
        before = len(results)
        for item in parser.results:
            canonical = _unwrap_search_url(item.get("url"))
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            results.append({
                "title": item.get("title", "")[:240],
                "url": canonical,
                "snippet": item.get("snippet", "")[:600],
            })
            if len(results) >= limit:
                break
        if len(results) >= limit or len(results) == before:
            break
    return results[:limit]

@app.get("/api/discover")
def discover_destinations():
    """Search public web indexes for destination links; this is not Facebook-account search."""
    query = str(request.args.get("q", "")).strip()[:180]
    platform = str(request.args.get("platform", "facebook")).lower()
    target_type = str(request.args.get("type", "group")).lower()
    try:
        limit = max(1, min(100, int(request.args.get("limit", "20"))))
    except (TypeError, ValueError):
        return jsonify({"error": "Result limit must be between 1 and 100."}), 400
    if not query:
        return jsonify({"error": "Enter a search topic."}), 400

    scopes = {
        ("facebook", "group"): "site:facebook.com/groups",
        ("facebook", "page"): "site:facebook.com",
        ("instagram", "profile"): "site:instagram.com",
        ("tiktok", "profile"): "site:tiktok.com",
        ("telegram", "channel"): "site:t.me",
        ("telegram", "group"): "site:t.me",
        ("x", "profile"): "(site:x.com OR site:twitter.com)",
        ("linkedin", "page"): "site:linkedin.com",
        ("youtube", "channel"): "site:youtube.com",
        ("pinterest", "profile"): "site:pinterest.com",
        ("other", "other"): "",
    }
    scope = scopes.get((platform, target_type))
    if scope is None:
        return jsonify({"error": "This channel type is not configured for public search yet."}), 400
    search_query = f"{scope} {query}".strip()
    if platform == "facebook" and target_type == "page":
        search_query += " -site:facebook.com/groups"

    results = []
    providers = []
    provider_errors = []
    api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()

    if api_key:
        try:
            params = urllib.parse.urlencode({
                "q": search_query, "count": min(20, limit),
                "safesearch": "moderate",
            })
            req = urllib.request.Request(
                "https://api.search.brave.com/res/v1/web/search?" + params,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                    "User-Agent": "smpost/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=18) as response:
                payload = json.loads(response.read().decode("utf-8"))
            web_results = (payload.get("web") or {}).get("results") or []
            for item in web_results:
                canonical = _allowed_destination_url(item.get("url"), platform, target_type)
                if not canonical:
                    continue
                results.append({
                    "title": str(item.get("title", "")).strip()[:240] or urllib.parse.urlparse(canonical).hostname,
                    "url": canonical,
                    "snippet": str(item.get("description", "")).strip()[:600],
                    "domain": urllib.parse.urlparse(canonical).hostname or "",
                })
            if results:
                providers.append("Brave")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                UnicodeDecodeError, OSError) as exc:
            provider_errors.append("Brave: " + exc.__class__.__name__)

    # Fill any shortfall: Brave returns at most 20 results per request, and its
    # index may return fewer matching Facebook group URLs than the user requested.
    if len(results) < limit:
        try:
            raw_results = _duckduckgo_search(search_query, limit)
            for item in raw_results:
                canonical = _allowed_destination_url(item.get("url"), platform, target_type)
                if not canonical:
                    continue
                results.append({
                    "title": item.get("title") or urllib.parse.urlparse(canonical).hostname,
                    "url": canonical,
                    "snippet": item.get("snippet", ""),
                    "domain": urllib.parse.urlparse(canonical).hostname or "",
                })
            providers.append("DuckDuckGo")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            provider_errors.append("DuckDuckGo: " + exc.__class__.__name__)

    unique = []
    seen = set()
    for item in results:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
        if len(unique) >= limit:
            break

    if not unique and provider_errors:
        return jsonify({
            "error": "Public search providers could not return results. Please retry.",
            "detail": "; ".join(provider_errors),
        }), 502

    source = " + ".join(providers) + (" public web index" if len(providers) == 1 else " public web indexes")
    return jsonify({
        "results": unique,
        "requested": limit,
        "returned": len(unique),
        "query": query,
        "platform": platform,
        "source": source or "No public search provider returned results",
    })


@app.after_request
def add_search_cors(response):
    origin = request.headers.get("Origin", "")
    allowed = {
        item.strip() for item in
        os.environ.get("ALLOWED_ORIGINS", "https://riobase.github.io").split(",")
        if item.strip()
    }
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

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
    networks = [x for x in networks if x in {"facebook", "instagram", "tiktok", "telegram", "x", "linkedin", "youtube", "pinterest", "other"}]
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
    supported_networks = {"facebook", "instagram", "tiktok", "telegram", "x", "linkedin", "youtube", "pinterest", "other"}
    if network not in supported_networks:
        return jsonify({"error": "Choose a supported platform."}), 400
    if target_type not in {"page", "group", "profile", "channel", "community", "other"} or not name:
        return jsonify({"error": "Enter a destination name and type."}), 400
    canonical = _allowed_destination_url(url, network, target_type)
    if not canonical:
        return jsonify({"error": "Use a valid HTTPS link for the selected platform."}), 400
    url = canonical
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
