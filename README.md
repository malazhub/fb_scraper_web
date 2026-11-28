# SocialMediaPosts

A responsive content-planning workspace built as a safe replacement for the previous browser-driven Facebook scraper.

## What this branch contains

- Responsive desktop/mobile dashboard branded **SocialMediaPosts**
- Password-protected workspace (configure `APP_PASSWORD`)
- CSRF-protected write endpoints and HTTP-only session cookies
- Upload support for common image/video formats
- Draft library
- Per-network schedule slots over a selected date range
- Destination directory for Facebook Pages, Instagram profiles and TikTok profiles
- Explicit disclosure that Facebook Group links are reference-only and platform publishing is not active

## Important: this is a planning MVP, not an auto-publisher

The app currently **does not publish posts**, run a background scheduler, connect social accounts, search public groups, or scrape public pages. The UI records planned dates; it does not dispatch jobs. Do not treat a saved schedule as a guaranteed post.

### Platform constraints verified 2026-11-28

- **Facebook Groups:** Meta removed the Groups API and the `publish_to_groups` permission in April 2024. There is no supported general-purpose API in this project for discovering arbitrary public groups or publishing to them. Group URLs can be saved as manual reference destinations only.
- **Facebook Pages:** Meta's Page Feed API supports Page publishing for a person with the Page's `CREATE_CONTENT` task and a Page access token with permissions including `pages_manage_posts`, `pages_read_engagement`, and `pages_show_list`. App review/advanced access may be required depending on who uses the app.
- **Instagram:** publishing is for eligible Professional (Business/Creator) accounts, not consumer personal accounts. A Meta app, OAuth authorization, publishing permissions, public media hosting, and any applicable review/authorization are required.
- **TikTok:** Direct Post requires an app with Content Posting API and approved `video.publish` scope plus creator authorization. Unaudited clients are restricted to private visibility and small usage caps. TikTok also says API clients must serve a broad creator audience; a tool built solely to post to the developer's own accounts is not an acceptable intended use. This means the proposed internal-only scheduler is not eligible for TikTok Direct Post as currently described.

No software can guarantee that a social platform will never restrict an account. This project deliberately avoids password/cookie automation, browser bots, CAPTCHA workarounds, fake-human behavior, mass group posting, and scraping.

## Local run

Python 3.11+ recommended.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export APP_PASSWORD='use-a-long-unique-password'
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export COOKIE_SECURE=false
python app.py
```

Open http://127.0.0.1:5000

On production, set `COOKIE_SECURE=true`, use HTTPS, set a strong `SECRET_KEY`, and configure `APP_PASSWORD` through the hosting provider's secret/environment-variable interface. Never commit credentials, OAuth tokens, or platform client secrets.

## Persistence and production deployment

The current SQLite database and uploaded media are stored under `DATA_DIR` (default: `instance/`). This is suitable for local testing only unless the host provides persistent storage.

Render Free web services have ephemeral filesystems: uploaded media and SQLite data can disappear on restart, redeploy, or spin-down. For a real deployment, configure a persistent storage layer first (for example managed PostgreSQL for records plus object storage for media, or a paid persistent disk). Set `DATA_DIR=/var/data` only after a disk is mounted there. Do not deploy this as a production content store on Render Free.

## Next work required before production

1. Choose durable PostgreSQL and object storage; add migrations and backups.
2. Add user accounts/roles if multiple people will use the service.
3. Add official OAuth flows and encrypted token storage for eligible Meta Pages and Instagram Professional accounts.
4. Submit the Meta app for the required permissions/review.
5. For TikTok, assess eligibility against its broad-audience intended-use policy before investing in the integration; the current owner-only use case is not eligible.
6. Implement a durable job queue, timezone-aware scheduling, retries, rate-limit handling, idempotency, audit logs and per-platform status reporting.
7. Add platform-specific content validation and an explicit final consent step before each API submission.
8. Run security review, upload malware/content validation, and end-to-end tests.

## Branch safety

This work is on branch `socialmediaposts-safe-v1`. The default `main` branch and existing deployment are unchanged. Review this branch before merging or deploying.
