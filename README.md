# smpost

A mobile-friendly social content planner designed to be served directly from GitHub Pages. The application is branded **smpost**. The repository name remains `smpost` to preserve existing repository links.

## GitHub Pages edition

The root `index.html` is a static, browser-based application.

- Create posts with text, photos and videos.
- Save Facebook, Instagram and TikTok destination links.
- Choose individual destinations for each platform.
- Set independent daily posting times for each platform.
- Generate dated actions over a selected date range.
- Review each planned action, open its destination, and mark it as manually completed.
- Search the public web or open a platform's own search page for manual review.
- Export/import a JSON backup and export a schedule CSV.
- Install as a basic PWA on supported browsers.

### Important scheduling distinction

The GitHub Pages edition stores a **plan** in the browser. It does not run a server-side/background scheduler, send posts when the browser is closed, or publish automatically. Data is local to the current browser and device. Export backups regularly.

GitHub Pages serves static files. It cannot execute Python or securely hold API secrets. Automatic publishing requires a separately deployed backend, durable database and media storage, a persistent job queue, OAuth connections and approved platform permissions.

## Platform limitations verified against official documentation (2026-11-28)

- **Facebook Groups:** Meta discontinued the former Groups API and `publish_to_groups` permission. There is no supported general-purpose API for discovering and publishing to arbitrary public Groups in this workflow. Group URLs are manual reference destinations only.
- **Facebook Pages:** Page publishing is supported through the Graph API for an authorized person with the Page's `CREATE_CONTENT` task and a Page access token / permissions including `pages_manage_posts`, `pages_read_engagement` and `pages_show_list`. App review or advanced access may be required.
- **Instagram:** Content publishing requires an eligible Professional (Business or Creator) account, Meta app authorization and applicable permissions. Consumer personal accounts are not supported by the Instagram API publishing flow.
- **TikTok:** Direct Post requires Content Posting API access, approved `video.publish` scope, creator authorization and app audit for public visibility. Unaudited clients are restricted to private visibility. TikTok's intended-use policy rejects an internal utility solely for publishing to accounts managed by its developer/team.

No software can guarantee zero account restrictions. This project deliberately avoids browser bots, password/cookie automation, random human-like clicks or scrolling, CAPTCHA bypasses, automatic group joining, and mass unsolicited posting.

## Enable the GitHub Pages site

1. Merge the tested Pages application into `main`.
2. Open repository **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Choose branch **`main`** and folder **`/ (root)`**.
5. Save. With the repository named `smpost`, the project site address is:

   `https://riobase.github.io/smpost/`

After this change is merged, confirm the Pages source is `main` and `/(root)`. The root `index.html` is the site entry point; GitHub Pages does not run the Flask/Python backend. No Render deployment is required for the static planner.

## Backend prototype

The root `app.py` is a separate Flask API/planning prototype. It offers a password-protected API for draft records, destination records, and media uploads. It is **not** connected to the GitHub Pages interface and is **not** an automatic social publisher. Do not use its SQLite/local upload storage as a production content store on an ephemeral host.

Local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export APP_PASSWORD='use-a-long-unique-password'
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export COOKIE_SECURE=false
python app.py
```

A production service needs HTTPS, secret environment variables, durable PostgreSQL and object storage, encrypted OAuth token storage, a reliable job queue, timezone handling, rate-limit-aware retries, idempotency, audit logs and monitoring. The included Render blueprint is a preview configuration; Render Free storage is ephemeral.

## Tests

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions validation workflow also checks Python tests and JavaScript syntax on pushes to this branch and pull requests to `main`.
