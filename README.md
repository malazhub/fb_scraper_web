# smpost

A single-page social publishing workspace. The public GitHub Pages address is:

**https://riobase.github.io/socialpost/**

The repository remains `riobase/socialpost`; do not rename it.

## Current interface

- Facebook-first destination discovery with a requested result count.
- Search results appear inside the page and can be added to the destination list.
- Select destinations with checkboxes.
- Add Instagram, TikTok, Telegram, X, LinkedIn, YouTube, Pinterest or a custom destination using **Add destination / channel**.
- Create one post with text and media.
- Set one publishing date and time for all selected destinations.
- Review and delete locally saved scheduled posts.
- Export and restore a local JSON backup.

## Enable in-page public destination search

GitHub Pages cannot run the Python search API. Deploy the included Flask service using the Render blueprint:

1. In Render, create a new Blueprint deployment from this repository and select `render.yaml`.
2. Add a **Brave Search API key** as the private environment variable `BRAVE_SEARCH_API_KEY`. Do not put the key in the browser or GitHub Pages.
3. Set `ALLOWED_ORIGINS` to `https://riobase.github.io`.
4. After Render deploys, copy the service URL (for example, `https://your-service.onrender.com`).
5. Paste that URL into **Search service URL** in smpost. It is saved in this browser.

Search results are publicly indexed web pages matching Facebook group URL patterns. They are not a private Facebook search API, do not include every group, and must be checked for relevance and whether the group accepts posts. The result count is a maximum requested count, not a guarantee.

## Publishing and scheduling: important limitation

The current GitHub Pages interface saves a schedule in this browser. It does **not** send a post automatically at the selected time. The Flask service currently supplies search and planning endpoints; it is not a production publisher or background scheduler.

Meta removed the general Facebook Groups API and `publish_to_groups` permission. There is no normal app permission that enables automatic publishing to arbitrary public Facebook Groups. Group destinations therefore cannot be auto-posted by this app.

Automatic publishing can be added only for supported destination types after the relevant provider integrations are approved and connected. For example, Facebook Page publishing requires Meta OAuth, a Page access token, Page permissions and app review/advanced access as applicable. Instagram, TikTok, X and Telegram each have separate account, API, scope and review requirements. Merely adding a channel URL does not authorize publishing.

To run reliable automatic publishing, the backend still needs durable storage, encrypted OAuth token storage, a persistent scheduler/queue, retry and idempotency handling, media hosting, and provider-specific publishing workers. Do not enter social passwords, session cookies or 2FA codes in smpost.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export COOKIE_SECURE=false
export BRAVE_SEARCH_API_KEY='your-search-api-key'
python app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```
