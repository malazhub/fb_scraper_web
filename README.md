# smpost

Single-page social publishing workspace:

**https://riobase.github.io/socialpost/**

The repository remains \`riobase/socialpost\`. Do not rename it.

## Public destination search backend

The frontend is hosted by GitHub Pages, which serves static files and cannot run the Python API. The search API is now implemented as a Cloudflare Worker in \`worker.mjs\`. It uses DuckDuckGo's public web index; no Brave API key or Render service is required.

### One-time deployment connection

A hosting account and deployment authorization are required for any live backend. This repository is configured for Cloudflare Workers:

1. Create or sign in to a Cloudflare account and enable its free \`workers.dev\` subdomain under **Workers & Pages**.
2. In Cloudflare, create an account API token with **Edit Cloudflare Workers** permission. Copy the account ID from the Cloudflare dashboard.
3. In GitHub, open **riobase/socialpost → Settings → Secrets and variables → Actions**. Add two repository secrets:
   - \`CLOUDFLARE_API_TOKEN\` — the token from step 2.
   - \`CLOUDFLARE_ACCOUNT_ID\` — the Cloudflare account ID.
4. Open **Actions → Deploy smpost search backend → Run workflow**. The workflow deploys \`worker.mjs\` and prints the Worker URL in its run summary.
5. Open **https://riobase.github.io/socialpost/**, paste that Worker URL into **Search backend URL**, and choose **Connect backend**.

After this one-time setup, pushes that change \`worker.mjs\` or \`wrangler.toml\` automatically redeploy the backend. The deployment workflow skips cleanly until the two secrets exist. The URL is stored in this browser only.

Cloudflare's official deployment instructions: https://developers.cloudflare.com/workers/ci-cd/external-cicd/github-actions/

### Search options in the app

The destination finder supports several platform rows in one search. Choose Facebook, Instagram, TikTok, Telegram, X, LinkedIn, YouTube, Pinterest, or Other, with the destination type appropriate to that platform. Add another platform row to search several networks together. Each result has one checkbox: checking it immediately saves the destination and selects it for the post; unchecking it removes it from the current post selection. There is no separate Add-result action or second manual destination-add step.

The optional location is included in each platform search. The requested result count is a maximum per platform; the Worker fetches up to five times that number of raw candidates before filtering and deduplicating destination links. Public search engines may still return fewer valid matches than requested.

Saved results retain their platform and destination type. Facebook post-result links are normalized to the parent group URL to avoid saving multiple posts from the same group as separate destinations.

### Scheduling in the app

The planner supports one date, a daily recurrence, or user-selected dates, plus a 24-hour grid from 00:00 to 23:00. Each checked hour creates a posting time. For example, check five hours to plan five daily slots. The current planner still saves its post plan and browser-uploaded media locally; it is not a cloud publishing scheduler.

### Search limitations

Search results are publicly indexed web pages matching a platform URL pattern. They are not a private Facebook search API, do not include every group, and cannot confirm whether a group accepts posts. The requested result count is a maximum, not a guarantee. DuckDuckGo may rate-limit or change its HTML, in which case search can temporarily fail.

## Publishing and scheduling

The current page stores post plans and media in this browser. The date/recurrence and 24-hour selections are planning data; they do **not** automatically send a post. The search Worker only provides public destination discovery. Closing the browser or turning off the computer does not run or publish saved plans.

Automatic publishing requires separate, approved API integrations and authorization for each platform, a secure token store, persistent scheduler/queue, media hosting, retries and delivery tracking. Adding a destination URL does not grant publishing permission. In particular, arbitrary Facebook Group auto-posting is not supported by Meta's current public publishing APIs. Do not enter social passwords, session cookies or two-factor codes in smpost.

## Tests

Python API tests (legacy backend):
\`\`\`bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
\`\`\`

Search Worker tests:
\`\`\`bash
node --test tests/test_worker.mjs
\`\`\`
