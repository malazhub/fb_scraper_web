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

### Search limitations

Search results are publicly indexed web pages matching a platform URL pattern. They are not a private Facebook search API, do not include every group, and cannot confirm whether a group accepts posts. The requested result count is a maximum, not a guarantee. DuckDuckGo may rate-limit or change its HTML, in which case search can temporarily fail.

## Publishing and scheduling

The current page stores post plans and media in this browser. The selected date/time is a plan; it does **not** automatically send a post. This search Worker only provides public destination discovery.

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
