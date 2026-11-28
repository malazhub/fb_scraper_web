import test from "node:test";
import assert from "node:assert/strict";
import worker from "../worker.mjs";

const originalFetch = globalThis.fetch;

test("status endpoint reports a configured search provider and CORS", async () => {
  const response = await worker.fetch(new Request("https://worker.example/api/status", {
    headers: { Origin: "https://riobase.github.io" }
  }));
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("access-control-allow-origin"), "https://riobase.github.io");
  const body = await response.json();
  assert.equal(body.search_configured, true);
  assert.equal(body.search_provider, "DuckDuckGo public web index");
});

test("discovery returns Facebook groups and filters unrelated URLs", async () => {
  const html = [
    '<div class="result"><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.facebook.com%2Fgroups%2Feducation-hub">Education <b>Hub</b></a><a class="result__snippet">A public education group.</a></div>',
    '<div class="result"><a class="result__a" href="https://www.facebook.com/education-page">Education Page</a></div>',
    '<div class="result"><a class="result__a" href="https://example.com/groups/education">Unrelated site</a></div>'
  ].join("");
  globalThis.fetch = async function () {
    return new Response(html, { status: 200, headers: { "content-type": "text/html" } });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=education&platform=facebook&type=group&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.returned, 1);
    assert.equal(body.results[0].title, "Education Hub");
    assert.equal(body.results[0].url, "https://www.facebook.com/groups/education-hub/");
    assert.equal(body.results[0].snippet, "A public education group.");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("search passes location and supports TikTok profile destinations", async () => {
  const originalFetch = globalThis.fetch;
  let capturedSearch = "";
  globalThis.fetch = async function (url) {
    capturedSearch = new URL(url).searchParams.get("q") || "";
    const html = '<div class="result"><a class="result__a" href="https://www.tiktok.com/@learnwithme">Learn with me</a><a class="result__snippet">Learning videos.</a></div>';
    return new Response(html, { status: 200, headers: { "content-type": "text/html" } });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=education&location=Beirut%2C%20Lebanon&platform=tiktok&type=profile&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.match(capturedSearch, /education/);
    assert.match(capturedSearch, /Beirut, Lebanon/);
    assert.equal(body.platform, "tiktok");
    assert.equal(body.location, "Beirut, Lebanon");
    assert.equal(body.returned, 1);
    assert.equal(body.results[0].url, "https://www.tiktok.com/@learnwithme");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("rejects other website origins", async () => {
  const response = await worker.fetch(new Request("https://worker.example/api/status", {
    headers: { Origin: "https://evil.example" }
  }));
  assert.equal(response.status, 403);
  assert.equal(response.headers.get("access-control-allow-origin"), null);
});

test("validates query and result limit", async () => {
  const empty = await worker.fetch(new Request("https://worker.example/api/discover?q=%20"));
  assert.equal(empty.status, 400);
  const tooMany = await worker.fetch(new Request(
    "https://worker.example/api/discover?q=education&limit=301"
  ));
  assert.equal(tooMany.status, 400);
});

test("search returns X communities and filters out profile URLs", async () => {
  globalThis.fetch = async function () {
    const html = [
      '<a class="result__a" href="https://x.com/i/communities/123456789">Public educators community</a>',
      '<a class="result__a" href="https://x.com/educatornews">Educator News profile</a>'
    ].join("");
    return new Response(html, { status: 200, headers: { "content-type": "text/html" } });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=educators&platform=x&type=community&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.returned, 1);
    assert.equal(body.results[0].url, "https://x.com/i/communities/123456789");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("LinkedIn group search only returns group URLs", async () => {
  globalThis.fetch = async function () {
    const html = [
      '<a class="result__a" href="https://www.linkedin.com/groups/123456/">Teachers community</a>',
      '<a class="result__a" href="https://www.linkedin.com/in/teacher-one/">Teacher profile</a>'
    ].join("");
    return new Response(html, { status: 200, headers: { "content-type": "text/html" } });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=teachers&platform=linkedin&type=group&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.returned, 1);
    assert.equal(body.results[0].url, "https://www.linkedin.com/groups/123456/");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("location-specific empty results retry without location and report broader results", async () => {
  let calls = 0;
  globalThis.fetch = async function (url) {
    calls += 1;
    const q = new URL(url).searchParams.get("q") || "";
    const html = q.includes("Beirut")
      ? '<html><body>No local index results</body></html>'
      : '<a class="result__a" href="https://www.tiktok.com/@learnwithme">Learn with me</a>';
    return new Response(html, { status: 200, headers: { "content-type": "text/html" } });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=education&location=Beirut&platform=tiktok&type=profile&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.ok(calls >= 2, 'should retry broader query after location-specific search');
    assert.equal(body.returned, 1);
    assert.equal(body.location_relaxed, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("falls back to DuckDuckGo Lite when HTML search is unavailable", async () => {
  const requests = [];
  globalThis.fetch = async function (url) {
    const parsed = new URL(url);
    requests.push(parsed.hostname);
    if (parsed.hostname === "html.duckduckgo.com") {
      return new Response("temporarily unavailable", { status: 503 });
    }
    return new Response(
      '<a class="result-link" href="https://www.tiktok.com/@fallbackcreator">Fallback Creator</a><div class="result-snippet">Public creator profile.</div>',
      { status: 200, headers: { "content-type": "text/html" } }
    );
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=education&platform=tiktok&type=profile&limit=10",
      { headers: { Origin: "https://riobase.github.io" } }
    ));
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.equal(body.returned, 1);
    assert.equal(body.results[0].url, "https://www.tiktok.com/@fallbackcreator");
    assert.deepEqual(requests, ["html.duckduckgo.com", "lite.duckduckgo.com"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("reports both provider HTTP failures in diagnostic detail", async () => {
  globalThis.fetch = async function (url) {
    const host = new URL(url).hostname;
    return new Response("unavailable", { status: host === "html.duckduckgo.com" ? 503 : 429 });
  };
  try {
    const response = await worker.fetch(new Request(
      "https://worker.example/api/discover?q=education&platform=tiktok&type=profile&limit=10"
    ));
    const body = await response.json();
    assert.equal(response.status, 502);
    assert.match(body.detail, /HTML HTTP 503/);
    assert.match(body.detail, /Lite HTTP 429/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
