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
    "https://worker.example/api/discover?q=education&limit=101"
  ));
  assert.equal(tooMany.status, 400);
});
