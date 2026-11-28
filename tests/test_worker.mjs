import test from "node:test";
import assert from "node:assert/strict";
import worker from "../worker.mjs";

const originalFetch = globalThis.fetch;
const origin = { headers: { Origin: "https://riobase.github.io" } };
const rss = (items) => '<?xml version="1.0"?><rss version="2.0"><channel>' +
  items.map((x) => '<item><title><![CDATA[' + x.title + ']]></title><link>' + x.url +
  '</link><description><![CDATA[' + (x.snippet || '') + ']]></description></item>').join("") +
  '</channel></rss>';

test("status reports Bing public RSS and CORS", async () => {
  const r = await worker.fetch(new Request("https://worker.example/api/status", origin));
  const b = await r.json();
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("access-control-allow-origin"), "https://riobase.github.io");
  assert.equal(b.search_configured, true);
  assert.equal(b.search_provider, "Bing public RSS search");
});

test("Facebook group discovery parses RSS and filters unrelated destinations", async () => {
  globalThis.fetch = async () => new Response(rss([
    {title:"Education Hub",url:"https://www.facebook.com/groups/education-hub",snippet:"A public education group."},
    {title:"Education page",url:"https://www.facebook.com/education-page"},
    {title:"Unrelated",url:"https://example.com/groups/education"}
  ]), {status:200});
  try {
    const r = await worker.fetch(new Request("https://worker.example/api/discover?q=education&platform=facebook&type=group&limit=10", origin));
    const b = await r.json();
    assert.equal(r.status,200);
    assert.equal(b.returned,1);
    assert.equal(b.results[0].title,"Education Hub");
    assert.equal(b.results[0].url,"https://www.facebook.com/groups/education-hub/");
    assert.equal(b.results[0].snippet,"A public education group.");
  } finally { globalThis.fetch = originalFetch; }
});

test("TikTok profile discovery includes location and sends RSS parameters", async () => {
  let captured;
  globalThis.fetch = async (url) => {
    captured = new URL(url);
    return new Response(rss([{title:"Learn with me",url:"https://www.tiktok.com/@learnwithme",snippet:"Learning videos."}]), {status:200});
  };
  try {
    const r = await worker.fetch(new Request("https://worker.example/api/discover?q=education&location=Beirut%2C%20Lebanon&platform=tiktok&type=profile&limit=10", origin));
    const b = await r.json();
    assert.equal(r.status,200);
    assert.match(captured.searchParams.get("q"),/Beirut, Lebanon/);
    assert.equal(captured.searchParams.get("format"),"rss");
    assert.equal(b.returned,1);
    assert.equal(b.results[0].url,"https://www.tiktok.com/@learnwithme");
  } finally { globalThis.fetch = originalFetch; }
});

test("location-specific empty RSS retries without location", async () => {
  let calls=0;
  globalThis.fetch = async (url) => {
    calls++;
    const q=new URL(url).searchParams.get("q");
    return new Response(rss(q.includes("Beirut") ? [] : [{title:"Creator",url:"https://www.tiktok.com/@creator"}]),{status:200});
  };
  try {
    const r=await worker.fetch(new Request("https://worker.example/api/discover?q=education&location=Beirut&platform=tiktok&type=profile&limit=10",origin));
    const b=await r.json();
    assert.equal(r.status,200);
    assert.equal(calls,2);
    assert.equal(b.location_relaxed,true);
    assert.equal(b.returned,1);
  } finally { globalThis.fetch=originalFetch; }
});

test("provider errors are reported instead of false zero results", async () => {
  globalThis.fetch=async()=>new Response("blocked",{status:503});
  try {
    const r=await worker.fetch(new Request("https://worker.example/api/discover?q=education&platform=facebook&type=group&limit=10",origin));
    const b=await r.json();
    assert.equal(r.status,502);
    assert.match(b.detail,/HTTP 503/);
  } finally { globalThis.fetch=originalFetch; }
});

test("rejects other website origins", async () => {
  const r=await worker.fetch(new Request("https://worker.example/api/status",{headers:{Origin:"https://evil.example"}}));
  assert.equal(r.status,403);
});

test("validates query and result limit", async () => {
  const empty=await worker.fetch(new Request("https://worker.example/api/discover?q=%20"));
  assert.equal(empty.status,400);
  const many=await worker.fetch(new Request("https://worker.example/api/discover?q=education&limit=301"));
  assert.equal(many.status,400);
});
