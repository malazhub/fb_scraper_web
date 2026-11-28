const ALLOWED_ORIGIN = "https://riobase.github.io";
const SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/";

const SEARCH_SCOPES = {
  "facebook:group": "site:facebook.com/groups",
  "facebook:page": "site:facebook.com -site:facebook.com/groups",
  "instagram:profile": "site:instagram.com",
  "tiktok:profile": "site:tiktok.com",
  "telegram:group": "site:t.me",
  "telegram:channel": "site:t.me",
  "x:profile": "(site:x.com OR site:twitter.com) -inurl:/status/ -inurl:/i/communities/",
  "x:community": "site:x.com/i/communities",
  "linkedin:page": "site:linkedin.com/company",
  "linkedin:group": "site:linkedin.com/groups",
  "linkedin:profile": "site:linkedin.com/in",
  "youtube:channel": "(site:youtube.com/@ OR site:youtube.com/channel OR site:youtube.com/c OR site:youtube.com/user)",
  "pinterest:profile": "site:pinterest.com -inurl:/pin/ -inurl:/board/",
  "pinterest:board": "site:pinterest.com -inurl:/pin/",
  "other:other": ""
};

function responseJson(payload, status, origin) {
  const headers = new Headers({
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff"
  });
  if (!origin || origin === ALLOWED_ORIGIN) {
    headers.set("access-control-allow-origin", ALLOWED_ORIGIN);
    headers.set("vary", "Origin");
    headers.set("access-control-allow-methods", "GET, OPTIONS");
    headers.set("access-control-allow-headers", "Content-Type");
    headers.set("access-control-max-age", "86400");
  }
  return new Response(status === 204 ? null : JSON.stringify(payload), { status: status || 200, headers });
}

function decodeEntities(value) {
  return String(value || "")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&nbsp;/gi, " ")
    .replace(/&#(\d+);/g, function (_, n) {
      const code = Number(n);
      return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : " ";
    })
    .replace(/&#x([0-9a-f]+);/gi, function (_, n) {
      const code = parseInt(n, 16);
      return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : " ";
    });
}

function stripMarkup(value) {
  return decodeEntities(String(value || "").replace(/<[^>]*>/g, " "))
    .replace(/\s+/g, " ")
    .trim();
}

function getAttribute(tag, name) {
  const match = String(tag || "").match(new RegExp(name + "\\s*=\\s*(['\"])(.*?)\\1", "i"));
  return match ? decodeEntities(match[2]) : "";
}

function unwrapSearchUrl(value) {
  try {
    let parsed = new URL(String(value || "").trim(), "https://duckduckgo.com");
    if (parsed.hostname === "duckduckgo.com" || parsed.hostname.endsWith(".duckduckgo.com")) {
      const target = parsed.searchParams.get("uddg");
      if (target) parsed = new URL(target);
    }
    return parsed.href;
  } catch (_) {
    return "";
  }
}

function allowedDestinationUrl(value, platform, targetType) {
  let parsed;
  try {
    parsed = new URL(unwrapSearchUrl(value));
  } catch (_) {
    return "";
  }
  const host = parsed.hostname.toLowerCase().replace(/\.$/, "");
  if (parsed.protocol !== "https:" || !host || parsed.username || parsed.password) return "";

  const domains = {
    facebook: ["facebook.com"],
    instagram: ["instagram.com"],
    tiktok: ["tiktok.com"],
    telegram: ["t.me"],
    x: ["x.com", "twitter.com"],
    linkedin: ["linkedin.com"],
    youtube: ["youtube.com", "youtu.be"],
    pinterest: ["pinterest.com"]
  };

  if (platform !== "other") {
    const allowed = domains[platform] || [];
    if (!allowed.some(function (domain) {
      return host === domain || host.endsWith("." + domain);
    })) return "";
  }

  const path = parsed.pathname.replace(/\/+$/, "") || "/";
  if (platform === "instagram" && targetType === "profile") {
    if (!/^\/(?!p$|reel$|tv$|explore$|stories$|direct$|accounts$|about$|developer$)[^/]+$/i.test(path)) return "";
  }
  if (platform === "tiktok" && targetType === "profile") {
    if (!/^\/@[^/]+$/i.test(path)) return "";
  }
  if (platform === "x" && targetType === "profile") {
    if (!/^\/(?!i$|home$|explore$|search$|settings$|messages$|notifications$|compose$|intent$|share$|hashtag$)[^/]+$/i.test(path)) return "";
  }
  if (platform === "x" && targetType === "community") {
    if (!/^\/i\/communities\/[^/]+$/i.test(path)) return "";
  }
  if (platform === "linkedin" && targetType === "page") {
    if (!/^\/(company|school)\/[^/]+/i.test(path)) return "";
  }
  if (platform === "linkedin" && targetType === "group") {
    if (!/^\/groups\/[^/]+/i.test(path)) return "";
  }
  if (platform === "linkedin" && targetType === "profile") {
    if (!/^\/in\/[^/]+/i.test(path)) return "";
  }
  if (platform === "youtube" && targetType === "channel") {
    if (!/^\/(?:@[^/]+|channel\/[^/]+|c\/[^/]+|user\/[^/]+)$/i.test(path)) return "";
  }
  if (platform === "pinterest" && targetType === "profile") {
    if (!/^\/(?!pin$|ideas$|_created$|_saved$)[^/]+$/i.test(path)) return "";
  }
  if (platform === "pinterest" && targetType === "board") {
    if (!/^\/(?!pin\/)[^/]+\/[^/]+$/i.test(path)) return "";
  }

  if (platform === "facebook" && targetType === "group") {
    const group = parsed.pathname.match(/^\/groups\/([^/]+)/i);
    if (!group) return "";
    // Search engines often return individual posts as group results.
    // Normalize each to the canonical group home URL.
    parsed.pathname = "/groups/" + group[1] + "/";
    parsed.search = "";
  }
  if (platform === "facebook" && targetType === "page" && /\/groups(?:\/|$)/i.test(parsed.pathname)) return "";
  parsed.hash = "";
  return parsed.href;
}

function extractResults(html) {
  const anchors = Array.from(String(html || "").matchAll(
    /<a\b[^>]*class\s*=\s*["'][^"']*\bresult__a\b[^"']*["'][^>]*>[\s\S]*?<\/a\s*>/gi
  ));
  return anchors.map(function (match, index) {
    const opening = match[0].match(/<a\b[^>]*>/i);
    const href = opening ? getAttribute(opening[0], "href") : "";
    const nextStart = anchors[index + 1] ? anchors[index + 1].index : String(html).length;
    const tail = String(html).slice(match.index + match[0].length, nextStart);
    const snippetMatch = tail.match(
      /<(?:a|div|td|span)\b[^>]*class\s*=\s*["'][^"']*\bresult__snippet\b[^"']*["'][^>]*>([\s\S]*?)<\/(?:a|div|td|span)\s*>/i
    );
    return {
      title: stripMarkup(match[0].replace(/^<a\b[^>]*>/i, "").replace(/<\/a\s*>$/i, "")),
      url: unwrapSearchUrl(href),
      snippet: snippetMatch ? stripMarkup(snippetMatch[1]) : ""
    };
  }).filter(function (item) { return item.title && item.url; });
}

async function searchPublicWeb(query, limit) {
  const results = [];
  const seen = new Set();

  for (let offset = 0; offset < limit; offset += 30) {
    const url = new URL(SEARCH_ENDPOINT);
    url.searchParams.set("q", query);
    url.searchParams.set("s", String(offset));
    url.searchParams.set("kl", "wt-wt");

    const upstream = await fetch(url.toString(), {
      headers: {
        "accept": "text/html,application/xhtml+xml",
        "accept-language": "en-US,en;q=0.9",
        "user-agent": "smpost-search/1.0 (+https://riobase.github.io/socialpost/)"
      },
      cf: { cacheTtl: 0, cacheEverything: false }
    });
    if (!upstream.ok) throw new Error("Search provider returned HTTP " + upstream.status);

    const html = await upstream.text();
    const pageResults = extractResults(html);
    const before = results.length;

    for (const item of pageResults) {
      if (seen.has(item.url)) continue;
      seen.add(item.url);
      results.push(item);
      if (results.length >= limit) break;
    }

    if (results.length >= limit || results.length === before) break;
  }

  return results;
}

export default {
  async fetch(request) {
    const origin = request.headers.get("Origin") || "";
    const url = new URL(request.url);

    if (origin && origin !== ALLOWED_ORIGIN) {
      return responseJson({ error: "Origin not allowed." }, 403, origin);
    }

    if (request.method === "OPTIONS") {
      return responseJson({ ok: true }, 204, origin);
    }
    if (request.method !== "GET") {
      return responseJson({ error: "Method not allowed." }, 405, origin);
    }

    if (url.pathname === "/api/status") {
      return responseJson({
        ok: true,
        service: "smpost-search-api",
        search_configured: true,
        search_provider: "DuckDuckGo public web index"
      }, 200, origin);
    }

    if (url.pathname !== "/api/discover") {
      return responseJson({ error: "Not found." }, 404, origin);
    }

    const query = String(url.searchParams.get("q") || "").trim().slice(0, 180);
    const location = String(url.searchParams.get("location") || "").trim().slice(0, 120);
    const platform = String(url.searchParams.get("platform") || "facebook").toLowerCase();
    const targetType = String(url.searchParams.get("type") || "group").toLowerCase();
    const requestedLimit = Number(url.searchParams.get("limit") || "20");

    if (!query) return responseJson({ error: "Enter a search topic." }, 400, origin);
    if (!Number.isInteger(requestedLimit) || requestedLimit < 1 || requestedLimit > 300) {
      return responseJson({ error: "Result limit must be between 1 and 300." }, 400, origin);
    }

    const scope = SEARCH_SCOPES[platform + ":" + targetType];
    if (scope === undefined) {
      return responseJson({ error: "This channel type is not configured for public search yet." }, 400, origin);
    }

    let searchQuery = (scope + " " + query + (location ? " " + location : "")).trim();
    if (platform === "facebook" && targetType === "page") searchQuery += " -site:facebook.com/groups";

    try {
      const candidateLimit = Math.min(300, requestedLimit * 5);
      const collect = (rawResults) => {
        const results = [];
        const seen = new Set();
        for (const item of rawResults) {
          const safeUrl = allowedDestinationUrl(item.url, platform, targetType);
          if (!safeUrl || seen.has(safeUrl)) continue;
          seen.add(safeUrl);
          results.push({
            title: item.title.slice(0, 240),
            url: safeUrl,
            snippet: item.snippet.slice(0, 600),
            domain: new URL(safeUrl).hostname
          });
          if (results.length >= requestedLimit) break;
        }
        return results;
      };

      let rawResults = await searchPublicWeb(searchQuery, candidateLimit);
      let results = collect(rawResults);
      let locationRelaxed = false;
      if (!results.length && location) {
        const broaderQuery = (scope + " " + query).trim();
        rawResults = await searchPublicWeb(broaderQuery, candidateLimit);
        results = collect(rawResults);
        locationRelaxed = results.length > 0;
      }

      return responseJson({
        results: results,
        requested: requestedLimit,
        returned: results.length,
        query: query,
        location: location,
        location_relaxed: locationRelaxed,
        platform: platform,
        type: targetType,
        source: "DuckDuckGo public web index"
      }, 200, origin);
    } catch (error) {
      return responseJson({
        error: "Public search provider could not return results. Please retry.",
        detail: String(error && error.message ? error.message : "Upstream search error").slice(0, 240)
      }, 502, origin);
    }
  }
};
