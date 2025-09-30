
#!/usr/bin/env python3
"""
BlockWireNews Article Bot (hardened)

Key features
- Pull headlines from RSS (scripts/sources.yaml)
- Fetch & extract article text (BeautifulSoup or Trafilatura toggle)
- Summarize with OpenAI (strict JSON schema enforced)
- Write Hugo posts under content/news/YYYY/MM/slug/index.md
- Guardrails: de-dupe (URL + title), domain whitelist, article age cap,
  min words, retries with exponential backoff, disclosure assertion,
  safe/unique slugs, body length clamps, grounding (ensure source citation).

Env:
  OPENAI_API_KEY   (required)
  OPENAI_MODEL     (optional, default: gpt-4.1-mini)
"""

import os
import re
import json as _json
import time
import hashlib
import datetime
import pathlib
import random
from pathlib import Path
from urllib.parse import urlparse

import requests
import feedparser
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_exponential


import yaml  # PyYAML

# ==== DEFAULT PROMPTS (used if prompts.yaml is missing or incomplete) ====
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful financial news summarizer. Stay faithful to source facts; "
    "do not invent numbers or quotes. Avoid advice. Keep tone neutral, professional, "
    "and readable. US English."
)

DEFAULT_USER_TEMPLATE = (
    "Summarize the following crypto news article faithfully and concisely for a general audience.\n"
    "Return a JSON object with keys: \"title\", \"description\", \"bullets\" (3–5 items), \"body\".\n"
    "- \"title\": punchy 60–70 characters, include the main entity/event.\n"
    "- \"description\": SEO meta, 140–155 chars, include a keyword like ‘crypto’, ‘Bitcoin’, ‘DeFi’.\n"
    "- \"bullets\": 3–5 item TL;DR with concrete facts.\n"
    "- \"body\": 700–900 words. Structure: intro; context; what happened; why it matters; market context; one-line takeaway.\n"
    "Include a short \"Sources\" section at the end with the provided URL.\n"
    "Do NOT add financial advice. Do NOT create data that isn’t in the text.\n\n"
    "Source title: {source_title}\n"
    "Source URL: {source_url}\n\n"
    "Article text:\n---\n{article_text}\n---\n\n"
    "Respond ONLY with a fenced JSON block:\n```json\n{ ... }\n```\n"
)

# Safely format the user prompt: only our three placeholders are substituted.
_DEF_PLACEHOLDERS = ("{source_title}", "{source_url}", "{article_text}")


def safe_format_user_template(tmpl: str, **kwargs) -> str:
    # Temporarily replace our placeholders with sentinels
    sentinel = {"{source_title}": "__ST__",
                "{source_url}": "__SU__", "{article_text}": "__AT__"}
    for k, v in sentinel.items():
        tmpl = tmpl.replace(k, v)
    # Escape all remaining braces so .format ignores them
    tmpl = tmpl.replace("{", "{{").replace("}", "}}")
    # Restore placeholders
    for k, v in sentinel.items():
        tmpl = tmpl.replace(v, k)
    return tmpl.format(**kwargs)

def load_prompts() -> dict:
    """Load prompts.yaml if present and merge with defaults to avoid KeyError."""
    merged = {"system": DEFAULT_SYSTEM_PROMPT, "user_template": DEFAULT_USER_TEMPLATE}
    try:
        y = read_yaml(PROMPTS_YAML)
        if isinstance(y, dict):
            if y.get("system"):
                merged["system"] = y["system"]
            if y.get("user_template"):
                merged["user_template"] = y["user_template"]
    except Exception:
        pass
    return merged

# Optional extractors
USE_TRAFILATURA = os.environ.get("BWN_USE_TRAFILATURA", "0") in ("1", "true", "True")
if USE_TRAFILATURA:
    try:
        import trafilatura  # type: ignore
    except Exception as e:
        raise SystemExit("BWN_USE_TRAFILATURA=1 but trafilatura not installed. Add to requirements.txt")


# ==== PATHS ====
REPO_ROOT = Path(__file__).resolve().parents[1]
CONTENT_ROOT = REPO_ROOT / "content" / "news"
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
POSTED_DB = DATA_DIR / "posted.json"
SOURCES_YAML = REPO_ROOT / "scripts" / "sources.yaml"
PROMPTS_YAML = REPO_ROOT / "scripts" / "prompts.yaml"


# ==== RUN CONFIG ====
RUN_CONFIG = {
    "max_candidates": 16,        # consider a few more
    "posts_per_run": 1,          # lower during testing to reduce LLM calls
    "min_words": 250,            # more forgiving for extraction/feeds
    "timeout": 12,               # HTTP timeout (seconds)
    "target_min_words": 500,     # temp lower floor for LLM body
    "target_max_words": 950,     # clamp long bodies
}

# Source/domain controls
ALLOWED_DOMAINS = {
    "coindesk.com", "www.coindesk.com",
    "theblock.co", "www.theblock.co",
    "cointelegraph.com", "www.cointelegraph.com",
    "decrypt.co", "www.decrypt.co",
    "bankless.com", "www.bankless.com",
    "cryptoslate.com", "www.cryptoslate.com",
    "bitcoinmagazine.com", "www.bitcoinmagazine.com",
}
MAX_ARTICLE_AGE_DAYS = 10

# Explicitly skip junk/aggregator/social domains
BLOCKED_DOMAINS = {
    "reddit.com", "www.reddit.com", "link.reddit.com",
    "news.google.com", "news.yahoo.com",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "youtube.com", "www.youtube.com", "youtu.be",
}

# Monetization / disclosure blocks
AFFILIATE_BLOCK = '{{< aff-cta >}}'

DISCLAIMER = "_This article is a summarized news brief for informational purposes only. Not financial advice._"


 # ==== UTILS ====
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return _json.load(f)
    return default


def save_json(path: Path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(txt: str) -> str:
    return re.sub(r"\s+", " ", (txt or "").strip())


def _norm_title(t: str) -> str:
    t = re.sub(r'[^A-Za-z0-9 ]+', '', t).lower().strip()
    t = re.sub(r'\s+', ' ', t)
    return t


def is_probable_duplicate_title(posted: dict, new_title: str) -> bool:
    nt = _norm_title(new_title)
    for _, meta in posted.items():
        mt = _norm_title(meta.get("title", ""))
        if not mt:
            continue
        # quick-and-simple similarity: prefix match and similar length
        if abs(len(nt) - len(mt)) < 10 and nt[:40] == mt[:40]:
            return True
    return False


 # ==== NETWORK / EXTRACTION ====
@retry(wait=wait_exponential(multiplier=1, min=2, max=16), stop=stop_after_attempt(3))
def fetch_url(url: str, timeout: int = 12) -> requests.Response:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BWNewsBot/1.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


@retry(wait=wait_exponential(multiplier=1, min=2, max=16), stop=stop_after_attempt(3))
def extract_main_text(url: str, timeout: int = 12) -> str:
    """
    Use Trafilatura (if enabled) or fallback to a lightweight BeautifulSoup heuristic.
    Fix: do not pass unsupported 'timeout' kwarg to trafilatura.fetch_url; fallback to requests if needed.
    """
    if USE_TRAFILATURA:
        try:
            # Some versions of trafilatura.fetch_url do not accept 'timeout'
            downloaded = trafilatura.fetch_url(url)
        except TypeError:
            downloaded = None

        if not downloaded:
            # Fallback: fetch with requests, then extract from raw HTML
            try:
                r = fetch_url(url, timeout=timeout)
                downloaded = r.text
            except Exception:
                return ""

        txt = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            include_formatting=False,
            favor_recall=True,
        )
        return clean_text(txt) if txt else ""

    # --- BeautifulSoup fallback ---
    from bs4 import BeautifulSoup  # import lazily
    try:
        r = fetch_url(url, timeout=timeout)
    except Exception:
        return ""
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "form", "aside"]):
        tag.decompose()
    nodes = []
    for sel in ["article", "main"]:
        node = soup.find(sel)
        if node:
            nodes.append(node)
    if not nodes:
        nodes = [soup]

    def score_block(block):
        text = " ".join([p.get_text(" ", strip=True)
                        for p in block.find_all("p")])
        return len(text), text

    best_len, best_text = 0, ""
    for node in nodes:
        for block in node.find_all(True):
            if block.name in ["article", "section", "div"]:
                l, t = score_block(block)
                if l > best_len:
                    best_len, best_text = l, t

    return clean_text(best_text)[:15000]

 # ==== LLM ====

# Optional: fake LLM mode for local/CI testing (set env BWN_FAKE_LLM=1)
FAKE_LLM = os.environ.get("BWN_FAKE_LLM", "0") in ("1", "true", "True")

def _fake_llm(article_text: str, source_title: str, source_url: str) -> dict:
    words = article_text.split()
    body = " ".join(words[:750]) if len(words) > 750 else article_text
    bullets = []
    for sent in re.split(r"(?<=[.!?])\s+", article_text)[:4]:
        s = sent.strip()
        if s:
            bullets.append(s[:140])
        if len(bullets) >= 4:
            break
    return {
        "title": source_title[:70] or "Crypto news brief",
        "description": clean_text(article_text)[:150] or "Daily crypto news brief.",
        "bullets": bullets or ["Key points from source."],
        "body": body or "Summary unavailable.",
    }


def _validate_llm_json(txt: str):
    """
    Accept either a fenced JSON block (```json ... ```) or raw JSON text.
    Validate required keys/types and normalize bullets.
    """
    m = re.search(r"```json(.*?)```", txt, re.S | re.I)
    payload = None
    if m:
        candidate = m.group(1).strip()
        try:
            payload = _json.loads(candidate)
        except Exception:
            payload = None
    if payload is None:
        # try parsing the whole content as JSON
        try:
            payload = _json.loads(txt.strip())
        except Exception as e:
            raise ValueError(f"Invalid JSON from LLM: {e}")

    for k in ["title", "description", "bullets", "body"]:
        if k not in payload:
            raise ValueError(f"Missing key: {k}")

    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(payload["description"], str):
        raise ValueError("description must be a string")
    if not isinstance(payload["bullets"], list):
        raise ValueError("bullets must be a list")
    if not isinstance(payload["body"], str) or len(payload["body"].split()) < 120:
        raise ValueError("body must be a string with >=120 words")

    payload["bullets"] = [str(b).strip()
        for b in payload["bullets"] if str(b).strip()][:5]
    return payload

@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(6))
def llm_summarize(article_text: str, source_title: str, source_url: str, prompts: dict) -> dict:
    """
    Returns dict with: title, description, bullets (list), body
    Body should include facts grounded in the article and a Sources section.
    """
    if FAKE_LLM:
        return _fake_llm(article_text, source_title, source_url)
    sys_prompt = prompts.get("system", DEFAULT_SYSTEM_PROMPT)
    user_prompt = safe_format_user_template(
        prompts.get("user_template", DEFAULT_USER_TEMPLATE),
        source_title=source_title,
        source_url=source_url,
        article_text=article_text[:12000],
    )
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.4,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    out = resp.choices[0].message.content or ""
    return _validate_llm_json(out)


# ==== FEEDS ====
def pick_sources():
    cfg = read_yaml(SOURCES_YAML)
    feeds = cfg["rss_feeds"]
    random.shuffle(feeds)
    return feeds


INTERNAL_LINKS = [
    ("What is DeFi?", "/pages/what-is-defi/"),
    ("How to store Bitcoin safely", "/pages/how-to-store-bitcoin-safely/")
]


def inject_internal_links(body: str, count: int = 2) -> str:
    import random
    picks = random.sample(INTERNAL_LINKS, k=min(count, len(INTERNAL_LINKS)))
    links_md = "\n\n".join([f"See also: [{t}]({u})" for t, u in picks])
    # add near the end, before Sources if present
    parts = body.split("\n### Sources", 1)
    if len(parts) == 2:
        return parts[0].rstrip() + "\n\n" + links_md + "\n\n### Sources" + parts[1]
    return body.rstrip() + "\n\n" + links_md + "\n"

def get_fresh_candidates(max_items=12):
    posted = load_json(POSTED_DB, default={})
    feed_urls = pick_sources()
    items = []
    ua = {"User-Agent": "Mozilla/5.0 (compatible; BWNewsBot/1.0)"}
    for fu in feed_urls:
        feed = feedparser.parse(fu)
        for e in feed.entries[:15]:
            url = e.get("link") or ""
            if not url:
                continue

            # Normalize host and filter blocked/aggregators quickly
            host = urlparse(url).netloc.lower()
            if host in BLOCKED_DOMAINS:
                continue

            # If not in allowed list, try to resolve one redirect to the original article
            if host not in ALLOWED_DOMAINS:
                try:
                    r = requests.get(url, headers=ua, timeout=8, allow_redirects=True)
                    if r.url:
                        url = r.url
                        host = urlparse(url).netloc.lower()
                except Exception:
                    pass
                # Re-check post-resolution
                if host in BLOCKED_DOMAINS or host not in ALLOWED_DOMAINS:
                    continue

            h = sha1(url)
            if h in posted:
                continue

            title = (e.get("title", "") or "").strip()
            if not title:
                continue

            # capture any feed-provided text as an emergency fallback
            summary_txt = ""
            if hasattr(e, "summary") and e.summary:
                summary_txt = re.sub(r"<[^>]+>", " ", str(e.summary))
            if hasattr(e, "content") and e.content and isinstance(e.content, list):
                try:
                    summary_txt = re.sub(r"<[^>]+>", " ", str(e.content[0].value)) or summary_txt
                except Exception:
                    pass

            published = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published = datetime.datetime(*e.published_parsed[:6], tzinfo=datetime.timezone.utc)

            items.append({
                "url": url,
                "title": title,
                "published": published,
                "summary": clean_text(summary_txt)[:15000]
            })
            if len(items) >= max_items:
                break
        if len(items) >= max_items:
            break
    return items


# ==== HUGO ====
def write_hugo_post(payload: dict):
    date = datetime.datetime.utcnow()
    y, m = date.strftime("%Y"), date.strftime("%m")
    slug = slugify(payload["title"])[:80] or slugify(urlparse(payload["source_url"]).path)[:80]

    # ensure unique slug
    post_dir = CONTENT_ROOT / y / m / slug
    base_slug = slug
    i = 2
    while (post_dir / "index.md").exists():
        slug = f"{base_slug}-{i}"
        post_dir = CONTENT_ROOT / y / m / slug
        i += 1

    post_dir.mkdir(parents=True, exist_ok=True)
    md_path = post_dir / "index.md"

    # Escape values for YAML front matter
    title_esc = (payload.get('title') or '').replace('"', "'")
    desc_esc  = (payload.get('description') or '').replace('"', "'")

    front_matter = f"""---
title: "{title_esc}"
date: {date.isoformat()}Z
draft: false
description: "{desc_esc}"
tags: ["crypto","news","blockchain"]
categories: ["News"]
source_url: "{payload['source_url']}"
canonicalURL: "{payload.get('canonical', payload['source_url'])}"
seo:
  meta_description: "{desc_esc}"
  og_type: "article"
  og_image: ""
---
"""

    bullets_section = ""
    if payload.get("bullets"):
        bullets_section = "### TL;DR\n" + "\n".join([f"- {b}" for b in payload["bullets"]]) + "\n\n"

    # Newsletter injection directly after TL;DR (or at top if TL;DR absent)
    newsletter_snippet = "{{< newsletter-inline >}}\n\n"
    if bullets_section:
        bullets_section = bullets_section + newsletter_snippet

    sources_section = f"""### Sources
- {payload['source_title']} — {payload['source_url']}
"""

    body = f"""{bullets_section if bullets_section else newsletter_snippet}{payload['body']}

{AFFILIATE_BLOCK}

{DISCLAIMER}

{sources_section}
"""

    # Disclosure present?
    assert "Not financial advice" in body, "Disclosure missing; aborting write."

    with md_path.open("w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write("\n")
        f.write(body)

    return str(md_path)


# ==== MAIN ====
def run_once():
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    candidates = get_fresh_candidates(RUN_CONFIG["max_candidates"])
    random.shuffle(candidates)
    if not candidates:
        print("No fresh candidates.")
        return 0

    posted = load_json(POSTED_DB, default={})
    made = 0
    for item in candidates:
        if made >= RUN_CONFIG["posts_per_run"]:
            break
        url = item["url"]

        # Domain whitelist
        host = urlparse(url).netloc.lower()
        if host not in ALLOWED_DOMAINS:
            print(f"[skip] domain not in whitelist: {host}")
            continue

        # Age filter
        pub = item.get("published")
        if pub:
            age_days = (datetime.datetime.now(datetime.timezone.utc) - pub).days
            if age_days > MAX_ARTICLE_AGE_DAYS:
                print(f"[skip] stale article ({age_days}d): {url}")
                continue

        # Title de-dupe
        if is_probable_duplicate_title(posted, item["title"]):
            print(f"[skip] probable duplicate title: {item['title']}")
            continue

        # Extract content (tolerant: return empty string on network errors)
        text = ""
        try:
            text = extract_main_text(url, timeout=RUN_CONFIG["timeout"])
        except Exception as e:
            print(f"[warn] fetch/extract failed {url}: {e}")
            text = ""

        if len(text.split()) < RUN_CONFIG["min_words"]:
            # try feed-provided summary as a fallback
            fallback = clean_text(item.get("summary", ""))
            if len(fallback.split()) >= max(220, RUN_CONFIG["min_words"] // 2):
                text = fallback
            else:
                print(f"[skip] too short: {url}")
                continue

        # Summarize
        try:
            llm = llm_summarize(
                article_text=text,
                source_title=item["title"],
                source_url=url,
                prompts=prompts,
            )
        except Exception as e:
            print(f"[warn] LLM failed: {e}")
            continue

        # Grounding: ensure body references the source
        body_text = llm.get("body", "")
        if url not in body_text:
            body_text = body_text.rstrip() + f"\n\n### Sources\n- {item['title']} — {url}\n"

        # Length clamps
        words = body_text.split()
        if len(words) < RUN_CONFIG["target_min_words"]:
            print(f"[warn] body too short after LLM: {len(words)} words; skipping {url}")
            continue
        if len(words) > RUN_CONFIG["target_max_words"]:
            body_text = " ".join(words[: RUN_CONFIG['target_max_words']]) + " …"
        body_text = inject_internal_links(body_text, count=2)

        payload = {
            "title": llm["title"].strip() or item["title"],
            "description": llm.get("description", "").strip()[:155],
            "bullets": llm.get("bullets", [])[:5],
            "body": body_text.strip(),
            "source_url": url,
            "source_title": item["title"],
        }

        outpath = write_hugo_post(payload)
        print(f"[ok] wrote {outpath}")
        posted[sha1(url)] = {"url": url, "title": item["title"], "ts": now_iso()}
        made += 1
        time.sleep(random.uniform(1.0, 2.0))

    if made:
        save_json(POSTED_DB, posted)
    return made


if __name__ == "__main__":
    Path(CONTENT_ROOT).mkdir(parents=True, exist_ok=True)
    made = run_once()
    print(f"Created {made} post(s).")
