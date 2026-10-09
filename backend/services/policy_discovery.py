import httpx
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

MAX_POLICY_PAGES = 16

# How many link-followed sub-pages (hop 1) to allow per run. Raised from
# 6 — at 6, topics whose links appeared earlier on a page (photos,
# messages) could exhaust the budget before topics whose links appeared
# later (security, government requests) ever got a chance to be fetched.
MAX_LINK_FOLLOWS = 14

# Generic (not company-specific) words that, when they appear in a link's
# visible anchor text on an already-fetched OFFICIAL page, mark it as
# worth following. Many companies structure their privacy hub as a page
# of links out to dedicated sub-pages per topic (AI, security, retention,
# etc.) rather than one long inline document — plain-text extraction
# throws those links away, so without this, discovery depends entirely on
# live search happening to surface the exact right sub-page URL.
TOPIC_LINK_HINTS = [
    "photo", "video", "image", "camera",
    "message", "messaging", "chat",
    "personal data", "personal information", "information we collect",
    "share", "sharing", "third part", "partner",
    "artificial intelligence", "generative ai", "ai model",
    "machine learning", "training",
    "government", "law enforcement", "legal request", "transparency",
    "security", "secure", "encrypt",
    "retention", "delete", "deletion",
]

# Below this many usable characters, a "successful" HTTP fetch is
# treated as a JS-rendered shell rather than real content, and we
# escalate to a headless browser that actually executes JS.
MIN_USABLE_CONTENT_CHARS = 600

# Real single-page privacy policies (Meta's, for example) can run to
# 100K+ characters. 30,000 was silently truncating away entire topic
# sections. This is generous but still bounded.
MAX_CONTENT_CHARS = 150000


# Generic, company-independent hints that a URL is a dedicated policy
# page worth prioritizing. NOTE: this used to be a list of Apple's own
# URL paths (e.g. "/legal/privacy/data/en/photos"), which meant this
# "priority" boost only ever fired for apple.com. These are now generic
# path fragments that show up across many companies' privacy hubs.
PRIORITY_POLICY_URL_TERMS = [
    "/privacy-policy",
    "/privacy/policy",
    "/legal/privacy",
    "/privacy-notice",
    "/data-policy",
    "/privacypolicy",
    "/privacy-center",
    "/privacycenter",
    "/data/privacyinfo",
    "/app-privacy-details",
    "/government-information-requests",
    "/government-requests",
    "/law-enforcement",
    "/transparency",
    "/security",
    "/data-retention",
]


# ============================================================
# PAGE RETRIEVAL
# ============================================================

def _render_with_browser(url: str) -> str:
    """
    Returns the post-JavaScript rendered HTML via a headless browser, or
    "" on failure. This is the ONLY way to see real content (or real
    links) on JS single-page apps like Meta's privacy center — the raw
    HTML from a plain HTTP fetch is just an empty shell for these.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"]
            )
            page = browser.new_page()

            page.goto(
                url,
                wait_until="networkidle",
                timeout=30000
            )

            html = page.content()
            browser.close()

        return html

    except Exception as e:
        print(f"Browser fallback failed for {url}: {e}")
        return ""


def get_page_text_with_browser(url: str) -> str:
    """Backward-compatible wrapper returning just the rendered text."""
    return _html_to_text(_render_with_browser(url))


def _fetch_html(url: str) -> str:
    """
    Try requests, then httpx. Returns raw HTML, or "" on failure or
    when the response is a PDF (handled separately elsewhere).
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/152.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(5, 15),
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            print(f"Skipping PDF for HTML parser: {url}")
            return ""

        return response.text

    except requests.exceptions.Timeout:
        print(f"requests timed out for {url}")

    except requests.exceptions.RequestException as e:
        print(f"requests failed for {url}: {e}")

    # httpx fallback
    try:
        with httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=15.0,
        ) as client:
            response = client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                print(f"Skipping PDF for HTML parser: {url}")
                return ""

            return response.text

    except Exception as e:
        print(f"httpx also failed for {url}: {e}")

    return ""


def _html_to_text(html: str) -> str:
    """
    Reverted from "\n" back to " ". The newline approach (tried briefly)
    correctly isolated menu/nav junk from adjacent real sentences, but
    it ALSO broke apart headings from the content that followed them
    across the whole corpus — e.g. "Here's the information we collect:"
    separated from the list that used to immediately follow it, which
    silently killed eligibility for sentences that depended on that
    merged context. Too broad a change for the one narrow problem it
    was meant to solve. See _sentence_split in legal_analyzer.py for the
    narrower, scoped fix instead.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for element in soup([
        "script", "style", "noscript", "nav", "footer", "header"
    ]):
        element.decompose()

    return soup.get_text(separator=" ", strip=True)


def _extract_topic_links(html: str, base_url: str, limit: int = MAX_LINK_FOLLOWS) -> list:
    """
    Pull same-domain links off an already-fetched official page whose
    VISIBLE anchor text suggests they lead to a dedicated topic sub-page
    (e.g. "How Meta uses information for generative AI models"). Generic
    across companies — matches on topic wording, not any specific site's
    URL structure.
    """
    if not html:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []

    base_domain = _extract_root_domain(base_url)
    found = []
    seen = set()

    for a in soup.find_all("a", href=True):
        text = (a.get_text() or "").strip()
        href = a["href"].strip()

        if not text or not href or len(text) > 120:
            continue

        text_lower = text.lower()
        if not any(hint in text_lower for hint in TOPIC_LINK_HINTS):
            continue

        absolute_url = urljoin(base_url, href)

        if _extract_root_domain(absolute_url) != base_domain:
            continue

        normalized = normalize_url(absolute_url)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        found.append({"title": text, "url": absolute_url, "snippet": ""})

        if len(found) >= limit:
            break

    return found


def get_page_content(url: str) -> dict:
    """
    Download an official webpage and extract readable text, plus any
    topic-relevant outbound links found on it (see _extract_topic_links).

    Slow, unavailable, or unsupported pages are skipped rather than
    stopping the entire policy-discovery pipeline.

    Important: a fetch can "succeed" (HTTP 200) and still return almost
    no real content, because many privacy hubs (Meta's included) are
    JavaScript single-page apps whose body only renders after JS runs.
    A thin static-HTML result is therefore escalated to a real headless
    browser, not just outright request failures/timeouts.
    """

    html = _fetch_html(url)
    text = _html_to_text(html)

    if len(text.strip()) < MIN_USABLE_CONTENT_CHARS:
        print(
            f"Only {len(text.strip())} usable chars from static fetch "
            f"(likely a JS-rendered page) — trying headless browser: {url}"
        )

        browser_html = _render_with_browser(url)
        browser_text = _html_to_text(browser_html)

        if len(browser_text.strip()) > len(text.strip()):
            text = browser_text
            # IMPORTANT: also swap in the browser-rendered HTML for link
            # extraction below. The static HTML that triggered this
            # fallback is a near-empty JS-app shell with no real <a> tags
            # in it — using it for link discovery would silently miss
            # every link on exactly the pages that most need this path
            # (e.g. Meta's privacy center, which links out to a dedicated
            # "how generative AI training works" sub-page that only
            # exists in the post-JS DOM).
            html = browser_html

    if len(text.strip()) < MIN_USABLE_CONTENT_CHARS:
        print(f"Skipping page with insufficient extractable content: {url}")
        return {"text": "", "links": []}

    links = _extract_topic_links(html, url)

    return {
        "text": text[:MAX_CONTENT_CHARS],
        "links": links,
    }


def get_page_text(url: str) -> str:
    """Backward-compatible wrapper returning just the text."""
    return get_page_content(url)["text"]


# ============================================================
# OFFICIAL DOMAIN CHECK
# ============================================================

def is_official_source(url: str, app_name: str) -> bool:
    """
    Determine whether a URL plausibly belongs to the requested
    app/company without maintaining a company-specific domain list.
    """

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().strip()
        path = parsed.path.lower().strip()
    except Exception:
        return False

    if not domain:
        return False

    domain = domain.split(":")[0]

    if domain.startswith("www."):
        domain = domain[4:]

    # Reject ordinary Instagram content/profile pages hosted directly on
    # the bare instagram.com domain (e.g. instagram.com/username). This
    # does NOT affect subdomains like help.instagram.com or
    # privacycenter.instagram.com, which are the real policy hosts.
    if domain == "instagram.com":
        if not path.startswith("/help/") and not path.startswith("/legal/"):
            return False

    app_key = "".join(
        character
        for character in str(app_name or "").lower()
        if character.isalnum()
    )

    if not app_key:
        return False

    domain_labels = [
        "".join(character for character in label if character.isalnum())
        for label in domain.split(".")
    ]

    if app_key in domain_labels:
        return True

    return False


# ============================================================
# PRIORITY DETECTION
# ============================================================

def is_priority_policy(result: dict) -> bool:
    url = str(result.get("url", "")).lower()
    return any(term in url for term in PRIORITY_POLICY_URL_TERMS)


# ============================================================
# POLICY RELEVANCE SCORING
# ============================================================

def policy_relevance_score(result: dict) -> int:
    title = str(result.get("title", "")).lower()
    url = str(result.get("url", "")).lower()
    snippet = str(result.get("snippet", "")).lower()

    text = title + " " + url + " " + snippet

    keywords = {
        "privacy policy": 20,
        "privacy": 8,
        "personal data": 10,
        "data sharing": 10,
        "data": 3,
        "terms": 4,
        "legal": 4,
        "privacy controls": 7,
        "photos": 12,
        "photo": 8,
        "camera": 8,
        "messages": 12,
        "imessage": 15,
        "message": 7,
        "artificial intelligence": 10,
        "apple intelligence": 12,
        "ai": 4,
        "training": 10,
        "training data": 15,
        "security": 7,
        "retention": 8,
        "deletion": 8,
        "government": 8,
        "law enforcement": 10,
        "transparency": 8,
    }

    score = 0
    for keyword, points in keywords.items():
        if keyword in text:
            score += points

    if is_priority_policy(result):
        score += 200

    return score


# ============================================================
# URL DEDUPLICATION
# ============================================================

def normalize_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    return url.rstrip("/").lower()


# ============================================================
# GENERIC URL PROBING (no per-company hardcoding)
# ============================================================

# Common path patterns companies use for their privacy hub. These are
# guesses, not facts — every guess still has to pass is_official_source()
# and then actually return usable content via get_page_text(). Bad
# guesses just quietly produce nothing and get dropped.
COMMON_PRIVACY_PATHS = [
    "/privacy",
    "/privacy/",
    "/privacy-policy",
    "/privacy-policy/",
    "/legal/privacy",
    "/legal/privacy/",
    "/policies/privacy",
    "/privacycenter/policy",
    "/data-policy",
    "/security",
    "/safety",
    "/legal/security",
    "/law-enforcement",
    "/government-requests",
    "/records",
]


def _extract_root_domain(url: str) -> str:
    """
    e.g. https://help.instagram.com/xyz -> instagram.com
    """
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return ""

    netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]

    parts = netloc.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return netloc


def discover_root_domain(search_results, app_name: str) -> str:
    """
    Find the company's own root domain from whichever live search
    result first passes the official-source check. This lets us probe
    common privacy-page paths on the right domain without ever having
    hardcoded it.
    """
    for result in search_results or []:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", "")).strip()
        if url and is_official_source(url, app_name):
            domain = _extract_root_domain(url)
            if domain:
                return domain

    return ""


def probe_common_privacy_paths(domain: str) -> list:
    """
    Generate candidate privacy-page URLs for ANY domain by guessing
    common path patterns. Works the same way for Apple, Instagram, or
    a company nobody has ever manually added to this codebase.
    """
    if not domain:
        return []

    candidates = []
    for base in (f"https://www.{domain}", f"https://{domain}"):
        for path in COMMON_PRIVACY_PATHS:
            candidates.append(
                {
                    "title": f"{domain} privacy page (probed)",
                    "url": base + path,
                    "snippet": "",
                }
            )

    return candidates


# ============================================================
# MAIN POLICY DISCOVERY
# ============================================================

def retrieve_policies(search_results, app_name: str):
    if not search_results:
        return []

    # Generic, company-independent enrichment: guess common privacy
    # paths on the company's own domain and add them as extra
    # candidates. Any optional hand-written seeds in search_service.py
    # are just an accelerator on top of this, never a requirement.
    root_domain = discover_root_domain(search_results, app_name)
    if root_domain:
        print(f"Probing common privacy paths on: {root_domain}")
        search_results = list(search_results) + probe_common_privacy_paths(root_domain)

    official_candidates = []

    for result in search_results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", "")).strip()
        if not url:
            continue

        if not is_official_source(url, app_name):
            continue

        official_candidates.append(result)

    unique_candidates = []
    seen_candidate_urls = set()

    for result in official_candidates:
        normalized = normalize_url(result.get("url", ""))
        if not normalized or normalized in seen_candidate_urls:
            continue
        seen_candidate_urls.add(normalized)
        unique_candidates.append(result)

    priority_candidates = [r for r in unique_candidates if is_priority_policy(r)]
    general_candidates = [r for r in unique_candidates if not is_priority_policy(r)]

    priority_candidates.sort(key=policy_relevance_score, reverse=True)
    general_candidates.sort(key=policy_relevance_score, reverse=True)

    ordered_candidates = priority_candidates + general_candidates

    print(f"Official policy candidates: {len(unique_candidates)}")
    print(f"Priority policy candidates: {len(priority_candidates)}")

    policy_pages = []
    seen_retrieved_urls = set()
    discovered_links = []  # topic-relevant sub-pages found via hop-0 pages

    # ---- Hop 0: the original + probed candidates ----
    for result in ordered_candidates:
        url = str(result.get("url", "")).strip()
        normalized_url = normalize_url(url)

        if not normalized_url or normalized_url in seen_retrieved_urls:
            continue

        seen_retrieved_urls.add(normalized_url)

        label = "PRIORITY" if is_priority_policy(result) else "policy"
        print(f"Retrieving {label} page: {url}")

        content = get_page_content(url)
        page_text = content["text"]

        if not page_text:
            continue

        print(f"  -> retrieved {len(page_text)} chars")

        if content["links"]:
            print(
                f"  -> found {len(content['links'])} topic-relevant "
                f"linked sub-page(s) on this page"
            )
            discovered_links.extend(content["links"])

        policy_pages.append(
            {
                "title": result.get("title", ""),
                "url": url,
                "official": True,
                "content": page_text,
                "priority": is_priority_policy(result),
                "discovery_score": policy_relevance_score(result),
            }
        )

        if len(policy_pages) >= MAX_POLICY_PAGES:
            break

    # ---- Hop 1: follow topic-relevant links found on hop-0 pages ----
    # Bounded and deliberately shallow — this exists to catch dedicated
    # sub-pages a policy hub links out to (e.g. a separate "how generative
    # AI training works" page), not to crawl the site generally.
    if discovered_links and len(policy_pages) < MAX_POLICY_PAGES:
        followed = 0

        for link in discovered_links:
            if followed >= MAX_LINK_FOLLOWS:
                break

            if len(policy_pages) >= MAX_POLICY_PAGES:
                break

            url = str(link.get("url", "")).strip()
            normalized_url = normalize_url(url)

            if not normalized_url or normalized_url in seen_retrieved_urls:
                continue

            if not is_official_source(url, app_name):
                continue

            seen_retrieved_urls.add(normalized_url)
            followed += 1

            print(f"Following linked sub-page: {url} (\"{link.get('title', '')}\")")

            content = get_page_content(url)
            page_text = content["text"]

            if not page_text:
                continue

            print(f"  -> retrieved {len(page_text)} chars")

            policy_pages.append(
                {
                    "title": link.get("title", ""),
                    "url": url,
                    "official": True,
                    "content": page_text,
                    "priority": True,  # a linked sub-page is inherently topic-specific
                    "discovery_score": policy_relevance_score(link),
                }
            )

    print(f"Successfully retrieved {len(policy_pages)} official policy pages.")

    return policy_pages