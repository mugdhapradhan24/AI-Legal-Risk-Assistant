import hashlib
from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode

import httpx
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# ============================================================
# CONFIGURATION
# ============================================================

MAX_POLICY_PAGES = 16

# Hop-1 (links found on already-fetched official pages).
MAX_LINK_FOLLOWS = 6

# Hard caps per report, so one request can never run away on a small host.
# MAX_TOTAL_FETCHES counts every attempt (including fast 404s).
# MAX_BROWSER_FETCHES counts headless-Chromium launches, which are the
# expensive, memory-hungry part.
MAX_TOTAL_FETCHES = 20
MAX_BROWSER_FETCHES = 4

# (connect, read) seconds for the plain HTTP fetch.
STATIC_TIMEOUT = (4, 10)

# Generic words that, when they appear in a link's visible text on an
# already-fetched OFFICIAL page, mark it as worth following.
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

# Below this many usable characters, a successful HTTP fetch is treated as
# a JS-rendered shell rather than real content.
MIN_USABLE_CONTENT_CHARS = 600

MAX_CONTENT_CHARS = 150000

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


class FetchBudget:
    """Tracks how much expensive work one report has used."""

    def __init__(self, max_attempts: int, max_browser: int):
        self.max_attempts = max_attempts
        self.max_browser = max_browser
        self.attempts = 0
        self.browser = 0

    def can_attempt(self) -> bool:
        return self.attempts < self.max_attempts

    def can_render(self) -> bool:
        return self.browser < self.max_browser


# ============================================================
# PAGE RETRIEVAL
# ============================================================

def _render_with_browser(url: str) -> str:
    """Render a JS-heavy page, always releasing Chromium afterwards."""
    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            page = browser.new_page()
            page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in ("image", "font", "media")
                else route.continue_(),
            )
            page.goto(url, wait_until="domcontentloaded", timeout=20000)

            # Wait until the app has actually rendered text, instead of a
            # fixed sleep. Gives up quietly after 8s.
            try:
                page.wait_for_function(
                    "document.body && document.body.innerText.length > 2000",
                    timeout=8000,
                )
            except Exception:
                pass

            return page.content()
    except Exception as e:
        print(f"Browser fallback failed for {url}: {e}")
        return ""
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass


def get_page_text_with_browser(url: str) -> str:
    """Backward-compatible wrapper returning just the rendered text."""
    return _html_to_text(_render_with_browser(url))


def _fetch_html(url: str):
    """
    Plain HTTP fetch. Returns (html, browser_worth_trying).

    browser_worth_trying is True only when the site blocked us (401/403/429),
    where a real browser might get through. A 404 or a dead domain is NOT
    worth a browser: there is nothing there to render.
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
            timeout=STATIC_TIMEOUT,
            allow_redirects=True,
        )

        status = response.status_code

        if status in (404, 410):
            return "", False

        if status in (401, 403, 429):
            print(f"Blocked ({status}) for {url}")
            return "", True

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type:
            print(f"Skipping PDF for HTML parser: {url}")
            return "", False

        return response.text, False

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {url}: {e}")
        return "", False

    except requests.exceptions.RequestException as e:
        print(f"requests failed for {url}: {e}")

    # Connection/timeout/SSL problem: one quick second attempt.
    try:
        with httpx.Client(
            headers=headers,
            follow_redirects=True,
            timeout=8.0,
        ) as client:
            response = client.get(url)

            if response.status_code in (404, 410):
                return "", False

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type:
                return "", False

            return response.text, False

    except Exception as e:
        print(f"httpx also failed for {url}: {e}")

    return "", False


def _html_to_text(html: str) -> str:
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
    Same-domain links whose visible text suggests a dedicated topic
    sub-page (e.g. "How Meta uses information for generative AI models").
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

        identity = fetch_identity(absolute_url)
        if not identity or identity in seen:
            continue

        seen.add(identity)
        found.append({"title": text, "url": absolute_url, "snippet": ""})

        if len(found) >= limit:
            break

    return found


def get_page_content(url: str, budget: FetchBudget = None, allow_browser: bool = True) -> dict:
    """
    Fetch one official page and return {"text": ..., "links": [...]}.

    The headless browser is used ONLY when:
      - the plain fetch returned a real HTML page that is nearly empty
        (a JavaScript app shell, e.g. Meta's privacy center), or
      - the site blocked the plain fetch (401/403/429),
    and only while the per-report browser budget allows it.
    A 404 or unreachable page never triggers a browser launch.
    """
    html, blocked = _fetch_html(url)
    text = _html_to_text(html)

    thin = len(text.strip()) < MIN_USABLE_CONTENT_CHARS
    looks_like_js_shell = thin and bool(html.strip())

    if thin and (looks_like_js_shell or blocked) and allow_browser:
        if budget is None or budget.can_render():
            if budget is not None:
                budget.browser += 1

            print(
                f"Only {len(text.strip())} usable chars from static fetch "
                f"- trying headless browser: {url}"
            )

            browser_html = _render_with_browser(url)
            browser_text = _html_to_text(browser_html)

            if len(browser_text.strip()) > len(text.strip()):
                text = browser_text
                # Link extraction must use the rendered HTML: the static
                # shell has no real <a> tags in it.
                html = browser_html
        else:
            print(f"Browser budget used up; skipping render for {url}")

    if len(text.strip()) < MIN_USABLE_CONTENT_CHARS:
        print(f"Skipping page with insufficient extractable content: {url}")
        return {"text": "", "links": []}

    return {
        "text": text[:MAX_CONTENT_CHARS],
        "links": _extract_topic_links(html, url),
    }


def get_page_text(url: str) -> str:
    """Backward-compatible wrapper returning just the text."""
    return get_page_content(url)["text"]


# ============================================================
# OFFICIAL DOMAIN CHECK
# ============================================================

def is_official_source(url: str, app_name: str) -> bool:
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

    # Reject ordinary Instagram content/profile pages on the bare domain.
    # Subdomains such as help.instagram.com and privacycenter.instagram.com
    # are the real policy hosts and are unaffected.
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

    return app_key in domain_labels


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


def fetch_identity(url: str) -> str:
    """
    Treat UI-only policy section parameters (?section_id=...) as the same
    document, but keep every other query parameter so genuinely different
    documents are not merged.
    """
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    ui_only = {"section_id", "subpage", "annotations[0]"}
    query = parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(k, v) for k, v in query if k not in ui_only]
    return normalize_url(
        urlunparse(parsed._replace(query=urlencode(filtered), fragment=""))
    )


# ============================================================
# GENERIC URL PROBING (no per-company hardcoding)
# ============================================================

# Guesses, not facts. Kept short on purpose: every guess costs a network
# round-trip, and most companies only use one or two of these.
COMMON_PRIVACY_PATHS = [
    "/privacy",
    "/privacy-policy",
    "/legal/privacy",
    "/legal/privacy-policy",
    "/policies/privacy",
    "/data-policy",
    "/security",
    "/safety",
    "/law-enforcement",
    "/government-requests",
]


def _extract_root_domain(url: str) -> str:
    """e.g. https://help.instagram.com/xyz -> instagram.com"""
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
    if not domain:
        return []

    return [
        {
            "title": f"{domain} privacy page (probed)",
            "url": f"https://www.{domain}{path}",
            "snippet": "",
            "probed": True,
        }
        for path in COMMON_PRIVACY_PATHS
    ]


# ============================================================
# MAIN POLICY DISCOVERY
# ============================================================

def retrieve_policies(search_results, app_name: str):
    if not search_results:
        return []

    root_domain = discover_root_domain(search_results, app_name)
    if root_domain:
        print(f"Probing common privacy paths on: {root_domain}")
        search_results = list(search_results) + probe_common_privacy_paths(root_domain)

    unique_candidates = []
    seen_candidate_ids = set()

    for result in search_results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", "")).strip()
        if not url or not is_official_source(url, app_name):
            continue

        identity = fetch_identity(url)
        if not identity or identity in seen_candidate_ids:
            continue

        seen_candidate_ids.add(identity)
        unique_candidates.append(result)

    # Real search results / seeds first, guessed URLs last. Within each
    # group: priority policy pages first, then by relevance score.
    ordered_candidates = sorted(
        unique_candidates,
        key=lambda r: (
            bool(r.get("probed")),
            not is_priority_policy(r),
            -policy_relevance_score(r),
        ),
    )

    print(f"Official policy candidates: {len(unique_candidates)}")
    print(
        f"Priority policy candidates: "
        f"{sum(1 for r in unique_candidates if is_priority_policy(r))}"
    )

    budget = FetchBudget(MAX_TOTAL_FETCHES, MAX_BROWSER_FETCHES)

    policy_pages = []
    seen_retrieved = set()
    seen_content_hashes = set()
    discovered_links = []

    def add_page(title, url, page_text, priority, score):
        content_hash = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
        if content_hash in seen_content_hashes:
            print(f"Skipping duplicate policy text: {url}")
            return False
        seen_content_hashes.add(content_hash)

        policy_pages.append(
            {
                "title": title,
                "url": url,
                "official": True,
                "content": page_text,
                "priority": priority,
                "discovery_score": score,
            }
        )
        return True

    # ---- Hop 0: search results, seeds, then probes ----
    for result in ordered_candidates:
        if len(policy_pages) >= MAX_POLICY_PAGES:
            break

        if not budget.can_attempt():
            print("Fetch budget reached; stopping hop 0.")
            break

        url = str(result.get("url", "")).strip()
        identity = fetch_identity(url)

        if identity in seen_retrieved:
            continue
        seen_retrieved.add(identity)

        budget.attempts += 1
        label = "PRIORITY" if is_priority_policy(result) else "policy"
        print(f"Retrieving {label} page: {url}")

        content = get_page_content(url, budget)
        page_text = content["text"]

        if not page_text:
            continue

        if not add_page(
            result.get("title", ""),
            url,
            page_text,
            is_priority_policy(result),
            policy_relevance_score(result),
        ):
            continue

        print(f"  -> retrieved {len(page_text)} chars")

        if content["links"]:
            print(f"  -> found {len(content['links'])} topic-relevant linked sub-page(s)")
            discovered_links.extend(content["links"])

    # ---- Hop 1: follow topic links found on hop-0 pages ----
    followed = 0

    for link in discovered_links:
        if followed >= MAX_LINK_FOLLOWS:
            break

        if len(policy_pages) >= MAX_POLICY_PAGES:
            break

        if not budget.can_attempt():
            print("Fetch budget reached; stopping hop 1.")
            break

        url = str(link.get("url", "")).strip()
        identity = fetch_identity(url)

        if not identity or identity in seen_retrieved:
            continue

        if not is_official_source(url, app_name):
            continue

        seen_retrieved.add(identity)
        followed += 1
        budget.attempts += 1

        print(f"Following linked sub-page: {url} ({link.get('title', '')})")

        content = get_page_content(url, budget)
        page_text = content["text"]

        if not page_text:
            continue

        if add_page(
            link.get("title", ""),
            url,
            page_text,
            True,
            policy_relevance_score(link),
        ):
            print(f"  -> retrieved {len(page_text)} chars")

    print(
        f"Successfully retrieved {len(policy_pages)} official policy pages "
        f"({budget.attempts} fetch attempts, {budget.browser} browser renders)."
    )

    return policy_pages