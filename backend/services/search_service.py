import time
from ddgs import DDGS


# ============================================================
# CONFIGURATION
# ============================================================

MAX_RESULTS_PER_QUERY = 6
RETRIES_PER_QUERY = 2
RETRY_DELAY_SECONDS = 1
MAX_LIVE_QUERIES = 9


# ============================================================
# TRUSTED OFFICIAL SEED SOURCES
# ============================================================

OFFICIAL_POLICY_SEEDS = {
    "apple": [
        {
            "title": "Apple Privacy Policy",
            "url": "https://www.apple.com/legal/privacy/en-ww/",
            "snippet": "Official Apple Privacy Policy.",
        },
        {
            "title": "Apple Intelligence Training Data",
            "url": "https://www.apple.com/legal/ai-regulations/training-data/",
            "snippet": (
                "Official Apple information about datasets "
                "used for Apple Intelligence and foundation models."
            ),
        },
        {
            "title": "Apple Government Information Requests",
            "url": (
                "https://www.apple.com/privacy/"
                "government-information-requests/"
            ),
            "snippet": (
                "Official Apple information about government "
                "and law-enforcement requests."
            ),
        },
        {
            "title": "Apple Transparency Report",
            "url": "https://www.apple.com/legal/transparency/",
            "snippet": "Official Apple transparency information.",
        },
        {
            "title": "Apple Data and Privacy",
            "url": "https://privacy.apple.com/data/privacyinfo",
            "snippet": "Official Apple Data and Privacy information.",
        },
        {
            "title": "Apple App Privacy Details",
            "url": (
                "https://developer.apple.com/"
                "app-store/app-privacy-details/"
            ),
            "snippet": "Official Apple App Privacy documentation.",
        },

        # Dedicated topic sources
        {
            "title": "Apple Photos & Privacy",
            "url": "https://www.apple.com/legal/privacy/data/en/photos/",
            "snippet": (
                "Official Apple privacy information about "
                "Photos and photo-library data."
            ),
        },
        {
            "title": "Apple Messages & Privacy",
            "url": "https://www.apple.com/legal/privacy/data/en/messages/",
            "snippet": (
                "Official Apple privacy information about "
                "Messages and iMessage."
            ),
        },
    ],
    "instagram": [
        {
            "title": "Meta Privacy Policy (Instagram)",
            "url": "https://privacycenter.instagram.com/policy",
            "snippet": "Official Meta Privacy Policy covering Instagram.",
        },
        {
            "title": "Instagram Help Center — Privacy and Safety",
            "url": "https://help.instagram.com/477434105621119",
            "snippet": "Official Instagram Help Center privacy and safety hub.",
        },
    ],
}


# ============================================================
# HELPERS
# ============================================================

def _normalize_app_name(app_name):
    return str(app_name or "").strip()


def _add_result(
    results,
    seen_urls,
    title,
    url,
    snippet=""
):
    if not url:
        return

    url = str(url).strip()

    if not url:
        return

    if url in seen_urls:
        return

    seen_urls.add(url)

    results.append(
        {
            "title": str(title or "").strip(),
            "url": url,
            "snippet": str(snippet or "").strip(),
        }
    )


def _get_seed_results(app_name):

    key = _normalize_app_name(
        app_name
    ).lower()

    return list(
        OFFICIAL_POLICY_SEEDS.get(
            key,
            []
        )
    )


# ============================================================
# LIVE SEARCH
# ============================================================

def _run_ddgs_query(query):

    for attempt in range(
        1,
        RETRIES_PER_QUERY + 1
    ):

        try:

            with DDGS() as ddgs:

                raw_results = ddgs.text(
                    query,
                    max_results=MAX_RESULTS_PER_QUERY
                )

                results = list(
                    raw_results or []
                )

                if results:
                    return results

                print(
                    f"Search returned no results for "
                    f"'{query}' "
                    f"(attempt {attempt}/"
                    f"{RETRIES_PER_QUERY})"
                )

        except Exception as exc:

            print(
                f"Search unavailable for '{query}' "
                f"(attempt {attempt}/"
                f"{RETRIES_PER_QUERY}): {exc}"
            )

        if attempt < RETRIES_PER_QUERY:
            time.sleep(
                RETRY_DELAY_SECONDS
            )

    return []


# ============================================================
# SEARCH QUERIES
# ============================================================

def _build_queries(app_name):

    app_name = _normalize_app_name(
        app_name
    )

    return [
        f"{app_name} official privacy policy",
        f"{app_name} official photos videos privacy",
        f"{app_name} official messages communications privacy",
        f"{app_name} official personal data collection privacy",
        f"{app_name} official data sharing third parties privacy",
        f"{app_name} official AI model training privacy",
        f"{app_name} official government law enforcement data requests",
        f"{app_name} official security encryption privacy",
        f"{app_name} official data retention deletion privacy",
    ][:MAX_LIVE_QUERIES]

# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def search_web(app_name):
    """
    Discover privacy/legal sources for a company.

    Trusted official URLs are used as a reliable base.
    Live web search is used only as additional discovery.
    """

    app_name = _normalize_app_name(
        app_name
    )

    if not app_name:
        return []

    results = []
    seen_urls = set()

    # --------------------------------------------------------
    # 1. Trusted official seeds
    # --------------------------------------------------------

    seed_results = _get_seed_results(
        app_name
    )

    for item in seed_results:

        _add_result(
            results,
            seen_urls,
            item.get("title", ""),
            item.get("url", ""),
            item.get("snippet", ""),
        )

    if seed_results:

        print(
            f"Loaded {len(seed_results)} trusted "
            f"official seed source(s) for '{app_name}'."
        )

    # --------------------------------------------------------
    # 2. Live search enrichment
    # --------------------------------------------------------

    for query in _build_queries(
        app_name
    ):

        search_results = _run_ddgs_query(
            query
        )

        for item in search_results:

            if not isinstance(
                item,
                dict
            ):
                continue

            _add_result(
                results,
                seen_urls,
                item.get("title", ""),
                item.get("href", ""),
                item.get("body", ""),
            )

    print(
        f"Search/discovery completed for "
        f"'{app_name}'. "
        f"Unique sources available: "
        f"{len(results)}"
    )

    return results