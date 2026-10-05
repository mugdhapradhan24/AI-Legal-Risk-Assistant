import re
from urllib.parse import urlparse


# ============================================================
# TOPIC DEFINITIONS
# ============================================================

TOPIC_CONFIG = {
    "Photos & Videos": {
        "keywords": [
            "photos", "photo", "videos", "video", "images", "image",
            "camera", "media", "media files", "photo library",
            "video library", "visual content", "metadata",
        ],
        "preferred_url_terms": [
            "/privacy", "/legal/privacy", "/photos", "/photo",
            "/video", "/media", "/camera",
        ],
        "preferred_title_terms": [
            "photos", "photo", "videos", "video", "media", "camera",
            "privacy policy", "privacy",
        ],
        "negative_terms": [
            "government request", "law enforcement",
            "transparency report", "model training",
        ],
    },

    "Messages & Text": {
        "keywords": [
            "messages", "message", "messaging", "text messages",
            "text message", "communications", "communication", "chat",
            "chats", "sms", "email", "message content", "message contents",
            "end-to-end encryption", "end to end encryption",
            "encrypted messages", "private messages",
        ],
        "preferred_url_terms": [
            "/privacy", "/legal/privacy", "/message", "/messages",
            "/messaging", "/chat", "/security",
        ],
        "preferred_title_terms": [
            "messages", "message", "messaging", "chat",
            "private messaging", "message privately", "privacy policy",
        ],
        "negative_terms": [
            "government request", "transparency report",
            "advertising policy",
        ],
    },

    "Personal Data": {
        "keywords": [
            "personal data", "personal information", "information we collect",
            "data we collect", "account information", "profile information",
            "device information", "contact information", "payment information",
            "transaction information", "usage information", "usage data",
            "location information", "location data", "financial information",
            "identifiers", "ip address", "phone number", "email address",
        ],
        "preferred_url_terms": [
            "/privacy", "/legal/privacy", "/privacy-policy", "/data",
        ],
        "preferred_title_terms": [
            "privacy policy", "privacy", "personal data",
            "personal information", "data we collect", "information we collect",
        ],
        "negative_terms": [
            "government request", "transparency report", "law enforcement",
        ],
    },

    "Data Sharing": {
        "keywords": [
            "share personal data", "share personal information",
            "sharing personal data", "sharing personal information",
            "share information", "sharing information", "data sharing",
            "service providers", "third parties", "third-party", "partners",
            "affiliates", "affiliated companies", "related companies",
            "vendors", "processors", "business partners",
            "disclose information", "disclosure of information",
            "sell personal data", "sell personal information", "does not sell",
        ],
        "preferred_url_terms": [
            "/privacy", "/legal/privacy", "/privacy-policy",
            "/data-sharing", "/sharing",
        ],
        "preferred_title_terms": [
            "privacy policy", "data sharing", "sharing information",
            "sharing personal data", "privacy",
        ],
        "negative_terms": [
            "government request", "law enforcement", "transparency report",
        ],
    },

    "AI / Model Training": {
        "keywords": [
            "artificial intelligence", "generative ai",
            "generative artificial intelligence", "ai model", "ai models",
            "model training", "training data", "train our models",
            "train models", "training our models", "machine learning",
            "foundation model", "foundation models", "large language model",
            "large language models", "user interactions",
            "publicly available data", "licensed data", "synthetic data",
        ],
        "preferred_url_terms": [
            "/ai", "/artificial-intelligence", "/model", "/training",
            "/privacy", "/legal/privacy",
        ],
        "preferred_title_terms": [
            "artificial intelligence", "ai", "model training",
            "training data", "machine learning", "foundation models",
            "privacy policy",
        ],
        "negative_terms": [
            "government request", "law enforcement", "transparency report",
        ],
    },

    "Government / Legal Disclosure": {
        "keywords": [
            "law enforcement", "government", "government request",
            "government requests", "government authorities", "legal request",
            "legal requests", "legal process", "lawful request",
            "lawful requests", "court order", "subpoena", "required by law",
            "national security", "emergency request", "emergency requests",
            "disclose information", "disclosure to authorities",
            "transparency report",
        ],
        "preferred_url_terms": [
            "/government", "/law-enforcement", "/law_enforcement", "/legal",
            "/transparency", "/records", "/privacy",
        ],
        "preferred_title_terms": [
            "government requests", "government request", "law enforcement",
            "legal requests", "legal request", "transparency", "government",
        ],
        "negative_terms": [
            "model training", "training data",
        ],
    },

    "Storage & Security": {
        "keywords": [
            "security", "data security", "information security",
            "data storage", "store personal data", "stored personal data",
            "stored information", "protect personal data",
            "protect personal information", "protect your data",
            "encryption", "encrypted", "end-to-end encryption",
            "end to end encryption", "encryption at rest",
            "encryption in transit", "security safeguards",
            "security measures", "technical safeguards",
            "administrative safeguards", "physical safeguards",
            "access controls", "secure storage", "stored securely",
        ],
        "preferred_url_terms": [
            "/security", "/privacy", "/legal/privacy", "/encryption", "/safety",
        ],
        "preferred_title_terms": [
            "security", "privacy policy", "privacy", "encryption",
            "secure", "safety",
        ],
        "negative_terms": [
            "government request", "law enforcement", "transparency report",
            "model training",
        ],
    },

    "Retention & Deletion": {
        "keywords": [
            "retention", "data retention", "retain personal data",
            "retain personal information", "retain information", "retained",
            "retention period", "retention periods", "kept for",
            "kept as long as", "retain for", "delete personal data",
            "delete personal information", "delete your data",
            "delete your information", "data deletion", "deletion",
            "delete account", "delete your account", "account deletion",
            "request deletion", "right to deletion", "right to delete",
        ],
        "preferred_url_terms": [
            "/privacy", "/legal/privacy", "/privacy-policy", "/delete",
            "/deletion", "/retention", "/account",
        ],
        "preferred_title_terms": [
            "privacy policy", "privacy", "retention", "deletion",
            "delete account", "delete your account",
        ],
        "negative_terms": [
            "government request", "law enforcement", "transparency report",
            "model training",
        ],
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_page_fields(page):
    title = normalize_text(page.get("title", ""))
    url = normalize_text(page.get("url", ""))
    content = normalize_text(page.get("content", ""))
    return title, url, content


def get_domain(url):
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_official_source(page):
    if not page:
        return False
    if "official" in page:
        return bool(page.get("official"))
    url = page.get("url", "").strip()
    domain = get_domain(url)
    return bool(url and domain)


# ============================================================
# MARKETING / FEATURE PAGE DETECTION (generic, not company-specific)
# ============================================================

# Path fragments that indicate an actual legal/policy document.
LEGAL_PATH_INDICATORS = [
    "/legal", "/privacy", "/policy", "/policies", "/data-policy",
    "/terms", "/gdpr", "/privacyinfo", "/data/privacyinfo",
]

# Path fragments that indicate a marketing/how-to/feature page instead
# of a policy document. This is intentionally generic — most companies
# use some version of these path conventions for their marketing site,
# completely separate from their legal/privacy pages.
MARKETING_PATH_INDICATORS = [
    "/features/", "/feature/", "/product/", "/products/", "/shop/",
    "/discover/", "/explore/", "/download", "/get-started",
    "/create/", "/inspiration/", "/blog/", "/stories/", "/newsroom",
]


# Subdomain-level marketing indicators. A path-only check misses pages
# like blog.whatsapp.com/some-post, where "blog" is the SUBDOMAIN, not
# a path segment — these are never policy documents regardless of path.
MARKETING_SUBDOMAIN_INDICATORS = [
    "blog.", "news.", "newsroom.", "stories.", "press.",
]


def is_marketing_page(page):
    """
    A page can be on an official company domain and still not be a
    policy document at all — e.g. a "how our messaging feature works"
    marketing page. Such pages often repeat topic keywords (photo,
    video, message) far more densely than the actual legal text,
    which lets them win on raw keyword scoring despite being the
    wrong kind of source entirely.

    This check is path-based, not company-specific, so it applies the
    same way regardless of which app is being analyzed.
    """
    _, url, _ = get_page_fields(page)

    if any(term in url for term in LEGAL_PATH_INDICATORS):
        return False

    domain = get_domain(url)
    if any(domain.startswith(term) for term in MARKETING_SUBDOMAIN_INDICATORS):
        return True

    return any(term in url for term in MARKETING_PATH_INDICATORS)


def marketing_page_penalty(page):
    return 120 if is_marketing_page(page) else 0


# ============================================================
# KEYWORD SCORING
# ============================================================

def keyword_score(title, url, content, keywords):
    score = 0
    for keyword in keywords:
        keyword = normalize_text(keyword)
        if not keyword:
            continue
        if keyword in title:
            score += 12
        elif keyword in url:
            score += 8
        elif keyword in content:
            score += 2
    return score


# ============================================================
# TOPIC-SPECIFIC SOURCE PRIORITY
# ============================================================

def preferred_source_score(page, topic):
    config = TOPIC_CONFIG.get(topic, {})
    preferred_urls = config.get("preferred_url_terms", [])
    preferred_titles = config.get("preferred_title_terms", [])

    title, url, _ = get_page_fields(page)

    score = 0

    for term in preferred_urls:
        term = normalize_text(term)
        if term and term in url:
            score += 25

    for term in preferred_titles:
        term = normalize_text(term)
        if term and term in title:
            score += 35

    if is_official_source(page):
        score += 10

    return score


# ============================================================
# NEGATIVE RELEVANCE
# ============================================================

def negative_source_penalty(page, topic):
    config = TOPIC_CONFIG.get(topic, {})
    negative_terms = config.get("negative_terms", [])

    title, url, content = get_page_fields(page)

    penalty = 0

    for term in negative_terms:
        term = normalize_text(term)
        if not term:
            continue
        if term in title:
            penalty += 30
        elif term in url:
            penalty += 20
        elif term in content:
            penalty += 4

    return penalty


# ============================================================
# SOURCE SCOPE / QUALITY
# ============================================================

BROAD_TOPICS = {
    "Personal Data", "Data Sharing", "Storage & Security",
    "Retention & Deletion",
}


SPECIALIZED_TOPIC_TERMS = {
    "Photos & Videos": [
        "photo", "photos", "video", "videos", "camera", "cameras",
        "media", "image", "images", "visual", "data-collection",
    ],
    "Messages & Text": [
        "message", "messages", "messaging", "chat", "chats", "sms",
        "communication", "communications", "end-to-end", "encryption",
    ],
    "AI / Model Training": [
        "artificial intelligence", "generative ai", "machine learning",
        "model training", "training data", "foundation model",
        "ai-solutions", "ai solutions", "responsible ai",
    ],
    "Government / Legal Disclosure": [
        "law enforcement", "law-enforcement", "government request",
        "government requests", "legal request", "legal requests",
        "legal process", "transparency", "subpoena", "court order",
        "guidelines-for-law-enforcement",
    ],
}


BROAD_SOURCE_TERMS = [
    "privacy policy", "privacy notice", "privacy center", "privacy centre",
    "data policy", "data privacy", "/privacy-policy", "/legal/privacy",
    "/privacy/", "privacy.",
]


NARROW_SCOPE_TERMS = [
    "drivers", "driver", "delivery people", "delivery partner",
    "merchant", "merchants", "advertiser", "advertisers", "developer",
    "developers", "enterprise", "workspace", "cloud", "autonomous",
    "self-driving", "vehicle data", "ai solutions", "ai-solutions",
    "business users", "business customer",
]


ARCHIVE_TERMS = [
    "/archive/", "/archives/", "archived privacy", "privacy archive",
    "historical privacy", "previous privacy policy", "previous version",
]


def classify_source_scope(page):
    title, url, _ = get_page_fields(page)
    combined = title + " " + url

    if any(term in combined for term in NARROW_SCOPE_TERMS):
        return "narrow"

    if any(term in combined for term in BROAD_SOURCE_TERMS):
        return "broad"

    return "unknown"


def source_scope_adjustment(page, topic):
    title, url, _ = get_page_fields(page)
    combined = title + " " + url

    adjustment = 0

    if any(term in combined for term in ARCHIVE_TERMS):
        adjustment -= 100

    if topic in BROAD_TOPICS:
        scope = classify_source_scope(page)
        if scope == "broad":
            adjustment += 35
        elif scope == "narrow":
            adjustment -= 40
        return adjustment

    specialized_terms = SPECIALIZED_TOPIC_TERMS.get(topic, [])
    if any(term in combined for term in specialized_terms):
        adjustment += 60

    return adjustment


# ============================================================
# FINAL TOPIC SCORE
# ============================================================

def calculate_topic_score(page, topic):
    config = TOPIC_CONFIG.get(topic, {})
    keywords = config.get("keywords", [])

    title, url, content = get_page_fields(page)

    base_score = keyword_score(title, url, content, keywords)
    source_bonus = preferred_source_score(page, topic)
    penalty = negative_source_penalty(page, topic)
    scope_adjustment = source_scope_adjustment(page, topic)
    marketing_penalty = marketing_page_penalty(page)

    final_score = (
        base_score
        + source_bonus
        - penalty
        + scope_adjustment
        - marketing_penalty
    )

    return max(0, final_score)


# ============================================================
# SOURCE SELECTION
# ============================================================

def select_topic_sources(policy_pages, topic, max_sources=3):
    if not policy_pages:
        return []

    scored_pages = []

    for page in policy_pages:
        if not is_official_source(page):
            continue

        score = calculate_topic_score(page, topic)

        MIN_TOPIC_SCORE = 15

        if score < MIN_TOPIC_SCORE:
            continue

        scored_pages.append((score, page))

    scored_pages.sort(key=lambda item: item[0], reverse=True)

    print(f"\n  Candidate scores for {topic}:")
    for score, page in scored_pages[:10]:
        print(
            f"    {score:>4} | "
            f"{page.get('title', '')} | "
            f"{page.get('url', '')}"
        )

    selected = []
    seen_urls = set()

    for score, page in scored_pages:
        url = page.get("url", "").strip()
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        selected.append(page)

        if len(selected) >= max_sources:
            break

    return selected


# ============================================================
# EVIDENCE BUILDING
# ============================================================

def build_topic_evidence(policy_pages, topic):
    selected_pages = select_topic_sources(policy_pages, topic, max_sources=3)

    if not selected_pages:
        return []

    evidence = []

    for page in selected_pages:
        title = page.get("title", "")
        url = page.get("url", "")
        content = page.get("content", "")

        topic_score = calculate_topic_score(page, topic)
        source_scope = classify_source_scope(page)

        evidence.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "topic_score": topic_score,
                "relevance_score": topic_score,
                "official": is_official_source(page),
                "source_scope": source_scope,
            }
        )

    return evidence


# ============================================================
# MAIN FUNCTION
# ============================================================

def retrieve_evidence(policy_pages):
    topics = [
        "Photos & Videos", "Messages & Text", "Personal Data",
        "Data Sharing", "AI / Model Training",
        "Government / Legal Disclosure", "Storage & Security",
        "Retention & Deletion",
    ]

    evidence_by_topic = {topic: [] for topic in topics}

    if not policy_pages:
        return evidence_by_topic

    for topic in topics:
        print(f"Retrieving evidence for: {topic}")

        topic_evidence = build_topic_evidence(policy_pages, topic)

        if not topic_evidence:
            print("  No sufficiently relevant official source found.")
            evidence_by_topic[topic] = []
            continue

        print(f"  Found {len(topic_evidence)} strong source(s)")

        for item in topic_evidence:
            print(f"    Score: {item['topic_score']} | {item['title']}")

        evidence_by_topic[topic] = topic_evidence

    return evidence_by_topic