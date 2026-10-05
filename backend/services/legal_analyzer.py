import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# How many OpenAI rewrite calls to run concurrently. Up to 8 topics may
# each need one — doing them one after another means paying the full
# round-trip latency 8 times in a row. These are independent calls (no
# topic's rewrite depends on another's), so they're a clean fit for a
# thread pool.
REWRITE_WORKERS = 4


# --------------------------------------------------
# Environment / client
# --------------------------------------------------

load_dotenv("backend/.env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)


# --------------------------------------------------
# Topics
# --------------------------------------------------

TOPICS = [
    "Photos & Videos",
    "Messages & Text",
    "Personal Data",
    "Data Sharing",
    "AI / Model Training",
    "Government / Legal Disclosure",
    "Storage & Security",
    "Retention & Deletion",
]


TOPIC_QUESTIONS = {
    "Photos & Videos": (
        "What does the company explicitly do, or explicitly not do, with users' "
        "photos or videos?"
    ),
    "Messages & Text": (
        "What does the company explicitly collect, retain, access, process, "
        "encrypt, or disclose about message content or message metadata?"
    ),
    "Personal Data": (
        "What personal data does the company explicitly collect or receive?"
    ),
    "Data Sharing": (
        "What personal data does the company explicitly share or disclose "
        "outside the company, with whom, and under what conditions?"
    ),
    "AI / Model Training": (
        "What does the company explicitly say about using users' private or "
        "personal data, content, prompts, or interactions to train AI models?"
    ),
    "Government / Legal Disclosure": (
        "What does the company explicitly say about disclosing user data to "
        "government or law-enforcement authorities?"
    ),
    "Storage & Security": (
        "What storage or security protections are explicitly described for "
        "user data?"
    ),
    "Retention & Deletion": (
        "What retention periods, deletion controls, backup retention, or "
        "deletion exceptions are explicitly described?"
    ),
}


TOPIC_PHRASES = {
    "Photos & Videos": {
        "strong": [
            "photos and videos",
            "your photos",
            "your videos",
            "photo and video",
            "photos or videos",
            "store your photos",
            "share your photos",
            "use your photos",
            "sell your photos",
            "photo content",
            "video content",
        ],
        "support": [
            "photo", "video", "image", "media", "camera", "memories",
            "advertising", "share", "store", "access", "permission",
        ],
    },
    "Messages & Text": {
        "strong": [
            "message content",
            "message log",
            "message logs",
            "send and receive messages",
            "sender and recipient",
            "stored text message",
            "private message content",
            "routing information",
            "end-to-end encrypted",
            "end to end encrypted",
            "read your messages",
            "access your messages",
        ],
        "support": [
            "messages", "messaging", "chat", "sms", "text message",
            "sender", "recipient", "routing", "calls and messages",
            "communication",
        ],
    },
    "Personal Data": {
        "strong": [
            "personal information",
            "personal data",
            "information we collect",
            "we collect information",
            "we collect data",
            "we also collect",
            "you provide us with personal information",
            "information google collects",
        ],
        "support": [
            "name", "email", "phone number", "ip address", "location",
            "device", "account", "payment information", "identifiers",
        ],
    },
    "Data Sharing": {
        "strong": [
            "when google shares your information",
            "share personal information outside",
            "we provide personal information to our affiliates",
            "we provide personal information",
            "we'll share personal information",
            "we will share personal information",
            "share your personal information",
            "disclose personal information",
            "service providers",
            "affiliates and other trusted businesses",
        ],
        "support": [
            "share", "sharing", "disclose", "disclosure", "third party",
            "third parties", "affiliate", "service provider", "partner",
            "advertiser", "consent",
        ],
    },
    "AI / Model Training": {
        "strong": [
            "used for generative ai model training",
            "used for model training",
            "train generative ai models",
            "train or fine-tune",
            "training generative ai models",
            "customer data for training",
            "model training outside your domain",
            "not used for training",
            "not used to train",
        ],
        "support": [
            "training", "train", "generative ai", "foundation model",
            "model", "prompt", "customer data", "content", "permission",
        ],
    },
    "Government / Legal Disclosure": {
        "strong": [
            "disclose user information",
            "we disclose",
            "we may disclose",
            "may disclose",
            "produce information",
            "provide user information",
            "provide personal information",
            "compelled to disclose",
            "may be compelled to disclose",
            "required by law",
            "court order",
            "subpoena",
            "warrant",
        ],
        "support": [
            "government", "law", "legal", "request", "disclose",
            "warrant", "subpoena", "court order", "emergency",
        ],
    },
    "Storage & Security": {
        "strong": [
            "keeping your information secure",
            "we use encryption",
            "encryption to keep your data private",
            "restrict access to personal information",
            "unauthorized access",
            "security features",
            "physical security measures",
            "encryption at rest",
            "encrypted when stored at rest",
        ],
        "support": [
            "security", "secure", "encrypt", "encryption", "protect",
            "storage", "stored", "access", "safeguard",
        ],
    },
    "Retention & Deletion": {
        "strong": [
            "retaining your information",
            "we retain the data",
            "how google retains data",
            "retention period",
            "retention periods",
            "delete data",
            "deletion process",
            "data can remain",
            "backup storage",
            "up to 6 months",
            "up to 9 months",
            "up to 18 months",
            "around 2 months",
            "retained for longer periods",
        ],
        "support": [
            "retain", "retained", "retention", "delete", "deleted",
            "deletion", "backup", "months", "years", "anonymized",
            "anonymised",
        ],
    },
}


NARROW_HINT_TERMS = [
    "workspace",
    "google photos",
    "messenger",
    "messages",
    "vanish mode",
    "cloud",
    "driver",
    "drivers",
    "delivery",
    "merchant",
    "enterprise",
    "education",
    "gemini",
    "autonomous",
    "vehicle",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _json_from_model_text(raw_output: str) -> dict:
    if not raw_output:
        raise ValueError("The AI returned an empty response.")

    raw_output = raw_output.strip()

    if raw_output.startswith("```"):
        raw_output = (
            raw_output
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    return json.loads(raw_output)


def _safe_conditions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _sentence_case(text: str) -> str:
    """
    Capitalizes only the first letter for display purposes. Some scraped
    sentences legitimately start lowercase (a stylized pull-quote, or a
    fragment captured mid-clause) — this doesn't change the fact, just
    how it reads. The verbatim evidence_quote shown separately is never
    touched by this.
    """
    text = str(text or "")
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
        if ch not in " \t\"'":
            break
    return text


def _not_disclosed_finding(
    topic: str,
    quote: str = "",
    source: str = "",
) -> dict:
    return {
        "topic": topic,
        "status": "Not disclosed",
        "finding": (
            "The supplied official evidence does not establish a verified "
            "conclusion for this topic."
        ),
        "conditions": [],
        "simple_explanation": (
            "The official evidence reviewed does not directly answer this "
            "question."
        ),
        "evidence_quote": quote,
        "evidence_source": source,
    }


def _normalize_text(value: str) -> str:
    value = str(value or "")
    value = value.replace("\u2018", "'")
    value = value.replace("\u2019", "'")
    value = value.replace("\u201c", '"')
    value = value.replace("\u201d", '"')
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _relevance_score(item: dict) -> float:
    value = item.get(
        "relevance_score",
        item.get(
            "topic_score",
            item.get("score", 0),
        ),
    )

    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def detect_app_name(evidence_by_topic: dict) -> str:
    for evidence_items in evidence_by_topic.values():
        if not isinstance(evidence_items, list):
            continue

        for item in evidence_items:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "")).strip()

            if title:
                return title.split()[0].strip("|:-")

    return ""


def _sentence_split(text: str) -> list[str]:
    """
    Primary split: period/!/? followed by a capital/digit/quote, same as
    the original approach (merging related heading+content together is
    usually HELPFUL, not harmful — most of the pipeline's best evidence
    depends on that).

    Narrow secondary pass: a nav/menu dump glued to a real sentence (no
    punctuation between list items) produces one long merged "sentence"
    that the structural-noise filter correctly flags as junk — but a
    genuine fact can be trapped inside it. For exactly that shape
    (long AND flagged noisy), try a looser split (no capital-letter
    requirement) and keep any resulting piece that is NOT itself noisy.
    This is additive — it doesn't remove or alter anything else.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'])",
        text,
    )

    sentences = []

    for part in parts:
        part = part.strip()

        if len(part) < 20:
            continue

        sentences.append(part)

        if len(part) > 200 and _is_structural_noise(part):
            loose_parts = re.split(r"(?<=[.!?])\s+", part)

            if len(loose_parts) > 1:
                for loose in loose_parts:
                    loose = loose.strip()
                    if len(loose) >= 20 and not _is_structural_noise(loose):
                        sentences.append(loose)

    return sentences


def _phrase_score(
    text: str,
    topic: str,
) -> float:
    lowered = _normalize_text(text)
    config = TOPIC_PHRASES[topic]

    score = 0.0

    for phrase in config["strong"]:
        if phrase in lowered:
            score += 20.0

    for phrase in config["support"]:
        if phrase in lowered:
            score += 3.0

    action_terms = [
        "collect", "use", "share", "disclose", "retain", "delete",
        "encrypt", "access", "train", "store", "process", "provide",
    ]

    for term in action_terms:
        if term in lowered:
            score += 1.5

    if topic == "Photos & Videos":
        direct_media_patterns = [
            r"\b(?:collect|use|access|upload|store|process|receive|capture|record)\w*\b.{0,40}\b(?:photo|photos|image|images|video|videos|recording|recordings)\b",
            r"\b(?:photo|photos|image|images|video|videos|recording|recordings)\b.{0,40}\b(?:collect|use|access|upload|store|process|receive|capture|record)\w*\b",
        ]

        if any(
            re.search(pattern, lowered)
            for pattern in direct_media_patterns
        ):
            score += 30.0
    return score


def _infer_scope(
    app_name: str,
    topic: str,
    title: str,
    url: str,
    source_scope_hint: str,
    quote: str = "",
) -> str:
    """
    Infer whether the selected evidence describes a company-wide practice
    ("broad") or a specific product/feature/user-group carve-out
    ("narrow").

    IMPORTANT: this must check the QUOTE TEXT itself, not just the page's
    title/URL. A generic "Privacy Policy" page can still contain a
    sentence that is scoped to one specific feature (e.g. "Messenger's
    vanish mode"), and reporting that as a company-wide practice is
    misleading. Title/URL-only scope inference misses this entirely.
    """
    hint = str(source_scope_hint or "unknown").lower().strip()

    combined = _normalize_text(
        f"{title} {url} {quote}"
    )

    # A narrow-hint term appearing ANYWHERE (title, URL, or the actual
    # quoted sentence) marks this as narrow, regardless of what the
    # upstream page-level hint said.
    for term in NARROW_HINT_TERMS:
        if term in combined:
            if topic == "Government / Legal Disclosure":
                continue
            return "narrow"

    if hint == "narrow":
        return "narrow"

    if hint == "broad":
        return "broad"

    broad_signals = [
        "privacy policy",
        "/privacy?",
        "/privacy/",
        "privacy notice",
        "privacy & terms",
    ]

    if any(signal in combined for signal in broad_signals):
        return "broad"

    return "unknown"


def _practice_from_quote(
    topic: str,
    quote: str,
) -> str:
    text = _normalize_text(quote)

    if not text:
        return "unknown"

    negative_patterns = [
        r"\bdo not\b",
        r"\bdoes not\b",
        r"\bdon't\b",
        r"\bdoesn't\b",
        r"\bnot used\b",
        r"\bnot share\b",
        r"\bnot disclose\b",
        r"\bnot sell\b",
        r"\bcannot access\b",
        r"\bcan't access\b",
    ]

    positive_patterns = {
        "Personal Data": [
            r"\bwe collect\b",
            r"\bwe also collect\b",
            r"\byou provide us\b",
        ],
        "Data Sharing": [
            r"\bwe share\b",
            r"\bwe'll share\b",
            r"\bwe will share\b",
            r"\bwe provide personal information\b",
            r"\bdisclose\b",
        ],
        "Government / Legal Disclosure": [
            r"\bwe disclose\b",
            r"\bwe may disclose\b",
            r"\bmay disclose\b",
            r"\bproduce information\b",
            r"\bprovide user information\b",
            r"\bprovide personal information\b",
            r"\bcompelled to disclose\b",
            r"\brequired by law\b",
        ],
        "Messages & Text": [
            r"\bcollect\b",
            r"\bretain\b",
            r"\baccess\b",
            r"\bprocess\b",
            r"\bstore\b",
            r"\bdisclose\b",
        ],
        "Photos & Videos": [
            r"\bstore\b",
            r"\bshare\b",
            r"\buse\b",
            r"\baccess\b",
            r"\bcollect\b",
            r"\bprocess\b",
        ],
        "AI / Model Training": [
            r"\btrain\b",
            r"\btraining\b",
            r"\bused for\b",
        ],
    }

    if topic in {
        "Storage & Security",
        "Retention & Deletion",
    }:
        return "descriptive"

    if topic == "Messages & Text":
        metadata_markers = [
            "message log",
            "message logs",
            "sender and recipient",
            "sender",
            "recipient",
            "routing information",
            "date and time",
            "duration",
            "types and volumes",
            "phone number",
        ]
        content_markers = [
            "message content",
            "content of messages",
            "read your messages",
            "access your messages",
            "stored text message",
            "private message content",
        ]

        has_metadata = any(marker in text for marker in metadata_markers)
        has_content = any(marker in text for marker in content_markers)

        if has_metadata and not has_content:
            return "metadata_only"

    has_negative = any(
        re.search(pattern, text)
        for pattern in negative_patterns
    )

    has_positive = any(
        re.search(pattern, text)
        for pattern in positive_patterns.get(topic, [])
    )

    if has_negative and has_positive:
        return "mixed"

    if has_negative:
        return "does_not_occur"

    if has_positive:
        return "occurs"

    if topic == "AI / Model Training" and "without permission" in text:
        return "mixed"

    return "descriptive"


def _map_status(
    topic: str,
    scope: str,
    practice: str,
) -> str:
    if practice == "unknown":
        return "Not disclosed"

    if scope in {
        "narrow",
        "unknown",
    }:
        return "Limited"

    if topic in {
        "Storage & Security",
        "Retention & Deletion",
    }:
        return "Yes"

    if practice == "occurs":
        return "Yes"

    if practice == "metadata_only":
        return "Limited"

    if practice == "does_not_occur":
        return "No"

    if practice in {
        "mixed",
        "descriptive",
    }:
        return "Limited"

    return "Not disclosed"


# A heading-dump sentence ("WhatsApp Legal Info Information We Collect
# Information Shared by You...") is mostly Title-Case words strung
# together with no real verb — this threshold catches that shape
# generically, without naming any specific heading text.
STRUCTURAL_NOISE_CAP_RATIO = 0.4
STRUCTURAL_NOISE_MIN_WORDS = 6

# Generic "pointer" phrasing: the sentence directs the reader elsewhere
# instead of stating a fact itself. Company-independent — these phrases
# show up in scraped text from almost any site's navigation/cross-links.
_STRUCTURAL_NOISE_POINTER_PATTERNS = [
    r"\bplease see\b",
    r"\bsee\b.{0,60}\bfor more\b",
    r"\bread more\b",
    r"\bclick here\b",
    r"\bfor more information\b",
    r"\bto find out more\b",
    r"\bto learn more\b",
    r"\blearn more about\b",
    r"\bmore resources\b",
    r"^table of contents\b",
]


def _is_structural_noise(sentence: str) -> bool:
    """
    Generic filter for scraped-but-not-substantive text: page
    table-of-contents dumps, bare section headings, and "see X for
    more"-style pointers. These can contain real topic keywords and
    pass is_topic_eligible(), but they describe no actual practice —
    quoting one as a "finding" risks the rewrite model inventing a
    confident claim out of a heading or a dead-end pointer.
    """
    text = str(sentence or "").strip()

    if not text:
        return True

    lowered = text.lower()

    if any(
        re.search(pattern, lowered)
        for pattern in _STRUCTURAL_NOISE_POINTER_PATTERNS
    ):
        return True

    # Many privacy policies are written in Q&A format, using a question
    # as a SECTION HEADING ("How do we safeguard your information?")
    # with the actual answer living in the following paragraph — not in
    # the heading itself. A short sentence that is ENTIRELY a question
    # like this has no factual content of its own and should never be
    # quoted as if it were a disclosure.
    if text.rstrip().endswith("?"):
        heading_question_starts = (
            "how ", "what ", "why ", "when ", "where ", "who ", "which ",
            "do ", "does ", "is ", "are ", "can ", "will ", "should ",
        )
        word_count = len(text.split())
        if lowered.startswith(heading_question_starts) and word_count <= 15:
            return True

    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)

    if len(words) >= STRUCTURAL_NOISE_MIN_WORDS:
        # Skip the first word: sentence-initial capitalization is normal
        # and doesn't indicate a heading dump by itself.
        rest = words[1:]

        if rest:
            capitalized = sum(1 for w in rest if w[0].isupper())
            ratio = capitalized / len(rest)

            if ratio >= STRUCTURAL_NOISE_CAP_RATIO:
                return True

    return False


def is_topic_eligible(topic: str, text: str) -> bool:
    t = text.lower()

    if topic == "Photos & Videos":
        media_terms = [
        "photo", "photos", "video", "videos",
        "image", "images", "camera", "dashcam",
        "recording", "recordings"
        ]

        handling_terms = [
        "collect", "collected", "collects",
        "access", "accessed", "accesses",
        "upload", "uploaded", "uploads",
        "store", "stored", "stores",
        "process", "processed", "processes",
        "receive", "received", "receives",
        "capture", "captured", "captures",
        "record", "recorded", "records",
        ]

        has_media = any(term in t for term in media_terms)
        has_handling = any(term in t for term in handling_terms)

        return has_media and has_handling

    if topic == "Messages & Text":
        strong_message_terms = [
        "message content",
        "content of messages",
        "messages you send",
        "messages you receive",
        "send and receive messages",
        "send or receive messages",
        "in-app messages",
        "in-app messaging",
        "conversations in the app",
        "chat messages",
        "message log",
        "message logs",
        "call and message log",
        "communications information",
        "sender and recipient",
        "sender and recipient email",
        "message metadata",
        ]

        return any(term in t for term in strong_message_terms)

    if topic == "Personal Data":
        collection_terms = [
            "we collect", "collect information", "collects information",
            "information we collect", "personal information we collect",
            "data we collect", "information you provide",
            "location information", "device information",
            "account information"
        ]
        return any(term in t for term in collection_terms)

    if topic == "Data Sharing":
        sharing_terms = [
            "we share", "we disclose",
            "share personal information",
            "disclose personal information",
            "share your information",
            "disclose your information",
            "third parties", "service providers"
        ]
        return any(term in t for term in sharing_terms)

    if topic == "AI / Model Training":
        ai_terms = [
            "artificial intelligence", "generative ai",
            "machine learning", "model", "models"
        ]

        training_terms = [
            "train", "training", "improve our models",
            "model improvement"
        ]

        data_terms = [
            "personal information", "personal data",
            "user data", "customer data", "your data",
            "your information", "your content",
            "prompt", "prompts",
            "information", "data", "content"
        ]

        return (
            any(term in t for term in ai_terms)
            and any(term in t for term in training_terms)
            and any(term in t for term in data_terms)
        )

    if topic == "Government / Legal Disclosure":
        government_terms = [
            "law enforcement", "government", "government authorities",
            "legal request", "subpoena", "court order",
            "search warrant", "national security"
        ]

        disclosure_terms = [
            "disclose", "disclosure", "provide user information",
            "provide information", "produce information",
            "compel", "compelled"
        ]

        return (
            any(term in t for term in government_terms)
            and any(term in t for term in disclosure_terms)
        )

    if topic == "Storage & Security":
        # NOTE: this used to require exact multi-word phrases like
        # "we use encryption" or "protect your data". Real policy text
        # often expresses the same idea differently (e.g. "protect your
        # account using ... advanced technology such as encryption"),
        # which never matched any of those exact phrases and silently
        # produced zero eligible sentences. Loosened to short terms,
        # same pattern as the other topics; _phrase_score() below still
        # ranks quality among whatever passes this gate.
        # NOTE: bare "security" / "secure" were dropped — they match
        # generic marketing boilerplate like "safety, security and
        # integrity" that's actually about something else entirely
        # (e.g. inter-company data sharing), not an actual security
        # practice. What's left requires genuinely technical language.
        security_action_terms = [
            "encrypt", "encrypted", "encryption",
            "safeguard", "safeguards", "authentication",
            "two-factor", "2-step", "2 step",
            "access control", "restrict access", "restricted access",
            "unauthorized access", "firewall", "data breach",
            "security measures", "security safeguards",
            "security features", "security practices",
        ]

        return any(term in t for term in security_action_terms)

    if topic == "Retention & Deletion":
        retention_practice_terms = [
        "we retain",
        "we keep",
        "may retain",
        "retained for",
        "retain your",
        "retain personal",
        "retention period",
        "how long we retain",
        "how long we keep",
        "after account deletion",
        "following account deletion",
        "when you delete your account",
        "when you delete your information",
        "we delete",
        "how long we keep your information",
        "how long we retain your information",
        "delete your account",
        "delete your information",
        "deletion of your account",
        "retention periods",
        "deletion process",
        "deleted from our servers",
        "removed from our servers",
        "retained only",
        "retain certain information",
        ]

        return any(term in t for term in retention_practice_terms)


# --------------------------------------------------
# Deterministic evidence selection
# --------------------------------------------------

def select_best_evidence(
    topic: str,
    evidence_items: list[dict],
    app_name: str,
) -> dict | None:
    ranked = []

    # Checked once per call, not once per sentence.
    STRICT_TOPICS = {
        "AI / Model Training",
        "Storage & Security",
        "Retention & Deletion",
    }

    current_app = str(app_name or "").lower().strip()

    known_products = {
        "instagram",
        "facebook",
        "messenger",
        "whatsapp",
    }

    other_products = known_products - {current_app}

    for item in evidence_items:
        if not isinstance(item, dict):
            continue

        content = str(
            item.get("content", "")
        ).strip()

        url = str(
            item.get("url", "")
        ).strip()

        if not content or not url:
            continue

        title = str(
            item.get("title", "")
        ).strip()

        source_scope_hint = str(
            item.get(
                "source_scope",
                "unknown",
            )
        ).lower().strip()

        sentences = _sentence_split(content)

        if not sentences:
            continue

        sentence_scores = []

        # IMPORTANT: every check below runs INSIDE this loop, once per
        # sentence. A prior version of this file had these checks
        # accidentally de-indented to run only once, after the loop
        # finished, using whatever sentence/index was left over from the
        # LAST sentence in the list — which meant every topic that isn't
        # in STRICT_TOPICS silently stopped comparing candidates at all
        # and just grabbed the page's trailing paragraph every time.
        for index, sentence in enumerate(sentences):

            if _is_structural_noise(sentence):
                continue

            # --------------------------------------------------
            # Reject evidence explicitly scoped to another product
            # --------------------------------------------------

            sentence_lower = sentence.lower()

            explicitly_other_product = any(
                product in sentence_lower
                for product in other_products
            )

            mentions_current_product = (
                current_app
                and current_app in sentence_lower
            )

            if (
                explicitly_other_product
                and not mentions_current_product
            ):
                continue

            # --------------------------------------------------
            # Strict eligibility for sensitive topics
            # --------------------------------------------------

            if topic in STRICT_TOPICS:
                if not is_topic_eligible(
                    topic,
                    sentence,
                ):
                    continue

            # --------------------------------------------------
            # Score every surviving sentence
            # --------------------------------------------------

            score = _phrase_score(
                sentence,
                topic,
            )

            if score > 0:
                sentence_scores.append(
                    (
                        score,
                        index,
                    )
                )

        if not sentence_scores:
            continue

        sentence_scores.sort(
            key=lambda value: value[0],
            reverse=True,
        )

        best_sentence_score, best_index = sentence_scores[0]

        quote_parts = [sentences[best_index]]

        if topic == "Data Sharing":
            for offset in range(1, 5):
                index = best_index + offset
                if index >= len(sentences):
                    break

                next_sentence = sentences[index]
                next_score = _phrase_score(next_sentence, topic)

                if next_score >= 3:
                    quote_parts.append(next_sentence)
                elif offset > 1:
                    break

        elif topic == "Government / Legal Disclosure":
            for offset in range(1, 4):
                index = best_index + offset
                if index >= len(sentences):
                    break

                next_sentence = sentences[index]
                if _phrase_score(next_sentence, topic) >= 3:
                    quote_parts.append(next_sentence)

        else:
            if best_index + 1 < len(sentences):
                next_sentence = sentences[best_index + 1]
                if _phrase_score(next_sentence, topic) >= 3:
                    quote_parts.append(next_sentence)

        quote = " ".join(quote_parts).strip()

        source_score = _relevance_score(
            item
        )

        total_score = (
            best_sentence_score * 10
            + min(
                source_score,
                250,
            )
        )

        if topic == "Government / Legal Disclosure":
            normalized_quote = _normalize_text(quote)
            disclosure_markers = [
                "we disclose",
                "we may disclose",
                "may disclose",
                "produce information",
                "provide user information",
                "provide personal information",
                "compelled to disclose",
                "required by law",
            ]
            if not any(marker in normalized_quote for marker in disclosure_markers):
                total_score -= 1000

        scope = _infer_scope(
            app_name,
            topic,
            title,
            url,
            source_scope_hint,
            quote,
        )

        ranked.append(
            {
                "score": total_score,
                "title": title,
                "url": url,
                "quote": quote,
                "scope": scope,
                "source_scope_hint": source_scope_hint,
            }
        )

    if not ranked:
        return None

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[0]


# --------------------------------------------------
# LLM rewriting only
# --------------------------------------------------

def _sanitize_data_sharing_rewrite(text: str, quote: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text

    quote_lower = _normalize_text(quote)
    quote_has_exclusivity = any(
        term in quote_lower
        for term in [
            "only if",
            "only when",
            "solely",
            "exclusively",
        ]
    )

    if quote_has_exclusivity:
        return text

    text = re.sub(r"\bonly\s+if\b", "when", text, flags=re.IGNORECASE)
    text = re.sub(r"\bonly\s+when\b", "when", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsolely\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bexclusively\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def summarize_fixed_evidence(
    topic: str,
    app_name: str,
    quote: str,
    scope: str,
) -> dict:
    prompt = f"""
You are rewriting ONE verified privacy-policy quote for a consumer.

APP:
{app_name}

TOPIC:
{topic}

QUESTION:
{TOPIC_QUESTIONS[topic]}

SCOPE:
{scope}

VERIFIED QUOTE:
{quote}

Rules:

1. Use ONLY the verified quote.
2. Do not add any fact that is not directly stated or necessarily implied.
3. Preserve all limits and qualifications.
4. Preserve product/service/user-group scope.
5. Do not change message metadata into message content.
6. Do not turn encryption into "the company cannot access the data".
7. Do not turn a conditional restriction into an absolute restriction.
8. conditions may contain ONLY qualifications explicitly visible in the quote.
9. For Data Sharing, never turn one listed sharing condition into an exclusive rule such as "only shares when..." unless the quote explicitly says it is the only condition.
10. For Messages & Text, if the quote identifies logs/metadata but not message content, explicitly preserve that distinction.
11. Keep the finding concise and factual.
12. Do not choose Yes / No / Limited / Not disclosed. Python does that.

Return ONLY JSON:

{{
  "finding": "strictly quote-supported factual finding",
  "conditions": [],
  "simple_explanation": "plain-English version of exactly the same fact"
}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite verified privacy evidence. "
                        "Never add facts beyond the quote. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

        result = _json_from_model_text(
            response
            .choices[0]
            .message.content
        )

        finding = str(
            result.get(
                "finding",
                "",
            )
        ).strip()

        conditions = _safe_conditions(
            result.get(
                "conditions",
                [],
            )
        )

        simple = str(
            result.get(
                "simple_explanation",
                "",
            )
        ).strip()

        if not finding:
            raise ValueError(
                "Empty finding from rewrite model."
            )

        if topic == "Data Sharing":
            finding = _sanitize_data_sharing_rewrite(
                finding,
                quote,
            )
            simple = _sanitize_data_sharing_rewrite(
                simple,
                quote,
            )

        return {
            "finding": _sentence_case(finding),
            "conditions": conditions,
            "simple_explanation": _sentence_case(
                simple
                or finding
            ),
        }

    except Exception as exc:
        print(
            f"Evidence rewrite failed for "
            f"{topic}: {exc}"
        )

        return {
            "finding": _sentence_case(quote),
            "conditions": [],
            "simple_explanation": _sentence_case(quote),
        }


# --------------------------------------------------
# Main analyzer
# --------------------------------------------------

def analyze_text(
    evidence_by_topic: dict,
    app_name: str | None = None,
) -> dict:
    if not isinstance(
        evidence_by_topic,
        dict,
    ):
        raise TypeError(
            "analyze_text() expected "
            "evidence_by_topic to be a dictionary."
        )

    if not app_name:
        app_name = detect_app_name(
            evidence_by_topic
        )

    # ---- Pass 1: deterministic selection + status for every topic ----
    # This is all CPU-only (sentence scoring, regex matching) — fast
    # regardless of order, so it stays a simple sequential loop.
    pending_findings = {}  # topic -> finding dict (final, or pre-rewrite)
    rewrite_jobs = {}  # topic -> (status, quote, scope, selected) needing an LLM call

    for topic in TOPICS:
        raw_items = evidence_by_topic.get(topic, [])

        if not isinstance(raw_items, list):
            raw_items = []

        if app_name.lower() == "instagram":
            print(f"\n[INSTAGRAM EVIDENCE] {topic}")
            print(f"Raw items: {len(raw_items)}")

        selected = select_best_evidence(topic, raw_items, app_name)

        if not selected:
            pending_findings[topic] = _not_disclosed_finding(topic)
            continue

        quote = selected["quote"]
        scope = selected["scope"]
        practice = _practice_from_quote(topic, quote)
        status = _map_status(topic, scope, practice)

        if app_name.lower() == "instagram":
            print("QUOTE:", quote)
            print("SCOPE:", scope)
            print("PRACTICE:", practice)
            print("STATUS:", status)

        if status == "Not disclosed":
            pending_findings[topic] = _not_disclosed_finding(
                topic, quote, selected["url"]
            )
            continue

        # Needs an LLM rewrite — defer to the concurrent batch below
        # instead of calling it here and blocking the loop.
        rewrite_jobs[topic] = (status, quote, scope, selected)

    # ---- Pass 2: run all needed OpenAI rewrite calls CONCURRENTLY ----
    # Previously each of up to 8 topics made this call one after another,
    # paying full network round-trip latency 8 times in sequence. These
    # calls are fully independent of each other.
    if rewrite_jobs:
        with ThreadPoolExecutor(max_workers=REWRITE_WORKERS) as executor:
            future_to_topic = {
                executor.submit(
                    summarize_fixed_evidence, topic, app_name, quote, scope
                ): topic
                for topic, (status, quote, scope, selected) in rewrite_jobs.items()
            }

            for future in future_to_topic:
                topic = future_to_topic[future]
                status, quote, scope, selected = rewrite_jobs[topic]

                try:
                    rewritten = future.result()
                except Exception as exc:
                    print(f"Rewrite batch failed for {topic}: {exc}")
                    rewritten = {
                        "finding": _sentence_case(quote),
                        "conditions": [],
                        "simple_explanation": _sentence_case(quote),
                    }

                pending_findings[topic] = {
                    "topic": topic,
                    "status": status,
                    "finding": rewritten["finding"],
                    "conditions": rewritten["conditions"],
                    "simple_explanation": rewritten["simple_explanation"],
                    "evidence_quote": quote,
                    "evidence_source": selected["url"],
                }

    # Assemble in the original topic order regardless of completion order.
    findings = [pending_findings[topic] for topic in TOPICS]

    return {
        "app_name": app_name,
        "findings": findings,
    }