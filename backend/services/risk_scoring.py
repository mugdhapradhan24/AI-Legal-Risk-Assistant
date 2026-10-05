"""
Converts an `analyze_text()` result into a single risk score + level for
the risk thermometer, plus a per-topic breakdown for drill-down display.

IMPORTANT DESIGN NOTE: "Yes" does not mean the same thing for every
topic. "Yes" on Storage & Security (strong encryption disclosed) is
GOOD news. "Yes" on AI / Model Training (your content trains their
models) is a real privacy cost. Scoring every topic the same way would
be misleading, so topics are split into two groups with OPPOSITE
status->risk mappings, then combined with per-topic weights.
"""

# --------------------------------------------------
# Topic groups
# --------------------------------------------------

# More disclosure of the practice = MORE risk to the user.
EXPOSURE_TOPICS = {
    "Photos & Videos",
    "Messages & Text",
    "Personal Data",
    "Data Sharing",
    "AI / Model Training",
    "Government / Legal Disclosure",
}

# More disclosure of the practice = LESS risk to the user (these are
# protective measures — finding out the company encrypts your data or
# has clear deletion controls is reassuring, not alarming).
PROTECTIVE_TOPICS = {
    "Storage & Security",
    "Retention & Deletion",
}


# --------------------------------------------------
# Status -> risk points (0 = no risk, 3 = high risk)
# --------------------------------------------------

EXPOSURE_STATUS_RISK = {
    "Yes": 3,
    "Limited": 2,
    "Not disclosed": 2,  # opacity is a risk signal, but not as bad as a confirmed "Yes"
    "No": 0,
}

PROTECTIVE_STATUS_RISK = {
    "Yes": 0,
    "Limited": 1,
    "Not disclosed": 2,  # no visibility into protections is itself a risk
    "No": 3,
}


# --------------------------------------------------
# Topic weights — how much each topic counts toward the overall score.
# Government/Legal Disclosure is weighted low: being legally compelled
# to respond to a valid warrant is normal, expected behavior for any
# legitimate company, not really a mark against it specifically.
# --------------------------------------------------

TOPIC_WEIGHTS = {
    "Photos & Videos": 1.0,
    "Messages & Text": 1.3,
    "Personal Data": 1.1,
    "Data Sharing": 1.2,
    "AI / Model Training": 1.3,
    "Government / Legal Disclosure": 0.6,
    "Storage & Security": 1.1,
    "Retention & Deletion": 0.9,
}

MAX_RISK_POINTS = 3.0


def _topic_risk_points(topic: str, status: str) -> float:
    status = str(status or "").strip()

    if topic in PROTECTIVE_TOPICS:
        return PROTECTIVE_STATUS_RISK.get(status, 2)

    # Default to EXPOSURE_TOPICS behavior for anything unrecognized —
    # fails toward flagging more risk, not less, if a new topic is ever
    # added without being explicitly categorized above.
    return EXPOSURE_STATUS_RISK.get(status, 2)


def _risk_level(score_0_100: float) -> str:
    if score_0_100 < 34:
        return "Low"
    if score_0_100 < 67:
        return "Medium"
    return "High"


def badge_class_for(topic: str, status: str) -> str:
    """
    Maps a (topic, status) pair to a CSS class name for the status pill
    next to each topic in the report, using the SAME topic-aware logic
    as the thermometer score — so a "Yes" on Storage & Security reads
    as reassuring (green) while a "Yes" on AI / Model Training reads as
    a real cost to the user (red), instead of both looking identical.
    """
    status = str(status or "").strip()
    points = _topic_risk_points(topic, status)

    if status == "Not disclosed":
        return "badge-grey"

    if points <= 0.5:
        return "badge-green"
    if points <= 1.5:
        return "badge-amber"
    return "badge-red"


def calculate_risk_score(analysis: dict) -> dict:
    """
    Input: the dict returned by legal_analyzer.analyze_text() —
    {"app_name": ..., "findings": [{"topic": ..., "status": ..., ...}, ...]}

    Output:
    {
        "app_name": str,
        "score": float (0-100, higher = riskier),
        "level": "Low" | "Medium" | "High",
        "topics": [
            {
                "topic": str,
                "status": str,
                "risk_points": float (0-3),
                "weight": float,
                "group": "exposure" | "protective",
            },
            ...
        ],
    }
    """
    findings = analysis.get("findings", []) if isinstance(analysis, dict) else []

    topic_breakdown = []
    weighted_sum = 0.0
    weight_total = 0.0

    for finding in findings:
        if not isinstance(finding, dict):
            continue

        topic = str(finding.get("topic", "")).strip()
        status = str(finding.get("status", "")).strip()

        if not topic:
            continue

        weight = TOPIC_WEIGHTS.get(topic, 1.0)
        points = _topic_risk_points(topic, status)

        weighted_sum += points * weight
        weight_total += weight

        topic_breakdown.append(
            {
                "topic": topic,
                "status": status,
                "risk_points": points,
                "weight": weight,
                "group": "protective" if topic in PROTECTIVE_TOPICS else "exposure",
            }
        )

    if weight_total == 0:
        score = 0.0
    else:
        # Normalize to 0-100: (weighted average of 0-3 points) / 3 * 100
        score = (weighted_sum / weight_total) / MAX_RISK_POINTS * 100

    score = round(score, 1)

    return {
        "app_name": analysis.get("app_name", "") if isinstance(analysis, dict) else "",
        "score": score,
        "level": _risk_level(score),
        "topics": topic_breakdown,
    }