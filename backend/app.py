
import html
import os
import re
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from backend.services.search_service import search_web
from backend.services.legal_analyzer import analyze_text
from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.policy_discovery import retrieve_policies
from backend.services.risk_scoring import (
    calculate_risk_score,
    badge_class_for,
)
from backend.services.report_cache import (
    load_report,
    save_report,
    list_reports,
)


# --------------------------------------------------
# Application setup
# --------------------------------------------------

load_dotenv("backend/.env")

app = FastAPI(title="AI Legal Risk Assistant")

templates = Jinja2Templates(directory="backend/templates")

# Render uses cached reports by default.
# Local development allows fresh analysis.
_default_live = "0" if os.getenv("RENDER") else "1"

ALLOW_LIVE_ANALYSIS = (
    os.getenv("ALLOW_LIVE_ANALYSIS", _default_live) == "1"
)

print(
    f"[startup] live_analysis={ALLOW_LIVE_ANALYSIS} "
    f"on_render={bool(os.getenv('RENDER'))} "
    f"cached_reports={list_reports()}"
)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to AI Legal Risk Assistant",
        "key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "live_analysis_enabled": ALLOW_LIVE_ANALYSIS,
        "cached_reports": list_reports(),
    }


# --------------------------------------------------
# HEAD routes
# --------------------------------------------------

@app.head("/")
@app.head("/legal-analysis")
def head_ok():
    return Response(status_code=200)


# --------------------------------------------------
# Search
# --------------------------------------------------

@app.get("/search")
def search(query: str):
    return search_web(query)


# --------------------------------------------------
# Browser health check
# --------------------------------------------------

@app.get("/health/browser")
def health_browser():
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )

            try:
                page = browser.new_page()

                page.goto(
                    "https://privacycenter.instagram.com/policy",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

                page.wait_for_timeout(4000)

                body_chars = len(page.inner_text("body"))

            finally:
                browser.close()

        return {
            "ok": True,
            "body_chars": body_chars,
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


# --------------------------------------------------
# Search health check
# --------------------------------------------------

@app.get("/health/search")
def health_search(query: str = "instagram"):
    from backend.services.policy_discovery import (
        is_official_source,
    )

    results = search_web(query)

    official = [
        r.get("url", "")
        for r in results
        if is_official_source(r.get("url", ""), query)
    ]

    return {
        "total_results": len(results),
        "official_results": len(official),
        "official_urls": official[:20],
    }


# --------------------------------------------------
# Text cleanup helpers
# --------------------------------------------------

def _clean_text(value):
    """
    Decode HTML entities in cached/model-generated text.
    Preserve the original analysis in the cache.
    """
    value = str(value or "")

    for _ in range(4):
        decoded = html.unescape(value)

        if decoded == value:
            break

        value = decoded

    return re.sub(r"\s+", " ", value).strip()


def _excerpt(value, limit=360):
    """
    Create a readable display excerpt.
    The original evidence remains unchanged in the cache.
    """
    value = _clean_text(value)

    if len(value) <= limit:
        return value

    shortened = value[:limit].rsplit(" ", 1)[0]

    return shortened.rstrip(" ,;:") + " … [excerpt]"


# --------------------------------------------------
# V1 presentation and attribution safeguards
# --------------------------------------------------

def _prepare_v1_findings(analysis: dict, app_name: str) -> dict:
    """
    Prepare readable findings for the report.

    Does not modify the saved analysis.

    Narrows unsupported product-specific claims and
    prepares display-only excerpts.
    """

    prepared = dict(analysis or {})
    cleaned_findings = []

    is_instagram = (
        str(app_name or "").strip().lower() == "instagram"
    )

    for original in prepared.get("findings", []):
        if not isinstance(original, dict):
            continue

        finding = dict(original)

        topic = _clean_text(finding.get("topic"))
        status = _clean_text(finding.get("status"))

        raw_finding = _clean_text(
            finding.get("finding")
        )

        raw_simple = _clean_text(
            finding.get("simple_explanation")
        )

        evidence_quote = _clean_text(
            finding.get("evidence_quote")
        )

        quote_lower = evidence_quote.lower()

        display_finding = _excerpt(raw_finding, 320)
        display_simple = _excerpt(raw_simple, 280)
        display_quote = _excerpt(evidence_quote, 700)

        # ------------------------------------------
        # Product attribution: Messenger
        # ------------------------------------------

        if is_instagram and topic == "Messages & Text":

            messenger_specific = (
                "messenger" in quote_lower
                and "instagram" not in quote_lower
            )

            if messenger_specific:
                status = "Not disclosed"

                display_finding = (
                    "The reviewed evidence does not establish "
                    "this messaging practice for Instagram "
                    "specifically."
                )

                display_simple = (
                    "The cited passage refers to Messenger. "
                    "It cannot be used to confirm how "
                    "Instagram chats are protected or processed."
                )

        # ------------------------------------------
        # Product attribution: Encryption
        # ------------------------------------------

        if is_instagram and topic == "Storage & Security":

            encryption_claim = (
                "end-to-end encrypt" in quote_lower
                or "end to end encrypt" in quote_lower
            )

            instagram_specific = (
                "instagram" in quote_lower
            )

            if encryption_claim and not instagram_specific:
                # Generic Meta messaging evidence does not
                # establish encryption for Instagram chats.
                # This classification is conservative for
                # the specific claim, not an audit of all
                # security measures.
                status = "Not disclosed"

                display_finding = (
                    "The reviewed passage describes encryption "
                    "across Meta messaging products but does "
                    "not establish the protection available "
                    "for Instagram chats specifically."
                )

                display_simple = (
                    "This evidence is not enough to confirm "
                    "that Instagram conversations are "
                    "end-to-end encrypted. The status of "
                    "specific Instagram chat protections "
                    "needs separate verification."
                )

        # ------------------------------------------
        # Readable Personal Data summary
        # ------------------------------------------

        if topic == "Personal Data":

            if (
                "content and messages" in quote_lower
                or "device information" in quote_lower
                or "types of content" in quote_lower
            ):
                display_finding = (
                    "Meta's policy describes collecting "
                    "information about content, interactions, "
                    "app usage and devices. The information "
                    "collected depends on the features used."
                )

                display_simple = (
                    "The app may collect information about "
                    "what you share, how you use its features "
                    "and the devices you use. The exact "
                    "information depends on your activity."
                )

        # ------------------------------------------
        # Readable AI summary
        # ------------------------------------------

        if topic == "AI / Model Training":

            if (
                "develop and improve ai at meta" in quote_lower
                and "public content" in quote_lower
            ):
                display_finding = (
                    "Meta's policy identifies developing and "
                    "improving AI as a purpose for processing "
                    "information, potentially including public "
                    "content shared on Meta products."
                )

                display_simple = (
                    "Public posts, comments or other disclosed "
                    "information may be used to develop and "
                    "improve Meta AI. This passage does not "
                    "establish that every private message "
                    "is used to train AI."
                )

        # ------------------------------------------
        # Store display-only fields
        # ------------------------------------------

        finding["status"] = status

        finding["display_finding"] = display_finding
        finding["display_simple"] = display_simple
        finding["display_quote"] = display_quote

        cleaned_findings.append(finding)

    prepared["findings"] = cleaned_findings

    return prepared


# --------------------------------------------------
# Report rendering
# --------------------------------------------------

def _render_report(
    request: Request,
    query: str,
    sources: int,
    analysis: dict,
    generated_at: str = "",
):
    # Apply display and attribution safeguards.
    prepared_analysis = _prepare_v1_findings(
        analysis,
        query,
    )

    # Score the statuses actually displayed.
    risk = calculate_risk_score(prepared_analysis)

    findings_with_risk = []

    for finding in prepared_analysis.get("findings", []):
        enriched = dict(finding)

        enriched["badge_class"] = badge_class_for(
            finding.get("topic", ""),
            finding.get("status", ""),
        )

        findings_with_risk.append(enriched)

    return templates.TemplateResponse(
        request=request,
        name="report.html",
        context={
            "request": request,
            "query": query,
            "sources": sources,
            "analysis": prepared_analysis,
            "findings": findings_with_risk,
            "risk": risk,
            "generated_at": generated_at,
        },
    )


# --------------------------------------------------
# MAIN REPORT ENDPOINT — GET
# --------------------------------------------------

@app.get("/legal-analysis")
def legal_analysis(
    request: Request,
    query: str,
    refresh: bool = False,
):
    query = query.strip()

    if not query:
        return HTMLResponse(
            "Please enter an app name.",
            status_code=400,
        )

    # ------------------------------------------
    # 1. Load cached report
    # ------------------------------------------

    if not (refresh and ALLOW_LIVE_ANALYSIS):

        cached = load_report(query)

        if cached:
            return _render_report(
                request=request,
                query=query,
                sources=cached.get("sources", 0),
                analysis=cached["analysis"],
                generated_at=cached.get(
                    "generated_at", ""
                ),
            )

    # ------------------------------------------
    # 2. Render: cached reports only
    # ------------------------------------------

    if not ALLOW_LIVE_ANALYSIS:

        available_reports = list_reports()

        items = "".join(
            (
                f'<li><a href="/legal-analysis?query={quote(name)}">'
                f'{html.escape(name)}</a></li>'
            )
            for name in available_reports
        )

        if not items:
            items = (
                "<li>No reports have been "
                "generated yet.</li>"
            )

        return HTMLResponse(
            f"""
            <div style="
                font-family:Arial,sans-serif;
                max-width:640px;
                margin:60px auto;
                padding:20px;
            ">
                <h2>
                    No report for
                    "{html.escape(query)}" yet
                </h2>

                <p>
                    This demo serves
                    pre-generated reports.
                </p>

                <ul>{items}</ul>
            </div>
            """,
            status_code=404,
        )

    # ------------------------------------------
    # 3. Fresh local analysis
    # ------------------------------------------

    search_results = search_web(query)

    policy_pages = retrieve_policies(
        search_results,
        query,
    )

    evidence = retrieve_evidence(policy_pages)

    analysis = analyze_text(
        evidence,
        query,
    )

    # Save the original analysis.
    payload = save_report(
        query,
        len(policy_pages),
        analysis,
    )

    # Render with V1 presentation safeguards.
    return _render_report(
        request=request,
        query=query,
        sources=len(policy_pages),
        analysis=analysis,
        generated_at=payload.get(
            "generated_at", ""
        ),
    )
