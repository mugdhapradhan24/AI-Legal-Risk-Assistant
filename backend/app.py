import html
import os
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from backend.services.search_service import search_web
from backend.services.legal_analyzer import analyze_text
from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.policy_discovery import retrieve_policies
from backend.services.risk_scoring import calculate_risk_score, badge_class_for
from backend.services.report_cache import load_report, save_report, list_reports

# Load environment variables
load_dotenv("backend/.env")

app = FastAPI()
templates = Jinja2Templates(directory="backend/templates")

# On a small host (e.g. Render) set ALLOW_LIVE_ANALYSIS=0. Visitors then get
# instant cached reports and the memory-hungry pipeline (page fetching,
# headless Chromium, AI calls) can never run inside a web request and crash
# the service. Locally the default (1) runs the full pipeline for any app.
#
# Render sets RENDER=true automatically, so on Render live analysis defaults
# to OFF even if you forget to set the variable. To force it on or off
# anywhere, set ALLOW_LIVE_ANALYSIS=1 or ALLOW_LIVE_ANALYSIS=0 explicitly.
_default_live = "0" if os.getenv("RENDER") else "1"
ALLOW_LIVE_ANALYSIS = os.getenv("ALLOW_LIVE_ANALYSIS", _default_live) == "1"

# Shows up in the Render "Logs" tab at startup, so you can see exactly what
# the deployed app thinks its situation is.
print(
    f"[startup] live_analysis={ALLOW_LIVE_ANALYSIS} "
    f"on_render={bool(os.getenv('RENDER'))} "
    f"cached_reports={list_reports()}"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Legal Risk Assistant",
        "key_loaded": os.getenv("OPENAI_API_KEY") is not None,
        "live_analysis_enabled": ALLOW_LIVE_ANALYSIS,
        "cached_reports": list_reports(),
    }


# Uptime monitors and link checkers often probe with HEAD. Answer instantly
# with 200 instead of 405 (and never run any analysis for a HEAD request).
@app.head("/")
@app.head("/legal-analysis")
def head_ok():
    return Response(status_code=200)


@app.get("/search")
def search(query: str):
    return search_web(query)


@app.get("/health/browser")
def health_browser():
    """Does headless Chromium work on this host?"""
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--no-sandbox"],
            )
            page = browser.new_page()
            page.goto(
                "https://privacycenter.instagram.com/policy",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            page.wait_for_timeout(4000)
            body_chars = len(page.inner_text("body"))
            browser.close()

        return {"ok": True, "body_chars": body_chars}

    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/health/search")
def health_search(query: str = "instagram"):
    """How many official URLs does search find from this host?"""
    from backend.services.policy_discovery import is_official_source

    results = search_web(query)
    official = [
        r["url"] for r in results
        if is_official_source(r.get("url", ""), query)
    ]

    return {
        "total_results": len(results),
        "official_results": len(official),
        "official_urls": official[:20],
    }


def _render_report(
    request: Request,
    query: str,
    sources: int,
    analysis: dict,
    generated_at: str = "",
):
    # Risk score and badge colors are computed fresh each time (cheap and
    # deterministic), so scoring changes apply to cached reports too.
    risk = calculate_risk_score(analysis)

    findings_with_risk = []
    for finding in analysis.get("findings", []):
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
            "analysis": analysis,
            "findings": findings_with_risk,
            "risk": risk,
            "generated_at": generated_at,
        },
    )


@app.get("/legal-analysis")
def legal_analysis(request: Request, query: str, refresh: bool = False):
    query = query.strip()

    # 1. Cached report: instant, no heavy work.
    # (refresh=true re-runs the pipeline, but only where live analysis is allowed)
    if not (refresh and ALLOW_LIVE_ANALYSIS):
        cached = load_report(query)
        if cached:
            return _render_report(
                request,
                query,
                cached.get("sources", 0),
                cached["analysis"],
                cached.get("generated_at", ""),
            )

    # 2. Live analysis disabled (e.g. on Render): friendly message, no crash.
    if not ALLOW_LIVE_ANALYSIS:
        items = "".join(
            f'<li><a href="/legal-analysis?query={quote(name)}">{html.escape(name)}</a></li>'
            for name in list_reports()
        ) or "<li>No reports have been generated yet.</li>"

        return HTMLResponse(
            f"""
            <div style="font-family:Arial,sans-serif;max-width:640px;margin:60px auto;padding:0 20px">
              <h2>No report for "{html.escape(query)}" yet</h2>
              <p>This live demo serves pre-generated reports. Available now:</p>
              <ul>{items}</ul>
            </div>
            """,
            status_code=404,
        )

    # 3. Live analysis (local use): run the full pipeline and cache the result.
    search_results = search_web(query)
    policy_pages = retrieve_policies(search_results, query)
    evidence = retrieve_evidence(policy_pages)
    analysis = analyze_text(evidence, query)

    payload = save_report(query, len(policy_pages), analysis)

    return _render_report(
        request,
        query,
        len(policy_pages),
        analysis,
        payload.get("generated_at", ""),
    )