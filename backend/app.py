
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os

from backend.services.search_service import search_web
from backend.services.legal_analyzer import analyze_text
from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.policy_discovery import (
    retrieve_policies,
    is_official_source,
)
from backend.services.risk_scoring import (
    calculate_risk_score,
    badge_class_for,
)

# Load environment variables
load_dotenv("backend/.env")

app = FastAPI()
templates = Jinja2Templates(directory="backend/templates")


@app.get("/")
def home():
    return {
        "message": "Welcome to AI Legal Risk Assistant",
        "key_loaded": os.getenv("OPENAI_API_KEY") is not None
    }


@app.get("/search")
def search(query: str):
    return search_web(query)


# --------------------------------------------------
# HEALTH CHECK 1: Chromium browser
# --------------------------------------------------

@app.get("/health/browser")
def health_browser():
    """
    Check whether Playwright Chromium can launch
    and retrieve Instagram's privacy policy.

    This helps diagnose browser problems on Render.
    """
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage"],
            )

            try:
                page = browser.new_page()
                page.goto(
                    "https://privacycenter.instagram.com/policy",
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

                try:
                    page.wait_for_selector(
                        "body",
                        timeout=10000,
                    )
                except Exception:
                    pass

                body_chars = len(page.inner_text("body"))

                return {
                    "ok": True,
                    "body_chars": body_chars,
                    "page_url": page.url,
                }

            finally:
                browser.close()

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


# --------------------------------------------------
# HEALTH CHECK 2: Search discovery
# --------------------------------------------------

@app.get("/health/search")
def health_search(query: str = "instagram"):
    """
    Check search results and identify official
    policy sources discovered for the requested app.
    """
    try:
        results = search_web(query)

        official = [
            result["url"]
            for result in results
            if result.get("url")
            and is_official_source(
                result["url"],
                query,
            )
        ]

        return {
            "ok": True,
            "total_results": len(results),
            "official_results": len(official),
            "official_urls": official[:20],
        }

    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


# --------------------------------------------------
# EXISTING LEGAL ANALYSIS ROUTE
# --------------------------------------------------

@app.get("/legal-analysis")
def legal_analysis(request: Request, query: str):

    # Step 1: Find relevant policy pages
    search_results = search_web(query)

    # Step 2: Retrieve the actual policy pages
    policy_pages = retrieve_policies(
        search_results,
        query
    )

    evidence = retrieve_evidence(
        policy_pages
    )

    # Step 3: Analyze the retrieved evidence
    analysis = analyze_text(
        evidence,
        query,
    )

    # Step 4: Calculate risk score
    risk = calculate_risk_score(analysis)

    # Step 5: Attach risk badge classes
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
            "sources": len(policy_pages),
            "analysis": analysis,
            "findings": findings_with_risk,
            "risk": risk,
        }
    )
