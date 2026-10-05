from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi import Request
from fastapi.templating import Jinja2Templates
import os

from backend.services.search_service import search_web
from backend.services.legal_analyzer import analyze_text
from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.policy_discovery import retrieve_policies
from backend.services.risk_scoring import calculate_risk_score, badge_class_for

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

    # Step 4: Turn the findings into a risk score for the thermometer
    risk = calculate_risk_score(analysis)

    # Step 5: Attach a color-coded badge class to each finding, using
    # the SAME topic-aware risk logic as the thermometer (a "Yes" on
    # Storage & Security is good news and should read as green; a "Yes"
    # on AI / Model Training is a real cost and should read as red) —
    # rather than every status pill looking visually identical.
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