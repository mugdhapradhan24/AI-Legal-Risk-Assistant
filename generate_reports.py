"""
Pre-generate reports and store them in backend/cache/.

Run from the repo root, with your virtualenv active:

    python generate_reports.py                    # default list below
    python generate_reports.py instagram spotify  # just these

Then commit backend/cache/*.json and push. The live site serves them instantly.
"""

import sys
import traceback

from backend.services.search_service import search_web
from backend.services.policy_discovery import retrieve_policies
from backend.services.evidence_retrieval import retrieve_evidence
from backend.services.legal_analyzer import analyze_text
from backend.services.report_cache import save_report

DEFAULT_APPS = ["instagram", "whatsapp", "apple", "spotify"]


def generate(app_name: str) -> bool:
    print(f"\n{'=' * 60}\nGenerating report for: {app_name}\n{'=' * 60}")

    try:
        results = search_web(app_name)
        pages = retrieve_policies(results, app_name)

        if not pages:
            print(f"No policy pages retrieved for {app_name}; not caching.")
            return False

        evidence = retrieve_evidence(pages)
        analysis = analyze_text(evidence, app_name)

        save_report(app_name, len(pages), analysis)
        print(f"Saved report for {app_name} ({len(pages)} sources).")
        return True

    except Exception:
        print(f"FAILED for {app_name}:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    apps = sys.argv[1:] or DEFAULT_APPS
    ok = [a for a in apps if generate(a)]
    failed = [a for a in apps if a not in ok]

    print(f"\nDone. Saved: {ok or 'none'}. Failed: {failed or 'none'}.")