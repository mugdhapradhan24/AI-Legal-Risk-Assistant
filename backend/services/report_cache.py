"""
Tiny file-based cache for finished analyses.

Why: building a report means fetching ~16 pages, sometimes rendering them in
headless Chromium, then making AI calls. That takes minutes and a lot of
memory, which is fragile on a small host. Generate reports once (locally, where
it works), commit the JSON files, and let the live site serve them instantly.

Only the raw analysis is stored. The risk score and badge colors are
recomputed on every request, so changes to risk_scoring.py apply to old
reports without regenerating anything.
"""

import json
import re
import time
from pathlib import Path

# backend/services/report_cache.py -> backend/cache
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _slug(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(query or "").strip().lower()).strip("-")


def load_report(query: str):
    slug = _slug(query)
    if not slug:
        return None

    path = CACHE_DIR / f"{slug}.json"
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read cached report {path.name}: {exc}")
        return None


def save_report(query: str, sources: int, analysis: dict) -> dict:
    slug = _slug(query)
    if not slug:
        raise ValueError("Cannot cache a report with an empty query.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "query": query,
        "sources": sources,
        "generated_at": time.strftime("%Y-%m-%d"),
        "analysis": analysis,
    }

    (CACHE_DIR / f"{slug}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return payload


def list_reports() -> list:
    if not CACHE_DIR.exists():
        return []

    return sorted(path.stem for path in CACHE_DIR.glob("*.json"))