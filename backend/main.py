"""
main.py - CatchPhish backend API

Run from inside backend/:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000 in your browser (frontend is served automatically).
"""
import os
import json
import time
import uuid
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feature_extraction import extract_lexical_host_features, extract_content_features, is_trusted_domain
import risk_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
HISTORY_FILE = os.path.join(BASE_DIR, "scan_history.json")

app = FastAPI(title="CatchPhish API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    url: str
    mode: str = "fast"  # "fast" (Tier 1 only) or "deep" (Tier 1 + Tier 2)


def _load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def _save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-50:], f, indent=2)  # keep last 50


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/history")
def get_history():
    return list(reversed(_load_history()))


@app.delete("/api/history")
def clear_history():
    _save_history([])
    return {"status": "cleared"}


@app.post("/api/scan")
def scan(req: ScanRequest):
    t0 = time.time()
    url = req.url.strip()
    if not url:
        return {"error": "URL cannot be empty"}

    lexical_feats, domain, scheme = extract_lexical_host_features(url)

    # --- Tier 0: trusted domain registry (see feature_extraction.py for rationale) ---
    if is_trusted_domain(domain):
        result = {
            "url": url, "domain": domain, "scheme": scheme, "mode_requested": req.mode,
            "tier0_trusted": True,
            "fast": {"tier": "Safe", "risk_score": 0.0, "proba_legit": 1.0,
                     "reasons": [{"feature": "TrustedDomainRegistry", "value": domain,
                                  "impact": 0, "direction": "lowers risk"}],
                     "model": "Tier-0 Trusted Registry", "latency_ms": 0.1},
            "deep": None, "final_tier": "Safe", "final_risk_score": 0.0,
            "escalated": False, "fetch_error": None,
            "total_latency_ms": round((time.time() - t0) * 1000, 1),
            "id": str(uuid.uuid4())[:8], "timestamp": datetime.utcnow().isoformat(),
        }
        history = _load_history()
        history.append({"id": result["id"], "url": url, "tier": "Safe", "risk_score": 0.0,
                         "escalated": False, "timestamp": result["timestamp"]})
        _save_history(history)
        return result

    fast_result = risk_engine.score_fast(lexical_feats)
    fast_latency_ms = round((time.time() - t0) * 1000, 1)

    result = {
        "url": url,
        "domain": domain,
        "scheme": scheme,
        "mode_requested": req.mode,
        "fast": {**fast_result, "latency_ms": fast_latency_ms},
        "deep": None,
        "final_tier": fast_result["tier"],
        "final_risk_score": fast_result["risk_score"],
        "escalated": False,
        "fetch_error": None,
    }

    should_escalate = req.mode == "deep" or fast_result["tier"] == "Suspicious"
    if should_escalate:
        t1 = time.time()
        content_feats, meta = extract_content_features(url)
        if meta.get("error"):
            raw_err = meta["error"]
            if "NameResolution" in raw_err or "getaddrinfo" in raw_err:
                friendly = "domain does not resolve (site may not exist or is unreachable)"
            elif "Timeout" in raw_err or "timed out" in raw_err:
                friendly = "page took too long to respond (timed out)"
            elif "ConnectionError" in raw_err or "Max retries" in raw_err:
                friendly = "could not connect to the server"
            elif "SSL" in raw_err:
                friendly = "SSL/certificate error while connecting"
            else:
                friendly = raw_err[:100]
            result["fetch_error"] = friendly
        else:
            all_feats = {**lexical_feats, **content_feats}
            deep_result = risk_engine.score_deep(all_feats)
            deep_latency_ms = round((time.time() - t1) * 1000, 1)
            content_summary = {
                "page_title": meta.get("title") or "(no title found)",
                "has_password_field": bool(content_feats.get("HasPasswordField")),
                "has_hidden_fields": bool(content_feats.get("HasHiddenFields")),
                "has_external_form_submit": bool(content_feats.get("HasExternalFormSubmit")),
                "num_iframes": content_feats.get("NoOfiFrame", 0),
                "num_redirects": content_feats.get("NoOfURLRedirect", 0),
                "num_external_links": content_feats.get("NoOfExternalRef", 0),
                "num_self_links": content_feats.get("NoOfSelfRef", 0),
                "num_images": content_feats.get("NoOfImage", 0),
                "num_scripts": content_feats.get("NoOfJS", 0),
                "has_favicon": bool(content_feats.get("HasFavicon")),
                "is_responsive": bool(content_feats.get("IsResponsive")),
                "domain_title_match": content_feats.get("DomainTitleMatchScore", 0),
                "mentions_bank_pay_crypto": bool(
                    content_feats.get("Bank") or content_feats.get("Pay") or content_feats.get("Crypto")
                ),
                "has_social_links": bool(content_feats.get("HasSocialNet")),
                "has_copyright_notice": bool(content_feats.get("HasCopyrightInfo")),
                "robots_txt_present": bool(content_feats.get("Robots")),
            }
            result["deep"] = {**deep_result, "latency_ms": deep_latency_ms, "page_title": meta.get("title"),
                               "content_summary": content_summary}
            result["final_tier"] = deep_result["tier"]
            result["final_risk_score"] = deep_result["risk_score"]
            result["escalated"] = True

    result["total_latency_ms"] = round((time.time() - t0) * 1000, 1)
    result["id"] = str(uuid.uuid4())[:8]
    result["timestamp"] = datetime.utcnow().isoformat()

    history = _load_history()
    history.append({
        "id": result["id"],
        "url": url,
        "tier": result["final_tier"],
        "risk_score": result["final_risk_score"],
        "escalated": result["escalated"],
        "timestamp": result["timestamp"],
    })
    _save_history(history)

    return result


# Serve frontend static files at root (after API routes so /api/* takes priority)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")