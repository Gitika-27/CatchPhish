"""
risk_engine.py
Loads the trained Tier-1 / Tier-2 models and turns raw features into a
verdict: risk tier, numeric score, and a plain-language explanation.

Design choice (documented for the mentor Q&A):
We use the XGBoost model specifically for live scoring+explanation, not the
Stacking Ensemble, even though Stacking scored marginally higher in
benchmarking (see CatchPhish_Progress_Report). Reason: SHAP's fast
TreeExplainer works natively on XGBoost but not cleanly on a heterogeneous
stacked ensemble, and for a live demo, sub-100ms explainable output matters
more than a ~0.03% accuracy gain. The full benchmark comparison (incl.
Stacking) remains the evaluation of record in the report.
"""
import os
import joblib
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_tier1 = joblib.load(os.path.join(MODELS_DIR, "tier1_models.pkl"))
_tier2 = joblib.load(os.path.join(MODELS_DIR, "tier2_models.pkl"))
_feature_lists = joblib.load(os.path.join(MODELS_DIR, "feature_lists.pkl"))

LEXICAL_HOST_FEATURES = _feature_lists["lexical_host"]
CONTENT_FEATURES = _feature_lists["content"]

_xgb1 = _tier1["XGBoost"]
_xgb2 = _tier2["XGBoost"]
_explainer1 = shap.TreeExplainer(_xgb1)
_explainer2 = shap.TreeExplainer(_xgb2)


def risk_tier(proba_legit: float, low: float = 0.35, high: float = 0.65):
    risk_score = round((1 - proba_legit) * 100, 1)
    if proba_legit >= high:
        tier = "Safe"
    elif proba_legit <= low:
        tier = "Dangerous"
    else:
        tier = "Suspicious"
    return tier, risk_score


def _top_reasons(shap_row, feature_names, feature_values, top_n=5):
    """Returns top contributing features, each with direction (raises/lowers risk)."""
    pairs = list(zip(feature_names, shap_row, feature_values))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for name, sv, val in pairs[:top_n]:
        # positive SHAP value here pushes toward class 1 (legitimate) for XGBoost's
        # default margin on a binary target where 1=legit -> negative SHAP = raises phishing risk
        direction = "lowers risk" if sv > 0 else "raises risk"
        reasons.append({
            "feature": name,
            "value": round(float(val), 4) if isinstance(val, (int, float, np.floating)) else val,
            "impact": round(float(sv), 4),
            "direction": direction,
        })
    return reasons


def score_fast(feature_dict: dict):
    """Tier 1: lexical + host only. Returns verdict dict."""
    x = np.array([[feature_dict[f] for f in LEXICAL_HOST_FEATURES]])
    proba_legit = float(_xgb1.predict_proba(x)[0][1])
    tier, score = risk_tier(proba_legit)
    shap_vals = _explainer1.shap_values(x)[0]
    reasons = _top_reasons(shap_vals, LEXICAL_HOST_FEATURES, x[0])
    return {
        "tier": tier,
        "risk_score": score,
        "proba_legit": round(proba_legit, 4),
        "reasons": reasons,
        "model": "XGBoost (Tier-1 Fast)",
    }


def score_deep(feature_dict: dict):
    """Tier 2: lexical + host + content. Returns verdict dict."""
    all_features = LEXICAL_HOST_FEATURES + CONTENT_FEATURES
    x = np.array([[feature_dict[f] for f in all_features]])
    proba_legit = float(_xgb2.predict_proba(x)[0][1])
    tier, score = risk_tier(proba_legit)
    shap_vals = _explainer2.shap_values(x)[0]
    reasons = _top_reasons(shap_vals, all_features, x[0], top_n=8)
    return {
        "tier": tier,
        "risk_score": score,
        "proba_legit": round(proba_legit, 4),
        "reasons": reasons,
        "model": "XGBoost (Tier-2 Deep)",
    }
