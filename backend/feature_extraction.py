"""
feature_extraction.py
Converts a raw URL (and optionally its fetched HTML) into the same feature
columns CatchPhish's models were trained on (PhiUSIIL feature schema).

REVISION NOTE (train/serve skew fix):
An earlier version approximated TLDLegitimateProb and URLCharProb with
hand-guessed heuristics. During backend integration testing, real legitimate
sites (e.g. wikipedia.org) were misclassified as "Dangerous" -- root cause
was that these heuristics produced values on a completely different numeric
scale than what the model saw during training (e.g. URLCharProb ~0.9 live
vs ~0.05 in training data). Fix:
  - TLDLegitimateProb: now an EXACT lookup table built from the training
    corpus itself (models/tld_lookup.json), not a guess -- same scale,
    fully reproducible.
  - URLCharProb, CharContinuationRate: DROPPED. Both were computed offline
    by the dataset authors using proprietary corpus statistics that can't be
    faithfully reproduced on a fresh URL at inference time. Models were
    retrained without them (see code/04_retrain_v2_fixed.py) to eliminate
    the skew entirely rather than patch around it.
"""
import re
import os
import json
import ipaddress
from urllib.parse import urlparse

# --- Tier 0: trusted domain registry ---------------------------------------
# WHY THIS EXISTS (be ready to explain this to your mentor):
# PhiUSIIL's legitimate-URL examples have a DomainLength distribution starting
# at 8 chars, with only the bottom 1% at <=10 chars. Extremely short, globally
# famous root domains (github.com=10, google.com=10, x.com=5) are essentially
# out-of-distribution for the trained model -- it has seen too few examples
# like them to be confident. Rather than mask this with more feature hacks,
# we handle it the way real production systems do (e.g. Google Safe Browsing,
# browser allowlists): a fast trusted-registry check runs BEFORE the ML tiers.
# This is a legitimate hybrid rule+ML architecture, not a workaround -- and
# it's a good talking point on why real systems rarely rely on ML alone.
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "github.com", "gitlab.com", "microsoft.com",
    "apple.com", "amazon.com", "wikipedia.org", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
    "stackoverflow.com", "pypi.org", "npmjs.com", "python.org", "mozilla.org",
    "wordpress.com", "adobe.com", "salesforce.com", "oracle.com", "ibm.com",
    "cloudflare.com", "openai.com", "anthropic.com", "nytimes.com", "bbc.com",
    "cnn.com", "who.int", "un.org", "gov.in", "nic.in", "cit.edu.in",
}
# -----------------------------------------------------------------------------

_LOOKUP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "tld_lookup.json")
with open(_LOOKUP_PATH) as f:
    _tld_data = json.load(f)
TLD_LOOKUP = _tld_data["lookup"]
DEFAULT_TLD_PROB = _tld_data["default"]

LEXICAL_HOST_FEATURES = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength',
    'TLDLegitimateProb', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
    'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
    'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
]

CONTENT_FEATURES = [
    'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore',
    'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive',
    'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup',
    'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton',
    'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
    'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef',
    'NoOfEmptyRef', 'NoOfExternalRef'
]


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def is_trusted_domain(domain: str) -> bool:
    """Tier-0 check: is this domain (or its registrable parent) in the trusted registry?"""
    parts = domain.lower().split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in TRUSTED_DOMAINS:
            return True
    return False


def extract_lexical_host_features(url: str) -> dict:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.split("@")[-1]  # strip userinfo if present
    domain = netloc.split(":")[0]  # strip port
    full_url = url

    labels = domain.split(".") if domain else []
    tld = labels[-1].lower() if len(labels) >= 2 else ""
    n_subdomains = max(len(labels) - 2, 0)

    letters = sum(c.isalpha() for c in full_url)
    digits = sum(c.isdigit() for c in full_url)
    n_equals = full_url.count("=")
    n_qmark = full_url.count("?")
    n_amp = full_url.count("&")
    n_percent = full_url.count("%")
    other_special = sum(
        1 for c in full_url
        if not c.isalnum() and c not in ".-_/:?&=%"
    )

    has_obfuscation = 1 if (n_percent > 0 or "@" in full_url or _is_ip(domain)) else 0

    feats = {
        'URLLength': len(full_url),
        'DomainLength': len(domain),
        'IsDomainIP': 1 if _is_ip(domain) else 0,
        'TLDLength': len(tld),
        'TLDLegitimateProb': TLD_LOOKUP.get(tld, DEFAULT_TLD_PROB),
        'HasObfuscation': has_obfuscation,
        'NoOfObfuscatedChar': n_percent,
        'ObfuscationRatio': round(n_percent / len(full_url), 4) if full_url else 0,
        'NoOfLettersInURL': letters,
        'LetterRatioInURL': round(letters / len(full_url), 4) if full_url else 0,
        'NoOfDegitsInURL': digits,
        'DegitRatioInURL': round(digits / len(full_url), 4) if full_url else 0,
        'NoOfEqualsInURL': n_equals,
        'NoOfQMarkInURL': n_qmark,
        'NoOfAmpersandInURL': n_amp,
        'NoOfOtherSpecialCharsInURL': other_special,
        'SpacialCharRatioInURL': round(other_special / len(full_url), 4) if full_url else 0,
        'IsHTTPS': 1 if scheme == "https" else 0,
    }
    return feats, domain, scheme


def extract_content_features(url: str, timeout: float = 6.0) -> dict:
    """
    Fetches the live page and derives Tier-2 content-based features.
    Returns (features_dict, meta) where meta carries extra info (title, error).
    Raises no exception on failure -- returns a low-signal default feature set
    with meta['error'] set, so the API can still respond gracefully.
    """
    import requests
    from bs4 import BeautifulSoup
    from difflib import SequenceMatcher

    meta = {"error": None, "title": None, "final_url": url}
    defaults = {k: 0 for k in CONTENT_FEATURES}

    try:
        resp = requests.get(
            url if "://" in url else f"http://{url}",
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (CatchPhish-Scanner/1.0)"},
            allow_redirects=True,
        )
        meta["final_url"] = resp.url
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        lines = html.splitlines()
        title_tag = soup.title
        title = title_tag.get_text(strip=True) if title_tag else ""
        meta["title"] = title

        domain = urlparse(resp.url).netloc.split(":")[0]
        domain_title_score = round(SequenceMatcher(None, domain.lower(), title.lower()).ratio(), 4)
        url_title_score = round(SequenceMatcher(None, url.lower(), title.lower()).ratio(), 4)

        favicon = 1 if soup.find("link", rel=lambda v: v and "icon" in v.lower()) else 0
        viewport = soup.find("meta", attrs={"name": "viewport"})
        description = soup.find("meta", attrs={"name": "description"})

        forms = soup.find_all("form")
        has_ext_form = 0
        for f in forms:
            action = f.get("action", "")
            if action.startswith("http") and domain not in action:
                has_ext_form = 1
                break

        links = soup.find_all("a", href=True)
        self_ref, empty_ref, ext_ref = 0, 0, 0
        social_domains = ("facebook.com", "twitter.com", "x.com", "instagram.com",
                           "linkedin.com", "youtube.com", "t.me", "whatsapp.com")
        has_social = 0
        for a in links:
            href = a["href"].strip()
            if href in ("", "#") or href.startswith("javascript:"):
                empty_ref += 1
            elif href.startswith("http") and domain not in href:
                ext_ref += 1
                if any(sd in href for sd in social_domains):
                    has_social = 1
            else:
                self_ref += 1

        text_lower = soup.get_text(" ", strip=True).lower()
        robots_allowed = 1
        try:
            r = requests.get(f"{urlparse(resp.url).scheme}://{domain}/robots.txt", timeout=3)
            robots_allowed = 1 if r.status_code == 200 else 0
        except Exception:
            robots_allowed = 0

        feats = {
            'LineOfCode': len(lines),
            'LargestLineLength': max((len(l) for l in lines), default=0),
            'HasTitle': 1 if title else 0,
            'DomainTitleMatchScore': domain_title_score,
            'URLTitleMatchScore': url_title_score,
            'HasFavicon': favicon,
            'Robots': robots_allowed,
            'IsResponsive': 1 if viewport else 0,
            'NoOfURLRedirect': len(resp.history),
            'NoOfSelfRedirect': sum(1 for h in resp.history if domain in h.url),
            'HasDescription': 1 if description else 0,
            'NoOfPopup': html.lower().count("window.open("),
            'NoOfiFrame': len(soup.find_all("iframe")),
            'HasExternalFormSubmit': has_ext_form,
            'HasSocialNet': has_social,
            'HasSubmitButton': 1 if soup.find("input", {"type": "submit"}) or soup.find("button", {"type": "submit"}) else 0,
            'HasHiddenFields': 1 if soup.find("input", {"type": "hidden"}) else 0,
            'HasPasswordField': 1 if soup.find("input", {"type": "password"}) else 0,
            'Bank': 1 if "bank" in text_lower else 0,
            'Pay': 1 if ("pay" in text_lower or "payment" in text_lower) else 0,
            'Crypto': 1 if ("crypto" in text_lower or "bitcoin" in text_lower or "wallet" in text_lower) else 0,
            'HasCopyrightInfo': 1 if ("©" in html or "copyright" in text_lower) else 0,
            'NoOfImage': len(soup.find_all("img")),
            'NoOfCSS': len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style")),
            'NoOfJS': len(soup.find_all("script")),
            'NoOfSelfRef': self_ref,
            'NoOfEmptyRef': empty_ref,
            'NoOfExternalRef': ext_ref,
        }
        return feats, meta

    except Exception as e:
        meta["error"] = str(e)
        return defaults, meta
