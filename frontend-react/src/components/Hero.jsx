export default function Hero({ url, setUrl, mode, setMode, onScan, scanning, result }) {
  const hasResult = !!result;
  const tier = result?.final_tier;
  const score = result?.final_risk_score;
  const signalCount = result ? (result.deep || result.fast).reasons.length : null;

  return (
    <div className="hero">
      <div className="hero-eyebrow">
        <span className="rdot" />
        {hasResult
          ? `SCANNED: ${result.domain}`
          : "TWO-TIER REAL-TIME PHISHING DETECTION // CATCHPHISH"}
      </div>

      {!hasResult && (
        <h1 className="hero-headline">
          PASTE A URL.<br />
          <span className="accent">WE'LL FIND</span><br />
          THE RISK.
        </h1>
      )}

      {hasResult && (
        <h1 className="hero-headline">
          THIS URL HAS<br />
          <span className={tier === "Dangerous" ? "danger" : tier === "Safe" ? "safe" : "accent"}>
            {score}%
          </span><br />
          RISK SCORE.
        </h1>
      )}

      {hasResult ? (
        <div className={`verdict-banner ${tier}`}>
          VERDICT: {tier?.toUpperCase()} — {signalCount} SIGNALS ANALYZED
        </div>
      ) : (
        <div className="verdict-banner">
          SEE THE VERDICT IN UNDER 100ms
        </div>
      )}

      <p className="hero-desc">
        {hasResult ? (
          <>
            <b>{result.escalated ? "Tier-2 deep scan" : "Tier-1 fast scan"}</b> completed in{" "}
            <b>{result.total_latency_ms}ms</b>. Model reasoning is on the right — every verdict
            is explained with SHAP, not a black box.
          </>
        ) : (
          <>
            Most phishing detectors give you a black-box <b>yes/no</b>. CatchPhish reads URL
            structure instantly, escalates to a live page-content check only when uncertain, and
            explains exactly which signals drove the call.
          </>
        )}
      </p>

      <form className="scan-form" onSubmit={(e) => { e.preventDefault(); onScan(); }}>
        <input
          className="scan-input"
          type="text"
          placeholder="paste a url — e.g. paypal-secure-login.tk/verify"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          spellCheck={false}
          autoComplete="off"
        />
        <button className="btn-gold" type="submit" disabled={scanning}>
          <span className="live-dot" />
          {scanning ? "SCANNING…" : "RUN SCAN →"}
        </button>
      </form>

      <div className="mode-row">
        <button
          className={`mode-chip ${mode === "fast" ? "active" : ""}`}
          onClick={() => setMode("fast")}
          type="button"
        >
          FAST SCAN
        </button>
        <button
          className={`mode-chip ${mode === "deep" ? "active" : ""}`}
          onClick={() => setMode("deep")}
          type="button"
        >
          DEEP SCAN
        </button>
      </div>
    </div>
  );
}