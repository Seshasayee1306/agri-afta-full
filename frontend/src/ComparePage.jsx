import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./compare.css";

const BASE_URL = "http://127.0.0.1:8000";
const PAYLOAD_STORAGE_KEY = "last_predict_payload";
const RESULT_STORAGE_KEY = "last_predict_result";
const FIXED_CHALLENGER_KIND = "catboost";
const FIXED_CHALLENGER_LABEL = "Challenger Model";

function toDecisionText(pred) {
  return pred === 1 ? "Irrigation Needed" : "No Irrigation Needed";
}

export default function ComparePage() {
  const [payloadError, setPayloadError] = useState("");
  const [hasPayload, setHasPayload] = useState(false);
  const [mainResult, setMainResult] = useState(null);
  const [baselineResult, setBaselineResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareStatus, setCompareStatus] = useState("");
  const lastComparedPayloadSigRef = useRef("");

  const runBaselineCompare = useCallback(async (payload) => {
    setCompareLoading(true);
    setCompareStatus(`Running comparison with ${FIXED_CHALLENGER_LABEL}...`);
    setBaselineResult(null);

    try {
      // Send the same payload saved by Prediction Console so challenger uses
      // backend-derived defaults/features identical to model-serving logic.
      const comparePayload = {
        ...payload,
        challenger_kind: FIXED_CHALLENGER_KIND,
      };

      const response = await fetch(`${BASE_URL}/predict_catboost_compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(comparePayload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Baseline comparison failed");
      }

      setBaselineResult(data);
      setCompareStatus("Comparison updated.");
    } catch (error) {
      setBaselineResult({
        error: error?.message || "Comparison failed",
      });
      setCompareStatus("");
    } finally {
      setCompareLoading(false);
    }
  }, []);

  const hydrateFromStorage = useCallback((forceCompare = false) => {
    let parsedPayload = null;
    let parsedMainResult = null;

    try {
      const payloadRaw = localStorage.getItem(PAYLOAD_STORAGE_KEY);
      const resultRaw = localStorage.getItem(RESULT_STORAGE_KEY);

      if (payloadRaw) {
        parsedPayload = JSON.parse(payloadRaw);
        setHasPayload(true);
      } else {
        setHasPayload(false);
      }

      if (resultRaw) {
        parsedMainResult = JSON.parse(resultRaw);
        setMainResult(parsedMainResult);
      } else {
        setMainResult(null);
      }

      setPayloadError("");
      if (parsedPayload) {
        const payloadSignature = JSON.stringify(parsedPayload);
        if (forceCompare || payloadSignature !== lastComparedPayloadSigRef.current) {
          lastComparedPayloadSigRef.current = payloadSignature;
          runBaselineCompare(parsedPayload);
        }
      } else {
        setBaselineResult(null);
        setCompareStatus("");
        lastComparedPayloadSigRef.current = "";
      }
    } catch {
      setPayloadError("Auto-load failed. Run prediction once in Prediction Console.");
      setMainResult(null);
      setHasPayload(false);
      setBaselineResult(null);
      setCompareStatus("");
      lastComparedPayloadSigRef.current = "";
    }
  }, [runBaselineCompare]);

  useEffect(() => {
    hydrateFromStorage();
  }, [hydrateFromStorage]);

  useEffect(() => {
    const syncListener = (event) => {
      if (event.key === RESULT_STORAGE_KEY) {
        hydrateFromStorage(true);
        return;
      }
      if (event.key === PAYLOAD_STORAGE_KEY) {
        hydrateFromStorage();
      }
    };
    const predictSyncListener = () => hydrateFromStorage(true);
    const focusListener = () => hydrateFromStorage();
    const visibilityListener = () => {
      if (!document.hidden) hydrateFromStorage();
    };

    window.addEventListener("storage", syncListener);
    window.addEventListener("agri-predict-updated", predictSyncListener);
    window.addEventListener("focus", focusListener);
    document.addEventListener("visibilitychange", visibilityListener);
    return () => {
      window.removeEventListener("storage", syncListener);
      window.removeEventListener("agri-predict-updated", predictSyncListener);
      window.removeEventListener("focus", focusListener);
      document.removeEventListener("visibilitychange", visibilityListener);
    };
  }, [hydrateFromStorage]);

  const agreementState = useMemo(() => {
    if (
      !mainResult ||
      !baselineResult ||
      baselineResult.error ||
      typeof mainResult.final_prediction !== "number" ||
      typeof baselineResult.final_prediction !== "number"
    ) {
      return null;
    }

    const matches = mainResult.final_prediction === baselineResult.final_prediction;
    return {
      matches,
      label: matches ? "Models agree" : "Models disagree",
      description: matches
        ? "Both models reached the same irrigation decision."
        : "Two models gave different decisions. Field validation is recommended.",
    };
  }, [mainResult, baselineResult]);

  return (
    <div className="dashboard-enter">
      <section className="hero-panel">
        <p className="hero-kicker">Model Comparison</p>
        <h2 className="hero-title">Main vs Challenger</h2>
        <p className="hero-copy">Compares latest prediction with the configured challenger model.</p>
      </section>

      <section className="panel result-panel">
        <div className="panel-heading panel-heading-inline">
          <div>
            <h3>Comparison Result</h3>
            <p>Main model on left, challenger on right.</p>
          </div>
          <div className="compare-actions">
            <p className="challenger-fixed-badge">Challenger: Configured</p>
            <button className="btn btn-secondary" onClick={() => hydrateFromStorage(true)} disabled={compareLoading}>
              {compareLoading ? "Syncing..." : "Sync Latest"}
            </button>
          </div>
        </div>

        {compareStatus && <p className="status-pill success">{compareStatus}</p>}
        {payloadError && <p className="status-pill danger">{payloadError}</p>}
        {baselineResult?.fallback_used && (
          <p className="status-pill danger">
            Selected challenger unavailable. Using fallback: {baselineResult.fallback_source || "legacy"}.
          </p>
        )}
        {!hasPayload && !payloadError && (
          <p className="status-pill danger">No saved prediction found. Run Prediction Console once first.</p>
        )}

        <div className="compare-grid">
          <article className="compare-column">
            <h4>Main Model</h4>
            {!mainResult && <p className="status-text">No main prediction found in local storage.</p>}
            {mainResult && (
              <div className="metrics-grid">
                <div className="metric-card">
                  <p className="metric-label">Final Decision</p>
                  <p className="metric-value">{toDecisionText(mainResult.final_prediction)}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Growth Stage</p>
                  <p className="metric-value">{mainResult.growth_stage || "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">AFTA Prediction</p>
                  <p className="metric-value">{mainResult.afta_prediction ?? "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Context Score</p>
                  <p className="metric-value">{mainResult.context_score ?? "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Stress Index</p>
                  <p className="metric-value">{mainResult.stress_index ?? "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Water (Liters)</p>
                  <p className="metric-value">{mainResult.recommended_water_liters ?? "n/a"}</p>
                </div>
              </div>
            )}
          </article>

          <article className="compare-column">
            <h4>{FIXED_CHALLENGER_LABEL}</h4>
            {!baselineResult && !compareLoading && <p className="status-text">Challenger result not available yet.</p>}
            {baselineResult?.error && <p className="status-pill danger">{baselineResult.error}</p>}
            {baselineResult && !baselineResult.error && (
              <div className="metrics-grid">
                <div className="metric-card">
                  <p className="metric-label">Decision</p>
                  <p className="metric-value">{baselineResult.prediction_text || toDecisionText(baselineResult.final_prediction)}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Probability</p>
                  <p className="metric-value">{baselineResult.probability ?? "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Confidence Band</p>
                  <p className="metric-value">{baselineResult.confidence_band ?? "n/a"}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Validation AUC</p>
                  <p className="metric-value">{baselineResult.validation_auc ?? "n/a"}</p>
                </div>
                {baselineResult.safety_override?.applied && (
                  <div className="metric-card compare-reason compare-alert">
                    <p className="metric-label">Safety Override</p>
                    <p className="metric-value">{baselineResult.safety_override.reason}</p>
                  </div>
                )}
                {Array.isArray(baselineResult.field_warnings) && baselineResult.field_warnings.length > 0 && (
                  <div className="metric-card compare-reason compare-alert">
                    <p className="metric-label">Field Warnings</p>
                    <p className="metric-value">{baselineResult.field_warnings.join(" | ")}</p>
                  </div>
                )}
                <div className="metric-card compare-reason">
                  <p className="metric-label">Decision Reason</p>
                  <p className="metric-value">{baselineResult.decision_reason || "n/a"}</p>
                </div>
              </div>
            )}
          </article>
        </div>

        {agreementState && (
          <div className={`decision-card ${agreementState.matches ? "success" : "danger"}`}>
            <p className="decision-title">{agreementState.label}</p>
            <p className="decision-subtitle">{agreementState.description}</p>
          </div>
        )}
      </section>
    </div>
  );
}
