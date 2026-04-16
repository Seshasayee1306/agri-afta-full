import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function toNumberOrNull(value) {
  const text = String(value ?? "").trim();
  if (!text) return null;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : null;
}

function validateRange(name, value, lo, hi) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return `${name} must be a valid number`;
  }
  if (value < lo || value > hi) {
    return `${name} must be between ${lo} and ${hi}`;
  }
  return null;
}

export default function Predict() {
  const [sowingDate, setSowingDate] = useState("");
  const [currentDate, setCurrentDate] = useState("");

  const [soilMoisture, setSoilMoisture] = useState("");
  const [temperature, setTemperature] = useState("");
  const [humidity, setHumidity] = useState("");
  const [ph, setPh] = useState("");

  const [region, setRegion] = useState("");
  const [cropType, setCropType] = useState("");
  const [soilType, setSoilType] = useState("");

  const [result, setResult] = useState(null);
  const [lastPayload, setLastPayload] = useState(null);
  const [explainResult, setExplainResult] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState(null);
  const [sensorSyncStatus, setSensorSyncStatus] = useState("");
  const [predictLoading, setPredictLoading] = useState(false);

  const baseUrl = "http://127.0.0.1:8000";

  const fetchExplain = async (payload) => {
    setExplainLoading(true);
    try {
      const res = await fetch(baseUrl + "/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setExplainResult(data);
      try {
        localStorage.setItem("last_explain_result", JSON.stringify(data));
      } catch {
        // ignore storage errors
      }
    } catch (err) {
      console.error(err);
      setExplainResult({ error: "Explanation failed" });
    } finally {
      setExplainLoading(false);
    }
  };

  const handlePredict = async () => {
    const payload = {
      sowing_date: sowingDate,
      current_date: currentDate,
      soil_moisture: toNumberOrNull(soilMoisture),
      temperature: toNumberOrNull(temperature),
      humidity: toNumberOrNull(humidity),
      ph: toNumberOrNull(ph),
      region: String(region || "").trim(),
      crop_type: String(cropType || "").trim(),
      soil_type: String(soilType || "").trim(),
    };

    if (!payload.sowing_date || !payload.current_date) {
      setResult({ error: "Sowing Date and Current Date are required." });
      return;
    }
    if (!payload.region || !payload.crop_type || !payload.soil_type) {
      setResult({ error: "Region, Crop Type, and Soil Type are required." });
      return;
    }

    const sowingDateObj = new Date(`${payload.sowing_date}T00:00:00`);
    const currentDateObj = new Date(`${payload.current_date}T00:00:00`);
    if (Number.isNaN(sowingDateObj.getTime()) || Number.isNaN(currentDateObj.getTime())) {
      setResult({ error: "Invalid date format. Use valid dates." });
      return;
    }
    if (currentDateObj < sowingDateObj) {
      setResult({ error: "Current Date must be on or after Sowing Date." });
      return;
    }

    const fieldChecks = [
      validateRange("Soil Moisture", payload.soil_moisture, 0, 100),
      validateRange("Temperature", payload.temperature, -20, 70),
      validateRange("Humidity", payload.humidity, 0, 100),
      validateRange("pH", payload.ph, 0, 14),
    ].filter(Boolean);

    if (fieldChecks.length > 0) {
      setResult({ error: fieldChecks[0] });
      return;
    }

    setPredictLoading(true);
    try {
      const res = await fetch(baseUrl + "/predict_full_intelligent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) {
        setResult({ error: data?.error || "Prediction failed" });
        return;
      }

      setResult(data);
      setExplainResult(null);
      setFeedbackStatus(null);
      setLastPayload(payload);
      try {
        localStorage.setItem("last_predict_payload", JSON.stringify(payload));
        localStorage.setItem("last_predict_result", JSON.stringify(data));
        window.dispatchEvent(new Event("agri-predict-updated"));
      } catch {
        // ignore storage errors
      }

      fetchExplain(payload);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    } finally {
      setPredictLoading(false);
    }
  };

  const fetchLatestSensorReadings = async () => {
    setSensorSyncStatus("Fetching latest ESP32 readings...");
    try {
      const res = await fetch(baseUrl + "/sensor_readings/latest");
      const data = await res.json();
      if (!res.ok) {
        setSensorSyncStatus(data?.error || "Failed to fetch sensor readings");
        return;
      }

      setSoilMoisture(String(data.soil_moisture ?? ""));
      setTemperature(String(data.temperature ?? ""));
      setHumidity(String(data.humidity ?? ""));
      setPh(String(data.ph ?? ""));
      setSensorSyncStatus(
        "Updated from " +
          (data.device_id || "esp32") +
          " at " +
          (data.received_at || "unknown time")
      );
    } catch (err) {
      console.error(err);
      setSensorSyncStatus("Failed to fetch sensor readings");
    }
  };

  useEffect(() => {
    fetchLatestSensorReadings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitFeedback = async (label) => {
    if (!lastPayload) {
      alert("Run prediction first.");
      return;
    }
    setFeedbackStatus("submitting");
    try {
      const res = await fetch(baseUrl + "/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastPayload, label }),
      });
      const data = await res.json();
      if (!res.ok) {
        setFeedbackStatus("error: " + (data?.error || "failed"));
        return;
      }
      setFeedbackStatus("saved");
    } catch (err) {
      console.error(err);
      setFeedbackStatus("error: failed");
    }
  };

  const handleExplain = async () => {
    if (!lastPayload) {
      alert("Run prediction first.");
      return;
    }
    fetchExplain(lastPayload);
  };

  const predictionToneClass =
    result?.final_prediction === 1 ? "decision-card danger" : "decision-card success";

  return (
    <div className="dashboard-enter">
      <section className="hero-panel">
        <p className="hero-kicker">Precision Operations Suite</p>
        <h2 className="hero-title">Field Prediction Workflow</h2>
        <p className="hero-copy">
          Sync live sensor telemetry, attach crop context, and run the full intelligent irrigation decision engine.
        </p>
      </section>

      <section className="panel form-panel">
        <div className="panel-heading">
          <h3>Crop Timeline</h3>
          <p>Set growth window dates to align stage-aware predictions.</p>
        </div>
        <div className="input-grid input-grid-two">
          <label className="field">
            <span>Sowing Date</span>
            <input type="date" value={sowingDate} onChange={(e) => setSowingDate(e.target.value)} />
          </label>
          <label className="field">
            <span>Current Date</span>
            <input type="date" value={currentDate} onChange={(e) => setCurrentDate(e.target.value)} />
          </label>
        </div>
      </section>

      <section className="panel form-panel">
        <div className="panel-heading panel-heading-inline">
          <div>
            <h3>Sensor Inputs</h3>
            <p>Use the latest ESP32 readings or manually override before prediction.</p>
          </div>
          <button onClick={fetchLatestSensorReadings} className="btn btn-secondary">
            Pull Latest From ESP32
          </button>
        </div>
        {sensorSyncStatus && <p className="status-text">{sensorSyncStatus}</p>}
        <div className="input-grid input-grid-four">
          <label className="field">
            <span>Soil Moisture (%)</span>
            <input type="number" value={soilMoisture} onChange={(e) => setSoilMoisture(e.target.value)} />
          </label>
          <label className="field">
            <span>Temperature (°C)</span>
            <input type="number" value={temperature} onChange={(e) => setTemperature(e.target.value)} />
          </label>
          <label className="field">
            <span>Humidity (%)</span>
            <input type="number" value={humidity} onChange={(e) => setHumidity(e.target.value)} />
          </label>
          <label className="field">
            <span>pH</span>
            <input type="number" step="0.01" value={ph} onChange={(e) => setPh(e.target.value)} />
          </label>
        </div>
      </section>

      <section className="panel form-panel">
        <div className="panel-heading">
          <h3>Context Inputs</h3>
          <p>Add farm context used by lookup, scoring, and recommendation stages.</p>
        </div>
        <div className="input-grid input-grid-three">
          <label className="field">
            <span>Region</span>
            <input type="text" value={region} onChange={(e) => setRegion(e.target.value)} />
          </label>
          <label className="field">
            <span>Crop Type</span>
            <input type="text" value={cropType} onChange={(e) => setCropType(e.target.value)} />
          </label>
          <label className="field">
            <span>Soil Type</span>
            <input type="text" value={soilType} onChange={(e) => setSoilType(e.target.value)} />
          </label>
        </div>
      </section>

      <section className="action-row">
        <button onClick={handlePredict} className="btn btn-primary btn-large" disabled={predictLoading}>
          {predictLoading ? "Running Prediction..." : "Run Full Intelligent Prediction"}
        </button>
      </section>

      {result && (
        <section className="panel result-panel">
          <div className="panel-heading">
            <h3>Prediction Output</h3>
            <p>Decision details from stage model, AFTA blend, and contextual reasoning.</p>
          </div>

          {result.error ? (
            <div className="status-pill danger">{result.error}</div>
          ) : (
            <>
              <div className="metrics-grid">
                <div className="metric-card">
                  <p className="metric-label">Growth Stage</p>
                  <p className="metric-value">{result.growth_stage}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Stage Model</p>
                  <p className="metric-value">{result.stage_model_prediction}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">AFTA Combined</p>
                  <p className="metric-value">{result.afta_prediction}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Context Score</p>
                  <p className="metric-value">{result.context_score}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Stress Index</p>
                  <p className="metric-value">{result.stress_index}</p>
                </div>
              </div>

              <div className={predictionToneClass}>
                <p className="decision-title">
                  {result.final_prediction === 1
                    ? "Irrigation Required"
                    : "No Irrigation Needed"}
                </p>
                <p className="decision-subtitle">
                  {result.final_prediction === 1
                    ? "Recommendation generated based on current risk profile."
                    : "Current field conditions are stable for now."}
                </p>
                {result.final_prediction === 1 && (
                  <p className="decision-water">
                    Recommended Water: {result.recommended_water_liters} liters
                  </p>
                )}
              </div>

              <div className="afta-panel">
                <h4>Global vs Local AFTA</h4>
                <p>
                  <strong>Global Prediction:</strong> {result.afta_global_prediction}
                  {typeof result.afta_global_probability === "number"
                    ? " (p=" + result.afta_global_probability + ")"
                    : ""}
                </p>
                <p>
                  <strong>Local Prediction:</strong> {result.afta_local_prediction}
                  {typeof result.afta_local_probability === "number"
                    ? " (p=" + result.afta_local_probability + ")"
                    : ""}
                </p>
                <p>
                  <strong>Combined Prediction:</strong> {result.afta_combined_prediction}
                  {typeof result.afta_combined_probability === "number"
                    ? " (p=" + result.afta_combined_probability + ")"
                    : ""}
                </p>
                <p><strong>Decision Mode:</strong> {result.afta_decision_mode || "n/a"}</p>
                <p>
                  <strong>Local Model:</strong> {result.afta_local_model_name || "not selected"} (
                  {result.afta_local_model_available ? "loaded" : "fallback to global"})
                </p>
              </div>

              <div className="feedback-panel">
                <h4>Feedback for Retraining</h4>
                <div className="feedback-actions">
                  <button onClick={() => submitFeedback(1)} className="btn btn-success">
                    Label as irrigation needed
                  </button>
                  <button onClick={() => submitFeedback(0)} className="btn btn-danger">
                    Label as no irrigation needed
                  </button>
                </div>
                {feedbackStatus === "submitting" && <p className="status-text">Saving feedback...</p>}
                {feedbackStatus === "saved" && (
                  <p className="status-pill success">Feedback saved successfully.</p>
                )}
                {feedbackStatus && feedbackStatus.startsWith("error") && (
                  <p className="status-pill danger">{feedbackStatus}</p>
                )}
              </div>

              <div className="llm-panel">
                <div className="panel-heading panel-heading-inline">
                  <div>
                    <h4>LLM Explanation</h4>
                    <p>Natural language rationale for the prediction decision.</p>
                  </div>
                  <button
                    onClick={handleExplain}
                    disabled={explainLoading}
                    className="btn btn-tertiary"
                  >
                    {explainLoading ? "Explaining..." : "Generate Explanation"}
                  </button>
                </div>

                {explainResult && explainResult.error && (
                  <pre className="code-box">{JSON.stringify(explainResult, null, 2)}</pre>
                )}

                {explainResult && explainResult.llm_explanation && (
                  <div className="markdown-card">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {explainResult.llm_explanation}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}
