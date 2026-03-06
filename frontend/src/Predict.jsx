import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

  const fetchExplain = async (payload) => {
    setExplainLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/explain", {
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
      soil_moisture: Number(soilMoisture),
      temperature: Number(temperature),
      humidity: Number(humidity),
      ph: Number(ph),
      region,
      crop_type: cropType,
      soil_type: soilType
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/predict_full_intelligent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      setResult(data);
      setExplainResult(null);
      setFeedbackStatus(null);
      setLastPayload(payload);
      try {
        localStorage.setItem("last_predict_payload", JSON.stringify(payload));
        localStorage.setItem("last_predict_result", JSON.stringify(data));
      } catch {
        // ignore storage errors
      }

      // Automatically run explanation using the exact same payload as prediction.
      fetchExplain(payload);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }
  };

  const submitFeedback = async (label) => {
    if (!lastPayload) {
      alert("Run prediction first.");
      return;
    }
    setFeedbackStatus("submitting");
    try {
      const res = await fetch("http://127.0.0.1:8000/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastPayload, label }),
      });
      const data = await res.json();
      if (!res.ok) {
        setFeedbackStatus(`error: ${data?.error || "failed"}`);
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

  return (
    <div style={{ maxWidth: "900px", margin: "40px auto", fontFamily: "Inter" }}>
      <h2>🌾 Full Intelligent Irrigation System</h2>

      <h3>📅 Crop Stage</h3>
      <input type="date" value={sowingDate} onChange={e => setSowingDate(e.target.value)} />
      <input type="date" value={currentDate} onChange={e => setCurrentDate(e.target.value)} style={{ marginLeft: "10px" }} />

      <h3>🌡 Sensor Values</h3>
      <input type="number" placeholder="Soil Moisture (%)" value={soilMoisture} onChange={e => setSoilMoisture(e.target.value)} />
      <input type="number" placeholder="Temperature (°C)" value={temperature} onChange={e => setTemperature(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="number" placeholder="Humidity (%)" value={humidity} onChange={e => setHumidity(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="number" step="0.01" placeholder="pH" value={ph} onChange={e => setPh(e.target.value)} style={{ marginLeft: "10px" }} />

      <h3>🌍 Context</h3>
      <input type="text" placeholder="Region" value={region} onChange={e => setRegion(e.target.value)} />
      <input type="text" placeholder="Crop Type" value={cropType} onChange={e => setCropType(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="text" placeholder="Soil Type" value={soilType} onChange={e => setSoilType(e.target.value)} style={{ marginLeft: "10px" }} />

      <br /><br />

      <button
        onClick={handlePredict}
        style={{
          padding: "12px 25px",
          background: "#2563eb",
          color: "white",
          border: "none",
          borderRadius: "8px",
          fontWeight: "600"
        }}
      >
        🚀 Run Full Intelligent Prediction
      </button>

      {result && (
        <div style={{ marginTop: "30px", padding: "20px", background: "#f1f5f9", borderRadius: "10px" }}>
          <h3>📊 Results</h3>
          <p><strong>Growth Stage:</strong> {result.growth_stage}</p>
          <p><strong>Stage Model:</strong> {result.stage_model_prediction}</p>
          <p><strong>AFTA Model (Combined):</strong> {result.afta_prediction}</p>
          <div
            style={{
              marginTop: "14px",
              padding: "14px",
              borderRadius: "10px",
              border: "1px solid #cbd5e1",
              background: "#ffffff"
            }}
          >
            <h4 style={{ marginTop: 0, marginBottom: "10px" }}>🌐 Global vs 🧩 Local AFTA</h4>
            <p style={{ margin: "6px 0" }}>
              <strong>Global AFTA Prediction:</strong> {result.afta_global_prediction}
              {typeof result.afta_global_probability === "number" ? ` (p=${result.afta_global_probability})` : ""}
            </p>
            <p style={{ margin: "6px 0" }}>
              <strong>Local Stage AFTA Prediction:</strong> {result.afta_local_prediction}
              {typeof result.afta_local_probability === "number" ? ` (p=${result.afta_local_probability})` : ""}
            </p>
            <p style={{ margin: "6px 0" }}>
              <strong>Combined AFTA Prediction:</strong> {result.afta_combined_prediction}
              {typeof result.afta_combined_probability === "number" ? ` (p=${result.afta_combined_probability})` : ""}
            </p>
            <p style={{ margin: "6px 0" }}>
              <strong>AFTA Decision Mode:</strong> {result.afta_decision_mode || "n/a"}
            </p>
            <p style={{ margin: "6px 0" }}>
              <strong>Local Model:</strong>{" "}
              {result.afta_local_model_name || "not selected"}{" "}
              ({result.afta_local_model_available ? "loaded" : "fallback to global"})
            </p>
          </div>
          <p><strong>Context Score:</strong> {result.context_score}</p>
          <p><strong>Stress Index:</strong> {result.stress_index}</p>

          <h2>
            {result.final_prediction === 1
              ? "💧 Irrigation Required"
              : "✅ No Irrigation Needed"}
          </h2>

          {result.final_prediction === 1 && (
            <h3>
              💦 Recommended Water: {result.recommended_water_liters} Liters
            </h3>
          )}

          <div style={{ marginTop: "20px" }}>
            <h3>✅ Feedback (for retraining)</h3>
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
              <button
                onClick={() => submitFeedback(1)}
                style={{
                  padding: "10px 14px",
                  background: "#16a34a",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontWeight: "600",
                  cursor: "pointer",
                }}
              >
                It needed irrigation (Label=1)
              </button>
              <button
                onClick={() => submitFeedback(0)}
                style={{
                  padding: "10px 14px",
                  background: "#dc2626",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  fontWeight: "600",
                  cursor: "pointer",
                }}
              >
                No irrigation needed (Label=0)
              </button>
            </div>
            {feedbackStatus === "submitting" && (
              <p style={{ marginTop: "10px" }}>Saving feedback...</p>
            )}
            {feedbackStatus === "saved" && (
              <p style={{ marginTop: "10px" }}>Feedback saved (and sent to S3 if configured).</p>
            )}
            {feedbackStatus && feedbackStatus.startsWith("error") && (
              <p style={{ marginTop: "10px", color: "#b91c1c" }}>{feedbackStatus}</p>
            )}
          </div>

          <div style={{ marginTop: "20px" }}>
            <button
              onClick={handleExplain}
              disabled={explainLoading}
              style={{
                padding: "10px 18px",
                background: "#111827",
                color: "white",
                border: "none",
                borderRadius: "8px",
                fontWeight: "600",
                cursor: explainLoading ? "not-allowed" : "pointer"
              }}
            >
              {explainLoading ? "⏳ Explaining..." : "🧠 Explain with LLM"}
            </button>
          </div>

          {explainResult && (
            <div style={{ marginTop: "20px" }}>
              <h3>🧠 Explanation</h3>
              {explainResult.error && (
                <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(explainResult, null, 2)}</pre>
              )}
              {explainResult.llm_explanation && (
                <div
                  style={{
                    marginTop: "10px",
                    padding: "16px",
                    borderRadius: "12px",
                    background: "#fff",
                    border: "1px solid #e2e8f0",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
                    lineHeight: "1.7",
                    fontSize: "16px",
                    color: "#1e293b",
                  }}
                >
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {explainResult.llm_explanation}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
