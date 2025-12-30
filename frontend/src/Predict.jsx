import React, { useState } from "react";

export default function Predict() {
  const [features, setFeatures] = useState(""); // CSV-like input
  const [result, setResult] = useState(null);

  // ✅ ADDED STATES
  const [actualLabel, setActualLabel] = useState("");
  const [statusMsg, setStatusMsg] = useState("");
  const [lastFeatureArray, setLastFeatureArray] = useState(null);

  const handlePredict = async () => {
    if (!features) return;

    // Convert comma-separated input to array of floats
    const arr = features.split(",").map((x) => parseFloat(x.trim()));

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: arr }),
      });
      const data = await res.json();
      setResult(data);

      // ✅ STORE FEATURES FOR TRAINING LATER
      setLastFeatureArray(arr);
      setActualLabel("");
      setStatusMsg("");
    } catch (err) {
      console.error(err);
      setResult({ error: "Failed to fetch prediction" });
    }
  };

  // ✅ ADDED FUNCTION — SAVE LABELED DATA
  const submitForTraining = async () => {
    if (!lastFeatureArray) {
      setStatusMsg("No prediction data available");
      return;
    }

    if (actualLabel === "") {
      setStatusMsg("Please select the actual outcome");
      return;
    }

    const payload = {
      features: lastFeatureArray,
      label: Number(actualLabel),
    };

    try {
      const res = await fetch("http://127.0.0.1:8001/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to save data");

      setStatusMsg("✅ Data saved for future training");
      setActualLabel("");
    } catch (err) {
      setStatusMsg(`❌ ${err.message}`);
    }
  };

  return (
    <div>
      <h2>Predict Water Need</h2>
      <input
        type="text"
        placeholder="Enter features comma separated"
        value={features}
        onChange={(e) => setFeatures(e.target.value)}
        style={{ width: "400px", marginRight: "10px" }}
      />
      <button onClick={handlePredict}>Predict</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>Prediction Result:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>

          {/* ✅ ADDED LABEL CONFIRMATION SECTION */}
          <div style={{ marginTop: "15px" }}>
            <h4>Confirm Actual Outcome (for future training)</h4>

            <select
              value={actualLabel}
              onChange={(e) => setActualLabel(e.target.value)}
            >
              <option value="">Select actual result</option>
              <option value="1">Needs Water</option>
              <option value="0">No Water Needed</option>
            </select>

            <button
              onClick={submitForTraining}
              style={{ marginLeft: "10px" }}
            >
              Save for Training
            </button>
          </div>
        </div>
      )}

      {/* ✅ STATUS MESSAGE */}
      {statusMsg && <p style={{ marginTop: "10px" }}>{statusMsg}</p>}
    </div>
  );
}