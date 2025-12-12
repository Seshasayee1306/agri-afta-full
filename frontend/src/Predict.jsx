import React, { useState } from "react";

export default function Predict() {
  const [features, setFeatures] = useState(""); // CSV-like input
  const [result, setResult] = useState(null);

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
    } catch (err) {
      console.error(err);
      setResult({ error: "Failed to fetch prediction" });
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
        </div>
      )}
    </div>
  );
}
