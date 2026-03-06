import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function Compare() {
    const [features, setFeatures] = useState("");
    const [predictResult, setPredictResult] = useState(null);
    const [explainResult, setExplainResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleCompare = async () => {
        if (!features) return;
        setLoading(true);
        setPredictResult(null);
        setExplainResult(null);

        const arr = features.split(",").map((x) => parseFloat(x.trim()));

        try {
            // Parallel requests
            const [predRes, explRes] = await Promise.all([
                fetch("http://127.0.0.1:8000/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ features: arr }),
                }),
                fetch("http://127.0.0.1:8000/explain", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ features: arr }),
                })
            ]);

            const predData = await predRes.json();
            const explData = await explRes.json();

            setPredictResult(predData);
            setExplainResult(explData);
        } catch (err) {
            console.error(err);
            alert("Error fetching data");
        } finally {
            setLoading(false);
        }
    };

    const getLabel = (result) => {
        // Check for invalid data / errors
        if (result.error) {
            return "❌ Invalid Data";
        }
        return result.prediction === 1 ? "Needs Water 💧" : "No Irrigation Needed 🌿";
    };

    const getConfidence = (result) => {
        // 1. If explicit score exists (New Backend), use it directly
        if (result.confidence_score !== undefined) {
            return (result.confidence_score * 100).toFixed(2) + "%";
        }

        // 2. Fallback for old API
        if (result.accuracy !== undefined) {
            // Basic naive inversion
            const p = result.accuracy;
            const conf = result.prediction === 1 ? p : 1 - p;
            return (conf * 100).toFixed(2) + "%";
        }

        if (result.probability !== undefined) {
            const p = result.probability;
            const conf = result.prediction === 1 ? p : 1 - p;
            return (conf * 100).toFixed(2) + "%";
        }

        return "N/A";
    };

    return (
        <div style={{ maxWidth: "1000px", margin: "40px auto", padding: "20px", fontFamily: "Inter, sans-serif" }}>
            <h2 style={{ fontSize: "28px", fontWeight: "700", textAlign: "center" }}>
                ⚖️ Compare Accuracy: Predict vs Explain
            </h2>

            <div style={{ marginTop: "20px", textAlign: "center" }}>
                <input
                    type="text"
                    placeholder="Enter 12 comma-separated features"
                    value={features}
                    onChange={(e) => setFeatures(e.target.value)}
                    style={{
                        width: "60%",
                        padding: "12px",
                        fontSize: "16px",
                        borderRadius: "12px",
                        border: "1px solid #d1d5db",
                    }}
                />
                <br />
                <button
                    onClick={handleCompare}
                    disabled={loading}
                    style={{
                        marginTop: "15px",
                        padding: "12px 30px",
                        borderRadius: "10px",
                        border: "none",
                        fontSize: "16px",
                        fontWeight: "600",
                        background: loading ? "#94a3b8" : "#2563eb",
                        color: "white",
                        cursor: loading ? "not-allowed" : "pointer",
                    }}
                >
                    {loading ? "Running Comparison..." : "Run Comparison"}
                </button>
            </div>

            {predictResult && explainResult && (
                <div style={{ display: "flex", gap: "20px", marginTop: "40px" }}>

                    {/* LEFT: STANDARD PREDICT */}
                    <div style={{
                        flex: 1,
                        padding: "20px",
                        borderRadius: "12px",
                        background: predictResult.error ? "#fee2e2" : "#f8fafc",
                        border: predictResult.error ? "2px solid #ef4444" : "1px solid #e2e8f0"
                    }}>
                        <h3 style={{ borderBottom: "2px solid #cbd5e1", paddingBottom: "10px" }}>⚡ Standard Predict</h3>
                        <p style={{ fontSize: "18px", fontWeight: "bold", marginTop: "20px" }}>
                            Prediction: <span style={{ color: predictResult.error ? "red" : (predictResult.prediction === 1 ? "blue" : "green") }}>
                                {getLabel(predictResult)}
                            </span>
                        </p>
                        {predictResult.accuracy !== undefined && (
                            <p style={{ fontSize: "16px", color: "#64748b" }}>
                                Confidence: <strong>{getConfidence(predictResult)}</strong>
                            </p>
                        )}
                        <pre style={{ background: "#eee", padding: "10px", borderRadius: "8px", fontSize: "12px" }}>
                            {JSON.stringify(predictResult, null, 2)}
                        </pre>
                    </div>

                    {/* RIGHT: LLM EXPLAIN */}
                    <div style={{
                        flex: 1,
                        padding: "20px",
                        borderRadius: "12px",
                        background: explainResult.llm_explanation?.includes("Cannot explain invalid data") || explainResult.error ? "#fee2e2" : "#f0fdf4",
                        border: explainResult.llm_explanation?.includes("Cannot explain invalid data") || explainResult.error ? "2px solid #ef4444" : "1px solid #ecfccb"
                    }}>
                        <h3 style={{ borderBottom: "2px solid #86efac", paddingBottom: "10px" }}>🧠 LLM Explain</h3>
                        <p style={{ fontSize: "18px", fontWeight: "bold", marginTop: "20px" }}>
                            Prediction: <span style={{ color: (explainResult.llm_explanation?.includes("Cannot explain invalid data") || explainResult.error) ? "red" : (explainResult.prediction === 1 ? "blue" : "green") }}>
                                {getLabel(explainResult)}
                            </span>
                        </p>
                        {explainResult.probability !== undefined && (
                            <p style={{ fontSize: "16px", color: "#64748b" }}>
                                Confidence: <strong>{getConfidence(explainResult)}</strong>
                            </p>
                        )}
                        <div style={{ marginTop: "15px", padding: "10px", background: "white", borderRadius: "8px", border: "1px solid #ddd" }}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {explainResult.llm_explanation}
                            </ReactMarkdown>
                        </div>
                    </div>
                </div>
            )}

            {/* CONSISTENCY CHECK */}
            {predictResult && explainResult && (
                <div style={{
                    marginTop: "30px", padding: "15px", borderRadius: "10px", textAlign: "center",
                    background: predictResult.prediction === explainResult.prediction ? "#dcfce7" : "#fee2e2",
                    border: `2px solid ${predictResult.prediction === explainResult.prediction ? "#22c55e" : "#ef4444"}`
                }}>
                    <h3 style={{ margin: 0 }}>
                        {predictResult.prediction === explainResult.prediction
                            ? "✅ Both Models Agree!"
                            : "⚠️ Mismatch Detected!"}
                    </h3>
                </div>
            )}
        </div>
    );
}
