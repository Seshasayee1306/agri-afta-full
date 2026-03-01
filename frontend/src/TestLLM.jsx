import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function TestLLM() {
  const [payloadText, setPayloadText] = useState(() => {
    try {
      return localStorage.getItem("last_predict_payload") || "";
    } catch {
      return "";
    }
  });
  const [result, setResult] = useState(null);

  const handleExplain = async () => {
    if (!payloadText) return;
    let payload;
    try {
      payload = JSON.parse(payloadText);
    } catch (e) {
      setResult({ error: "Payload must be valid JSON (copied from Predict page)" });
      return;
    }

    try {
      const res = await fetch("http://127.0.0.1:8000/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Failed to fetch explanation" });
    }
  };

  return (
    <div
      style={{
        maxWidth: "900px",
        margin: "40px auto",
        padding: "20px",
        fontFamily: "Inter, sans-serif",
      }}
    >
      <h2 style={{ fontSize: "28px", fontWeight: "700" }}>
        🌱 AI Crop Watering Explanation Tester
      </h2>

      <div style={{ marginTop: "20px" }}>
        <textarea
          placeholder='Paste the JSON payload from the Predict page (auto-filled if available)'
          value={payloadText}
          onChange={(e) => setPayloadText(e.target.value)}
          rows={8}
          style={{
            width: "100%",
            padding: "12px",
            fontSize: "14px",
            borderRadius: "12px",
            border: "1px solid #d1d5db",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
          }}
        />

        <button
          onClick={handleExplain}
          style={{
            marginTop: "15px",
            padding: "12px 20px",
            borderRadius: "10px",
            border: "none",
            fontSize: "16px",
            fontWeight: "600",
            background: "#2563eb",
            color: "white",
            cursor: "pointer",
          }}
        >
          🔍 Explain Prediction
        </button>
      </div>

      {result && (
        <div style={{ marginTop: "30px" }}>
          <h3 style={{ fontSize: "22px", fontWeight: 700 }}>
            🧠 LLM Explanation Result
          </h3>

          {/* JSON preview box */}
          <div
            style={{
              background: "#f1f5f9",
              padding: "20px",
              borderRadius: "12px",
              marginTop: "15px",
              overflowX: "auto",
              fontSize: "14px",
            }}
          >
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>

          {/* Markdown explanation */}
          {result.llm_explanation && (
            <div
              style={{
                marginTop: "30px",
                padding: "20px",
                borderRadius: "12px",
                background: "#fff",
                border: "1px solid #e2e8f0",
                boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
                lineHeight: "1.7",
                fontSize: "17px",
                color: "#1e293b",
              }}
            >
              <h3 style={{ marginBottom: "15px" }}>💬 LLM Says:</h3>

              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }) => (
                    <p style={{ marginBottom: "12px" }}>{children}</p>
                  ),
                  li: ({ children }) => (
                    <li style={{ marginBottom: "6px" }}>{children}</li>
                  ),
                }}
              >
                {result.llm_explanation}
              </ReactMarkdown>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
