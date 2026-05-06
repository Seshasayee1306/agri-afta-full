import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { IRRIGATION_API_BASE_URL } from "./api/config";

export default function TestLLM() {
  const [payloadText, setPayloadText] = useState(() => {
    try {
      return localStorage.getItem("last_predict_payload") || "";
    } catch {
      return "";
    }
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleExplain = async () => {
    if (!payloadText) return;

    let payload;
    try {
      payload = JSON.parse(payloadText);
    } catch {
      setResult({ error: "Payload must be valid JSON (copied from Prediction Console)" });
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(IRRIGATION_API_BASE_URL + "/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Failed to fetch explanation" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-enter">
      <section className="hero-panel hero-panel-compact">
        <p className="hero-kicker">Model Explainability Workspace</p>
        <h2 className="hero-title">LLM Explanation Inspector</h2>
        <p className="hero-copy">
          Paste payload JSON from the Prediction Console and inspect explanation output with raw API response.
        </p>
      </section>

      <section className="panel form-panel">
        <div className="panel-heading">
          <h3>Input Payload</h3>
          <p>Auto-filled from local storage when a prediction has already been run.</p>
        </div>

        <textarea
          className="payload-textarea"
          placeholder="Paste JSON payload here"
          value={payloadText}
          onChange={(e) => setPayloadText(e.target.value)}
          rows={10}
        />

        <div className="action-row left">
          <button onClick={handleExplain} className="btn btn-primary" disabled={loading}>
            {loading ? "Running Explanation..." : "Run Explanation"}
          </button>
        </div>
      </section>

      {result && (
        <section className="panel result-panel">
          <div className="panel-heading">
            <h3>Explanation Response</h3>
            <p>Raw response plus formatted markdown explanation.</p>
          </div>

          <pre className="code-box">{JSON.stringify(result, null, 2)}</pre>

          {result.llm_explanation && (
            <div className="markdown-card">
              <h4>LLM Narrative</h4>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.llm_explanation}
              </ReactMarkdown>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
