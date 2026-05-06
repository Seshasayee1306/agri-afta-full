import React, { useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { DISEASE_API_BASE_URL } from "./api/config";

export default function DiseasePredict() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const overlaySrc = useMemo(() => {
    if (!result?.overlay_base64) return "";
    return `data:image/png;base64,${result.overlay_base64}`;
  }, [result]);

  const maskSrc = useMemo(() => {
    if (!result?.mask_base64) return "";
    return `data:image/png;base64,${result.mask_base64}`;
  }, [result]);

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setResult(null);

    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result || "");
    reader.readAsDataURL(file);
  };

  const runDiseasePredict = async () => {
    if (!selectedFile) {
      alert("Please upload an image first.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);

      const res = await fetch(DISEASE_API_BASE_URL + "/predict_disease", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        setResult({ error: data?.error || "Disease prediction failed" });
        return;
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: "Disease prediction failed" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-enter">
      <section className="hero-panel">
        <p className="hero-kicker">Plant Health Vision Suite</p>
        <h2 className="hero-title">Disease Segmentation Workflow</h2>
        <p className="hero-copy">
          Upload a leaf image, detect likely diseased regions, and get a plain-language LLM explanation for why the disease label was chosen.
        </p>
      </section>

      <section className="panel form-panel">
        <div className="panel-heading">
          <h3>Leaf Image Input</h3>
          <p>Keep this page style aligned with your existing console while running a separate disease microservice.</p>
        </div>

        <div className="input-grid input-grid-two">
          <label className="field">
            <span>Upload Plant Image</span>
            <input type="file" accept="image/*" onChange={onFileChange} />
          </label>
          <div className="field">
            <span>Action</span>
            <button onClick={runDiseasePredict} className="btn btn-primary" disabled={loading}>
              {loading ? "Analyzing Image..." : "Run Disease Prediction"}
            </button>
          </div>
        </div>
      </section>

      {previewUrl && (
        <section className="panel result-panel">
          <div className="panel-heading">
            <h3>Uploaded Image</h3>
          </div>
          <img src={previewUrl} alt="Uploaded leaf" className="disease-image" />
        </section>
      )}

      {result && (
        <section className="panel result-panel">
          <div className="panel-heading">
            <h3>Disease Analysis Output</h3>
          </div>

          {result.error ? (
            <div className="status-pill danger">{result.error}</div>
          ) : (
            <>
              <div className="metrics-grid">
                <div className="metric-card">
                  <p className="metric-label">Disease Class</p>
                  <p className="metric-value">{result.disease_class}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Severity</p>
                  <p className="metric-value">{result.severity}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Confidence</p>
                  <p className="metric-value">{result.confidence}</p>
                </div>
                <div className="metric-card">
                  <p className="metric-label">Coverage (%)</p>
                  <p className="metric-value">{result.disease_coverage_percent}</p>
                </div>
              </div>

              <div className="input-grid input-grid-two">
                <div>
                  <h4>Overlay</h4>
                  {overlaySrc && <img src={overlaySrc} alt="Disease overlay" className="disease-image" />}
                </div>
                <div>
                  <h4>Mask</h4>
                  {maskSrc && <img src={maskSrc} alt="Disease mask" className="disease-image" />}
                </div>
              </div>

              <div className="llm-panel">
                <div className="panel-heading">
                  <h4>LLM Disease Explanation</h4>
                  <p>Why the model likely arrived at this disease from the image pattern.</p>
                </div>
                <div className="markdown-card">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {result.llm_explanation || "No explanation available."}
                  </ReactMarkdown>
                </div>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}
