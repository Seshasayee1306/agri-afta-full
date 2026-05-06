import React, { useState } from "react";
import { IRRIGATION_API_BASE_URL } from "./api/config";

export default function RetrainConsole() {
  const [formData, setFormData] = useState({
    farmer_id: "",
    region: "",
    reason: "",
  });
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.reason.trim()) {
      alert("Please provide a reason for the retrain request.");
      return;
    }

    setLoading(true);
    setStatus(null);

    try {
      const res = await fetch(`${IRRIGATION_API_BASE_URL}/retrain/request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const data = await res.json();
      if (!res.ok) {
        setStatus({ error: data.error || "Failed to submit request." });
      } else {
        setStatus({
          success: true,
          message: data.message,
          job_name: data.job_name,
        });
        setFormData({ farmer_id: "", region: "", reason: "" });
      }
    } catch (err) {
      console.error(err);
      setStatus({ error: "Network error. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-enter">
      <section className="hero-panel">
        <p className="hero-kicker">System Optimization Portal</p>
        <h2 className="hero-title">Request Model Retraining</h2>
        <p className="hero-copy">
          If you notice that the irrigation predictions are no longer accurate for your specific farm or region, 
          you can request an immediate federated retraining cycle.
        </p>
      </section>

      <div className="input-grid input-grid-two">
        <section className="panel form-panel">
          <div className="panel-heading">
            <h3>Farmer Feedback</h3>
            <p>Tell us why the current model results seem misleading.</p>
          </div>

          <form onSubmit={handleSubmit} className="retrain-form">
            <div className="input-grid">
              <label className="field">
                <span>Farmer ID / Name</span>
                <input
                  type="text"
                  name="farmer_id"
                  value={formData.farmer_id}
                  onChange={handleChange}
                  placeholder="e.g. Farmer John"
                />
              </label>

              <label className="field">
                <span>Region / Farm Location</span>
                <input
                  type="text"
                  name="region"
                  value={formData.region}
                  onChange={handleChange}
                  placeholder="e.g. North Fields"
                />
              </label>
            </div>

            <label className="field" style={{ marginTop: "1rem" }}>
              <span>What seems misleading?</span>
              <textarea
                name="reason"
                value={formData.reason}
                onChange={handleChange}
                placeholder="Describe the issues (e.g., 'Soil moisture is high but it recommends 25L water')"
                rows={5}
                style={{
                    width: '100%',
                    background: '#f8fbfc',
                    border: '1px solid #ccdde4',
                    borderRadius: '11px',
                    color: 'var(--text-strong)',
                    padding: '12px',
                    fontSize: '0.9rem',
                    resize: 'vertical'
                }}
              />
            </label>

            <div style={{ marginTop: "1.5rem" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading}
                style={{ width: "100%" }}
              >
                {loading ? "Submitting Request..." : "Submit Retrain Request"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel info-panel">
          <div className="panel-heading">
            <h3>Retraining Process</h3>
          </div>
          
          <div className="metric-card" style={{ marginBottom: '1rem' }}>
            <p className="metric-label">Job Type</p>
            <p className="metric-value" style={{ fontSize: '1.2rem' }}>Federated AFTA-Correct</p>
          </div>

          <div className="metric-card" style={{ marginBottom: '1rem' }}>
            <p className="metric-label">Execution Environment</p>
            <p className="metric-value" style={{ fontSize: '1.2rem' }}>Kubernetes Cluster</p>
          </div>

          <div className="status-pill info" style={{ padding: '1rem', height: 'auto', display: 'block', lineHeight: '1.4' }}>
            <strong>Note:</strong> Manual retraining bypasses the standard data volume gates and immediately attempts to integrate the latest labeled data from S3.
          </div>

          {status && (
            <div style={{ marginTop: "1.5rem" }}>
              {status.error ? (
                <div className="status-pill danger">{status.error}</div>
              ) : (
                <div className="status-pill success" style={{ background: 'rgba(0,200,100,0.1)', border: '1px solid rgba(0,200,100,0.3)', padding: '1rem', height: 'auto', display: 'block' }}>
                  <p style={{ fontWeight: '600', marginBottom: '0.5rem' }}>✓ {status.message}</p>
                  <p style={{ fontSize: '0.8rem', opacity: 0.8 }}>Kubernetes Job: <code>{status.job_name}</code></p>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
