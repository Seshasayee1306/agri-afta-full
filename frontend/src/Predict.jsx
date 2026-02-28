import React, { useState } from "react";

export default function Predict() {
  const [sensorFeatures, setSensorFeatures] = useState("");
  const [sowingDate, setSowingDate] = useState("");
  const [currentDate, setCurrentDate] = useState("");

  const [soilMoisture, setSoilMoisture] = useState("");
  const [temperature, setTemperature] = useState("");
  const [rainfall, setRainfall] = useState("");

  const [region, setRegion] = useState("");
  const [cropType, setCropType] = useState("");
  const [ndvi, setNdvi] = useState("0.5");
  const [diseaseStatus, setDiseaseStatus] = useState("None");
  const [humidity, setHumidity] = useState("");

  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const arr = sensorFeatures.split(",").map(x => parseFloat(x.trim()));

    if (arr.length !== 12 || arr.some(isNaN)) {
      alert("Enter exactly 12 valid sensor values");
      return;
    }

    const payload = {
      sowing_date: sowingDate,
      current_date: currentDate,
      soil_moisture: Number(soilMoisture),
      temperature: Number(temperature),
      rainfall: Number(rainfall),

      soil_humidity: 50,
      air_temp: Number(temperature),
      air_humidity: Number(humidity),
      ph: 6.5,
      nitrogen: 40,
      phosphorus: 30,
      potassium: 50,

      sensor_features: arr,

      context: {
        region,
        crop_type: cropType,
        ndvi: Number(ndvi),
        disease_status: diseaseStatus,
        temperature: Number(temperature),
        rainfall: Number(rainfall),
        humidity: Number(humidity)
      }
    };

    try {
      const res = await fetch("http://127.0.0.1:8000/predict_full_intelligent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    }
  };

  return (
    <div style={{ maxWidth: "900px", margin: "40px auto", fontFamily: "Inter" }}>
      <h2>🌾 Full Intelligent Irrigation System</h2>

      <h3>📡 Sensor Input</h3>
      <input
        type="text"
        placeholder="Enter 12 sensor values comma separated"
        value={sensorFeatures}
        onChange={(e) => setSensorFeatures(e.target.value)}
        style={{ width: "100%", padding: "10px", marginBottom: "10px" }}
      />

      <h3>📅 Crop Stage</h3>
      <input type="date" value={sowingDate} onChange={e => setSowingDate(e.target.value)} />
      <input type="date" value={currentDate} onChange={e => setCurrentDate(e.target.value)} style={{ marginLeft: "10px" }} />

      <h3>🌡 Environmental Data</h3>
      <input type="number" placeholder="Soil Moisture" value={soilMoisture} onChange={e => setSoilMoisture(e.target.value)} />
      <input type="number" placeholder="Temperature" value={temperature} onChange={e => setTemperature(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="number" placeholder="Rainfall" value={rainfall} onChange={e => setRainfall(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="number" placeholder="Humidity" value={humidity} onChange={e => setHumidity(e.target.value)} style={{ marginLeft: "10px" }} />

      <h3>🌍 Context Data</h3>
      <input type="text" placeholder="Region" value={region} onChange={e => setRegion(e.target.value)} />
      <input type="text" placeholder="Crop Type" value={cropType} onChange={e => setCropType(e.target.value)} style={{ marginLeft: "10px" }} />
      <input type="number" step="0.01" placeholder="NDVI" value={ndvi} onChange={e => setNdvi(e.target.value)} style={{ marginLeft: "10px" }} />

      <select value={diseaseStatus} onChange={e => setDiseaseStatus(e.target.value)} style={{ marginLeft: "10px" }}>
        <option value="None">No Disease</option>
        <option value="Mild">Mild Disease</option>
        <option value="Severe">Severe Disease</option>
      </select>

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
          <p><strong>AFTA Model:</strong> {result.afta_prediction}</p>
          <p><strong>Context Score:</strong> {result.context_score}</p>
          <p><strong>Stress Index:</strong> {result.stress_index}</p>

          <h2>
            {result.final_prediction === 1
              ? "💧 Irrigation Required"
              : "✅ No Irrigation Needed"}
          </h2>

          <h3>
            💦 Recommended Water: {result.recommended_water_liters} Liters
          </h3>
        </div>
      )}
    </div>
  );
}