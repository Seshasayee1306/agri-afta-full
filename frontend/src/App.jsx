import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Predict from "./Predict";
import TestLLM from "./TestLLM";
import Compare from "./Compare";

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ padding: "10px", fontFamily: "Arial, sans-serif" }}>
        <nav style={{ marginBottom: "20px", display: "flex", gap: "20px" }}>
          <Link to="/" style={{ textDecoration: "none", fontWeight: "bold" }}>Predict</Link>
          <Link to="/llm" style={{ textDecoration: "none", fontWeight: "bold" }}>Test LLM</Link>
          <Link to="/compare" style={{ textDecoration: "none", fontWeight: "bold", color: "#d97706" }}>Compare Accuracy</Link>
        </nav>

        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/llm" element={<TestLLM />} />
          <Route path="/compare" element={<Compare />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
