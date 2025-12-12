import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import Predict from "./Predict";
import TestLLM from "./TestLLM";

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ padding: "10px", fontFamily: "Arial, sans-serif" }}>
        <nav style={{ marginBottom: "20px" }}>
          <Link to="/" style={{ marginRight: "10px" }}>Predict</Link>
          <Link to="/llm">Test LLM</Link>
        </nav>

        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/llm" element={<TestLLM />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
