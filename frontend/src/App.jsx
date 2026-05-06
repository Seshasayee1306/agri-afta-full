import React from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import Predict from "./Predict";
import TestLLM from "./TestLLM";
import DiseasePredict from "./DiseasePredict";
import RetrainConsole from "./RetrainConsole";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <div className="backdrop backdrop-one" />
        <div className="backdrop backdrop-two" />

        <header className="topbar">
          <div className="brand">
            <div className="brand-mark" aria-hidden="true">
              AF
            </div>
            <div>
              <p className="brand-eyebrow">Irrigation Intelligence</p>
              <h1 className="brand-title">AgriFlow Control Center</h1>
            </div>
          </div>

          <nav className="topnav" aria-label="Primary navigation">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `topnav-link ${isActive ? "topnav-link-active" : ""}`
              }
              end
            >
              Prediction Console
            </NavLink>
            <NavLink
              to="/llm"
              className={({ isActive }) =>
                `topnav-link ${isActive ? "topnav-link-active" : ""}`
              }
            >
              LLM Inspector
            </NavLink>
            <NavLink
              to="/disease"
              className={({ isActive }) =>
                `topnav-link ${isActive ? "topnav-link-active" : ""}`
              }
            >
              Disease Console
            </NavLink>
            <NavLink
              to="/retrain"
              className={({ isActive }) =>
                `topnav-link ${isActive ? "topnav-link-active" : ""}`
              }
            >
              Retrain System
            </NavLink>
          </nav>
        </header>

        <main className="page-wrap">
          <Routes>
            <Route path="/" element={<Predict />} />
            <Route path="/llm" element={<TestLLM />} />
            <Route path="/disease" element={<DiseasePredict />} />
            <Route path="/retrain" element={<RetrainConsole />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
