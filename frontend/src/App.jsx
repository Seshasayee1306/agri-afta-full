import React from "react";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import Predict from "./Predict";
import TestLLM from "./TestLLM";
import ComparePage from "./ComparePage";

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
              to="/compare"
              className={({ isActive }) =>
                `topnav-link ${isActive ? "topnav-link-active" : ""}`
              }
            >
              Model Compare
            </NavLink>
          </nav>
        </header>

        <main className="page-wrap">
          <Routes>
            <Route path="/" element={<Predict />} />
            <Route path="/llm" element={<TestLLM />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
