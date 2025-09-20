import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Dialog } from "./components/ui/dialog";
import {
  Satellite,
  Globe,
  Activity,
  Bot,
  TrendingUp,
  AlertTriangle,
  BarChart3,
} from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);

  const [satellites, setSatellites] = useState([]);
  const [selectedSatellite, setSelectedSatellite] = useState(null);
  const [satelliteSummary, setSatelliteSummary] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [showAIAgent, setShowAIAgent] = useState(false);
  const [agentQuery, setAgentQuery] = useState("");
  const [agentResponse, setAgentResponse] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [constellationFilter, setConstellationFilter] = useState("all");

  // Initialize 2D Canvas Globe (lighter than Three.js)
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = 800;
    canvas.height = 600;

    let animationTime = 0;

    const drawGlobe = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Earth
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = 180;

      // Earth gradient
      const earthGradient = ctx.createRadialGradient(
        centerX - 50,
        centerY - 50,
        0,
        centerX,
        centerY,
        radius
      );
      earthGradient.addColorStop(0, "#4a90e2");
      earthGradient.addColorStop(0.7, "#2563eb");
      earthGradient.addColorStop(1, "#1e40af");

      ctx.fillStyle = earthGradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw grid lines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.lineWidth = 1;

      // Longitude lines
      for (let i = 0; i < 12; i++) {
        const angle = (i * Math.PI) / 6;
        const radiusX = Math.abs(radius * Math.cos(angle));
        if (radiusX > 0) {
          ctx.beginPath();
          ctx.ellipse(centerX, centerY, radiusX, radius, 0, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // Latitude lines
      for (let i = 1; i < 6; i++) {
        const y = (radius * i) / 6;
        const radiusX = Math.abs(
          radius * Math.sin(Math.acos(Math.min(y / radius, 1)))
        );
        if (radiusX > 0) {
          ctx.beginPath();
          ctx.ellipse(
            centerX,
            centerY - y,
            radiusX,
            radius * 0.3,
            0,
            0,
            Math.PI * 2
          );
          ctx.stroke();
          ctx.beginPath();
          ctx.ellipse(
            centerX,
            centerY + y,
            radiusX,
            radius * 0.3,
            0,
            0,
            Math.PI * 2
          );
          ctx.stroke();
        }
      }

      // Draw satellites
      satellites.forEach((satellite, index) => {
        if (
          constellationFilter !== "all" &&
          satellite.constellation !== constellationFilter
        )
          return;

        const angle =
          (index / satellites.length) * Math.PI * 2 + animationTime * 0.01;
        const orbitRadius = radius + 80 + (index % 3) * 30;
        const x = centerX + Math.cos(angle) * orbitRadius;
        const y = centerY + Math.sin(angle) * orbitRadius * 0.6;

        // Draw orbit
        ctx.strokeStyle = `rgba(${getConstellationColorRGB(
          satellite.constellation
        )}, 0.3)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(
          centerX,
          centerY,
          orbitRadius,
          orbitRadius * 0.6,
          0,
          0,
          Math.PI * 2
        );
        ctx.stroke();

        // Draw satellite
        ctx.fillStyle = getConstellationColorHex(satellite.constellation);
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw satellite ID
        ctx.fillStyle = "white";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(satellite.satellite_id, x, y - 15);

        // Store click area for interaction
        satellite._clickArea = { x, y, radius: 15 };
      });

      animationTime++;
      animationRef.current = requestAnimationFrame(drawGlobe);
    };

    drawGlobe();

    // Handle canvas clicks
    const handleCanvasClick = (event) => {
      const rect = canvas.getBoundingClientRect();
      const clickX = event.clientX - rect.left;
      const clickY = event.clientY - rect.top;

      satellites.forEach((satellite) => {
        if (satellite._clickArea) {
          const distance = Math.sqrt(
            Math.pow(clickX - satellite._clickArea.x, 2) +
              Math.pow(clickY - satellite._clickArea.y, 2)
          );
          if (distance <= satellite._clickArea.radius) {
            handleSatelliteClick(satellite);
          }
        }
      });
    };

    canvas.addEventListener("click", handleCanvasClick);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      canvas.removeEventListener("click", handleCanvasClick);
    };
  }, [satellites, constellationFilter]);

  // Fetch satellites data
  useEffect(() => {
    fetchSatellites();
  }, []);

  const fetchSatellites = async () => {
    try {
      const response = await axios.get(`${API}/satellites`);
      setSatellites(response.data);
    } catch (error) {
      console.error("Error fetching satellites:", error);
    }
  };

  const fetchSatelliteSummary = async (satelliteId) => {
    try {
      const response = await axios.get(
        `${API}/satellite/${satelliteId}/summary`
      );
      setSatelliteSummary(response.data);
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error("Error fetching satellite summary:", error);
    }
  };

  const handleSatelliteClick = (satellite) => {
    setSelectedSatellite(satellite);
    fetchSatelliteSummary(satellite.satellite_id);
  };

  const handleAIQuery = async () => {
    if (!agentQuery.trim()) return;

    setAgentLoading(true);
    try {
      // Submit query
      const response = await axios.post(`${API}/agent/query`, {
        prompt: agentQuery,
        context: {
          satellite_ids: selectedSatellite
            ? [selectedSatellite.satellite_id]
            : [],
        },
      });

      const jobId = response.data.job_id;

      // Poll for results
      const pollResults = async () => {
        try {
          const resultResponse = await axios.get(
            `${API}/agent/result/${jobId}`
          );
          const job = resultResponse.data;

          if (job.status === "completed") {
            setAgentResponse(job.results);
            setAgentLoading(false);
          } else if (job.status === "failed") {
            setAgentResponse({ error: job.error });
            setAgentLoading(false);
          } else {
            setTimeout(pollResults, 2000);
          }
        } catch (error) {
          console.error("Error polling results:", error);
          setAgentLoading(false);
        }
      };

      setTimeout(pollResults, 1000);
    } catch (error) {
      console.error("Error submitting AI query:", error);
      setAgentLoading(false);
    }
  };

  const getConstellationColorRGB = (constellation) => {
    const colors = {
      GPS: "34, 197, 94", // Green
      GLONASS: "239, 68, 68", // Red
      Galileo: "59, 130, 246", // Blue
      BeiDou: "234, 179, 8", // Yellow
    };
    return colors[constellation] || "156, 163, 175";
  };

  const getConstellationColorHex = (constellation) => {
    const colors = {
      GPS: "#22c55e",
      GLONASS: "#ef4444",
      Galileo: "#3b82f6",
      BeiDou: "#eab308",
    };
    return colors[constellation] || "#9ca3af";
  };

  const filteredSatellites =
    constellationFilter === "all"
      ? satellites
      : satellites.filter((sat) => sat.constellation === constellationFilter);

  const renderSimpleChart = (data, title, color) => {
    if (!data.length) return null;

    const maxValue = Math.max(...data.map((d) => d.value));
    const chartHeight = 200;

    return (
      <div className="simple-chart">
        <h4>{title}</h4>
        <div className="chart-container" style={{ height: chartHeight }}>
          <svg width="100%" height={chartHeight}>
            {data.map((point, index) => {
              const x = (index / (data.length - 1)) * 100;
              const y =
                chartHeight - (point.value / maxValue) * (chartHeight - 20);
              return (
                <g key={index}>
                  <circle cx={`${x}%`} cy={y} r="3" fill={color} />
                  {index > 0 && (
                    <line
                      x1={`${((index - 1) / (data.length - 1)) * 100}%`}
                      y1={
                        chartHeight -
                        (data[index - 1].value / maxValue) * (chartHeight - 20)
                      }
                      x2={`${x}%`}
                      y2={y}
                      stroke={color}
                      strokeWidth="2"
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  const renderPredictionCharts = () => {
    if (!predictions.length) return null;

    const clockData = predictions.map((p, i) => ({
      index: i,
      value: p.pred_clock_error_m,
      time: new Date(p.timestamp).toLocaleTimeString(),
    }));

    const orbitData = predictions.map((p, i) => ({
      index: i,
      value: p.pred_orbit_error_m,
      time: new Date(p.timestamp).toLocaleTimeString(),
    }));

    return (
      <div className="charts-container">
        {renderSimpleChart(clockData, "Clock Error (m)", "#3b82f6")}
        {renderSimpleChart(orbitData, "Orbit Error (m)", "#ef4444")}
      </div>
    );
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Satellite className="logo-icon" />
            <h1>GNSS Mission Control</h1>
          </div>
          <div className="header-controls">
            <select
              value={constellationFilter}
              onChange={(e) => setConstellationFilter(e.target.value)}
              className="constellation-filter"
            >
              <option value="all">All Constellations</option>
              <option value="GPS">GPS</option>
              <option value="GLONASS">GLONASS</option>
              <option value="Galileo">Galileo</option>
              <option value="BeiDou">BeiDou</option>
            </select>
            <Button onClick={() => setShowAIAgent(true)} className="ai-button">
              <Bot size={16} />
              AI Agent
            </Button>
          </div>
        </div>
      </header>

      <div className="main-content">
        {/* 2D Globe Canvas */}
        <div className="globe-container">
          <canvas ref={canvasRef} className="globe-canvas" />

          {/* Satellite List Overlay */}
          <div className="satellite-list">
            <h3>Active Satellites ({filteredSatellites.length})</h3>
            {filteredSatellites.map((satellite) => (
              <div
                key={satellite.satellite_id}
                className={`satellite-item ${
                  selectedSatellite?.satellite_id === satellite.satellite_id
                    ? "selected"
                    : ""
                }`}
                onClick={() => handleSatelliteClick(satellite)}
              >
                <div
                  className="constellation-dot"
                  style={{
                    backgroundColor: getConstellationColorHex(
                      satellite.constellation
                    ),
                  }}
                />
                <span className="satellite-id">{satellite.satellite_id}</span>
                <Badge variant="outline" className="constellation-badge">
                  {satellite.constellation}
                </Badge>
                <Activity size={12} className="status-icon" />
              </div>
            ))}
          </div>
        </div>

        {/* Right Panel */}
        {selectedSatellite && (
          <div className="right-panel">
            <Card className="satellite-card">
              <div className="satellite-header">
                <h2>{selectedSatellite.satellite_id}</h2>
                <Badge
                  variant="secondary"
                  style={{
                    backgroundColor: `${getConstellationColorHex(
                      selectedSatellite.constellation
                    )}20`,
                    color: getConstellationColorHex(
                      selectedSatellite.constellation
                    ),
                  }}
                >
                  {selectedSatellite.constellation}
                </Badge>
              </div>

              {satelliteSummary && (
                <div className="satellite-stats">
                  <div className="stat-item">
                    <span className="stat-label">Peak Clock Error</span>
                    <span className="stat-value">
                      {satelliteSummary.summary.peak_clock_error?.toFixed(4) ||
                        "0.0000"}
                      m
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Peak Orbit Error</span>
                    <span className="stat-value">
                      {satelliteSummary.summary.peak_orbit_error?.toFixed(2) ||
                        "0.00"}
                      m
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Avg Clock Error</span>
                    <span className="stat-value">
                      {satelliteSummary.summary.avg_clock_error?.toFixed(4) ||
                        "0.0000"}
                      m
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Data Points</span>
                    <span className="stat-value">
                      {satelliteSummary.summary.data_points || 0}
                    </span>
                  </div>
                </div>
              )}

              <div className="action-buttons">
                <Button onClick={() => setShowAIAgent(true)} variant="outline">
                  <Bot size={16} />
                  Ask AI Agent
                </Button>
                <Button variant="outline">
                  <BarChart3 size={16} />
                  Analyze
                </Button>
              </div>
            </Card>

            {/* Charts */}
            {renderPredictionCharts()}
          </div>
        )}
      </div>

      {/* AI Agent Dialog */}
      {showAIAgent && (
        <div
          className="ai-dialog-overlay"
          onClick={() => setShowAIAgent(false)}
        >
          <div className="ai-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="ai-dialog-header">
              <h2>
                <Bot size={20} /> AI Agent Analysis
              </h2>
              <button
                className="close-button"
                onClick={() => setShowAIAgent(false)}
              >
                ×
              </button>
            </div>

            <div className="ai-examples">
              <h4>Example Queries:</h4>
              <div className="example-buttons">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setAgentQuery("Compare G01 vs R01 clock error volatility")
                  }
                >
                  Compare satellites volatility
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setAgentQuery("Find satellites with orbit error >1m")
                  }
                >
                  Find high orbit errors
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setAgentQuery(
                      "Analyze GPS constellation performance trends"
                    )
                  }
                >
                  GPS performance analysis
                </Button>
              </div>
            </div>

            <div className="ai-input">
              <Input
                placeholder="Ask me about satellite performance, errors, comparisons..."
                value={agentQuery}
                onChange={(e) => setAgentQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleAIQuery()}
              />
              <Button onClick={handleAIQuery} disabled={agentLoading}>
                {agentLoading ? "Analyzing..." : "Analyze"}
              </Button>
            </div>

            {agentResponse && (
              <div className="ai-response">
                <h4>Analysis Results:</h4>
                {agentResponse.error ? (
                  <div className="error-message">
                    <AlertTriangle size={16} />
                    {agentResponse.error}
                  </div>
                ) : (
                  <div>
                    <p className="response-text">
                      {agentResponse.text_summary}
                    </p>

                    {agentResponse.charts &&
                      agentResponse.charts.length > 0 && (
                        <div className="response-charts">
                          {agentResponse.charts.map((chart, index) => (
                            <div key={index} className="chart-result">
                              <h5>{chart.name}</h5>
                              <img
                                src={`${BACKEND_URL}${chart.url}`}
                                alt={chart.name}
                              />
                            </div>
                          ))}
                        </div>
                      )}

                    {agentResponse.statistics && (
                      <div className="response-stats">
                        <h5>Key Statistics:</h5>
                        <pre>
                          {JSON.stringify(agentResponse.statistics, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
