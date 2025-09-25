import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
import { Button } from "./components/ui/button";
import { Card } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import {
  Satellite,
  Bot,
  AlertTriangle,
  BarChart3,
  X,
  Activity,
} from "lucide-react";
import ThreeGlobe from "./components/ThreeGlobe";
import PredictionChart from "./components/PredictionChart";
import DecryptedText from "./components/DecryptedText";
import "./components/DecryptedText.css";
import videoBackground from "./assets/galaxy3.mp4";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";
const API = `${BACKEND_URL}/api`;

/*
// PROTOTYPE CONFIGURATION (Commented Out)
const USE_MOCK_DATA = true;
const MOCK_DATA_CONFIG = {
  HISTORICAL_HOURS: 12,
  HISTORICAL_INTERVAL_MINUTES: 15,
  PREDICTION_HOURS: 24,
  PREDICTION_INTERVAL_MINUTES: 15,
  ERROR_RANGES: {
    GPS: { clock: [1.0, 2.0], orbit: [2.5, 4.5] },
    GLONASS: { clock: [1.5, 2.5], orbit: [3.5, 5.5] },
    Galileo: { clock: [0.8, 1.6], orbit: [2.0, 3.5] },
    BeiDou: { clock: [1.3, 2.2], orbit: [3.0, 4.8] },
  },
};
*/

function App() {
  const [satellites, setSatellites] = useState([]);
  const [selectedSatellite, setSelectedSatellite] = useState(null);
  const [satelliteSummary, setSatelliteSummary] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [showAIAgent, setShowAIAgent] = useState(false);
  const [agentQuery, setAgentQuery] = useState("");
  const [agentResponse, setAgentResponse] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [constellationFilter, setConstellationFilter] = useState("all");
  const [expandedChart, setExpandedChart] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSatellites();
  }, []);

  const fetchSatellites = async () => {
    try {
      setError(null);
      const response = await axios.get(`${API}/satellites`);
      setSatellites(response.data);
    } catch (error) {
      console.error("Error fetching satellites:", error);
      setError("Could not connect to the mission control server.");
    }
  };

  const fetchSatelliteSummary = async (satelliteId) => {
    try {
      setError(null);
      const response = await axios.get(
        `${API}/satellite/${satelliteId}/summary`
      );
      setSatelliteSummary(response.data);
      setPredictions(response.data.predictions || []);
    } catch (error) {
      console.error("Error fetching satellite data:", error);
      setError(`Failed to fetch data for satellite ${satelliteId}.`);
      setSatelliteSummary(null);
      setPredictions([]);
    }
  };

  /*
  // All mock data generation has been commented out.
  const generateMockData = (satelliteId) => {
    const now = new Date();
    const constellation =
      satelliteId.charAt(0) === "G"
        ? "GPS"
        : satelliteId.charAt(0) === "R"
        ? "GLONASS"
        : satelliteId.charAt(0) === "E"
        ? "Galileo"
        : "BeiDou";

    const errorRange =
      MOCK_DATA_CONFIG.ERROR_RANGES[constellation] ||
      MOCK_DATA_CONFIG.ERROR_RANGES.GPS;

    const historical = [];
    const historicalPoints =
      (MOCK_DATA_CONFIG.HISTORICAL_HOURS * 60) /
      MOCK_DATA_CONFIG.HISTORICAL_INTERVAL_MINUTES;

    for (let i = historicalPoints; i >= 0; i--) {
      const timestamp = new Date(
        now.getTime() -
          i * MOCK_DATA_CONFIG.HISTORICAL_INTERVAL_MINUTES * 60 * 1000
      );

      const trend = Math.sin((i / historicalPoints) * Math.PI * 2) * 0.3;
      const noise = (Math.random() - 0.5) * 0.4;

      const clockError =
        (errorRange.clock[0] + errorRange.clock[1]) / 2 + trend + noise;
      const orbitError =
        (errorRange.orbit[0] + errorRange.orbit[1]) / 2 +
        trend * 2 +
        noise * 1.5;

      historical.push({
        timestamp: timestamp.toISOString(),
        clock_error: Math.max(0, clockError),
        orbit_error: Math.max(0, orbitError),
      });
    }

    const predicted = [];
    const predictionPoints =
      (MOCK_DATA_CONFIG.PREDICTION_HOURS * 60) /
      MOCK_DATA_CONFIG.PREDICTION_INTERVAL_MINUTES;

    const lastHistorical = historical[historical.length - 1];
    let lastClockError = lastHistorical.clock_error;
    let lastOrbitError = lastHistorical.orbit_error;

    for (let i = 1; i <= predictionPoints; i++) {
      const timestamp = new Date(
        now.getTime() +
          i * MOCK_DATA_CONFIG.PREDICTION_INTERVAL_MINUTES * 60 * 1000
      );

      const timeProgression = i / predictionPoints;
      const uncertainty = 1 + timeProgression * 0.5;
      const drift = Math.sin(i * 0.1) * 0.1 * timeProgression;
      const randomWalk = (Math.random() - 0.5) * 0.2 * uncertainty;

      lastClockError += drift + randomWalk * 0.5;
      lastOrbitError += drift * 2 + randomWalk;

      predicted.push({
        timestamp: timestamp.toISOString(),
        pred_clock_error_m: Math.max(0, lastClockError),
        pred_orbit_error_m: Math.max(0, lastOrbitError),
      });
    }

    const clockErrors = historical.map((h) => h.clock_error);
    const orbitErrors = historical.map((h) => h.orbit_error);

    const peakClockError = Math.max(...clockErrors);
    const peakOrbitError = Math.max(...orbitErrors);
    const avgClockError =
      clockErrors.reduce((sum, val) => sum + val, 0) / clockErrors.length;
    const avgOrbitError =
      orbitErrors.reduce((sum, val) => sum + val, 0) / orbitErrors.length;

    // This would need historicalData state to be added back if used
    // setHistoricalData(historical);
    setPredictions(predicted);
    setSatelliteSummary({
      satellite_id: satelliteId,
      constellation: constellation,
      summary: {
        message: "Prototype mock data",
        data_source: "mock",
        peak_clock_error: peakClockError,
        peak_orbit_error: peakOrbitError,
        avg_clock_error: avgClockError,
        avg_orbit_error: avgOrbitError,
        data_points: historical.length,
      },
    });
  };
  */

  const handleSatelliteClick = (satellite) => {
    setSelectedSatellite(satellite);
    fetchSatelliteSummary(satellite.satellite_id);
  };

  const handleCloseRightPanel = () => {
    setSelectedSatellite(null);
    setSatelliteSummary(null);
    setPredictions([]);
  };

  const handleChartExpand = (chartData, chartTitle, yLabel) => {
    setExpandedChart({
      data: chartData,
      title: chartTitle,
      yLabel: yLabel,
    });
  };

  const handleChartClose = () => {
    setExpandedChart(null);
  };

  const handleAIQuery = async () => {
    if (!agentQuery.trim()) return;
    setAgentLoading(true);
    setAgentResponse(null);
    try {
      const response = await axios.post(`${API}/agent/query`, {
        prompt: agentQuery,
        context: {
          satellite_ids: selectedSatellite
            ? [selectedSatellite.satellite_id]
            : [],
        },
      });

      const jobId = response.data.job_id;
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
            setAgentResponse({ error: job.error || "Analysis failed." });
            setAgentLoading(false);
          } else {
            setTimeout(pollResults, 2000);
          }
        } catch (pollError) {
          console.error("Error polling AI results:", pollError);
          setAgentResponse({ error: "Could not retrieve analysis results." });
          setAgentLoading(false);
        }
      };
      setTimeout(pollResults, 500);
    } catch (error) {
      console.error("Error submitting AI query:", error);
      setAgentResponse({ error: "Failed to submit query to AI Agent." });
      setAgentLoading(false);
    }
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

  // This function now renders both charts.
  // The orbit chart will be empty if the API doesn't provide pred_orbit_error_m.
  const renderPredictionCharts = () => {
    if (!predictions.length) return null;

    const clockData = {
      predicted: predictions.map((p) => ({
        timestamp: p.timestamp,
        value: p.pred_clock_error_m,
      })),
    };

    const orbitData = {
      predicted: predictions.map((p) => ({
        timestamp: p.timestamp,
        value: p.pred_orbit_error_m, // This will be undefined if not in API response
      })),
    };

    return (
      <div className="charts-container">
        <div
          className="chart-wrapper"
          onClick={() =>
            handleChartExpand(
              clockData,
              "Clock Error Prediction",
              "Error (nanoseconds)"
            )
          }
        >
          <PredictionChart
            historicalData={[]}
            predictedData={clockData.predicted}
            title="Clock Error Prediction"
            yLabel="Error (nanoseconds)"
            isExpandable={true}
          />
        </div>
        <div
          className="chart-wrapper"
          onClick={() =>
            handleChartExpand(
              orbitData,
              "Orbit Error Prediction",
              "Error (meters)"
            )
          }
        >
          <PredictionChart
            historicalData={[]}
            predictedData={orbitData.predicted}
            title="Orbit Error Prediction"
            yLabel="Error (meters)"
            isExpandable={true}
          />
        </div>
      </div>
    );
  };

  return (
    <div
      className="App"
      style={{ position: "relative", width: "100vw", height: "100vh" }}
    >
      <video
        autoPlay
        loop
        muted
        playsInline
        className="background-video"
        src={videoBackground}
      />

      <header className="header" style={{ position: "relative", zIndex: 10 }}>
        <div className="header-content">
          <div className="logo">
            <div style={{ display: "flex", alignItems: "center" }}>
              <Satellite className="logo-icon" />
              <DecryptedText
                text=" GNSS Mission Control"
                animateOn="view"
                revealDirection="center"
                maxIterations={25}
                encryptedClassName="scramble-text-encrypted"
                className="scramble-text-font"
              />
            </div>
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

      <div className="main-content" style={{ position: "relative", zIndex: 5 }}>
        <div className="globe-container">
          <ThreeGlobe satellites={filteredSatellites} />

          <div
            className="satellite-list"
            style={{
              position: "absolute",
              top: "20px",
              left: "20px",
              zIndex: 10,
              background: "rgba(0, 0, 0, 0.8)",
              backdropFilter: "blur(20px)",
              borderRadius: "12px",
              padding: "16px",
              maxHeight: "70vh",
              overflowY: "auto",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <h3>Active Satellites ({filteredSatellites.length})</h3>
            {error && (
              <div className="error-message-inline">
                <AlertTriangle size={14} /> {error}
              </div>
            )}
            <div className="satellite-list-items">
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
                  <Badge
                    variant="outline"
                    className="constellation-badge"
                    style={{ color: "white" }}
                  >
                    {satellite.constellation}
                  </Badge>
                  <Activity size={12} className="status-icon" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {selectedSatellite && (
          <div
            className="right-panel scrollable"
            style={{ position: "relative", zIndex: 10 }}
          >
            <Button
              onClick={handleCloseRightPanel}
              className="close-panel-button"
              variant="ghost"
              size="sm"
              style={{
                position: "absolute",
                top: "16px",
                right: "16px",
                zIndex: 11,
                backgroundColor: "rgba(0, 0, 0, 0.5)",
                borderRadius: "50%",
                width: "32px",
                height: "32px",
                padding: "0",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <X size={16} />
            </Button>

            <div className="right-panel-content">
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
                      <span className="stat-label">Avg Clock Error</span>
                      <span className="stat-value">
                        {satelliteSummary.summary.avg_clock_error?.toFixed(4) ||
                          "0.0000"}{" "}
                        ns
                      </span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">Peak Orbit Error</span>
                      <span className="stat-value">
                        {satelliteSummary.summary.peak_orbit_error?.toFixed(
                          2
                        ) || "0.00"}{" "}
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
                  <Button
                    onClick={() => setShowAIAgent(true)}
                    variant="outline"
                  >
                    <Bot size={16} />
                    Ask AI Agent
                  </Button>
                  <Button variant="outline">
                    <BarChart3 size={16} />
                    Analyze
                  </Button>
                </div>
              </Card>

              {renderPredictionCharts()}
            </div>
          </div>
        )}
      </div>

      {showAIAgent && (
        <div
          className="ai-dialog-overlay"
          onClick={() => setShowAIAgent(false)}
          style={{ zIndex: 1000 }}
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
                    setAgentQuery("Compare G01 vs R01 orbit error")
                  }
                >
                  Compare satellite errors
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    setAgentQuery("Find satellites with peak orbit error > 2m")
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

      {expandedChart && (
        <div
          className="chart-modal-overlay"
          onClick={handleChartClose}
          style={{ zIndex: 1000 }}
        >
          <div className="chart-modal" onClick={(e) => e.stopPropagation()}>
            <div className="chart-modal-header">
              <h2>
                {expandedChart.title} - {selectedSatellite?.satellite_id}
              </h2>
              <button className="close-button" onClick={handleChartClose}>
                ×
              </button>
            </div>
            <div className="chart-modal-content">
              <PredictionChart
                historicalData={[]}
                predictedData={expandedChart.data.predicted}
                title={expandedChart.title}
                yLabel={expandedChart.yLabel}
                isExpanded={true}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
