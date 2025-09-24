import React from "react";
import Plot from "react-plotly.js";

export default function PredictionChart({
  historicalData = [],
  predictedData = [],
  title,
  yLabel,
}) {
  if (!historicalData.length && !predictedData.length) return null;

  // Create connection point between historical and predicted data
  const connectionPoint =
    historicalData.length > 0 && predictedData.length > 0
      ? [
          {
            timestamp: historicalData[historicalData.length - 1].timestamp,
            value: historicalData[historicalData.length - 1].value,
          },
        ]
      : [];

  // Prepare data traces
  const traces = [];

  // Historical data trace
  if (historicalData.length > 0) {
    traces.push({
      x: historicalData.map((d) => d.timestamp),
      y: historicalData.map((d) => d.value),
      type: "scatter",
      mode: "lines",
      name: "Historical Error (Last 12h)",
      line: {
        color: "rgba(255, 255, 255, 0.8)",
        width: 2,
      },
      hovertemplate:
        "<b>Historical</b><br>" +
        "Time: %{x}<br>" +
        "Error: %{y:.4f}" +
        "<extra></extra>",
    });
  }

  // Predicted data trace (with connection to historical)
  if (predictedData.length > 0) {
    const predictionX = [
      ...connectionPoint.map((d) => d.timestamp),
      ...predictedData.map((d) => d.timestamp),
    ];
    const predictionY = [
      ...connectionPoint.map((d) => d.value),
      ...predictedData.map((d) => d.value),
    ];

    traces.push({
      x: predictionX,
      y: predictionY,
      type: "scatter",
      mode: "lines",
      name: "Predicted Error (Next 24h)",
      line: {
        color: "#4ade80",
        width: 3,
        dash: "dashdot",
      },
      fill: "tozeroy",
      fillcolor: "rgba(74, 222, 128, 0.15)",
      hovertemplate:
        "<b>Predicted</b><br>" +
        "Time: %{x}<br>" +
        "Error: %{y:.4f}" +
        "<extra></extra>",
    });
  }

  // Calculate time range for better x-axis formatting
  const allTimes = [
    ...historicalData.map((d) => new Date(d.timestamp)),
    ...predictedData.map((d) => new Date(d.timestamp)),
  ];

  const minTime =
    allTimes.length > 0
      ? Math.min(...allTimes.map((t) => t.getTime()))
      : Date.now();
  const maxTime =
    allTimes.length > 0
      ? Math.max(...allTimes.map((t) => t.getTime()))
      : Date.now();

  // Add vertical line at the transition point
  const transitionShapes = [];
  if (historicalData.length > 0 && predictedData.length > 0) {
    const transitionTime = historicalData[historicalData.length - 1].timestamp;
    transitionShapes.push({
      type: "line",
      x0: transitionTime,
      x1: transitionTime,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: {
        color: "rgba(255, 255, 255, 0.3)",
        width: 2,
        dash: "dot",
      },
    });
  }

  return (
    <div style={{ marginBottom: "20px" }}>
      <Plot
        data={traces}
        layout={{
          title: {
            text: title,
            font: { size: 20, color: "#fff" },
            x: 0.02,
            y: 0.95,
            xanchor: "left",
            yanchor: "top",
          },
          plot_bgcolor: "#18181b",
          paper_bgcolor: "#18181b",
          font: { color: "#fff", size: 12 },
          xaxis: {
            title: {
              text: "Time (UTC)",
              font: { size: 14 },
            },
            showgrid: true,
            gridcolor: "rgba(255, 255, 255, 0.1)",
            gridwidth: 1,
            tickfont: { size: 11 },
            tickformat: "%H:%M\n%d/%m",
            range: [new Date(minTime), new Date(maxTime)],
            type: "date",
          },
          yaxis: {
            title: {
              text: yLabel,
              font: { size: 14 },
            },
            showgrid: true,
            gridcolor: "rgba(255, 255, 255, 0.1)",
            gridwidth: 1,
            tickfont: { size: 11 },
            zeroline: true,
            zerolinecolor: "rgba(255, 255, 255, 0.3)",
            zerolinewidth: 1,
          },
          legend: {
            x: 0.02,
            y: 0.85,
            orientation: "v",
            font: { color: "#fff", size: 12 },
            bgcolor: "rgba(35, 35, 35, 0.8)",
            bordercolor: "#444",
            borderwidth: 1,
            itemsizing: "constant",
            itemwidth: 30,
          },
          margin: { t: 60, l: 60, r: 30, b: 60 },
          height: 350,
          hovermode: "x unified",
          hoverlabel: {
            bgcolor: "rgba(0, 0, 0, 0.8)",
            bordercolor: "#fff",
            font: { color: "#fff", size: 12 },
          },
          shapes: transitionShapes,
          annotations:
            transitionShapes.length > 0
              ? [
                  {
                    x: historicalData[historicalData.length - 1].timestamp,
                    y: 0.95,
                    yref: "paper",
                    text: "Now",
                    showarrow: false,
                    font: {
                      color: "rgba(255, 255, 255, 0.7)",
                      size: 10,
                    },
                    xanchor: "center",
                  },
                ]
              : [],
        }}
        config={{
          displayModeBar: false,
          responsive: true,
        }}
        style={{ width: "100%", height: "350px" }}
        useResizeHandler={true}
      />
    </div>
  );
}
