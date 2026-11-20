/**
 * HealthPredict Visualizations & Charts Engine
 * Renders animated Plotly.js speedometer gauges, biometric radar profiles,
 * SHAP waterfall charts, and Chart.js longitudinal telemetry trends.
 */

class ChartEngine {
  /** Renders an animated Plotly Speedometer Gauge */
  static renderGauge(domId, title, valuePct, riskColor, lowCutoff = 30, highCutoff = 60) {
    const el = document.getElementById(domId);
    if (!el) return;

    const data = [{
      type: "indicator",
      mode: "gauge+number",
      value: valuePct,
      number: {
        suffix: "%",
        font: { size: 26, color: "#0F172A", family: "Plus Jakarta Sans, sans-serif" }
      },
      title: {
        text: `<b>${title}</b>`,
        font: { size: 14, color: "#475569", family: "Plus Jakarta Sans, sans-serif" }
      },
      gauge: {
        axis: {
          range: [0, 100],
          tickwidth: 1,
          tickcolor: "#94A3B8",
          tickvals: [0, lowCutoff, highCutoff, 100],
          ticktext: ['0', `${lowCutoff}%`, `${highCutoff}%`, '100%'],
          tickfont: { size: 10, color: "#64748B" }
        },
        bar: { color: riskColor, thickness: 0.32 },
        bgcolor: "#FFFFFF",
        borderwidth: 1,
        bordercolor: "#E2E8F0",
        steps: [
          { range: [0, lowCutoff], color: "rgba(16, 185, 129, 0.15)" },
          { range: [lowCutoff, highCutoff], color: "rgba(245, 158, 11, 0.15)" },
          { range: [highCutoff, 100], color: "rgba(239, 68, 68, 0.15)" }
        ],
        threshold: {
          line: { color: "#0F172A", width: 3 },
          thickness: 0.75,
          value: valuePct
        }
      }
    }];

    const layout = {
      height: 180,
      margin: { l: 20, r: 20, t: 30, b: 10 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { family: "Plus Jakarta Sans, sans-serif" },
      autosize: true
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(domId, data, layout, config);
  }

  /** Renders 5-Axis Biometric Vulnerability Radar Profile */
  static renderRadarChart(domId, userData, lifeScore) {
    const el = document.getElementById(domId);
    if (!el) return;

    const categories = ['Glycemic Strain', 'Blood Pressure', 'Lipid Strain', 'Body Mass (BMI)', 'Lifestyle Strain'];

    const valGlycemic = Math.min(100, Math.max(10, (((userData.glucose || 90) - 70) / 130) * 100));
    const valBP = Math.min(100, Math.max(10, (((userData.blood_pressure_sys || 120) - 90) / 90) * 100));
    const valChol = Math.min(100, Math.max(10, (((userData.cholesterol || 200) - 140) / 160) * 100));
    const valBMI = Math.min(100, Math.max(10, (((userData.bmi || 24) - 18.5) / 16.5) * 100));
    const valLifestyle = Math.max(10, 100 - (lifeScore?.score || 80));

    const currentValues = [valGlycemic, valBP, valChol, valBMI, valLifestyle];
    const optimalValues = [20, 25, 25, 25, 15];

    const data = [
      {
        type: 'scatterpolar',
        r: [...optimalValues, optimalValues[0]],
        theta: [...categories, categories[0]],
        fill: 'toself',
        fillcolor: 'rgba(16, 185, 129, 0.20)',
        line: { color: '#10B981', width: 1.5, dash: 'dash' },
        name: 'Optimal Target'
      },
      {
        type: 'scatterpolar',
        r: [...currentValues, currentValues[0]],
        theta: [...categories, categories[0]],
        fill: 'toself',
        fillcolor: 'rgba(2, 132, 199, 0.35)',
        line: { color: '#0284C7', width: 2.5 },
        name: 'Your Profile'
      }
    ];

    const layout = {
      polar: {
        radialaxis: { visible: true, range: [0, 100], tickfont: { size: 9, color: '#94A3B8' } },
        angularaxis: { tickfont: { size: 11, color: '#334155', family: 'Plus Jakarta Sans, sans-serif' } }
      },
      showlegend: true,
      legend: { orientation: "h", yanchor: "bottom", y: -0.28, xanchor: "center", x: 0.5 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      height: 320,
      margin: { l: 35, r: 35, t: 25, b: 45 },
      autosize: true
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(domId, data, layout, config);
  }

  /** Renders Interactive SHAP Feature Contribution Waterfall Bar Chart */
  static renderSHAPChart(domId, diseaseName, features, shapValues) {
    const el = document.getElementById(domId);
    if (!el) return;

    // Take top 8 features by absolute magnitude
    const paired = features.map((f, i) => ({
      feature: f,
      shap: shapValues[i] || 0,
      absShap: Math.abs(shapValues[i] || 0)
    }));

    paired.sort((a, b) => b.absShap - a.absShap);
    const topPaired = paired.slice(0, 8).reverse(); // Reverse for top-down display

    const yFeatures = topPaired.map(p => p.feature);
    const xShap = topPaired.map(p => p.shap);
    const colors = topPaired.map(p => p.shap > 0 ? '#EF4444' : '#10B981');
    const textValues = topPaired.map(p => (p.shap > 0 ? `+${p.shap.toFixed(3)}` : p.shap.toFixed(3)));

    const data = [{
      type: 'bar',
      x: xShap,
      y: yFeatures,
      orientation: 'h',
      marker: { color: colors, borderRadius: 4 },
      text: textValues,
      textposition: 'auto',
      hoverinfo: 'x+y'
    }];

    const layout = {
      title: {
        text: `<b>Feature Impact on ${diseaseName.toUpperCase()} (SHAP Log-Odds)</b>`,
        font: { size: 14, color: '#0F172A', family: 'Plus Jakarta Sans, sans-serif' }
      },
      xaxis: {
        title: "Model Output Contribution (Red = Risk Driver, Green = Protective)",
        zeroline: true,
        zerolinewidth: 2,
        zerolinecolor: '#94A3B8',
        tickfont: { size: 10, color: '#64748B' }
      },
      yaxis: {
        tickfont: { size: 11, color: '#334155' }
      },
      margin: { l: 150, r: 20, t: 50, b: 40 },
      height: 340,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      autosize: true
    };

    const config = { responsive: true, displayModeBar: false };
    Plotly.newPlot(domId, data, layout, config);
  }

  /** Renders Longitudinal Risk Trend Chart using Chart.js */
  static renderHistoryTrend(canvasId, timeline) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (window._historyTrendChart) {
      window._historyTrendChart.destroy();
    }

    if (!timeline || timeline.length === 0) {
      return;
    }

    // Sort ascending for chronological trend
    const sorted = [...timeline].reverse();
    const labels = sorted.map((t, idx) => {
      try {
        const d = new Date(t.created_at);
        return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
      } catch (e) {
        return `#${idx + 1}`;
      }
    });

    const overallScores = sorted.map(t => t.overall_risk_score);
    const diabetesScores = sorted.map(t => Math.round((t.diabetes_prob || 0) * 100));
    const heartScores = sorted.map(t => Math.round((t.heart_prob || 0) * 100));
    const hyperScores = sorted.map(t => Math.round((t.hypertension_prob || 0) * 100));

    const ctx = canvas.getContext('2d');
    window._historyTrendChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Overall Composite Risk',
            data: overallScores,
            borderColor: '#0284C7',
            backgroundColor: 'rgba(2, 132, 199, 0.1)',
            borderWidth: 3,
            fill: true,
            tension: 0.3
          },
          {
            label: 'Diabetes Risk',
            data: diabetesScores,
            borderColor: '#7C3AED',
            borderWidth: 1.8,
            borderDash: [5, 5],
            tension: 0.3
          },
          {
            label: 'Heart Disease Risk',
            data: heartScores,
            borderColor: '#DC2626',
            borderWidth: 1.8,
            borderDash: [5, 5],
            tension: 0.3
          },
          {
            label: 'Hypertension Risk',
            data: hyperScores,
            borderColor: '#F59E0B',
            borderWidth: 1.8,
            borderDash: [5, 5],
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { boxWidth: 12, font: { family: 'Plus Jakarta Sans' } } },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          y: {
            min: 0,
            max: 100,
            ticks: { callback: v => `${v}%` },
            grid: { color: 'rgba(226, 232, 240, 0.6)' }
          },
          x: {
            grid: { display: false }
          }
        }
      }
    });
  }
}

window.ChartEngine = ChartEngine;

