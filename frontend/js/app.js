/**
 * HealthPredict Main Application Controller
 * Manages SPA state, reactive form telemetry, counterfactual simulations,
 * XAI attributions, batch processing, and history persistence.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // --- Global Application State ---
  const state = {
    currentView: 'view-dashboard',
    lowThresh: 0.30,
    highThresh: 0.60,
    presets: {},
    userData: {
      patient_name: "Anonymous User",
      age: 46,
      gender: "Male",
      height_cm: 175.0,
      weight_kg: 82.0,
      bmi: 26.8,
      physical_activity: "Moderate (3-4 days/week)",
      sleep_hours: 7.0,
      diet_quality: "Balanced",
      stress_level: "Moderate",
      alcohol_intake: "Occasional / Light",
      is_smoker: false,
      cigs_per_day: 0,
      blood_pressure_sys: 128.0,
      blood_pressure_dia: 84.0,
      resting_heart_rate: 72.0,
      max_heart_rate: 152.0,
      glucose: 105.0,
      cholesterol: 215.0,
      skin_thickness: 24.0,
      insulin: 95.0,
      diabetes_pedigree: 0.45,
      pregnancies: 0,
      chest_pain_type: 0,
      exercise_angina: 0,
      st_depression: 0.6,
      st_slope: 1,
      num_major_vessels: 0,
      thalassemia: 2,
      bp_meds: false,
      has_diabetes: false,
      notes: ""
    },
    latestPredictions: null,
    shapDisease: 'diabetes',
    simDebounceTimer: null,
    batchData: null
  };

  // --- Initialize App ---
  await initApp();

  async function initApp() {
    setupNavigation();
    setupFormInputs();
    setupSimulator();
    setupBatch();
    setupHistory();
    setupSensitivitySliders();
    
    try {
      state.presets = await API.getPresets();
      renderPresetCards();
    } catch (e) {
      console.warn("Could not load presets:", e);
    }

    // Auto-run initial prediction for default state
    await runFullAssessment(false);
    await loadHistoryStats();
    await loadBenchmarks();
  }

  // --- Navigation & View Switching ---
  function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetView = btn.dataset.view;
        switchView(targetView);
      });
    });

    // Dark Mode Toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        document.documentElement.classList.toggle('dark');
        const isDark = document.documentElement.classList.contains('dark');
        themeToggle.innerHTML = isDark ? '<i data-lucide="sun" class="w-5 h-5 text-amber-400"></i>' : '<i data-lucide="moon" class="w-5 h-5 text-slate-600"></i>';
        lucide.createIcons();
      });
    }

    // Print Medical Summary
    const printBtn = document.getElementById('btn-print-report');
    if (printBtn) {
      printBtn.addEventListener('click', () => {
        switchView('view-results');
        setTimeout(() => window.print(), 300);
      });
    }
  }

  function switchView(viewId) {
    state.currentView = viewId;
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));

    const target = document.getElementById(viewId);
    if (target) target.classList.add('active');

    const activeNav = document.querySelector(`.nav-btn[data-view="${viewId}"]`);
    if (activeNav) activeNav.classList.add('active');

    // Trigger chart resizes or specific view loaders
    if (viewId === 'view-results' && state.latestPredictions) {
      renderResultsView(state.latestPredictions);
    } else if (viewId === 'view-shap') {
      loadSHAPExplanation(state.shapDisease);
    } else if (viewId === 'view-history') {
      loadHistoryTable();
    }

    lucide.createIcons();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // --- Form Inputs & Live Telemetry ---
  function setupFormInputs() {
    const form = document.getElementById('assessment-form');
    if (!form) return;

    // Synchronize inputs with state
    const syncInputs = () => {
      state.userData.patient_name = document.getElementById('inp-patient-name')?.value || "Anonymous User";
      state.userData.age = parseInt(document.getElementById('inp-age')?.value) || 45;
      state.userData.gender = document.getElementById('inp-gender')?.value || "Male";
      state.userData.height_cm = parseFloat(document.getElementById('inp-height')?.value) || 170;
      state.userData.weight_kg = parseFloat(document.getElementById('inp-weight')?.value) || 70;
      state.userData.physical_activity = document.getElementById('inp-activity')?.value || "Moderate (3-4 days/week)";
      state.userData.sleep_hours = parseFloat(document.getElementById('inp-sleep')?.value) || 7;
      state.userData.diet_quality = document.getElementById('inp-diet')?.value || "Balanced";
      state.userData.stress_level = document.getElementById('inp-stress')?.value || "Moderate";
      state.userData.alcohol_intake = document.getElementById('inp-alcohol')?.value || "Occasional / Light";
      state.userData.is_smoker = document.getElementById('inp-smoker')?.checked || false;
      state.userData.cigs_per_day = parseInt(document.getElementById('inp-cigs')?.value) || 0;
      state.userData.blood_pressure_sys = parseFloat(document.getElementById('inp-sbp')?.value) || 120;
      state.userData.blood_pressure_dia = parseFloat(document.getElementById('inp-dbp')?.value) || 80;
      state.userData.resting_heart_rate = parseFloat(document.getElementById('inp-rhr')?.value) || 72;
      state.userData.max_heart_rate = parseFloat(document.getElementById('inp-mhr')?.value) || 150;
      state.userData.glucose = parseFloat(document.getElementById('inp-glucose')?.value) || 90;
      state.userData.cholesterol = parseFloat(document.getElementById('inp-chol')?.value) || 200;
      state.userData.skin_thickness = parseFloat(document.getElementById('inp-skin')?.value) || 20;
      state.userData.insulin = parseFloat(document.getElementById('inp-insulin')?.value) || 80;
      state.userData.diabetes_pedigree = parseFloat(document.getElementById('inp-pedigree')?.value) || 0.45;
      state.userData.pregnancies = parseInt(document.getElementById('inp-pregnancies')?.value) || 0;
      state.userData.chest_pain_type = parseInt(document.getElementById('inp-cp')?.value) || 0;
      state.userData.exercise_angina = parseInt(document.getElementById('inp-exang')?.value) || 0;
      state.userData.st_depression = parseFloat(document.getElementById('inp-oldpeak')?.value) || 0.5;
      state.userData.st_slope = parseInt(document.getElementById('inp-slope')?.value) || 1;
      state.userData.num_major_vessels = parseInt(document.getElementById('inp-vessels')?.value) || 0;
      state.userData.thalassemia = parseInt(document.getElementById('inp-thal')?.value) || 2;
      state.userData.bp_meds = document.getElementById('inp-bpmeds')?.checked || false;
      state.userData.has_diabetes = document.getElementById('inp-diabetes-hist')?.checked || false;
      state.userData.notes = document.getElementById('inp-notes')?.value || "";

      updateLiveTelemetry();
    };

    form.querySelectorAll('input, select').forEach(input => {
      input.addEventListener('input', syncInputs);
      input.addEventListener('change', syncInputs);
    });

    // Toggle smoking cigs count visibility
    const smokerCheckbox = document.getElementById('inp-smoker');
    const cigsContainer = document.getElementById('cigs-per-day-container');
    if (smokerCheckbox && cigsContainer) {
      smokerCheckbox.addEventListener('change', () => {
        cigsContainer.style.display = smokerCheckbox.checked ? 'block' : 'none';
      });
    }

    // Run Assessment Button
    const runBtn = document.getElementById('btn-run-assessment');
    if (runBtn) {
      runBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        syncInputs();
        const saveToDb = document.getElementById('inp-save-history')?.checked || true;
        await runFullAssessment(saveToDb);
        switchView('view-results');
      });
    }

    // Initial telemetry calculation
    populateFormWithUserData(state.userData);
    updateLiveTelemetry();
  }

  function updateLiveTelemetry() {
    // 1. BMI
    const h = state.userData.height_cm / 100.0;
    const w = state.userData.weight_kg;
    const bmi = h > 0 ? (w / (h * h)).toFixed(1) : 0;
    state.userData.bmi = parseFloat(bmi);

    const bmiBadge = document.getElementById('telemetry-bmi');
    if (bmiBadge) {
      let cat = 'Normal Weight';
      let color = '#10B981';
      let bg = '#ECFDF5';
      if (bmi < 18.5) { cat = 'Underweight'; color = '#0284C7'; bg = '#E0F2FE'; }
      else if (bmi >= 25 && bmi < 30) { cat = 'Overweight'; color = '#F59E0B'; bg = '#FFFBEB'; }
      else if (bmi >= 30) { cat = 'Obese'; color = '#EF4444'; bg = '#FEF2F2'; }
      
      bmiBadge.innerHTML = `<span style="background:${bg}; color:${color};" class="telemetry-chip">BMI: <b>${bmi} kg/m²</b> (${cat})</span>`;
    }

    // 2. MAP & Pulse Pressure
    const sbp = state.userData.blood_pressure_sys;
    const dbp = state.userData.blood_pressure_dia;
    const map = (dbp + (sbp - dbp) / 3).toFixed(1);
    const pp = (sbp - dbp).toFixed(0);

    const mapBadge = document.getElementById('telemetry-hemodynamics');
    if (mapBadge) {
      mapBadge.innerHTML = `<span class="telemetry-chip bg-slate-100 text-slate-700">MAP: <b>${map} mmHg</b></span> <span class="telemetry-chip bg-slate-100 text-slate-700">Pulse Press: <b>${pp} mmHg</b></span>`;
    }
  }

  function populateFormWithUserData(data) {
    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.value = val !== undefined ? val : '';
    };
    const setCheck = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.checked = Boolean(val);
    };

    setVal('inp-patient-name', data.patient_name);
    setVal('inp-age', data.age);
    setVal('inp-gender', data.gender);
    setVal('inp-height', data.height_cm);
    setVal('inp-weight', data.weight_kg);
    setVal('inp-activity', data.physical_activity);
    setVal('inp-sleep', data.sleep_hours);
    setVal('inp-diet', data.diet_quality);
    setVal('inp-stress', data.stress_level);
    setVal('inp-alcohol', data.alcohol_intake);
    setCheck('inp-smoker', data.is_smoker);
    setVal('inp-cigs', data.cigs_per_day);
    setVal('inp-sbp', data.blood_pressure_sys);
    setVal('inp-dbp', data.blood_pressure_dia);
    setVal('inp-rhr', data.resting_heart_rate);
    setVal('inp-mhr', data.max_heart_rate);
    setVal('inp-glucose', data.glucose);
    setVal('inp-chol', data.cholesterol);
    setVal('inp-skin', data.skin_thickness);
    setVal('inp-insulin', data.insulin);
    setVal('inp-pedigree', data.diabetes_pedigree);
    setVal('inp-pregnancies', data.pregnancies);
    setVal('inp-cp', data.chest_pain_type);
    setVal('inp-exang', data.exercise_angina);
    setVal('inp-oldpeak', data.st_depression);
    setVal('inp-slope', data.st_slope);
    setVal('inp-vessels', data.num_major_vessels);
    setVal('inp-thal', data.thalassemia);
    setCheck('inp-bpmeds', data.bp_meds);
    setCheck('inp-diabetes-hist', data.has_diabetes);
    setVal('inp-notes', data.notes);

    const cigsContainer = document.getElementById('cigs-per-day-container');
    if (cigsContainer) {
      cigsContainer.style.display = data.is_smoker ? 'block' : 'none';
    }

    updateLiveTelemetry();
  }

  // --- Run Full Assessment ---
  async function runFullAssessment(saveToHistory = false) {
    const runBtn = document.getElementById('btn-run-assessment');
    if (runBtn) {
      runBtn.innerHTML = `<i data-lucide="loader-2" class="w-5 h-5 animate-spin"></i> Analyzing Biomarkers...`;
      runBtn.disabled = true;
    }

    try {
      const res = await API.predictAll(state.userData, saveToHistory, state.lowThresh, state.highThresh);
      state.latestPredictions = res;
      renderResultsView(res);
      renderDashboardSummary(res);
      if (saveToHistory) {
        showToast("Assessment saved to patient history!", "success");
        await loadHistoryStats();
      }
    } catch (err) {
      showToast(`Assessment Error: ${err.message}`, "error");
    } finally {
      if (runBtn) {
        runBtn.innerHTML = `<i data-lucide="activity" class="w-5 h-5"></i> Run Comprehensive Risk Screening`;
        runBtn.disabled = false;
        lucide.createIcons();
      }
    }
  }

  // --- Render Results View ---
  function renderResultsView(res) {
    if (!res) return;

    // Overall Risk Header Card
    const overallCard = document.getElementById('overall-risk-banner');
    if (overallCard) {
      overallCard.style.borderColor = res.overall_color;
      overallCard.innerHTML = `
        <div class="flex flex-col md:flex-row items-center justify-between gap-4 p-6 bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-2xl shadow-xl">
          <div>
            <span class="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-white/10 text-slate-200 mb-2">
              <span class="pulse-indicator"></span> Multimodal Risk Composite
            </span>
            <h2 class="text-2xl md:text-3xl font-extrabold tracking-tight">Overall Composite Risk Index: ${res.overall_risk_score}%</h2>
            <p class="text-slate-300 text-sm mt-1 max-w-xl">
              Screening aggregate across metabolic, coronary, and hemodynamic vectors.
            </p>
          </div>
          <div class="flex items-center gap-3">
            <span style="background:${res.overall_color};" class="px-5 py-2.5 rounded-xl font-extrabold text-white text-base shadow-lg">
              ${res.overall_risk_category}
            </span>
          </div>
        </div>
      `;
    }

    // Render Gauges
    ChartEngine.renderGauge('gauge-diabetes', 'Diabetes Risk', res.diabetes.probability_pct, res.diabetes.color, state.lowThresh * 100, state.highThresh * 100);
    ChartEngine.renderGauge('gauge-heart', 'Heart Disease Risk', res.heart.probability_pct, res.heart.color, state.lowThresh * 100, state.highThresh * 100);
    ChartEngine.renderGauge('gauge-hypertension', 'Hypertension Risk', res.hypertension.probability_pct, res.hypertension.color, state.lowThresh * 100, state.highThresh * 100);
    ChartEngine.renderGauge('gauge-overall', 'Composite Score', res.overall_risk_score, res.overall_color, state.lowThresh * 100, state.highThresh * 100);

    // Render Radar Chart
    ChartEngine.renderRadarChart('radar-chart-container', state.userData, res.lifestyle_score);

    // Render Lifestyle Score Box
    const lifeBox = document.getElementById('lifestyle-score-box');
    if (lifeBox && res.lifestyle_score) {
      const ls = res.lifestyle_score;
      let deductionsHtml = '';
      if (ls.deductions && ls.deductions.length > 0) {
        deductionsHtml = ls.deductions.map(d => `
          <div class="flex items-start gap-2 text-xs py-1 border-b border-slate-100 last:border-0">
            <span class="text-red-500 font-bold">${d[1]} pts</span>
            <span class="text-slate-600 font-medium"><b>${d[0]}:</b> ${d[2]}</span>
          </div>
        `).join('');
      } else {
        deductionsHtml = `<p class="text-xs text-emerald-600 font-semibold">No negative lifestyle deductions detected!</p>`;
      }

      lifeBox.innerHTML = `
        <div class="flex items-center justify-between mb-3">
          <div>
            <h4 class="text-sm font-bold text-slate-800">Lifestyle Quality Index</h4>
            <span class="text-xs text-slate-500">${ls.rating}</span>
          </div>
          <span style="background:${ls.badge_bg}; color:${ls.color};" class="text-xl font-extrabold px-3 py-1 rounded-xl">
            ${ls.score} / 100
          </span>
        </div>
        <div class="mt-2 space-y-1">
          ${deductionsHtml}
        </div>
      `;
    }

    // Render Recommendations
    const recoContainer = document.getElementById('recommendations-container');
    if (recoContainer && res.recommendations) {
      recoContainer.innerHTML = res.recommendations.map(r => `
        <div class="p-4 rounded-xl border border-slate-200 bg-white hover:border-slate-300 transition shadow-sm mb-3">
          <div class="flex items-center gap-3 mb-1.5">
            <span class="text-xl">${r.icon}</span>
            <h5 class="text-sm font-bold text-slate-900">${r.title}</h5>
            <span style="background:${r.badge_bg}; color:${r.color};" class="ml-auto text-xs font-semibold px-2.5 py-0.5 rounded-full">
              ${r.domain}
            </span>
          </div>
          <p class="text-xs text-slate-600 leading-relaxed pl-8">
            ${r.text}
          </p>
        </div>
      `).join('');
    }
  }

  // --- Dashboard Summary ---
  function renderDashboardSummary(res) {
    if (!res) return;
    const setVal = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.innerText = val;
    };
    const setBadge = (id, cat, color, bg) => {
      const el = document.getElementById(id);
      if (el) {
        el.innerText = cat;
        el.style.color = color;
        el.style.backgroundColor = bg;
      }
    };

    setVal('dash-val-diabetes', `${res.diabetes.probability_pct}%`);
    setBadge('dash-cat-diabetes', res.diabetes.category, res.diabetes.color, res.diabetes.badge_bg);

    setVal('dash-val-heart', `${res.heart.probability_pct}%`);
    setBadge('dash-cat-heart', res.heart.category, res.heart.color, res.heart.badge_bg);

    setVal('dash-val-hypertension', `${res.hypertension.probability_pct}%`);
    setBadge('dash-cat-hypertension', res.hypertension.category, res.hypertension.color, res.hypertension.badge_bg);

    setVal('dash-val-overall', `${res.overall_risk_score}%`);
    setBadge('dash-cat-overall', res.overall_risk_category, res.overall_color, '#F8FAFC');
  }

  // --- Preset Personas ---
  function renderPresetCards() {
    const container = document.getElementById('presets-grid');
    if (!container || !state.presets) return;

    container.innerHTML = Object.entries(state.presets).map(([key, p]) => `
      <div class="glass-card glass-card-hover p-5 cursor-pointer border border-slate-200" data-preset="${key}">
        <div class="flex items-center justify-between mb-2">
          <h4 class="font-bold text-slate-900 text-sm">${p.name}</h4>
          <span class="text-xs bg-sky-50 text-sky-700 font-semibold px-2 py-0.5 rounded-full">Preset</span>
        </div>
        <p class="text-xs text-slate-500 mb-4 h-10 overflow-hidden">${p.tagline}</p>
        <button class="btn-load-preset w-full text-xs font-bold py-2 px-3 rounded-lg bg-slate-100 hover:bg-sky-600 hover:text-white transition text-slate-700 flex items-center justify-center gap-1.5" data-preset="${key}">
          <i data-lucide="download" class="w-3.5 h-3.5"></i> Load Profile
        </button>
      </div>
    `).join('');

    container.querySelectorAll('.btn-load-preset').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const pKey = btn.dataset.preset;
        loadPreset(pKey);
      });
    });

    lucide.createIcons();
  }

  function loadPreset(pKey) {
    const p = state.presets[pKey];
    if (!p) return;
    state.userData = { ...p.data };
    populateFormWithUserData(state.userData);
    showToast(`Loaded persona: "${p.name}"`, "info");
    runFullAssessment(false);
    switchView('view-assessment');
  }

  // --- Counterfactual Simulator ---
  function setupSimulator() {
    const sliders = [
      'sim-weight-loss',
      'sim-sbp-reduction',
      'sim-glu-reduction',
      'sim-chol-reduction'
    ];
    const toggles = [
      'sim-quit-smoking',
      'sim-boost-activity'
    ];

    const runSimulation = async () => {
      const weightLoss = parseFloat(document.getElementById('sim-weight-loss')?.value) || 0;
      const sbpRed = parseFloat(document.getElementById('sim-sbp-reduction')?.value) || 0;
      const gluRed = parseFloat(document.getElementById('sim-glu-reduction')?.value) || 0;
      const cholRed = parseFloat(document.getElementById('sim-chol-reduction')?.value) || 0;
      const quitSmoke = document.getElementById('sim-quit-smoking')?.checked || false;
      const boostAct = document.getElementById('sim-boost-activity')?.checked || false;

      // Update slider label values
      document.getElementById('val-sim-weight').innerText = `${weightLoss} kg`;
      document.getElementById('val-sim-sbp').innerText = `${sbpRed} mmHg`;
      document.getElementById('val-sim-glu').innerText = `${gluRed} mg/dL`;
      document.getElementById('val-sim-chol').innerText = `${cholRed} mg/dL`;

      const req = {
        base_data: state.userData,
        weight_loss_kg: weightLoss,
        sbp_reduction: sbpRed,
        glucose_reduction: gluRed,
        chol_reduction: cholRed,
        quit_smoking: quitSmoke,
        boost_activity: boostAct
      };

      try {
        const simRes = await API.simulateWhatIf(req, state.lowThresh, state.highThresh);
        renderSimulationResults(simRes);
      } catch (err) {
        console.error("Simulation error:", err);
      }
    };

    const debounceSim = () => {
      clearTimeout(state.simDebounceTimer);
      state.simDebounceTimer = setTimeout(runSimulation, 150);
    };

    sliders.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', debounceSim);
    });
    toggles.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', debounceSim);
    });

    const resetBtn = document.getElementById('btn-reset-sim');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        sliders.forEach(id => { const el = document.getElementById(id); if (el) el.value = 0; });
        toggles.forEach(id => { const el = document.getElementById(id); if (el) el.checked = false; });
        runSimulation();
      });
    }
  }

  function renderSimulationResults(simRes) {
    if (!simRes) return;
    const { base_preds, sim_preds, overall_drop, diabetes_drop, heart_drop, hypertension_drop, simulated_bmi, simulated_sbp, simulated_glucose } = simRes;

    const setDeltaBadge = (id, drop) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (drop > 0) {
        el.innerHTML = `<span class="text-emerald-600 font-bold bg-emerald-50 px-2 py-0.5 rounded text-xs">▼ -${drop}% Risk</span>`;
      } else {
        el.innerHTML = `<span class="text-slate-400 font-medium text-xs">0.0% (No change)</span>`;
      }
    };

    setDeltaBadge('sim-delta-overall', overall_drop);
    setDeltaBadge('sim-delta-diabetes', diabetes_drop);
    setDeltaBadge('sim-delta-heart', heart_drop);
    setDeltaBadge('sim-delta-hypertension', hypertension_drop);

    const projectedBox = document.getElementById('sim-projected-summary');
    if (projectedBox) {
      projectedBox.innerHTML = `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-xs text-slate-500">Simulated BMI</span>
            <h5 class="text-base font-extrabold text-slate-800">${simulated_bmi} kg/m²</h5>
          </div>
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-xs text-slate-500">Simulated SBP</span>
            <h5 class="text-base font-extrabold text-slate-800">${simulated_sbp} mmHg</h5>
          </div>
          <div class="bg-white p-3 rounded-xl border border-slate-200">
            <span class="text-xs text-slate-500">Simulated Glucose</span>
            <h5 class="text-base font-extrabold text-slate-800">${simulated_glucose} mg/dL</h5>
          </div>
          <div class="bg-emerald-50 p-3 rounded-xl border border-emerald-200">
            <span class="text-xs text-emerald-700 font-bold">Total Risk Drop</span>
            <h5 class="text-base font-extrabold text-emerald-700">▼ ${overall_drop}%</h5>
          </div>
        </div>
      `;
    }
  }

  // --- Explainable AI (SHAP Lab) ---
  document.querySelectorAll('.shap-disease-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.shap-disease-tab').forEach(t => t.classList.remove('active', 'bg-sky-600', 'text-white'));
      tab.classList.add('active', 'bg-sky-600', 'text-white');
      state.shapDisease = tab.dataset.disease;
      loadSHAPExplanation(state.shapDisease);
    });
  });

  async function loadSHAPExplanation(disease) {
    const container = document.getElementById('shap-chart-container');
    const narrativeEl = document.getElementById('shap-narrative-box');
    const topIncEl = document.getElementById('shap-top-increasing');
    const topDecEl = document.getElementById('shap-top-decreasing');

    if (container) {
      container.innerHTML = `<div class="p-8 text-center text-slate-400"><i data-lucide="loader-2" class="w-6 h-6 animate-spin mx-auto mb-2"></i>Computing Shapley Attribution Values...</div>`;
      lucide.createIcons();
    }

    try {
      const xaiRes = await API.explainSHAP(disease, state.userData);
      ChartEngine.renderSHAPChart('shap-chart-container', disease, xaiRes.features, xaiRes.shap_values);

      if (narrativeEl) {
        narrativeEl.innerHTML = `<p class="text-sm text-slate-700 leading-relaxed whitespace-pre-line">${xaiRes.narrative.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')}</p>`;
      }

      if (topIncEl) {
        topIncEl.innerHTML = xaiRes.top_increasing.map(item => `
          <div class="flex items-center justify-between p-2.5 rounded-lg bg-red-50 border border-red-100 text-xs mb-2">
            <span class="font-bold text-red-900">${item.feature}</span>
            <span class="font-extrabold text-red-600">+${item.shap_value}</span>
          </div>
        `).join('') || '<p class="text-xs text-slate-400">No significant risk drivers</p>';
      }

      if (topDecEl) {
        topDecEl.innerHTML = xaiRes.top_decreasing.map(item => `
          <div class="flex items-center justify-between p-2.5 rounded-lg bg-emerald-50 border border-emerald-100 text-xs mb-2">
            <span class="font-bold text-emerald-900">${item.feature}</span>
            <span class="font-extrabold text-emerald-600">${item.shap_value}</span>
          </div>
        `).join('') || '<p class="text-xs text-slate-400">No significant protective drivers</p>';
      }
    } catch (err) {
      if (container) container.innerHTML = `<div class="p-4 text-red-500 text-sm">SHAP Error: ${err.message}</div>`;
    }
  }

  // --- Batch Screening ---
  function setupBatch() {
    const dropzone = document.getElementById('batch-dropzone');
    const fileInput = document.getElementById('batch-file-input');
    const downloadTemplateBtn = document.getElementById('btn-download-template');
    const downloadEnrichedBtn = document.getElementById('btn-download-enriched');

    if (downloadTemplateBtn) {
      downloadTemplateBtn.addEventListener('click', () => {
        window.location.href = `${window.location.origin}/api/batch/sample-template`;
      });
    }

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('border-sky-500', 'bg-sky-50'); });
      dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('border-sky-500', 'bg-sky-50'); });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-sky-500', 'bg-sky-50');
        if (e.dataTransfer.files.length > 0) {
          handleBatchUpload(e.dataTransfer.files[0]);
        }
      });
      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
          handleBatchUpload(fileInput.files[0]);
        }
      });
    }

    if (downloadEnrichedBtn) {
      downloadEnrichedBtn.addEventListener('click', () => {
        if (!state.batchData || !state.batchData.enriched_csv) return;
        const blob = new Blob([state.batchData.enriched_csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `healthpredict_screened_${new Date().toISOString().slice(0,10)}.csv`;
        a.click();
      });
    }
  }

  async function handleBatchUpload(file) {
    const statusBox = document.getElementById('batch-status-box');
    const tableContainer = document.getElementById('batch-results-table');
    const downloadEnrichedBtn = document.getElementById('btn-download-enriched');

    if (statusBox) statusBox.innerHTML = `<span class="text-sky-600 font-bold flex items-center gap-2"><i data-lucide="loader-2" class="w-4 h-4 animate-spin"></i> Processing ${file.name}...</span>`;
    lucide.createIcons();

    try {
      const res = await API.uploadBatchCSV(file, state.lowThresh, state.highThresh);
      state.batchData = res;

      if (statusBox) {
        statusBox.innerHTML = `
          <div class="flex flex-wrap items-center gap-4 text-xs font-semibold text-slate-700 bg-sky-50 p-3 rounded-xl border border-sky-200">
            <span>File: <b>${res.filename}</b></span>
            <span>Total Patients: <b>${res.summary.total_patients}</b></span>
            <span class="text-emerald-700">Low Risk: <b>${res.summary.low_risk_count}</b></span>
            <span class="text-amber-700">Moderate Risk: <b>${res.summary.moderate_risk_count}</b></span>
            <span class="text-red-700">High Risk: <b>${res.summary.high_risk_count}</b></span>
            <span>Avg Risk: <b>${res.summary.avg_risk_score}%</b></span>
          </div>
        `;
      }

      if (downloadEnrichedBtn) downloadEnrichedBtn.style.display = 'inline-flex';

      // Render Table
      if (tableContainer) {
        tableContainer.innerHTML = `
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs border-collapse">
              <thead>
                <tr class="bg-slate-100 text-slate-700 font-bold border-b border-slate-200">
                  <th class="p-2.5">#</th>
                  <th class="p-2.5">Patient Name</th>
                  <th class="p-2.5">Age/Sex</th>
                  <th class="p-2.5">BMI</th>
                  <th class="p-2.5">BP (mmHg)</th>
                  <th class="p-2.5">Glucose</th>
                  <th class="p-2.5">Diabetes %</th>
                  <th class="p-2.5">Heart %</th>
                  <th class="p-2.5">Hypertension %</th>
                  <th class="p-2.5">Overall Score</th>
                </tr>
              </thead>
              <tbody>
                ${res.records.map(r => `
                  <tr class="border-b border-slate-100 hover:bg-slate-50 transition">
                    <td class="p-2.5 font-medium text-slate-400">${r.row_index}</td>
                    <td class="p-2.5 font-bold text-slate-800">${r.patient_name}</td>
                    <td class="p-2.5 text-slate-600">${r.age} / ${r.gender}</td>
                    <td class="p-2.5 font-mono">${r.bmi}</td>
                    <td class="p-2.5 font-mono">${r.bp}</td>
                    <td class="p-2.5 font-mono">${r.glucose}</td>
                    <td class="p-2.5"><span class="font-bold">${r.diabetes_pct}%</span></td>
                    <td class="p-2.5"><span class="font-bold">${r.heart_pct}%</span></td>
                    <td class="p-2.5"><span class="font-bold">${r.hypertension_pct}%</span></td>
                    <td class="p-2.5">
                      <span style="background:${r.overall_color};" class="text-white font-extrabold px-2 py-0.5 rounded text-[11px]">
                        ${r.overall_score}% (${r.overall_category})
                      </span>
                    </td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        `;
      }

      showToast(`Batch processing completed for ${res.records.length} patients!`, "success");
    } catch (err) {
      if (statusBox) statusBox.innerHTML = `<span class="text-red-500 font-bold">Error: ${err.message}</span>`;
      showToast(err.message, "error");
    }
  }

  // --- Patient Assessment History ---
  function setupHistory() {
    const searchInp = document.getElementById('history-search');
    const filterSelect = document.getElementById('history-risk-filter');
    const refreshBtn = document.getElementById('btn-refresh-history');

    if (searchInp) {
      searchInp.addEventListener('input', () => {
        clearTimeout(state.historyDebounce);
        state.historyDebounce = setTimeout(loadHistoryTable, 300);
      });
    }

    if (filterSelect) {
      filterSelect.addEventListener('change', loadHistoryTable);
    }

    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        loadHistoryTable();
        loadHistoryStats();
      });
    }
  }

  async function loadHistoryStats() {
    try {
      const stats = await API.getHistoryStats();
      const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.innerText = val;
      };

      setVal('stat-total-assessments', stats.total_assessments);
      setVal('stat-avg-risk', `${stats.avg_overall_risk}%`);
      setVal('stat-high-risk-cnt', stats.high_risk_count);
      setVal('stat-avg-lifestyle', stats.avg_lifestyle_score);

      // Render Longitudinal Trend Chart
      ChartEngine.renderHistoryTrend('history-trend-canvas', stats.recent_timeline);
    } catch (err) {
      console.warn("Could not load history stats:", err);
    }
  }

  async function loadHistoryTable() {
    const search = document.getElementById('history-search')?.value || '';
    const riskFilter = document.getElementById('history-risk-filter')?.value || 'ALL';
    const container = document.getElementById('history-table-container');

    if (container) {
      container.innerHTML = `<div class="p-6 text-center text-slate-400"><i data-lucide="loader-2" class="w-5 h-5 animate-spin mx-auto mb-1"></i>Loading patient records...</div>`;
      lucide.createIcons();
    }

    try {
      const records = await API.getHistory(50, 0, search, riskFilter);
      if (!records || records.length === 0) {
        if (container) container.innerHTML = `<div class="p-8 text-center text-slate-400 text-sm">No assessment records found. Run an assessment to populate the database.</div>`;
        return;
      }

      if (container) {
        container.innerHTML = `
          <div class="overflow-x-auto">
            <table class="w-full text-left text-xs border-collapse">
              <thead>
                <tr class="bg-slate-100 text-slate-700 font-bold border-b border-slate-200">
                  <th class="p-3">ID</th>
                  <th class="p-3">Timestamp</th>
                  <th class="p-3">Patient Name</th>
                  <th class="p-3">Age/Sex</th>
                  <th class="p-3">BMI</th>
                  <th class="p-3">BP / Glucose</th>
                  <th class="p-3">Lifestyle</th>
                  <th class="p-3">Overall Risk</th>
                  <th class="p-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                ${records.map(r => {
                  const dateStr = new Date(r.created_at).toLocaleDateString() + ' ' + new Date(r.created_at).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                  return `
                    <tr class="border-b border-slate-100 hover:bg-slate-50 transition">
                      <td class="p-3 font-mono text-slate-400">#${r.id}</td>
                      <td class="p-3 text-slate-500">${dateStr}</td>
                      <td class="p-3 font-bold text-slate-900">${r.patient_name}</td>
                      <td class="p-3 text-slate-600">${r.age} / ${r.gender}</td>
                      <td class="p-3 font-mono">${r.bmi}</td>
                      <td class="p-3 font-mono">${r.blood_pressure_sys}/${r.blood_pressure_dia} | ${r.glucose}</td>
                      <td class="p-3"><span class="font-bold text-emerald-600">${r.lifestyle_score}</span> <span class="text-slate-400 text-[10px]">(${r.lifestyle_rating})</span></td>
                      <td class="p-3">
                        <span class="px-2.5 py-1 rounded text-white font-extrabold text-[11px]" style="background:${r.overall_risk_score >= 60 ? '#EF4444' : r.overall_risk_score >= 30 ? '#F59E0B' : '#10B981'}">
                          ${r.overall_risk_score}%
                        </span>
                      </td>
                      <td class="p-3 text-right">
                        <button class="btn-load-history-record text-sky-600 hover:text-sky-800 font-bold mr-3" data-id="${r.id}">Load</button>
                        <button class="btn-delete-history-record text-red-500 hover:text-red-700 font-bold" data-id="${r.id}">Delete</button>
                      </td>
                    </tr>
                  `;
                }).join('')}
              </tbody>
            </table>
          </div>
        `;

        container.querySelectorAll('.btn-load-history-record').forEach(b => {
          b.addEventListener('click', () => loadHistoricalRecordIntoForm(parseInt(b.dataset.id)));
        });
        container.querySelectorAll('.btn-delete-history-record').forEach(b => {
          b.addEventListener('click', () => deleteHistoricalRecord(parseInt(b.dataset.id)));
        });
      }
    } catch (err) {
      if (container) container.innerHTML = `<div class="p-4 text-red-500 text-sm">Error: ${err.message}</div>`;
    }
  }

  async function loadHistoricalRecordIntoForm(id) {
    try {
      const record = await API.getAssessmentById(id);
      if (record && record.input_data) {
        state.userData = { ...record.input_data };
        populateFormWithUserData(state.userData);
        showToast(`Loaded assessment #${id} (${record.patient_name})`, "info");
        await runFullAssessment(false);
        switchView('view-assessment');
      }
    } catch (err) {
      showToast(`Error loading record: ${err.message}`, "error");
    }
  }

  async function deleteHistoricalRecord(id) {
    if (!confirm(`Are you sure you want to delete assessment record #${id}?`)) return;
    try {
      await API.deleteAssessment(id);
      showToast(`Record #${id} deleted`, "info");
      await loadHistoryTable();
      await loadHistoryStats();
    } catch (err) {
      showToast(`Error deleting record: ${err.message}`, "error");
    }
  }

  // --- Model Benchmarks ---
  async function loadBenchmarks() {
    const container = document.getElementById('benchmarks-table-container');
    if (!container) return;

    try {
      const data = await API.getBenchmarks();
      if (!data) return;

      const grouped = data.grouped || {};
      const bestModels = data.best_models || {};
      
      let html = '';
      const diseases = ['diabetes', 'heart', 'hypertension'];
      
      for (const disease of diseases) {
        const rows = grouped[disease] || [];
        if (rows.length === 0) continue;
        
        const bestInfo = bestModels[disease] || {};
        const bestName = bestInfo.best_model_name || '';
        const dTitle = disease === 'heart' ? 'CORONARY HEART DISEASE' : disease.toUpperCase();

        html += `
          <div class="mb-6">
            <div class="flex items-center justify-between mb-2">
              <h4 class="font-extrabold text-slate-800 text-sm flex items-center gap-2">
                <span class="w-2.5 h-2.5 rounded-full bg-sky-500"></span> ${dTitle} Screening Models Comparison
              </h4>
              <span class="text-[11px] font-bold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200">
                Selected: <b>${bestName}</b>
              </span>
            </div>
            <div class="overflow-x-auto rounded-xl border border-slate-200">
              <table class="w-full text-left text-xs border-collapse">
                <thead>
                  <tr class="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
                    <th class="p-2.5">Model Algorithm</th>
                    <th class="p-2.5">Accuracy</th>
                    <th class="p-2.5">Precision</th>
                    <th class="p-2.5 text-sky-700">Recall (Sensitivity)</th>
                    <th class="p-2.5">F1-Score</th>
                    <th class="p-2.5 text-indigo-700">ROC-AUC</th>
                    <th class="p-2.5">Status</th>
                  </tr>
                </thead>
                <tbody>
                  ${rows.map(m => {
                    const isSelected = m.model_name === bestName;
                    return `
                      <tr class="border-b border-slate-100 ${isSelected ? 'bg-sky-50/60 font-bold' : ''}">
                        <td class="p-2.5 text-slate-900">${m.model_name}</td>
                        <td class="p-2.5 font-mono">${(m.accuracy || 0).toFixed(3)}</td>
                        <td class="p-2.5 font-mono">${(m.precision || 0).toFixed(3)}</td>
                        <td class="p-2.5 font-mono text-sky-800 font-bold">${(m.recall || 0).toFixed(3)}</td>
                        <td class="p-2.5 font-mono">${(m.f1 || 0).toFixed(3)}</td>
                        <td class="p-2.5 font-mono text-indigo-800 font-bold">${(m.roc_auc || 0).toFixed(3)}</td>
                        <td class="p-2.5">
                          ${isSelected ? '<span class="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded text-[10px] font-extrabold">★ SELECTED</span>' : '<span class="text-slate-400 text-[10px]">Benchmark</span>'}
                        </td>
                      </tr>
                    `;
                  }).join('')}
                </tbody>
              </table>
            </div>
          </div>
        `;
      }

      container.innerHTML = html;
    } catch (err) {
      console.warn("Could not load benchmarks:", err);
    }
  }

  // --- Sensitivity Sliders ---
  function setupSensitivitySliders() {
    const lowSlider = document.getElementById('setting-low-thresh');
    const highSlider = document.getElementById('setting-high-thresh');

    const updateCutoffs = () => {
      state.lowThresh = (parseInt(lowSlider?.value) || 30) / 100.0;
      state.highThresh = (parseInt(highSlider?.value) || 60) / 100.0;
      document.getElementById('val-setting-low').innerText = `${Math.round(state.lowThresh * 100)}%`;
      document.getElementById('val-setting-high').innerText = `${Math.round(state.highThresh * 100)}%`;
      runFullAssessment(false);
    };

    if (lowSlider) lowSlider.addEventListener('input', updateCutoffs);
    if (highSlider) highSlider.addEventListener('input', updateCutoffs);
  }

  // --- Toast Notification Helper ---
  function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    const bg = type === 'success' ? 'bg-emerald-600' : type === 'error' ? 'bg-red-600' : 'bg-slate-900';
    toast.className = `${bg} text-white px-4 py-3 rounded-xl shadow-2xl text-xs font-semibold flex items-center gap-2 pointer-events-auto transition transform translate-y-2 opacity-0`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.remove('translate-y-2', 'opacity-0');
    }, 20);

    setTimeout(() => {
      toast.classList.add('opacity-0', 'translate-y-2');
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }
});

