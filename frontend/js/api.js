/**
 * HealthPredict API Client Module
 * Provides asynchronous communication methods with the FastAPI REST backend.
 */

const API_BASE = window.location.origin;

class HealthPredictAPI {
  /** Helper for JSON requests */
  static async request(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    };

    const config = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers
      }
    };

    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Request failed with status ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error(`API Error on ${endpoint}:`, err);
      throw err;
    }
  }

  // --- System & Presets ---
  static async getHealth() {
    return this.request('/api/health');
  }

  static async getPresets() {
    return this.request('/api/predict/presets');
  }

  // --- Predictions ---
  static async predictAll(payload, saveToHistory = false, lowThresh = 0.30, highThresh = 0.60) {
    const query = new URLSearchParams({
      save_to_history: saveToHistory,
      low_thresh: lowThresh,
      high_thresh: highThresh
    });
    return this.request(`/api/predict/all?${query}`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  static async predictSingle(disease, payload, lowThresh = 0.30, highThresh = 0.60) {
    const query = new URLSearchParams({ low_thresh: lowThresh, high_thresh: highThresh });
    return this.request(`/api/predict/${disease}?${query}`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
  }

  // --- Explainability (SHAP) ---
  static async explainSHAP(disease, inputData) {
    return this.request('/api/explain/shap', {
      method: 'POST',
      body: JSON.stringify({
        disease: disease,
        input_data: inputData
      })
    });
  }

  // --- Counterfactual Simulator ---
  static async simulateWhatIf(interventionRequest, lowThresh = 0.30, highThresh = 0.60) {
    const query = new URLSearchParams({ low_thresh: lowThresh, high_thresh: highThresh });
    return this.request(`/api/simulate/what-if?${query}`, {
      method: 'POST',
      body: JSON.stringify(interventionRequest)
    });
  }

  // --- Patient Assessment History ---
  static async getHistory(limit = 50, offset = 0, search = '', riskFilter = 'ALL') {
    const query = new URLSearchParams({ limit, offset });
    if (search) query.append('search', search);
    if (riskFilter && riskFilter !== 'ALL') query.append('risk_filter', riskFilter);
    return this.request(`/api/history?${query}`);
  }

  static async getHistoryStats() {
    return this.request('/api/history/stats/summary');
  }

  static async getAssessmentById(id) {
    return this.request(`/api/history/${id}`);
  }

  static async deleteAssessment(id) {
    return this.request(`/api/history/${id}`, { method: 'DELETE' });
  }

  // --- Models & Benchmarks ---
  static async getBenchmarks() {
    return this.request('/api/models/benchmarks');
  }

  static async getMetadata() {
    return this.request('/api/models/metadata');
  }

  // --- Batch Upload ---
  static async uploadBatchCSV(file, lowThresh = 0.30, highThresh = 0.60) {
    const formData = new FormData();
    formData.append('file', file);
    const query = new URLSearchParams({ low_thresh: lowThresh, high_thresh: highThresh });

    const response = await fetch(`${API_BASE}/api/batch/upload?${query}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || 'Batch CSV upload failed');
    }
    return await response.json();
  }
}

window.API = HealthPredictAPI;

