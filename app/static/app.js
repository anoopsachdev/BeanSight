/**
 * Coffee Bean Analyzer — Frontend Application
 *
 * Handles:
 *  - Drag-and-drop image upload with preview
 *  - API calls to /api/predict
 *  - Animated probability bar chart rendering
 *  - History panel updates
 *  - Toast notifications
 *  - Health check polling
 */

// ── State ──────────────────────────────────────────────────────────────

let selectedFile = null;

// ── DOM References ─────────────────────────────────────────────────────

const uploadZone     = document.getElementById('upload-zone');
const uploadInput    = document.getElementById('upload-input');
const uploadPrompt   = document.getElementById('upload-prompt');
const uploadPreview  = document.getElementById('upload-preview');
const uploadActions  = document.getElementById('upload-actions');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results-section');
const loadingOverlay = document.getElementById('loading-overlay');
const toastContainer = document.getElementById('toast-container');
const healthDot      = document.getElementById('health-dot');
const historyList    = document.getElementById('history-list');

// ── Toast Notifications ────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastOut 0.3s ease-in forwards';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Loading State ──────────────────────────────────────────────────────

function setLoading(isLoading) {
  loadingOverlay.classList.toggle('active', isLoading);
  analyzeBtn.disabled = isLoading;
}

// ── Upload Handling ────────────────────────────────────────────────────

function handleFile(file) {
  if (!file) return;

  // Validate type
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!allowedTypes.includes(file.type)) {
    showToast('Unsupported file type. Use JPG, PNG, or WebP.', 'error');
    return;
  }

  // Validate size (10 MB)
  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Maximum size is 10 MB.', 'error');
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadPreview.src = e.target.result;
    uploadPreview.style.display = 'block';
    uploadPrompt.style.display = 'none';
    uploadActions.style.display = 'flex';
    uploadZone.classList.add('has-image');
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  selectedFile = null;
  uploadInput.value = '';
  uploadPreview.style.display = 'none';
  uploadPreview.src = '';
  uploadPrompt.style.display = 'block';
  uploadActions.style.display = 'none';
  uploadZone.classList.remove('has-image');
  resultsSection.classList.remove('active');
}

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

// Click to browse
uploadZone.addEventListener('click', () => {
  if (!selectedFile) uploadInput.click();
});

uploadInput.addEventListener('change', (e) => {
  handleFile(e.target.files[0]);
});

// Clear button
clearBtn.addEventListener('click', clearUpload);

// ── Analyze ────────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    showToast('Please select an image first.', 'error');
    return;
  }

  setLoading(true);

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error: ${response.status}`);
    }

    const result = await response.json();
    displayResults(result);
    showToast('Analysis complete!', 'success');
    fetchHistory();
  } catch (error) {
    console.error('Prediction error:', error);
    showToast(error.message || 'Failed to analyze image.', 'error');
  } finally {
    setLoading(false);
  }
});

// ── Display Results ────────────────────────────────────────────────────

function displayResults(result) {
  resultsSection.classList.add('active');

  // Roast results
  if (result.roast) {
    renderPrediction(
      'roast',
      result.roast.prediction,
      result.roast.confidence,
      result.roast.probabilities,
    );
  }

  // Defect results
  if (result.defect) {
    renderPrediction(
      'defect',
      result.defect.prediction,
      result.defect.confidence,
      result.defect.probabilities,
    );
  }

  // Inference time
  const statTime = document.getElementById('stat-time');
  if (statTime) statTime.textContent = `${result.inference_time_ms} ms`;
  const statTimePill = document.getElementById('stat-time-pill');
  if (statTimePill) statTimePill.textContent = `${result.inference_time_ms} ms`;
}

function renderPrediction(type, prediction, confidence, probabilities) {
  // Badge
  const badge = document.getElementById(`${type}-badge`);
  badge.textContent = prediction;
  badge.className = 'prediction-badge ' + getBadgeClass(type, prediction, confidence);

  // Confidence
  const confEl = document.getElementById(`${type}-confidence`);
  animateCounter(confEl, 0, confidence * 100, 800, (v) => `${v.toFixed(1)}%`);

  // Probability bars
  const barsContainer = document.getElementById(`${type}-bars`);
  barsContainer.innerHTML = '';

  // Sort probabilities descending
  const sorted = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a);

  sorted.forEach(([label, prob], index) => {
    const isTop = index === 0;
    const row = document.createElement('div');
    row.className = 'prob-bar-row';

    row.innerHTML = `
      <div class="prob-label" title="${label}">${label}</div>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-prediction' : ''}"
             style="width: 0%"></div>
      </div>
      <div class="prob-value ${isTop ? 'highlight' : ''}">
        ${(prob * 100).toFixed(1)}%
      </div>
    `;

    barsContainer.appendChild(row);

    // Animate bar width
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = row.querySelector('.prob-fill');
        fill.style.width = `${Math.max(prob * 100, 0.5)}%`;
      }, index * 50);
    });
  });
}

function getBadgeClass(type, prediction, confidence) {
  if (type === 'roast') {
    const map = { 'Dark': 'badge-dark', 'Green': 'badge-green', 'Light': 'badge-light', 'Medium': 'badge-medium' };
    return map[prediction] || 'badge-medium';
  }
  // Defect: color by confidence severity
  if (confidence >= 0.7) return 'badge-defect-high';
  if (confidence >= 0.4) return 'badge-defect-medium';
  return 'badge-defect-low';
}

function animateCounter(element, from, to, duration, formatter) {
  const start = performance.now();
  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (to - from) * eased;
    element.textContent = formatter(current);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// ── History ────────────────────────────────────────────────────────────

async function fetchHistory() {
  try {
    const response = await fetch('/api/history?limit=20');
    if (!response.ok) return;
    const data = await response.json();
    renderHistory(data.predictions);
  } catch {
    // Silent fail — history is non-critical
  }
}

function renderHistory(predictions) {
  if (!predictions || predictions.length === 0) {
    historyList.innerHTML = '<div class="history-empty">No analyses yet. Upload an image to get started.</div>';
    return;
  }

  // Group by pairs (roast + defect from same image)
  const grouped = [];
  const seen = new Set();

  predictions.forEach((p) => {
    const key = `${p.filename}-${p.timestamp?.substring(0, 19)}`;
    if (!seen.has(key)) {
      seen.add(key);
      const pair = predictions.filter(
        (q) => q.filename === p.filename && q.timestamp?.substring(0, 19) === p.timestamp?.substring(0, 19)
      );
      grouped.push(pair);
    }
  });

  historyList.innerHTML = grouped.slice(0, 10).map((pair) => {
    const main = pair[0];
    const secondary = pair[1];
    const timeStr = main.timestamp
      ? new Date(main.timestamp).toLocaleString(undefined, {
          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
        })
      : '';

    let details = `${main.analysis_type}: ${main.predicted_class}`;
    if (secondary) {
      details += ` · ${secondary.analysis_type}: ${secondary.predicted_class}`;
    }

    const thumbHtml = main.image_url
      ? `<img class="history-thumb" src="${main.image_url}" alt="Bean" loading="lazy" />`
      : `<div class="history-thumb" style="background:var(--bg-elevated);display:flex;align-items:center;justify-content:center;font-size:18px;border-radius:var(--radius-sm);">☕</div>`;

    return `
      <div class="history-item">
        ${thumbHtml}
        <div class="history-info">
          <div class="history-class">${main.predicted_class}</div>
          <div class="history-type">${details}</div>
        </div>
        <div class="history-confidence">${(main.confidence * 100).toFixed(1)}%</div>
        <div class="history-time">${timeStr}</div>
      </div>
    `;
  }).join('');
}

// ── Health Check ───────────────────────────────────────────────────────

async function checkHealth() {
  const statusText = document.getElementById('status-text');
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error();
    const data = await response.json();

    const allLoaded = data.models?.roast && data.models?.defect;
    const someLoaded = data.models?.roast || data.models?.defect;

    healthDot.className = 'status-indicator ' + (
      allLoaded ? 'healthy' : someLoaded ? 'degraded' : 'unhealthy'
    );
    healthDot.title = allLoaded
      ? 'All models loaded'
      : someLoaded
      ? 'Some models loaded'
      : 'No models loaded';

    if (statusText) {
      statusText.textContent = allLoaded ? 'Inference Engine Online' : someLoaded ? 'Partial Models Active' : 'Offline';
    }
  } catch {
    healthDot.className = 'status-indicator unhealthy';
    healthDot.title = 'API unreachable';
    if (statusText) {
      statusText.textContent = 'API Unreachable';
    }
  }
}

// ── Init ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  fetchHistory();

  // Poll health every 30s
  setInterval(checkHealth, 30000);
});
