/**
 * BeanSight — Industrial Quality-Control Telemetry Application
 * 
 * Handles:
 *  - Single & Batch inspection telemetry workflows
 *  - Quality Score (0-100) & SCAA Grade evaluation algorithms
 *  - Interactive sample loading & bounding box localization overlays
 *  - Collapsible 17-class defect & 4-roast spectrum distributions
 *  - Operational filter tabs (ALL / PASS / REVIEW / DEFECTS)
 *  - Live health check polling & Supabase telemetry syncing
 */

// ── State ──────────────────────────────────────────────────────────────

let selectedFile = null;
let rawHistoryData = [];
let activeFilter = 'all';

// ── DOM References ─────────────────────────────────────────────────────

const uploadZone          = document.getElementById('upload-zone');
const uploadInput         = document.getElementById('upload-input');
const uploadPrompt        = document.getElementById('upload-prompt');
const uploadPreview       = document.getElementById('upload-preview');
const uploadActions       = document.getElementById('upload-actions');
const analyzeBtn          = document.getElementById('analyze-btn');
const clearBtn            = document.getElementById('clear-btn');
const resultsSection      = document.getElementById('results-section');
const loadingOverlay      = document.getElementById('loading-overlay');
const toastContainer      = document.getElementById('toast-container');
const healthDot           = document.getElementById('health-dot');
const statusText          = document.getElementById('status-text');
const historyList         = document.getElementById('history-list');
const spectrumToggleBtn   = document.getElementById('btn-spectrum-toggle');
const spectrumCollapsible = document.getElementById('spectrum-collapsible');
const tabSingle           = document.getElementById('tab-single');
const tabBatch            = document.getElementById('tab-batch');
const singleView          = document.getElementById('single-mode-view');
const batchView           = document.getElementById('batch-mode-view');
const inspectedCounter    = document.getElementById('inspected-counter');

// ── Toast Notifications ────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 3500) {
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
  if (analyzeBtn) analyzeBtn.disabled = isLoading;
}

// ── Mode Switcher (Single vs Batch) ────────────────────────────────────

if (tabSingle && tabBatch) {
  tabSingle.addEventListener('click', () => {
    tabSingle.classList.add('active');
    tabBatch.classList.remove('active');
    singleView.style.display = 'block';
    batchView.style.display = 'none';
  });

  tabBatch.addEventListener('click', () => {
    tabBatch.classList.add('active');
    tabSingle.classList.remove('active');
    singleView.style.display = 'none';
    batchView.style.display = 'block';
    resultsSection.classList.remove('active');
  });
}

// ── Collapsible Spectrum Details ───────────────────────────────────────

if (spectrumToggleBtn && spectrumCollapsible) {
  spectrumToggleBtn.addEventListener('click', () => {
    const isHidden = spectrumCollapsible.style.display === 'none';
    spectrumCollapsible.style.display = isHidden ? 'block' : 'none';
    const arrow = spectrumToggleBtn.querySelector('.toggle-arrow');
    if (arrow) arrow.textContent = isHidden ? '▲' : '▼';
  });
}

// ── Upload & Inspection Handling ───────────────────────────────────────

function handleFile(file) {
  if (!file) return;

  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!allowedTypes.includes(file.type)) {
    showToast('Unsupported file type. Use JPG, PNG, or WebP.', 'error');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showToast('File too large. Maximum size is 10 MB.', 'error');
    return;
  }

  selectedFile = file;

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

// Drag & Drop
if (uploadZone) {
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
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  uploadZone.addEventListener('click', (e) => {
    if (e.target.closest('#upload-actions')) return;
    uploadInput.click();
  });
}

if (uploadInput) {
  uploadInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });
}

if (clearBtn) clearBtn.addEventListener('click', clearUpload);

// ── Quick Demo Sample Buttons ──────────────────────────────────────────

document.querySelectorAll('.btn-sample').forEach(btn => {
  btn.addEventListener('click', () => {
    const sampleType = btn.getAttribute('data-sample');
    loadDemoSample(sampleType);
  });
});

function loadDemoSample(type) {
  // Create synthetic high-res test canvas sample
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  // Draw background bean tone
  if (type === 'dark') {
    ctx.fillStyle = '#2b1b14';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#4a2f22';
    ctx.beginPath();
    ctx.ellipse(112, 112, 70, 95, Math.PI / 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#120a06';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(112, 35);
    ctx.bezierCurveTo(105, 112, 120, 112, 112, 190);
    ctx.stroke();
  } else if (type === 'defect') {
    ctx.fillStyle = '#7a7052';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#9e926e';
    ctx.beginPath();
    ctx.ellipse(112, 112, 75, 90, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#d4cbb0';
    ctx.fillRect(80, 80, 64, 64);
  } else {
    ctx.fillStyle = '#445b38';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#658354';
    ctx.beginPath();
    ctx.ellipse(112, 112, 72, 92, -Math.PI / 12, 0, 2 * Math.PI);
    ctx.fill();
  }

  canvas.toBlob((blob) => {
    const file = new File([blob], `sample_${type}.jpg`, { type: 'image/jpeg' });
    handleFile(file);
    showToast(`Loaded demo sample: ${type.toUpperCase()}`, 'info');
  }, 'image/jpeg');
}

// ── Quality Score Algorithm (SCAA Grade Standard) ──────────────────────

function calculateQualityScore(roastResult, defectResult) {
  let baseScore = 95;

  const defectName = defectResult?.prediction || '';
  const defectConf = defectResult?.confidence || 0;

  // SCAA Defect Severity Categories
  const primarySevere = ['Full Black', 'Full Sour', 'Fungus Damage', 'Severe Insect Damage'];
  const secondaryModerate = ['Partial Black', 'Partial Sour', 'Broken', 'Cut', 'Shell', 'Floater'];
  const millingProcessing = ['Husk', 'Parchment', 'Dry Cherry', 'Immature', 'Withered', 'Fade'];

  let severityText = 'Low Risk (Grade 1 SCAA Compliant)';
  let verdictText = 'SCAA Specialty Grade 1 Equivalent';
  let statusFlagClass = 'status-pass';
  let statusFlagText = '🟢 PASS — SPECIALTY GRADE';

  if (primarySevere.includes(defectName)) {
    const penalty = 35 * defectConf;
    baseScore -= penalty;
    severityText = 'Critical Risk (Primary SCAA Defect)';
    verdictText = 'Off-Grade / Reject: Severe cup tainting defect detected';
    statusFlagClass = 'status-defect';
    statusFlagText = '🔴 REJECT — CRITICAL DEFECT';
  } else if (secondaryModerate.includes(defectName)) {
    const penalty = 20 * defectConf;
    baseScore -= penalty;
    severityText = 'Moderate Risk (Secondary SCAA Defect)';
    verdictText = 'Exchange Grade 2: Minor cup impact, secondary sort advised';
    statusFlagClass = 'status-review';
    statusFlagText = '🟡 REVIEW — SECONDARY DEFECT';
  } else if (millingProcessing.includes(defectName)) {
    const penalty = 10 * defectConf;
    baseScore -= penalty;
    severityText = 'Low Severity (Milling / Husk Artifact)';
    verdictText = 'Specialty Grade 1: Tolerable threshing artifact';
    statusFlagClass = 'status-pass';
    statusFlagText = '🟢 PASS — SPECIALTY GRADE';
  }

  const finalScore = Math.max(Math.min(Math.round(baseScore), 99), 35);

  return {
    score: finalScore,
    severity: severityText,
    verdict: verdictText,
    flagClass: statusFlagClass,
    flagText: statusFlagText,
  };
}

// ── Run AI Quality Diagnostic ──────────────────────────────────────────

if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please select or drop an image first.', 'error');
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
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const result = await response.json();
      displayInspectionResults(result);
      showToast('Inspection telemetry complete!', 'success');
      fetchHistory();
      incrementInspectionCounter();
    } catch (error) {
      console.error('Inspection error:', error);
      showToast(error.message || 'Failed to inspect bean.', 'error');
    } finally {
      setLoading(false);
    }
  });
}

// ── Render Inspection Workspace ────────────────────────────────────────

function displayInspectionResults(result) {
  resultsSection.classList.add('active');

  // Random inspection ID
  const randId = Math.floor(10000 + Math.random() * 90000);
  const inspIdEl = document.getElementById('inspection-id');
  if (inspIdEl) inspIdEl.textContent = `INSPECTION #BN-${randId}`;

  // Left View Preview
  const resultImg = document.getElementById('result-view-img');
  if (resultImg && uploadPreview.src) {
    resultImg.src = uploadPreview.src;
  }

  // Calculate Quality Score
  const quality = calculateQualityScore(result.roast, result.defect);

  // Quality Score & Verdict
  const scoreEl = document.getElementById('quality-score');
  if (scoreEl) {
    animateCounter(scoreEl, 0, quality.score, 800, (v) => Math.round(v));
  }

  const verdictEl = document.getElementById('score-verdict');
  if (verdictEl) verdictEl.textContent = quality.verdict;

  const flagEl = document.getElementById('status-flag');
  if (flagEl) {
    flagEl.className = `status-flag ${quality.flagClass}`;
    flagEl.textContent = quality.flagText;
  }

  // Roast Attribute
  const roastBadge = document.getElementById('roast-badge');
  if (roastBadge && result.roast) {
    roastBadge.textContent = `${result.roast.prediction} Roast (${(result.roast.confidence * 100).toFixed(1)}%)`;
    roastBadge.className = 'attr-val ' + getRoastBadgeClass(result.roast.prediction);
  }

  // Defect Attribute
  const defectBadge = document.getElementById('defect-badge');
  if (defectBadge && result.defect) {
    defectBadge.textContent = `${result.defect.prediction} (${(result.defect.confidence * 100).toFixed(1)}%)`;
    defectBadge.className = 'attr-val ' + getDefectBadgeClass(result.defect.confidence);
  }

  // Primary Confidence
  const primaryConf = document.getElementById('primary-confidence');
  if (primaryConf) {
    const topConf = Math.max(result.roast?.confidence || 0, result.defect?.confidence || 0);
    primaryConf.textContent = `${(topConf * 100).toFixed(1)}%`;
  }

  // Severity Index
  const severityEl = document.getElementById('defect-severity');
  if (severityEl) severityEl.textContent = quality.severity;

  // Inference Latency
  const statTime = document.getElementById('stat-time');
  if (statTime) statTime.textContent = `${result.inference_time_ms} ms`;

  // Render Detailed Bars in Collapsible
  if (result.roast?.probabilities) {
    renderBars('roast-bars', result.roast.probabilities);
  }
  if (result.defect?.probabilities) {
    renderBars('defect-bars', result.defect.probabilities);
  }

  // Scroll to results
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderBars(containerId, probabilities) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a);

  sorted.forEach(([label, prob], index) => {
    const isTop = index === 0;
    const row = document.createElement('div');
    row.className = 'prob-bar-row';

    row.innerHTML = `
      <div class="prob-label" title="${label}">${label}</div>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-prediction' : ''}" style="width: 0%"></div>
      </div>
      <div class="prob-value ${isTop ? 'highlight' : ''}">
        ${(prob * 100).toFixed(1)}%
      </div>
    `;

    container.appendChild(row);

    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = row.querySelector('.prob-fill');
        if (fill) fill.style.width = `${Math.max(prob * 100, 0.5)}%`;
      }, index * 30);
    });
  });
}

function getRoastBadgeClass(prediction) {
  const map = {
    'Dark': 'badge-dark',
    'Medium': 'badge-medium',
    'Light': 'badge-light',
    'Green': 'badge-green',
  };
  return map[prediction] || 'badge-medium';
}

function getDefectBadgeClass(confidence) {
  if (confidence >= 0.7) return 'badge-defect-high';
  if (confidence >= 0.4) return 'badge-defect-medium';
  return 'badge-defect-low';
}

function animateCounter(element, from, to, duration, formatter) {
  const start = performance.now();
  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = from + (to - from) * eased;
    element.textContent = formatter(current);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function incrementInspectionCounter() {
  if (!inspectedCounter) return;
  const current = parseInt(inspectedCounter.textContent.replace(/,/g, ''), 10) || 1284;
  inspectedCounter.textContent = (current + 1).toLocaleString();
}

// ── Operational History & Filter Tabs ──────────────────────────────────

async function fetchHistory() {
  try {
    const response = await fetch('/api/history?limit=30');
    if (!response.ok) return;
    const data = await response.json();
    rawHistoryData = data.predictions || [];
    renderFilteredHistory();
  } catch {
    // History sync is non-blocking
  }
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.getAttribute('data-filter');
    renderFilteredHistory();
  });
});

function renderFilteredHistory() {
  if (!historyList) return;

  if (!rawHistoryData || rawHistoryData.length === 0) {
    historyList.innerHTML = '<div class="history-empty">No inspections logged yet. Run a scan above to start audit log.</div>';
    return;
  }

  // Filter logic
  let filtered = rawHistoryData;
  if (activeFilter === 'pass') {
    filtered = rawHistoryData.filter(p => p.confidence < 0.4 || p.predicted_class === 'Dark' || p.predicted_class === 'Medium' || p.predicted_class === 'Light');
  } else if (activeFilter === 'review') {
    filtered = rawHistoryData.filter(p => p.confidence >= 0.4 && p.confidence < 0.7);
  } else if (activeFilter === 'defect') {
    filtered = rawHistoryData.filter(p => p.confidence >= 0.7 && p.analysis_type === 'defect');
  }

  if (filtered.length === 0) {
    historyList.innerHTML = `<div class="history-empty">No records matching filter "${activeFilter.toUpperCase()}".</div>`;
    return;
  }

  // Group predictions by filename
  const grouped = new Map();
  filtered.forEach(p => {
    const key = p.filename || p.id;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(p);
  });

  historyList.innerHTML = Array.from(grouped.values()).map(records => {
    const main = records[0];
    const secondary = records[1] || null;

    const timeStr = main.timestamp
      ? new Date(main.timestamp).toLocaleString(undefined, {
          month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
        })
      : 'Just now';

    const isHighRisk = main.confidence >= 0.7 && main.analysis_type === 'defect';
    const isModerate = main.confidence >= 0.4 && main.confidence < 0.7;

    const statusPillHtml = isHighRisk
      ? `<span class="history-status-pill status-defect">🔴 DEFECT</span>`
      : isModerate
      ? `<span class="history-status-pill status-review">🟡 REVIEW</span>`
      : `<span class="history-status-pill status-pass">🟢 PASS</span>`;

    const details = `${main.analysis_type.toUpperCase()}: ${main.predicted_class}` +
      (secondary ? ` · ${secondary.analysis_type.toUpperCase()}: ${secondary.predicted_class}` : '');

    const thumbHtml = main.image_url
      ? `<img class="history-thumb" src="${main.image_url}" alt="Bean" loading="lazy" />`
      : `<div class="history-thumb" style="background:var(--bg-elevated);display:flex;align-items:center;justify-content:center;font-size:16px;">☕</div>`;

    return `
      <div class="history-item">
        ${thumbHtml}
        <div class="history-info">
          <div class="history-class">${main.predicted_class}</div>
          <div class="history-type">${details}</div>
        </div>
        ${statusPillHtml}
        <div class="history-confidence">${(main.confidence * 100).toFixed(1)}%</div>
        <div class="history-time">${timeStr}</div>
      </div>
    `;
  }).join('');
}

// ── Health Check Polling ───────────────────────────────────────────────

async function checkHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error();
    const data = await response.json();

    const allLoaded = data.models?.roast && data.models?.defect;
    const someLoaded = data.models?.roast || data.models?.defect;

    healthDot.className = 'status-indicator ' + (
      allLoaded ? 'healthy' : someLoaded ? 'degraded' : 'unhealthy'
    );
    healthDot.title = allLoaded ? 'All models loaded' : someLoaded ? 'Some models loaded' : 'No models loaded';

    if (statusText) {
      statusText.textContent = allLoaded ? 'Inference Engine Online' : someLoaded ? 'Partial Telemetry Active' : 'Offline';
    }
  } catch {
    healthDot.className = 'status-indicator unhealthy';
    healthDot.title = 'API unreachable';
    if (statusText) statusText.textContent = 'API Unreachable';
  }
}

// ── Init ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  fetchHistory();
  setInterval(checkHealth, 30000);
});
