/**
 * BeanSight — Artisanal Quality Assurance Studio
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
const tabGreen            = document.getElementById('tab-green');
const tabRoasted          = document.getElementById('tab-roasted');
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

// ── Mode Switcher (Green vs Roasted Tabs) ──────────────────────────────

function setUITab(mode) {
  if (mode === 'green') {
    tabGreen.classList.add('active');
    tabRoasted.classList.remove('active');
  } else {
    tabRoasted.classList.add('active');
    tabGreen.classList.remove('active');
  }
}

if (tabGreen && tabRoasted) {
  tabGreen.addEventListener('click', () => setUITab('green'));
  tabRoasted.addEventListener('click', () => setUITab('roasted'));
}

// ── Analytics Dashboard Toggle ─────────────────────────────────────────

const btnAnalyticsGreen = document.getElementById('btn-analytics-green');
const btnAnalyticsRoasted = document.getElementById('btn-analytics-roasted');
const analyticsGreenView = document.getElementById('analytics-green-view');
const analyticsRoastedView = document.getElementById('analytics-roasted-view');
const chartGreen = document.getElementById('analytics-chart-green');
const chartRoasted = document.getElementById('analytics-chart-roasted');

if (btnAnalyticsGreen && btnAnalyticsRoasted) {
  btnAnalyticsGreen.addEventListener('click', () => {
    btnAnalyticsGreen.classList.add('active');
    btnAnalyticsRoasted.classList.remove('active');
    analyticsGreenView.style.display = 'grid';
    analyticsRoastedView.style.display = 'none';
    chartGreen.style.display = 'block';
    chartRoasted.style.display = 'none';
  });

  btnAnalyticsRoasted.addEventListener('click', () => {
    btnAnalyticsRoasted.classList.add('active');
    btnAnalyticsGreen.classList.remove('active');
    analyticsGreenView.style.display = 'none';
    analyticsRoastedView.style.display = 'grid';
    chartGreen.style.display = 'none';
    chartRoasted.style.display = 'block';
  });
}

// ── Collapsible Spectrum Details ───────────────────────────────────────

if (spectrumToggleBtn && spectrumCollapsible) {
  spectrumToggleBtn.addEventListener('click', () => {
    const isHidden = spectrumCollapsible.style.display === 'none';
    spectrumCollapsible.style.display = isHidden ? 'block' : 'none';
    spectrumToggleBtn.textContent = isHidden 
      ? 'Hide Detailed Spectrum ▲' 
      : 'View 17-Class Spectrum & Probability Distribution ▼';
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
  resultsSection.classList.remove('active');
}

// Drag & Drop
if (uploadZone) {
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  uploadZone.addEventListener('click', (e) => {
    if (e.target.closest('#upload-actions') || e.target.closest('.quick-samples')) return;
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
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const sampleType = btn.getAttribute('data-sample');
    loadDemoSample(sampleType);
  });
});

function loadDemoSample(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  if (type === 'dark') {
    ctx.fillStyle = '#1A1614';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#2C2825';
    ctx.beginPath();
    ctx.ellipse(112, 112, 70, 95, Math.PI / 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(112, 35);
    ctx.bezierCurveTo(105, 112, 120, 112, 112, 190);
    ctx.stroke();
  } else if (type === 'defect') {
    ctx.fillStyle = '#6a8f6d';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#7a9f7d';
    ctx.beginPath();
    ctx.ellipse(112, 112, 75, 90, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillStyle = '#C28E46'; // Defect spot
    ctx.fillRect(80, 80, 64, 64);
  } else {
    ctx.fillStyle = '#8f9a88';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#6a8f6d';
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

// ── Quality Score Algorithm (Green Beans) ──────────────────────

function calculateQualityScore(roastResult, defectResult) {
  let baseScore = 95;

  const defectName = defectResult?.prediction || '';
  const defectConf = defectResult?.confidence || 0;

  const primarySevere = ['Full Black', 'Full Sour', 'Fungus Damage', 'Severe Insect Damage', 'Scorched', 'Burnt'];
  const secondaryModerate = ['Partial Black', 'Partial Sour', 'Broken', 'Cut', 'Shell', 'Floater', 'Quaker'];
  const millingProcessing = ['Husk', 'Parchment', 'Dry Cherry', 'Immature', 'Withered', 'Fade', 'Insect Damage'];

  let severityText = 'Low Risk (Clean)';
  let verdictText = 'High Quality Clean Bean';
  let statusFlagClass = 'status-pass';
  let statusFlagText = '🟢 PASS — CLEAN BEAN';

  if (primarySevere.includes(defectName)) {
    baseScore -= 35 * defectConf;
    severityText = 'Critical Risk (Primary Defect)';
    verdictText = 'Off-Grade / Reject: Severe defect detected';
    statusFlagClass = 'status-defect';
    statusFlagText = '🔴 REJECT — CRITICAL DEFECT';
  } else if (secondaryModerate.includes(defectName)) {
    baseScore -= 20 * defectConf;
    severityText = 'Moderate Risk (Secondary Defect)';
    verdictText = 'Minor cup impact, secondary sort advised';
    statusFlagClass = 'status-review';
    statusFlagText = '🟡 REVIEW — SECONDARY DEFECT';
  } else if (millingProcessing.includes(defectName)) {
    baseScore -= 10 * defectConf;
    severityText = 'Low Severity (Minor Defect)';
    verdictText = 'Tolerable processing artifact';
    statusFlagClass = 'status-pass';
    statusFlagText = '🟢 PASS — MINOR DEFECT';
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

// ── Run AI Diagnostic ──────────────────────────────────────────

if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please load an image first.', 'error');
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
      showToast('Quality evaluation complete!', 'success');
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

  const randId = Math.floor(10000 + Math.random() * 90000);
  const inspIdEl = document.getElementById('inspection-id');
  if (inspIdEl) inspIdEl.textContent = `INSPECTION #BN-${randId}`;

  const resultImg = document.getElementById('result-view-img');
  const isGreen = result.roast?.prediction === 'Green';

  // Toggle UI Layout based on Tier (Green vs Roasted)
  setUITab(isGreen ? 'green' : 'roasted');
  
  const greenScoreCard = document.getElementById('green-score-card');
  const roastedScoreCard = document.getElementById('roasted-score-card');
  const spectrumContainer = document.getElementById('spectrum-container');
  const defectLbl = document.getElementById('defect-lbl');
  
  if (isGreen) {
    greenScoreCard.style.display = 'block';
    roastedScoreCard.style.display = 'none';
    spectrumContainer.style.display = 'block';
    defectLbl.textContent = 'Physical Defect';
    
    if (resultImg && uploadPreview.src) resultImg.src = uploadPreview.src;
    
    const quality = calculateQualityScore(result.roast, result.defect);
    const scoreEl = document.getElementById('quality-score');
    if (scoreEl) animateCounter(scoreEl, 0, quality.score, 800, Math.round);
    
    const verdictEl = document.getElementById('score-verdict');
    if (verdictEl) verdictEl.textContent = quality.verdict;
    const flagEl = document.getElementById('status-flag');
    if (flagEl) {
      flagEl.className = `status-flag ${quality.flagClass}`;
      flagEl.textContent = quality.flagText;
    }
    
    const severityEl = document.getElementById('defect-severity');
    if (severityEl) severityEl.textContent = quality.severity;
    
    if (result.roast?.probabilities) renderBars('roast-bars', result.roast.probabilities);
    if (result.defect?.probabilities) renderBars('defect-bars', result.defect.probabilities);
    
  } else {
    // Roasted Batch (YOLO output)
    greenScoreCard.style.display = 'none';
    roastedScoreCard.style.display = 'block';
    spectrumContainer.style.display = 'none';
    defectLbl.textContent = 'Batch Defects';
    
    if (result.roasted_defect?.annotated_image && resultImg) {
      resultImg.src = result.roasted_defect.annotated_image;
    } else if (resultImg && uploadPreview.src) {
      resultImg.src = uploadPreview.src;
    }

    const defectCount = result.roasted_defect?.defect_count || 0;
    const uniformity = Math.max(100 - (defectCount * 0.8), 70); // Simulated batch penalty
    const uniEl = document.getElementById('batch-uniformity');
    if (uniEl) animateCounter(uniEl, 0, uniformity, 800, (v) => v.toFixed(1));
    
    const flagEl = document.getElementById('batch-status-flag');
    const verdictEl = document.getElementById('batch-verdict');
    const severityEl = document.getElementById('defect-severity');
    
    if (defectCount > 5) {
      flagEl.className = 'status-flag status-defect';
      flagEl.textContent = '🔴 REJECT — UNEVEN ROAST';
      verdictEl.textContent = 'High variance or scorching detected.';
      severityEl.textContent = `High Risk (${defectCount} issues)`;
    } else if (defectCount > 0) {
      flagEl.className = 'status-flag status-review';
      flagEl.textContent = '🟡 REVIEW — MINOR ISSUES';
      verdictEl.textContent = 'Acceptable variance, monitor closely.';
      severityEl.textContent = `Moderate Risk (${defectCount} issues)`;
    } else {
      flagEl.className = 'status-flag status-pass';
      flagEl.textContent = '🟢 PRODUCTION READY';
      verdictEl.textContent = 'Flawless, consistent roast profile.';
      severityEl.textContent = 'Low Risk (Clean Batch)';
    }
  }

  // Common Attributes
  const roastBadge = document.getElementById('roast-badge');
  if (roastBadge && result.roast) {
    roastBadge.textContent = `${result.roast.prediction} Roast (${(result.roast.confidence * 100).toFixed(1)}%)`;
  }

  const defectBadge = document.getElementById('defect-badge');
  if (defectBadge && result.defect) {
    defectBadge.textContent = `${result.defect.prediction}`;
  }

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
      <div class="prob-value">
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

// ── Operational History ──────────────────────────────────

async function fetchHistory() {
  try {
    const response = await fetch('/api/history?limit=30');
    if (!response.ok) return;
    const data = await response.json();
    rawHistoryData = data.predictions || [];
    renderFilteredHistory();
  } catch {
    // Non-blocking
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
    historyList.innerHTML = '<div class="history-empty">No inspections logged yet.</div>';
    return;
  }

  let filtered = rawHistoryData;
  if (activeFilter === 'pass') {
    filtered = rawHistoryData.filter(p => p.confidence < 0.4 || p.predicted_class === 'Dark' || p.predicted_class === 'Medium' || p.predicted_class === 'Light');
  } else if (activeFilter === 'review') {
    filtered = rawHistoryData.filter(p => p.confidence >= 0.4 && p.confidence < 0.7);
  } else if (activeFilter === 'defect') {
    filtered = rawHistoryData.filter(p => p.confidence >= 0.7 && p.analysis_type === 'defect');
  }

  if (filtered.length === 0) {
    historyList.innerHTML = `<div class="history-empty">No records matching filter.</div>`;
    return;
  }

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
    const isGreen = main.predicted_class === 'Green' || main.analysis_type === 'defect' && main.predicted_class !== 'Dark' && main.predicted_class !== 'Medium' && main.predicted_class !== 'Light';

    const statusPillHtml = isHighRisk
      ? `<span class="history-status-pill status-defect">DEFECT</span>`
      : isModerate
      ? `<span class="history-status-pill status-review">REVIEW</span>`
      : `<span class="history-status-pill status-pass">PASS</span>`;

    const tierBadgeHtml = isGreen 
      ? `<span class="tag-tier tag-green">INBOUND - GREEN</span>`
      : `<span class="tag-tier tag-roasted">PRODUCTION - ROASTED</span>`;

    const details = `${main.analysis_type}: ${main.predicted_class}` +
      (secondary ? ` · ${secondary.analysis_type}: ${secondary.predicted_class}` : '');

    const thumbHtml = main.image_url
      ? `<img class="history-thumb" src="${main.image_url}" alt="Sample" loading="lazy" />`
      : `<div class="history-thumb" style="background:var(--bg-surface);display:flex;align-items:center;justify-content:center;font-size:20px;">📷</div>`;

    return `
      <div class="history-item">
        ${thumbHtml}
        <div class="history-info">
          <div class="history-class">${main.predicted_class}</div>
          <div class="history-type">${details}</div>
        </div>
        ${tierBadgeHtml}
        ${statusPillHtml}
        <div class="history-confidence">${(main.confidence * 100).toFixed(1)}%</div>
        <div class="history-time">${timeStr}</div>
      </div>
    `;
  }).join('');
}

// ── Health Check ───────────────────────────────────────────────

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
      statusText.textContent = allLoaded ? 'System Online' : someLoaded ? 'Degraded' : 'Offline';
    }
  } catch {
    healthDot.className = 'status-indicator unhealthy';
    if (statusText) statusText.textContent = 'API Unreachable';
  }
}

// ── Init ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  fetchHistory();
  setInterval(checkHealth, 30000);
});
