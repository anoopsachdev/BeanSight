/**
 * BeanSight — Industrial Quality Inspection Platform
 * Frontend Telemetry & Decision Controller
 */

// ── State Management ───────────────────────────────────────────────────

let activeMode = 'green'; // 'green' | 'roasted'
let selectedFile = null;
let rawHistoryData = [];
let activeFilter = 'all';
let statsData = null;

// ── DOM Element References ─────────────────────────────────────────────

const tabGreen              = document.getElementById('tab-green');
const tabRoasted            = document.getElementById('tab-roasted');
const modeContextTag        = document.getElementById('mode-context-tag');
const dropzoneBox           = document.getElementById('dropzone-box');
const fileInput             = document.getElementById('file-input');
const dropzoneHeadline      = document.getElementById('dropzone-headline');
const samplePreviewContainer= document.getElementById('sample-preview-container');
const samplePreviewImg      = document.getElementById('sample-preview-img');
const sampleFilename        = document.getElementById('sample-filename');
const sampleFilesize        = document.getElementById('sample-filesize');
const analyzeBtn            = document.getElementById('analyze-btn');
const clearBtn              = document.getElementById('clear-btn');

const resultsPlaceholder    = document.getElementById('results-placeholder');
const resultsActiveContainer= document.getElementById('results-active-container');
const inspectionIdBadge     = document.getElementById('inspection-id-badge');

const statusVerdictBanner   = document.getElementById('status-verdict-banner');
const verdictIcon           = document.getElementById('verdict-icon');
const verdictTitle          = document.getElementById('verdict-title');
const verdictSubtitle       = document.getElementById('verdict-subtitle');
const verdictConfidence     = document.getElementById('verdict-confidence');

const evidenceImage         = document.getElementById('evidence-image');
const evidenceTag           = document.getElementById('evidence-tag');

const detRoastVal           = document.getElementById('det-roast-val');
const detDefectLabel        = document.getElementById('det-defect-label');
const detDefectVal          = document.getElementById('det-defect-val');
const detSeverityVal        = document.getElementById('det-severity-val');
const detLatencyVal         = document.getElementById('det-latency-val');

const recTitle              = document.getElementById('rec-title');
const recDesc               = document.getElementById('rec-desc');

const spectrumToggleBtn     = document.getElementById('spectrum-toggle-btn');
const spectrumArrow         = document.getElementById('spectrum-arrow');
const spectrumDrawer        = document.getElementById('spectrum-drawer');
const spectrumBarsList      = document.getElementById('spectrum-bars-list');

const statTotalCount        = document.getElementById('stat-total-count');
const statAvgLatency        = document.getElementById('stat-avg-latency');
const statDefectRate        = document.getElementById('stat-defect-rate');
const analyticsDistBars     = document.getElementById('analytics-distribution-bars');

const auditTableBody        = document.getElementById('audit-table-body');
const healthDot             = document.getElementById('health-dot');
const healthStatusText      = document.getElementById('health-status-text');
const loadingOverlay        = document.getElementById('loading-overlay');
const toastContainer        = document.getElementById('toast-container');

// ── Toast System ───────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 3500) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'toastIn 0.2s ease reverse forwards';
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

// ── Mode Switcher ──────────────────────────────────────────────────────

function setInspectionMode(mode) {
  activeMode = mode;
  if (mode === 'green') {
    tabGreen.classList.add('active');
    tabRoasted.classList.remove('active');
    modeContextTag.textContent = 'Mode: Green Bean Inbound';
    dropzoneHeadline.textContent = 'Upload green coffee sample for lot inspection';
  } else {
    tabRoasted.classList.add('active');
    tabGreen.classList.remove('active');
    modeContextTag.textContent = 'Mode: Roasted Batch QA';
    dropzoneHeadline.textContent = 'Upload roasted coffee batch for uniformity scan';
  }
}

if (tabGreen && tabRoasted) {
  tabGreen.addEventListener('click', () => setInspectionMode('green'));
  tabRoasted.addEventListener('click', () => setInspectionMode('roasted'));
}

// ── File Ingestion & Dropzone ──────────────────────────────────────────

function handleLoadedFile(file) {
  if (!file) return;

  const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!validTypes.includes(file.type)) {
    showToast('Unsupported file format. Please upload JPG, PNG, or WEBP.', 'error');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showToast('File exceeds 10 MB limit.', 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    samplePreviewImg.src = e.target.result;
    sampleFilename.textContent = file.name;
    sampleFilesize.textContent = `${(file.size / 1024).toFixed(1)} KB`;

    dropzoneBox.style.display = 'none';
    samplePreviewContainer.style.display = 'flex';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetWorkstation() {
  selectedFile = null;
  fileInput.value = '';
  samplePreviewImg.src = '';
  samplePreviewContainer.style.display = 'none';
  dropzoneBox.style.display = 'flex';
  analyzeBtn.disabled = true;

  resultsActiveContainer.style.display = 'none';
  resultsPlaceholder.style.display = 'flex';
  inspectionIdBadge.textContent = 'ID: Standby';
}

if (dropzoneBox) {
  dropzoneBox.addEventListener('click', () => fileInput.click());

  dropzoneBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneBox.classList.add('drag-over');
  });

  dropzoneBox.addEventListener('dragleave', () => {
    dropzoneBox.classList.remove('drag-over');
  });

  dropzoneBox.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneBox.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleLoadedFile(e.dataTransfer.files[0]);
    }
  });
}

if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleLoadedFile(e.target.files[0]);
    }
  });
}

if (clearBtn) {
  clearBtn.addEventListener('click', resetWorkstation);
}

// ── Demo Fixture Generators ────────────────────────────────────────────

document.querySelectorAll('.btn-demo').forEach(btn => {
  btn.addEventListener('click', () => {
    const fixtureType = btn.getAttribute('data-fixture');
    loadDemoFixture(fixtureType);
  });
});

function loadDemoFixture(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d');

  if (type === 'green_clean') {
    setInspectionMode('green');
    ctx.fillStyle = '#1e241c';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#5c7a52';
    ctx.beginPath();
    ctx.ellipse(112, 112, 68, 92, -0.2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#3e5636';
    ctx.lineWidth = 3;
    ctx.stroke();
  } else if (type === 'green_defect') {
    setInspectionMode('green');
    ctx.fillStyle = '#26231c';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#6b7858';
    ctx.beginPath();
    ctx.ellipse(112, 112, 70, 90, 0, 0, 2 * Math.PI);
    ctx.fill();
    // Parchment / Damage artifact
    ctx.fillStyle = '#d4c79b';
    ctx.fillRect(80, 80, 60, 50);
  } else if (type === 'roasted_dark') {
    setInspectionMode('roasted');
    ctx.fillStyle = '#110f0d';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#2c1e18';
    ctx.beginPath();
    ctx.ellipse(112, 112, 68, 92, 0.3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#140c08';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(112, 30);
    ctx.bezierCurveTo(105, 112, 120, 112, 112, 194);
    ctx.stroke();
  } else {
    // Roasted defect
    setInspectionMode('roasted');
    ctx.fillStyle = '#181412';
    ctx.fillRect(0, 0, 224, 224);
    ctx.fillStyle = '#422c22';
    ctx.beginPath();
    ctx.ellipse(112, 112, 72, 88, 0, 0, 2 * Math.PI);
    ctx.fill();
    // Quaker / Scorch mark
    ctx.fillStyle = '#c28e46';
    ctx.beginPath();
    ctx.ellipse(100, 100, 30, 25, 0.4, 0, 2 * Math.PI);
    ctx.fill();
  }

  canvas.toBlob((blob) => {
    const file = new File([blob], `fixture_${type}.jpg`, { type: 'image/jpeg' });
    handleLoadedFile(file);
    showToast(`Loaded test sample: ${type.replace('_', ' ').toUpperCase()}`, 'info');
  }, 'image/jpeg');
}

// ── Spectrum Drawer Toggle ─────────────────────────────────────────────

if (spectrumToggleBtn && spectrumDrawer) {
  spectrumToggleBtn.addEventListener('click', () => {
    const isClosed = spectrumDrawer.style.display === 'none' || !spectrumDrawer.style.display;
    spectrumDrawer.style.display = isClosed ? 'block' : 'none';
    spectrumArrow.textContent = isClosed ? '▲' : '▼';
  });
}

// ── Run Inspection Execution ───────────────────────────────────────────

if (analyzeBtn) {
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please ingest a sample first.', 'error');
      return;
    }

    loadingOverlay.classList.add('active');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error ${response.status}`);
      }

      const result = await response.json();
      renderInspectionResult(result);
      showToast('Inspection completed successfully.', 'success');

      // Refresh background statistics & audit log
      fetchHistoricalAnalytics();
      fetchAuditLogs();
    } catch (err) {
      console.error('Inspection failed:', err);
      showToast(err.message || 'Failed to inspect sample.', 'error');
    } finally {
      loadingOverlay.classList.remove('active');
    }
  });
}

// ── Render Inspection Decision & Evidence ──────────────────────────────

function renderInspectionResult(result) {
  resultsPlaceholder.style.display = 'none';
  resultsActiveContainer.style.display = 'flex';

  // Generate ID
  const randomSerial = Math.floor(100000 + Math.random() * 900000);
  inspectionIdBadge.textContent = `ID: QA-${randomSerial}`;

  const roast = result.roast || { prediction: 'Unknown', confidence: 0 };
  const defect = result.defect || { prediction: 'None', confidence: 0 };
  const isGreen = roast.prediction === 'Green';

  // Switch active tab if model indicates discrepancy
  if (isGreen && activeMode !== 'green') setInspectionMode('green');
  if (!isGreen && activeMode !== 'roasted') setInspectionMode('roasted');

  // Evidence View
  if (result.roasted_defect?.annotated_image) {
    evidenceImage.src = result.roasted_defect.annotated_image;
    evidenceTag.textContent = 'EVIDENCE: YOLOv8 BOUNDING BOX LOCALIZER';
  } else if (samplePreviewImg.src) {
    evidenceImage.src = samplePreviewImg.src;
    evidenceTag.textContent = isGreen ? 'EVIDENCE: 224px CHROMATIC RGB' : 'EVIDENCE: POST-ROAST VISUAL SCAN';
  }

  // Detection Grid Values
  detRoastVal.textContent = `${roast.prediction} (${(roast.confidence * 100).toFixed(1)}%)`;
  detLatencyVal.textContent = `${result.inference_time_ms || 0} ms`;

  // Decision & Status Evaluation Logic
  const primarySevereDefects = ['Full Black', 'Full Sour', 'Fungus Damage', 'Severe Insect Damage'];
  const secondaryDefects = ['Partial Black', 'Partial Sour', 'Broken', 'Cut', 'Shell', 'Floater'];

  if (isGreen) {
    detDefectLabel.textContent = 'Agricultural Defect';
    detDefectVal.textContent = `${defect.prediction} (${(defect.confidence * 100).toFixed(1)}%)`;

    if (primarySevereDefects.includes(defect.prediction) && defect.confidence >= 0.5) {
      statusVerdictBanner.className = 'status-verdict-banner fail';
      verdictIcon.textContent = '✕';
      verdictTitle.textContent = 'REJECT — CRITICAL DEFECT DETECTED';
      verdictSubtitle.textContent = `Primary agricultural defect identified: ${defect.prediction}`;
      verdictConfidence.textContent = `${(defect.confidence * 100).toFixed(1)}% CONF`;

      detSeverityVal.textContent = 'Critical Risk (Primary Off-Grade)';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: REJECT LOT';
      recDesc.textContent = 'Sample fails clean specialty grade requirements. Isolate and reject inbound harvest lot.';
    } else if (secondaryDefects.includes(defect.prediction) && defect.confidence >= 0.4) {
      statusVerdictBanner.className = 'status-verdict-banner review';
      verdictIcon.textContent = '⚠';
      verdictTitle.textContent = 'REVIEW REQUIRED — SECONDARY DEFECT';
      verdictSubtitle.textContent = `Secondary defect identified: ${defect.prediction}`;
      verdictConfidence.textContent = `${(defect.confidence * 100).toFixed(1)}% CONF`;

      detSeverityVal.textContent = 'Moderate Risk (Secondary Defect)';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: SECONDARY SORT';
      recDesc.textContent = 'Minor defect anomaly. Run lot through optical sorter or gravity table before roasting.';
    } else {
      statusVerdictBanner.className = 'status-verdict-banner pass';
      verdictIcon.textContent = '✓';
      verdictTitle.textContent = 'PASS — QUALITY STANDARD MET';
      verdictSubtitle.textContent = 'Clean green coffee sample with no severe defects';
      verdictConfidence.textContent = `${(roast.confidence * 100).toFixed(1)}% CONF`;

      detSeverityVal.textContent = 'Low Risk (Clean Sample)';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: ACCEPT SAMPLE';
      recDesc.textContent = 'Sample conforms to clean green coffee grade requirements. Approved for production inventory.';
    }
  } else {
    // Roasted Batch Mode
    detDefectLabel.textContent = 'Batch Defects Found';
    const defectCount = result.roasted_defect?.defect_count || 0;
    detDefectVal.textContent = `${defectCount} Defect(s) Flagged`;

    if (defectCount > 3) {
      statusVerdictBanner.className = 'status-verdict-banner fail';
      verdictIcon.textContent = '✕';
      verdictTitle.textContent = 'REJECT — HIGH DEFECT VARIANCE';
      verdictSubtitle.textContent = `Multiple defects detected in roasted batch (${defectCount} total)`;
      verdictConfidence.textContent = 'YOLOv8 ACTIVE';

      detSeverityVal.textContent = 'High Defect Variance';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: QUARANTINE BATCH';
      recDesc.textContent = 'Excessive scorching or quakers detected. Check burner temperature and drum rotation speed.';
    } else if (defectCount > 0) {
      statusVerdictBanner.className = 'status-verdict-banner review';
      verdictIcon.textContent = '⚠';
      verdictTitle.textContent = 'REVIEW REQUIRED — MINOR DEFECTS';
      verdictSubtitle.textContent = `Isolated defects detected (${defectCount} total)`;
      verdictConfidence.textContent = 'YOLOv8 ACTIVE';

      detSeverityVal.textContent = 'Minor Defect Variance';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: MANUAL QUALITY CHECK';
      recDesc.textContent = 'Slight thermal or bean variance. Perform cup tasting verification before packaging.';
    } else {
      statusVerdictBanner.className = 'status-verdict-banner pass';
      verdictIcon.textContent = '✓';
      verdictTitle.textContent = 'PASS — PRODUCTION ROAST UNIFORM';
      verdictSubtitle.textContent = `Flawless roast profile: ${roast.prediction}`;
      verdictConfidence.textContent = `${(roast.confidence * 100).toFixed(1)}% CONF`;

      detSeverityVal.textContent = 'Nominal (Uniform Batch)';
      recTitle.textContent = 'RECOMMENDED OPERATOR ACTION: APPROVE FOR PACKAGING';
      recDesc.textContent = 'Roast profile is consistent and defect-free. Commit batch to packaging line.';
    }
  }

  // Render Full Probability Spectrum
  renderProbabilitySpectrum(result);
}

function renderProbabilitySpectrum(result) {
  spectrumBarsList.innerHTML = '';

  const probs = (result.defect && result.defect.probabilities)
    ? result.defect.probabilities
    : (result.roast && result.roast.probabilities)
    ? result.roast.probabilities
    : {};

  const entries = Object.entries(probs).sort(([, a], [, b]) => b - a);

  if (entries.length === 0) {
    spectrumBarsList.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem;padding:8px;">No probability distribution available.</div>';
    return;
  }

  entries.forEach(([label, prob], index) => {
    const pct = (prob * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'spectrum-row';
    row.innerHTML = `
      <span class="spectrum-row-label" title="${label}">${label}</span>
      <div class="spectrum-track">
        <div class="spectrum-fill" style="width: ${Math.max(prob * 100, 1)}%; ${index === 0 ? 'background:var(--accent);' : 'background:var(--text-muted);'}"></div>
      </div>
      <span class="spectrum-row-val">${pct}%</span>
    `;
    spectrumBarsList.appendChild(row);
  });
}

// ── Historical Operational Analytics (Real Backend Data) ──────────────

async function fetchHistoricalAnalytics() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    statsData = await res.json();

    // 1. Total count
    statTotalCount.textContent = (statsData.total_predictions || 0).toLocaleString();

    // 2. Avg latency
    statAvgLatency.textContent = `${statsData.avg_inference_time_ms || 0} ms`;

    // 3. Defect rate
    const total = statsData.total_predictions || 0;
    const defectCounts = statsData.predictions_by_type?.defect || 0;
    const rate = total > 0 ? ((defectCounts / total) * 100).toFixed(1) : '0.0';
    statDefectRate.textContent = `${rate}%`;

    // 4. Render real distribution bars
    renderDistributionBars(statsData.top_predicted_classes || {});
  } catch (e) {
    console.warn('Analytics fetch skipped:', e);
  }
}

function renderDistributionBars(topClasses) {
  const entries = Object.entries(topClasses).sort(([, a], [, b]) => b - a);

  if (entries.length === 0) {
    analyticsDistBars.innerHTML = '<div style="color:var(--text-muted);font-size:0.8rem;text-align:center;padding:16px;">No prediction records recorded yet. Run evaluations to populate operational distribution.</div>';
    return;
  }

  const maxVal = Math.max(...entries.map(([, count]) => count), 1);

  analyticsDistBars.innerHTML = entries.map(([className, count]) => {
    const pct = Math.round((count / maxVal) * 100);
    return `
      <div class="chart-row">
        <span class="chart-row-label">${className}</span>
        <div class="chart-track">
          <div class="chart-fill" style="width: ${pct}%;"></div>
        </div>
        <span class="chart-row-stat">${count} sample(s)</span>
      </div>
    `;
  }).join('');
}

// ── Quality Audit Log Table (Real Supabase Records) ────────────────────

async function fetchAuditLogs() {
  try {
    const res = await fetch('/api/history?limit=30');
    if (!res.ok) return;
    const data = await res.json();
    rawHistoryData = data.predictions || [];
    renderAuditTable();
  } catch (e) {
    console.warn('Audit fetch skipped:', e);
  }
}

document.querySelectorAll('.audit-filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.audit-filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.getAttribute('data-filter');
    renderAuditTable();
  });
});

function renderAuditTable() {
  if (!auditTableBody) return;

  if (!rawHistoryData || rawHistoryData.length === 0) {
    auditTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="audit-empty-state">No evaluations recorded in database yet. Run an inspection to start the audit trail.</td>
      </tr>
    `;
    return;
  }

  // Filter
  let filtered = rawHistoryData;
  if (activeFilter === 'pass') {
    filtered = rawHistoryData.filter(r => r.confidence < 0.4 || ['Dark', 'Medium', 'Light', 'Green'].includes(r.predicted_class));
  } else if (activeFilter === 'review') {
    filtered = rawHistoryData.filter(r => r.confidence >= 0.4 && r.confidence < 0.7);
  } else if (activeFilter === 'defect') {
    filtered = rawHistoryData.filter(r => r.confidence >= 0.7 && r.analysis_type === 'defect');
  }

  if (filtered.length === 0) {
    auditTableBody.innerHTML = `
      <tr>
        <td colspan="7" class="audit-empty-state">No evaluations matching filter "${activeFilter.toUpperCase()}".</td>
      </tr>
    `;
    return;
  }

  // Group or render individual rows
  auditTableBody.innerHTML = filtered.map(row => {
    const isDefect = row.analysis_type === 'defect' && row.confidence >= 0.6;
    const isReview = row.confidence >= 0.4 && row.confidence < 0.6;

    const decisionBadge = isDefect
      ? `<span class="badge-tag defect">DEFECT</span>`
      : isReview
      ? `<span class="badge-tag review">REVIEW</span>`
      : `<span class="badge-tag pass">PASS</span>`;

    const modeBadge = row.analysis_type === 'defect'
      ? `<span class="badge-tag green-mode">GREEN LOT</span>`
      : `<span class="badge-tag roasted-mode">ROASTED</span>`;

    const timeStr = row.timestamp
      ? new Date(row.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : 'Recent';

    const thumbHtml = row.image_url
      ? `<img class="audit-thumb-img" src="${row.image_url}" alt="Thumb" loading="lazy" />`
      : `<div class="audit-thumb-placeholder">📷</div>`;

    return `
      <tr>
        <td class="audit-thumb-cell">${thumbHtml}</td>
        <td style="font-family:var(--font-mono);font-size:0.75rem;color:var(--text-primary);">${row.filename || 'sample.jpg'}</td>
        <td>${modeBadge}</td>
        <td style="font-weight:600;color:var(--text-primary);">${row.predicted_class}</td>
        <td style="font-family:var(--font-mono);">${(row.confidence * 100).toFixed(1)}%</td>
        <td>${decisionBadge}</td>
        <td style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);">${timeStr}</td>
      </tr>
    `;
  }).join('');
}

// ── Live Health Status Check ───────────────────────────────────────────

async function checkSystemHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error();
    const data = await res.json();

    const allOnline = data.models?.roast && data.models?.defect;
    const partial = data.models?.roast || data.models?.defect;

    if (allOnline) {
      healthDot.className = 'status-dot operational';
      healthStatusText.textContent = 'System Operational';
    } else if (partial) {
      healthDot.className = 'status-dot degraded';
      healthStatusText.textContent = 'Degraded Telemetry';
    } else {
      healthDot.className = 'status-dot offline';
      healthStatusText.textContent = 'Models Offline';
    }
  } catch {
    healthDot.className = 'status-dot offline';
    healthStatusText.textContent = 'Service Unreachable';
  }
}

// ── Initialization ─────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  checkSystemHealth();
  fetchHistoricalAnalytics();
  fetchAuditLogs();
  setInterval(checkSystemHealth, 30000);
});
