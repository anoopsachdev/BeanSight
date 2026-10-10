/**
 * BeanSight — frontend controller
 *
 * Talks to:
 *   POST /api/predict  (roast level → green classifier | roasted detector)
 *   GET  /api/history  (recent logged analyses)
 *   GET  /api/stats    (aggregates)
 *   GET  /health       (model load status)
 */

'use strict';

// ── Domain knowledge ────────────────────────────────────────────────────

const ROAST_ORDER = ['Green', 'Light', 'Medium', 'Dark'];

// SCA green-coffee defect categories (Category 1 = primary, Category 2 = secondary).
const SCA_CATEGORY_1 = new Set(['Full Black', 'Full Sour', 'Dry Cherry', 'Fungus Damage', 'Severe Insect Damage']);
const SCA_CATEGORY_2 = new Set(['Partial Black', 'Partial Sour', 'Parchment', 'Floater', 'Immature', 'Withered', 'Shell', 'Broken', 'Cut', 'Husk', 'Slight Insect Damage']);

const DEFECT_INFO = {
  'Broken':               'Fractured or chipped bean, usually from hulling or handling.',
  'Cut':                  'Bean cut by processing machinery, often with a discoloured edge.',
  'Dry Cherry':           'Bean still enclosed in part or all of the dried cherry husk.',
  'Fade':                 'Bean has lost its colour, typically from age or poor storage.',
  'Floater':              'Low-density, pale bean; usually a result of poor drying or storage.',
  'Full Black':           'Entire bean is black and opaque, usually from over-fermentation.',
  'Full Sour':            'Entire bean shows sour, fermented discolouration.',
  'Fungus Damage':        'Mould spots or powdery patches from fungal attack.',
  'Husk':                 'Fragment of dried cherry skin mixed in with the beans.',
  'Immature':             'Underdeveloped bean, small with greenish-yellow colour and clinging silverskin.',
  'Parchment':            'Bean still fully or partly covered by its parchment layer.',
  'Partial Black':        'Part of the bean (less than half) is black.',
  'Partial Sour':         'Part of the bean shows sour, fermented discolouration.',
  'Severe Insect Damage': 'Three or more insect (coffee berry borer) holes.',
  'Shell':                'Malformed bean with a hollow, shell-like shape.',
  'Slight Insect Damage': 'One or two small insect holes.',
  'Withered':             'Shrivelled, wrinkled bean caused by drought or poor nutrition.',
  // Roasted detector classes
  'Quaker':               'Underdeveloped bean that stays pale after roasting, usually from unripe cherries.',
  'Scorched':             'Surface burn marks from excessive heat contact during roasting.',
  'Normal':               'No visible defect.',
};

const ROASTED_CHIP_STYLE = { Normal: 'green', Broken: 'amber', Quaker: 'pink', Scorched: 'red' };

// ── DOM ─────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const dropzone      = $('dropzone');
const fileInput     = $('file-input');
const preview       = $('preview');
const previewImg    = $('preview-img');
const previewName   = $('preview-name');
const previewSize   = $('preview-size');
const analyzeBtn    = $('analyze-btn');
const clearBtn      = $('clear-btn');

const resultsEmpty  = $('results-empty');
const resultsLoading= $('results-loading');
const results       = $('results');
const latencyTag    = $('latency-tag');

const roastValue    = $('roast-value');
const roastConf     = $('roast-conf');
const roastBars     = $('roast-bars');
const roastNote     = $('roast-note');
const routeChip     = $('route-chip');

const stageGreen    = $('stage-green');
const greenDefect   = $('green-defect');
const greenChips    = $('green-chips');
const greenDesc     = $('green-desc');
const greenBars     = $('green-bars');
const greenToggle   = $('green-toggle');

const stageRoasted  = $('stage-roasted');
const annotatedWrap = $('annotated-wrap');
const annotatedImg  = $('annotated-img');
const rBeans        = $('r-beans');
const rNormal       = $('r-normal');
const rDefects      = $('r-defects');
const roastedChips  = $('roasted-chips');
const roastedDesc   = $('roasted-desc');

const statusDot     = $('status-dot');
const statusText    = $('status-text');
const kpiTotal      = $('kpi-total');
const kpiLatency    = $('kpi-latency');
const kpiSplit      = $('kpi-split');
const kpiModels     = $('kpi-models');
const historyBody   = $('history-body');
const defectDist    = $('defect-dist');
const toastContainer= $('toast-container');

// ── State ───────────────────────────────────────────────────────────────

let selectedFile = null;
let lastGreenProbs = null;
let showAllGreen = false;
let beanType = 'auto'; // 'auto' | 'green' | 'roasted'

const beanTypeGroup = $('bean-type');
const routeNote = $('route-note');

function setBeanType(value) {
  beanType = value;
  beanTypeGroup.querySelectorAll('.seg').forEach((b) => {
    const on = b.dataset.value === value;
    b.classList.toggle('active', on);
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  });
}

beanTypeGroup.addEventListener('click', (e) => {
  const btn = e.target.closest('.seg');
  if (btn) setBeanType(btn.dataset.value);
});

// ── Helpers ─────────────────────────────────────────────────────────────

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function pct(p, digits = 1) {
  return `${(Number(p || 0) * 100).toFixed(digits)}%`;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(message, type = 'info', duration = 3200) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add('out');
    setTimeout(() => el.remove(), 220);
  }, duration);
}

function renderBars(container, probs, { limit = Infinity, rowClass = 'bar', nameWidth } = {}) {
  const entries = Object.entries(probs || {}).sort(([, a], [, b]) => b - a).slice(0, limit);
  container.innerHTML = entries.map(([name, p], i) => `
    <div class="${rowClass}${i === 0 ? ' top' : ''}">
      <span class="name" title="${escapeHtml(name)}"${nameWidth ? ` style="width:${nameWidth}"` : ''}>${escapeHtml(name)}</span>
      <div class="track"><div class="fill" style="width:${Math.max(Number(p) * 100, 1.5)}%"></div></div>
      <span class="val">${pct(p)}</span>
    </div>`).join('');
}

function scaChipFor(defect) {
  if (SCA_CATEGORY_1.has(defect)) return '<span class="chip red"><span class="dot"></span>SCA Category 1 · primary defect</span>';
  if (SCA_CATEGORY_2.has(defect)) return '<span class="chip amber"><span class="dot"></span>SCA Category 2 · secondary defect</span>';
  return '<span class="chip"><span class="dot"></span>Dataset class (not an SCA category)</span>';
}

// ── File handling ───────────────────────────────────────────────────────

function acceptFile(file) {
  if (!file) return;
  const okTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
  if (!okTypes.includes(file.type)) {
    showToast('Unsupported file type. Please use JPG, PNG or WEBP.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('That file is larger than 10 MB.', 'error');
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    dropzone.hidden = true;
    preview.hidden = false;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearAll() {
  selectedFile = null;
  fileInput.value = '';
  previewImg.removeAttribute('src');
  preview.hidden = true;
  dropzone.hidden = false;
  analyzeBtn.disabled = true;

  results.hidden = true;
  resultsLoading.hidden = true;
  resultsEmpty.hidden = false;
  latencyTag.textContent = '';
}

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag');
  acceptFile(e.dataTransfer.files?.[0]);
});
fileInput.addEventListener('change', (e) => acceptFile(e.target.files?.[0]));
clearBtn.addEventListener('click', clearAll);

// Allow pasting an image from the clipboard anywhere on the page.
document.addEventListener('paste', (e) => {
  const item = Array.from(e.clipboardData?.items || []).find((i) => i.type.startsWith('image/'));
  if (item) acceptFile(item.getAsFile());
});

// ── Analyze ─────────────────────────────────────────────────────────────

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  resultsEmpty.hidden = true;
  results.hidden = true;
  resultsLoading.hidden = false;
  latencyTag.textContent = '';

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('bean_type', beanType);

  try {
    const res = await fetch('/api/predict', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${res.status})`);
    }
    const data = await res.json();
    renderResult(data);
    loadStats();
    loadHistory();
  } catch (err) {
    console.error(err);
    showToast(err.message || 'Analysis failed. Please try again.', 'error');
    resultsEmpty.hidden = false;
  } finally {
    resultsLoading.hidden = true;
    analyzeBtn.disabled = !selectedFile;
  }
});

// ── Rendering ───────────────────────────────────────────────────────────

function renderResult(data) {
  const roast = data.roast;
  const route = data.route || (roast?.prediction === 'Green' ? 'green' : 'roasted');

  // Stage 1 — roast level
  if (roast) {
    roastValue.textContent = roast.prediction;
    roastConf.textContent = `${pct(roast.confidence)} confidence`;
    const ordered = {};
    ROAST_ORDER.forEach((k) => { if (k in (roast.probabilities || {})) ordered[k] = roast.probabilities[k]; });
    renderBars(roastBars, ordered, { rowClass: 'mini-bar' });

    if (roast.confidence < 0.5) {
      roastNote.innerHTML = '<svg aria-hidden="true"><use href="#i-alert"/></svg><span>Low confidence: the roast classifier is unsure about this image. A single bean on a plain, light background gives the most reliable result.</span>';
      roastNote.hidden = false;
    } else {
      roastNote.hidden = true;
    }
  } else {
    roastValue.textContent = 'Unavailable';
    roastConf.textContent = 'Roast model not loaded';
    roastBars.innerHTML = '';
    roastNote.hidden = true;
  }

  // Route chip + explanation
  const manual = data.route_source === 'manual';
  const prefix = manual ? 'Selected manually' : 'Auto-detected';
  if (route === 'green') {
    routeChip.className = 'route-chip green';
    routeChip.innerHTML = `<svg aria-hidden="true"><use href="#i-arrow"/></svg> ${prefix}: green bean → 17-class defect classifier`;
  } else {
    routeChip.className = 'route-chip roasted';
    routeChip.innerHTML = `<svg aria-hidden="true"><use href="#i-arrow"/></svg> ${prefix}: roasted bean → YOLOv8 defect detector`;
  }

  const classifierRoute = data.classifier_route || (roast?.prediction === 'Green' ? 'green' : 'roasted');
  if (manual && classifierRoute !== route) {
    routeNote.innerHTML = `<svg aria-hidden="true"><use href="#i-alert"/></svg><span>The roast classifier predicted <strong>${escapeHtml(roast?.prediction || '—')}</strong> (${classifierRoute} bean), but you selected <strong>${route}</strong>, so the ${route === 'green' ? 'green defect classifier' : 'roasted defect detector'} was used instead.</span>`;
    routeNote.hidden = false;
  } else if (!manual && roast && roast.confidence < 0.7) {
    routeNote.innerHTML = `<svg aria-hidden="true"><use href="#i-alert"/></svg><span>Not the right bean type? The roast classifier was only ${pct(roast.confidence, 0)} sure. Set <strong>Bean type</strong> manually and analyze again.</span>`;
    routeNote.hidden = false;
  } else {
    routeNote.hidden = true;
  }

  // Stage 2
  stageGreen.hidden = route !== 'green';
  stageRoasted.hidden = route !== 'roasted';
  if (route === 'green') renderGreen(data.defect);
  else renderRoasted(data.roasted_defect, data.defect);

  latencyTag.textContent = data.inference_time_ms != null ? `${data.inference_time_ms} ms` : '';

  resultsEmpty.hidden = true;
  results.hidden = false;
}

function renderGreen(defect) {
  if (!defect) {
    greenDefect.textContent = 'Unavailable';
    greenChips.innerHTML = '';
    greenDesc.textContent = 'The green bean defect model is not loaded.';
    greenBars.innerHTML = '';
    greenToggle.hidden = true;
    return;
  }
  greenDefect.textContent = defect.prediction;
  greenChips.innerHTML = `${scaChipFor(defect.prediction)}<span class="chip"><span class="mono">${pct(defect.confidence)}</span> confidence</span>`;
  greenDesc.textContent = DEFECT_INFO[defect.prediction] || '';

  lastGreenProbs = defect.probabilities || {};
  showAllGreen = false;
  renderGreenBars();
  greenToggle.hidden = Object.keys(lastGreenProbs).length <= 5;
}

function renderGreenBars() {
  renderBars(greenBars, lastGreenProbs, { limit: showAllGreen ? Infinity : 5 });
  greenToggle.textContent = showAllGreen ? 'Show top 5 only' : `Show all ${Object.keys(lastGreenProbs).length} classes`;
}

greenToggle.addEventListener('click', () => {
  showAllGreen = !showAllGreen;
  renderGreenBars();
});

function renderRoasted(roasted, defectSummary) {
  if (!roasted) {
    annotatedWrap.hidden = true;
    rBeans.textContent = rNormal.textContent = rDefects.textContent = '—';
    roastedChips.innerHTML = '';
    roastedDesc.textContent = 'The roasted bean detector is not loaded.';
    return;
  }

  const beans = roasted.bean_count ?? (roasted.detections?.length || 0);
  const defects = roasted.defect_count ?? 0;
  const normal = roasted.normal_count ?? Math.max(beans - defects, 0);
  const classSummary = roasted.class_summary || roasted.defect_summary || {};

  if (roasted.annotated_image) {
    annotatedImg.src = roasted.annotated_image;
    annotatedWrap.hidden = false;
  } else {
    annotatedWrap.hidden = true;
  }

  rBeans.textContent = beans;
  rNormal.textContent = normal;
  rDefects.textContent = defects;

  roastedChips.innerHTML = Object.entries(classSummary)
    .sort(([a], [b]) => (a === 'Normal') - (b === 'Normal') || a.localeCompare(b))
    .map(([cls, n]) => `<span class="chip ${ROASTED_CHIP_STYLE[cls] || 'blue'}"><span class="dot"></span>${escapeHtml(cls)} <span class="mono">×${n}</span></span>`)
    .join('');

  if (beans === 0) {
    roastedDesc.textContent = 'No beans were detected. Try a closer, sharper photo of roasted beans on a plain background.';
  } else if (defects === 0) {
    roastedDesc.textContent = `All ${beans} detected bean${beans === 1 ? '' : 's'} look normal (mean detector confidence ${pct(roasted.mean_confidence)}).`;
  } else {
    const top = defectSummary?.prediction;
    const info = DEFECT_INFO[top] ? ` ${top}: ${DEFECT_INFO[top]}` : '';
    roastedDesc.textContent = `${defects} of ${beans} detected bean${beans === 1 ? '' : 's'} flagged as defective, most often "${top}" (mean confidence of defect boxes ${pct(defectSummary?.confidence)}).${info}`;
  }
}

// ── Activity: stats, history, health ────────────────────────────────────

function setActivityNotice(available) {
  const notice = $('activity-notice');
  if (!notice) return;
  notice.hidden = available;
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error();
    const s = await res.json();

    if (s.database_available === false) {
      setActivityNotice(false);
      kpiTotal.textContent = kpiLatency.textContent = kpiSplit.textContent = '—';
      defectDist.innerHTML = '<p class="muted small">Not available while the activity log is offline.</p>';
      return;
    }
    setActivityNotice(true);

    const byType = s.predictions_by_type || {};
    const total = s.total_samples ?? byType.roast ?? Math.round((s.total_predictions || 0) / 2);
    kpiTotal.textContent = Number(total).toLocaleString();
    kpiLatency.textContent = s.avg_inference_time_ms ? `${Math.round(s.avg_inference_time_ms)} ms` : '—';

    const rd = s.roast_distribution || {};
    const green = rd.Green || 0;
    const roasted = (rd.Light || 0) + (rd.Medium || 0) + (rd.Dark || 0);
    kpiSplit.textContent = green + roasted > 0 ? `${green} / ${roasted}` : '—';

    const dist = s.defect_distribution && Object.keys(s.defect_distribution).length
      ? s.defect_distribution
      : Object.fromEntries(Object.entries(s.top_predicted_classes || {}).filter(([k]) => !ROAST_ORDER.includes(k) && !['Normal', 'No Defect', 'None (Clean)'].includes(k)));

    const entries = Object.entries(dist).sort(([, a], [, b]) => b - a);
    if (!entries.length) {
      defectDist.innerHTML = '<p class="muted small">No defects logged yet.</p>';
    } else {
      const max = Math.max(...entries.map(([, n]) => n), 1);
      defectDist.innerHTML = entries.map(([name, n], i) => `
        <div class="bar${i === 0 ? ' top' : ''}">
          <span class="name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
          <div class="track"><div class="fill" style="width:${Math.max((n / max) * 100, 2)}%"></div></div>
          <span class="val">${n}</span>
        </div>`).join('');
    }
  } catch {
    kpiTotal.textContent = kpiLatency.textContent = kpiSplit.textContent = '—';
    defectDist.innerHTML = '<p class="muted small">Statistics unavailable.</p>';
  }
}

/**
 * Each analysis is stored as two rows (roast + defect) that share filename,
 * image_url and inference_time_ms. Pair them back up so the table shows one
 * line per analysed image.
 */
function groupHistory(rows) {
  const groups = [];
  const byKey = new Map();
  for (const row of rows) {
    const key = `${row.image_url || ''}|${row.filename || ''}|${row.inference_time_ms ?? ''}`;
    let g = byKey.get(key);
    if (!g || (g.roast && row.analysis_type === 'roast') || (g.defect && row.analysis_type !== 'roast')) {
      g = { key, timestamp: row.timestamp, filename: row.filename, image_url: row.image_url, roast: null, defect: null };
      byKey.set(key, g);
      groups.push(g);
    }
    if (row.analysis_type === 'roast') g.roast = row; else g.defect = row;
    if (row.image_url && !g.image_url) g.image_url = row.image_url;
  }
  return groups;
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history?limit=40');
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (data.database_available === false) {
      historyBody.innerHTML = '<tr><td colspan="5" class="table-empty">The activity log is temporarily offline. Analysis still works — results just are not being recorded right now.</td></tr>';
      return;
    }
    const groups = groupHistory(data.predictions || []).slice(0, 20);

    if (!groups.length) {
      historyBody.innerHTML = '<tr><td colspan="5" class="table-empty">No analyses logged yet. Upload an image to get started.</td></tr>';
      return;
    }

    historyBody.innerHTML = groups.map((g) => {
      const roastLabel = g.roast?.predicted_class || '—';
      const roastConfTxt = g.roast ? pct(g.roast.confidence, 0) : '';
      const isRoasted = g.defect?.analysis_type === 'roasted_defect' || (g.roast && g.roast.predicted_class !== 'Green');
      const defectLabel = g.defect?.predicted_class || '—';
      const defectConf = g.defect ? pct(g.defect.confidence) : '—';
      const when = g.timestamp ? new Date(g.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
      const thumb = g.image_url
        ? `<img class="thumb" src="${escapeHtml(g.image_url)}" alt="" loading="lazy" />`
        : '<div class="thumb-ph"><svg aria-hidden="true"><use href="#i-image"/></svg></div>';
      const defectClass = defectLabel === 'Normal' || defectLabel === 'No Defect' || defectLabel === 'None (Clean)' ? 'green' : (isRoasted ? 'amber' : 'red');

      return `
        <tr>
          <td>${thumb}</td>
          <td><div class="cell-main">${escapeHtml(roastLabel)}</div><div class="cell-sub mono">${roastConfTxt}</div></td>
          <td><div class="cell-main"><span class="chip ${defectClass}"><span class="dot"></span>${escapeHtml(defectLabel)}</span></div><div class="cell-sub">${isRoasted ? 'Roasted detector' : 'Green classifier'}</div></td>
          <td class="mono">${defectConf}</td>
          <td class="muted">${when}</td>
        </tr>`;
    }).join('');
  } catch {
    historyBody.innerHTML = '<tr><td colspan="5" class="table-empty">History unavailable.</td></tr>';
  }
}

async function loadHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error();
    const data = await res.json();
    const m = data.models || {};
    const loaded = ['roast', 'defect', 'roasted_yolo'].filter((k) => m[k]).length;

    kpiModels.textContent = `${loaded} / 3`;
    const dbDown = data.database && data.database.connected === false;
    if (loaded === 3 && dbDown) {
      statusDot.className = 'status-dot warn';
      statusText.textContent = 'Models ready · activity log offline';
      statusText.parentElement.title = data.database.error ? `Database: ${data.database.error}` : 'Database unreachable';
    } else if (loaded === 3) {
      statusDot.className = 'status-dot ok';
      statusText.textContent = 'All models loaded';
      statusText.parentElement.title = 'Live model status from /health';
    } else if (loaded > 0) {
      statusDot.className = 'status-dot warn';
      statusText.textContent = `${loaded} of 3 models loaded`;
    } else {
      statusDot.className = 'status-dot warn';
      statusText.textContent = 'Models loading…';
    }
    return loaded === 3;
  } catch {
    statusDot.className = 'status-dot bad';
    statusText.textContent = 'Service unreachable';
    kpiModels.textContent = '—';
    return false;
  }
}

// ── Init ────────────────────────────────────────────────────────────────

(async function init() {
  loadStats();
  loadHistory();
  // Models load in the background after a cold start; poll quickly until ready, then slowly.
  const ready = await loadHealth();
  if (ready) {
    setInterval(loadHealth, 60000);
    return;
  }
  let attempts = 0;
  const timer = setInterval(async () => {
    attempts += 1;
    if (await loadHealth() || attempts > 30) {
      clearInterval(timer);
      setInterval(loadHealth, 60000);
    }
  }, 2000);
})();
