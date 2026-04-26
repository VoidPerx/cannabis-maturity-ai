const API = '';

// ── Estado ──────────────────────────────────────────────────────────
let lastPrediction = null;

// ── DOM refs ─────────────────────────────────────────────────────────
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const preview    = document.getElementById('img-preview');
const resultBox  = document.getElementById('result-box');
const statsBox   = document.getElementById('stats-box');
const spinner    = document.getElementById('spinner');
const uploadBtn  = document.getElementById('upload-btn');

// ── Drag & Drop ───────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.classList.remove('hidden');
    uploadBtn.disabled = false;
    uploadBtn._file = file;
  };
  reader.readAsDataURL(file);
}

// ── Predict ───────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  const file = uploadBtn._file;
  if (!file) return;

  spinner.classList.remove('hidden');
  resultBox.innerHTML = '';
  uploadBtn.disabled = true;

  const form = new FormData();
  form.append('file', file);
  form.append('plant_id', document.getElementById('plant-id').value);
  form.append('strain', document.getElementById('strain').value);
  form.append('week_of_flower', document.getElementById('week').value || '');

  try {
    const res = await fetch(`${API}/predict/`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    lastPrediction = data;
    renderResult(data);
    loadStats();
  } catch (err) {
    resultBox.innerHTML = `<div class="error">Error: ${err.message}</div>`;
  } finally {
    spinner.classList.add('hidden');
    uploadBtn.disabled = false;
  }
});

// ── Render resultado ──────────────────────────────────────────────────
const STAGE_CONFIG = {
  early:         { label: 'Temprana',      color: '#4ade80', icon: '🌱' },
  mid:           { label: 'Media',         color: '#facc15', icon: '🌿' },
  late:          { label: 'Tardía',        color: '#f97316', icon: '🌸' },
  harvest_ready: { label: 'Lista para cosechar', color: '#ef4444', icon: '🌺' },
};

function renderResult(data) {
  const stage = STAGE_CONFIG[data.maturity_stage] || { label: data.maturity_stage, color: '#aaa', icon: '?' };
  const conf = (data.confidence * 100).toFixed(1);
  const t = data.trichomes;
  const s = data.stigmas;

  resultBox.innerHTML = `
    <div class="result-card" style="border-color: ${stage.color}">
      <div class="result-header" style="background: ${stage.color}22">
        <span class="stage-icon">${stage.icon}</span>
        <div>
          <h2 style="color:${stage.color}">${stage.label}</h2>
          <p class="conf">Confianza: <strong>${conf}%</strong></p>
        </div>
      </div>

      <div class="metrics-grid">
        <div class="metric-card stigma">
          <h4>Estigmas</h4>
          <div class="bar-row">
            <span>Naranjos</span>
            <div class="bar"><div style="width:${s.orange_pct}%; background:#f97316"></div></div>
            <span>${s.orange_pct}%</span>
          </div>
          <div class="bar-row">
            <span>Blancos</span>
            <div class="bar"><div style="width:${s.white_pct}%; background:#e5e7eb"></div></div>
            <span>${s.white_pct}%</span>
          </div>
        </div>

        <div class="metric-card trichome">
          <h4>Trichomas</h4>
          <div class="bar-row">
            <span>Ámbar</span>
            <div class="bar"><div style="width:${t.amber_pct}%; background:#d97706"></div></div>
            <span>${t.amber_pct}%</span>
          </div>
          <div class="bar-row">
            <span>Lechosos</span>
            <div class="bar"><div style="width:${t.milky_pct}%; background:#f3f4f6"></div></div>
            <span>${t.milky_pct}%</span>
          </div>
          <div class="bar-row">
            <span>Claros</span>
            <div class="bar"><div style="width:${t.clear_pct}%; background:#d1fae5"></div></div>
            <span>${t.clear_pct}%</span>
          </div>
        </div>
      </div>

      ${data.annotated_image_url ? `
        <div class="annotated-img">
          <p>Imagen anotada:</p>
          <img src="${data.annotated_image_url}" alt="Resultado anotado" />
        </div>` : ''}

      <div class="inference-time">⚡ ${data.total_inference_ms.toFixed(0)} ms</div>

      <div class="maturity-guide">
        ${renderMaturityGuide(data.maturity_stage)}
      </div>
    </div>`;
}

function renderMaturityGuide(current) {
  const stages = ['early', 'mid', 'late', 'harvest_ready'];
  const idx = stages.indexOf(current);
  return `
    <div class="guide-bar">
      ${stages.map((s, i) => {
        const c = STAGE_CONFIG[s];
        return `<div class="guide-step ${i <= idx ? 'active' : ''}" style="--color:${c.color}">
          <span>${c.icon}</span><small>${c.label}</small>
        </div>`;
      }).join('')}
    </div>`;
}

// ── Stats del dashboard ───────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch(`${API}/samples/stats`);
    const data = await res.json();
    statsBox.innerHTML = `
      <div class="stat"><span>${data.total_samples}</span><small>Muestras</small></div>
      <div class="stat"><span>${data.total_predictions}</span><small>Predicciones</small></div>
      <div class="stat"><span>${(data.avg_confidence * 100).toFixed(1)}%</span><small>Conf. promedio</small></div>
      ${Object.entries(data.stage_distribution).map(([k, v]) => {
        const c = STAGE_CONFIG[k] || {};
        return `<div class="stat"><span>${v}</span><small>${c.label || k} ${c.icon || ''}</small></div>`;
      }).join('')}
    `;
  } catch {}
}

loadStats();
setInterval(loadStats, 30000);
