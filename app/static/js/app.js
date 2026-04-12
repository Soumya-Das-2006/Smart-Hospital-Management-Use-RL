/* app.js — Smart Hospital Allocator frontend logic */

/* ── Utility ───────────────────────────────────────────────────────────── */
const API = {
  async post(path, body = {}) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  },
  async get(path) {
    const r = await fetch(path);
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  },
  async del(path, body = {}) {
    const r = await fetch(path, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  },
};

function toast(msg, type = 'info') {
  const c = document.getElementById('toastContainer');
  if (!c) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => el.remove(), 3700);
}

function setGlobalStatus(s) {
  const dot = document.getElementById('globalStatus');
  const label = document.getElementById('globalStatusLabel');
  if (!dot || !label) return;
  dot.className = `status-dot ${s}`;
  label.textContent = s;
}

/* ── Dashboard state ───────────────────────────────────────────────────── */
const state = {
  sessionId: null,
  done: false,
  stepCount: 0,
  totalReward: 0,
  lastInfo: {},
  lastObs: null,
  log: [],
};

function addLog(line, type = 'info') {
  state.log.push({ line, type });
  const box = document.getElementById('logBox');
  if (!box) return;
  const el = document.createElement('div');
  el.className = `log-line ${type}`;
  el.textContent = `[${new Date().toLocaleTimeString()}] ${line}`;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
}

/* ── Render functions ──────────────────────────────────────────────────── */
function renderMetrics(info, reward) {
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? '—';
  };

  const ts = info.timestep ?? 0;
  const shift = 480;
  set('mTimestep', `${ts}/${shift}`);
  set('mReward',   typeof reward === 'number' ? reward.toFixed(1) : (info.total_reward ?? 0).toFixed(1));
  set('mSurvived', info.patients_survived ?? 0);
  set('mDied',     info.patients_died ?? 0);
  set('mQueue',    info.queue_length ?? 0);
  set('mAdmitted', info.admitted_count ?? 0);

  const sr = (info.survival_rate ?? 0) * 100;
  set('mSurvivalRate', `${sr.toFixed(1)}%`);
  const ta = (info.triage_accuracy ?? 1) * 100;
  set('mTriageAcc', `${ta.toFixed(1)}%`);

  // Colour coding
  const diedEl = document.getElementById('mDied');
  if (diedEl) diedEl.className = `metric-value ${(info.patients_died ?? 0) > 3 ? 'red' : 'green'}`;
  const srEl = document.getElementById('mSurvivalRate');
  if (srEl) srEl.className = `metric-value ${sr >= 70 ? 'green' : sr >= 50 ? 'orange' : 'red'}`;

  // Progress bar for shift
  const pb = document.getElementById('shiftBar');
  if (pb) pb.style.width = `${Math.min((ts / shift) * 100, 100)}%`;
}

function renderResources(info) {
  const resources = [
    { id: 'rICU',     label: 'ICU Beds',   val: info.icu_beds_free,      max: 10,  cls: 'red' },
    { id: 'rGen',     label: 'Gen Beds',   val: info.general_beds_free,  max: 30,  cls: 'blue' },
    { id: 'rDocs',    label: 'Doctors',    val: info.doctors_available,  max: 8,   cls: 'green' },
    { id: 'rNurses',  label: 'Nurses',     val: info.nurses_available,   max: 15,  cls: 'purple' },
  ];

  resources.forEach(r => {
    const valEl = document.getElementById(`${r.id}Val`);
    const barEl = document.getElementById(`${r.id}Bar`);
    if (valEl) valEl.textContent = `${r.val ?? 0}/${r.max}`;
    if (barEl) {
      const pct = ((r.val ?? 0) / r.max) * 100;
      barEl.style.width = `${Math.min(pct, 100)}%`;
      barEl.className = `bar-fill ${pct < 25 ? 'red' : pct < 50 ? 'orange' : r.cls}`;
    }
  });
}

function renderQueue(obs) {
  const tbody = document.getElementById('queueBody');
  if (!tbody || !obs) return;
  tbody.innerHTML = '';

  const patients = obs.patients ?? [];
  let shown = 0;
  for (let i = 0; i < 20 && shown < 15; i++) {
    const base = i * 4;
    const sev  = Math.round(patients[base] * 5);
    const wait = Math.round(patients[base + 1] * 60);
    const tx   = Math.round(patients[base + 2] * 120);
    const age  = (patients[base + 3] ?? 0).toFixed(2);
    if (sev <= 0) continue;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${shown + 1}</td>
      <td><span class="sev-badge sev-${sev}">${sev}</span></td>
      <td>${wait} min</td>
      <td>${tx} min</td>
      <td>${age}</td>
    `;
    tbody.appendChild(tr);
    shown++;
  }

  if (shown === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:1.5rem">Queue is empty — system stable</td></tr>`;
  }
}

function renderSessionInfo() {
  const el = document.getElementById('sessionIdVal');
  if (el) el.textContent = state.sessionId ? state.sessionId.slice(0, 18) + '…' : 'None';
  const stepEl = document.getElementById('stepCountVal');
  if (stepEl) stepEl.textContent = state.stepCount;
}

function setActionsEnabled(enabled) {
  document.querySelectorAll('.action-btn').forEach(btn => {
    btn.disabled = !enabled || state.done;
  });
  const hBtn = document.getElementById('btnHeuristic');
  if (hBtn) hBtn.disabled = !enabled || state.done;
}

function applyStepResult(data) {
  state.done        = data.done;
  state.stepCount   = data.step_count ?? (state.stepCount + 1);
  state.totalReward = data.total_reward ?? state.totalReward;
  state.lastInfo    = data.info ?? {};
  state.lastObs     = data.observation ?? state.lastObs;

  renderMetrics(state.lastInfo, data.reward);
  renderResources(state.lastInfo);
  renderQueue(state.lastObs);
  renderSessionInfo();
  setActionsEnabled(!state.done);

  const actionName = data.action_name ?? `action ${data.action ?? ''}`;
  const rwSign = data.reward >= 0 ? '+' : '';
  addLog(`STEP  action=${actionName}  reward=${rwSign}${(data.reward ?? 0).toFixed(2)}  done=${state.done}`, state.done ? 'done' : 'step');

  if (state.done) {
    setGlobalStatus('done');
    toast(`Episode finished! Total reward: ${state.totalReward.toFixed(1)}`, 'info');
    addLog(`DONE  total_reward=${state.totalReward.toFixed(2)}  survived=${state.lastInfo.patients_survived ?? 0}  died=${state.lastInfo.patients_died ?? 0}`, 'done');
  }
}

/* ── Session control ───────────────────────────────────────────────────── */
async function doReset() {
  const seedInput = document.getElementById('seedInput');
  const seed = seedInput && seedInput.value !== '' ? parseInt(seedInput.value) : undefined;

  try {
    setGlobalStatus('running');
    if (state.sessionId) {
      try { await API.del('/api/session', { session_id: state.sessionId }); } catch (_) {}
    }

    const data = await API.post('/api/reset', seed !== undefined ? { seed } : {});
    state.sessionId   = data.session_id;
    state.done        = data.done;
    state.stepCount   = 0;
    state.totalReward = 0;
    state.lastInfo    = data.info ?? {};
    state.lastObs     = data.observation ?? null;

    document.getElementById('logBox').innerHTML = '';
    addLog(`RESET  session=${state.sessionId.slice(0, 16)}…  seed=${seed ?? 'random'}`, 'reset');

    renderMetrics(state.lastInfo, 0);
    renderResources(state.lastInfo);
    renderQueue(state.lastObs);
    renderSessionInfo();
    setActionsEnabled(true);
    setGlobalStatus('online');
    toast('New episode started!', 'success');
  } catch (err) {
    setGlobalStatus('error');
    toast(err.message, 'error');
    addLog(`ERROR  ${err.message}`, 'error');
  }
}

async function doClose() {
  if (!state.sessionId) return;
  try {
    await API.del('/api/session', { session_id: state.sessionId });
    state.sessionId = null;
    state.done = false;
    setActionsEnabled(false);
    renderSessionInfo();
    setGlobalStatus('idle');
    toast('Session closed', 'info');
    addLog('CLOSE  session released', 'info');
  } catch (err) {
    toast(err.message, 'error');
  }
}

async function doStep(actionId) {
  if (!state.sessionId || state.done) return;
  try {
    setGlobalStatus('running');
    const data = await API.post('/api/step', { session_id: state.sessionId, action: actionId });
    applyStepResult(data);
    if (!state.done) setGlobalStatus('online');
  } catch (err) {
    setGlobalStatus('error');
    toast(err.message, 'error');
    addLog(`ERROR  step=${actionId}  ${err.message}`, 'error');
  }
}

async function doHeuristicStep() {
  if (!state.sessionId || state.done) return;
  try {
    setGlobalStatus('running');
    const data = await API.post('/api/heuristic-step', { session_id: state.sessionId });
    applyStepResult(data);
    if (!state.done) setGlobalStatus('online');
  } catch (err) {
    setGlobalStatus('error');
    toast(err.message, 'error');
    addLog(`ERROR  heuristic-step  ${err.message}`, 'error');
  }
}

async function doAutoRun() {
  if (!state.sessionId) { toast('Create a session first', 'error'); return; }
  const btn = document.getElementById('btnAutoRun');
  if (btn) btn.disabled = true;

  addLog('AUTO-RUN  running heuristic agent continuously…', 'info');
  while (!state.done) {
    await doHeuristicStep();
    await new Promise(r => setTimeout(r, 80)); // slight delay for UI updates
  }
  if (btn) btn.disabled = false;
}

/* ── Download helpers ──────────────────────────────────────────────────── */
function downloadJSON(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadCSV(rows, headers, filename) {
  const lines = [headers.join(','), ...rows.map(r => r.map(v => `"${v}"`).join(','))];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadText(text, filename) {
  const blob = new Blob([text], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ── Admin panel actions ───────────────────────────────────────────────── */
async function downloadSessionStatus() {
  try {
    const data = await API.get('/api/status');
    downloadJSON(data, `hospital-status-${Date.now()}.json`);
    toast('Status downloaded', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

async function downloadEnvInfo() {
  try {
    const data = await API.get('/api/info');
    downloadJSON(data, 'hospital-env-info.json');
    toast('Env info downloaded', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

async function downloadEpisodeReport() {
  try {
    const seedInput = document.getElementById('adminSeed');
    const seed = seedInput && seedInput.value !== '' ? parseInt(seedInput.value) : undefined;
    const btn = document.getElementById('btnDemoRun');
    if (btn) { btn.disabled = true; btn.textContent = 'Running…'; }

    const data = await API.post('/api/demo', seed !== undefined ? { seed } : {});
    downloadJSON(data, `hospital-episode-report-${Date.now()}.json`);
    toast(`Episode done! Survival: ${data.survival_rate}%`, 'success');

    // Also show in results area
    const resultsEl = document.getElementById('demoResults');
    if (resultsEl) {
      resultsEl.innerHTML = `
        <div class="info-grid">
          <div class="info-item"><div class="info-label">Total Reward</div><div class="info-val">${data.total_reward}</div></div>
          <div class="info-item"><div class="info-label">Steps</div><div class="info-val">${data.steps}</div></div>
          <div class="info-item"><div class="info-label">Survived</div><div class="info-val green">${data.survived}</div></div>
          <div class="info-item"><div class="info-label">Died</div><div class="info-val red">${data.died}</div></div>
          <div class="info-item"><div class="info-label">Survival Rate</div><div class="info-val">${data.survival_rate}%</div></div>
          <div class="info-item"><div class="info-label">Triage Accuracy</div><div class="info-val">${data.triage_accuracy}%</div></div>
        </div>`;
    }
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    const btn = document.getElementById('btnDemoRun');
    if (btn) { btn.disabled = false; btn.textContent = '▶ Run & Download Report'; }
  }
}

async function downloadHourlyCSV() {
  try {
    const seedInput = document.getElementById('adminSeed');
    const seed = seedInput && seedInput.value !== '' ? parseInt(seedInput.value) : undefined;
    const data = await API.post('/api/demo', seed !== undefined ? { seed } : {});
    if (!data.hourly_snapshots?.length) { toast('No hourly data available', 'error'); return; }
    const headers = ['Hour', 'Queue', 'Survived', 'Died', 'Reward_So_Far', 'Survival_Rate_%'];
    const rows = data.hourly_snapshots.map(h => [h.hour, h.queue, h.survived, h.died, h.reward_so_far, h.survival_rate]);
    downloadCSV(rows, headers, `hospital-hourly-${Date.now()}.csv`);
    toast('Hourly CSV downloaded', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

async function downloadRewardsCSV() {
  try {
    const seedInput = document.getElementById('adminSeed');
    const seed = seedInput && seedInput.value !== '' ? parseInt(seedInput.value) : undefined;
    const data = await API.post('/api/demo', seed !== undefined ? { seed } : {});
    const headers = ['Step', 'Reward'];
    const rows = (data.rewards ?? []).map((r, i) => [i + 1, r.toFixed(4)]);
    downloadCSV(rows, headers, `hospital-rewards-${Date.now()}.csv`);
    toast('Rewards CSV downloaded', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

function downloadSessionLog() {
  if (!state.log.length) { toast('No log entries yet', 'error'); return; }
  const text = state.log.map(l => l.line).join('\n');
  downloadText(text, `hospital-session-log-${Date.now()}.txt`);
  toast('Log downloaded', 'success');
}

async function downloadConfigJSON() {
  try {
    const data = await API.get('/api/info');
    downloadJSON(data, 'hospital-env-config.json');
    toast('Config downloaded', 'success');
  } catch (err) { toast(err.message, 'error'); }
}

/* ── Init ──────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setActionsEnabled(false);

  // Wire reset / close buttons
  document.getElementById('btnReset')?.addEventListener('click', doReset);
  document.getElementById('btnClose')?.addEventListener('click', doClose);
  document.getElementById('btnHeuristic')?.addEventListener('click', doHeuristicStep);
  document.getElementById('btnAutoRun')?.addEventListener('click', doAutoRun);

  // Wire action buttons (data-action attribute)
  document.querySelectorAll('.action-btn[data-action]').forEach(btn => {
    btn.addEventListener('click', () => doStep(parseInt(btn.dataset.action)));
  });

  // Admin download buttons
  document.getElementById('btnDlStatus')?.addEventListener('click', downloadSessionStatus);
  document.getElementById('btnDlEnvInfo')?.addEventListener('click', downloadEnvInfo);
  document.getElementById('btnDemoRun')?.addEventListener('click', downloadEpisodeReport);
  document.getElementById('btnDlHourly')?.addEventListener('click', downloadHourlyCSV);
  document.getElementById('btnDlRewards')?.addEventListener('click', downloadRewardsCSV);
  document.getElementById('btnDlLog')?.addEventListener('click', downloadSessionLog);
  document.getElementById('btnDlConfig')?.addEventListener('click', downloadConfigJSON);

  setGlobalStatus('idle');
});
