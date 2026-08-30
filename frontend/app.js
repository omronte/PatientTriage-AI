/* ============================================================
   PatientTriage.ai — Application Logic (Vanilla JS) — V3
   Full-Stack: Connected to FastAPI backend with AI triage pipeline,
   surge simulation, deterioration detection, and audit logging.
   ============================================================ */

// ---- Configuration ----
const API_BASE = ''; // Same-origin (served by FastAPI)

// ---- Enums & Constants ----
const TriagePriority = Object.freeze({
  P1_RESUSCITATION: 1,
  P2_EMERGENT: 2,
  P3_URGENT: 3,
  P4_LESS_URGENT: 4,
  P5_NON_URGENT: 5,
});

const PRIORITY_LABELS = {
  1: 'Resuscitation',
  2: 'Emergent',
  3: 'Urgent',
  4: 'Less Urgent',
  5: 'Non-Urgent',
};

const PRIORITY_CSS_CLASS = {
  1: 'p1',
  2: 'p2',
  3: 'p3',
  4: 'p4',
  5: 'p5',
};

// Deterioration threshold (minutes)
const OVERDUE_THRESHOLD_MINUTES = 60;

// ---- Application State ----
const appState = {
  patients: [],
  selectedPatientId: null,
  showOnlyUnreviewed: false,
  isNetworkOnline: true,
  isLoading: false,
  backendConnected: false,
};

// ---- DOM References ----
const DOM = {};

function cacheDOMReferences() {
  DOM.offlineBanner = document.getElementById('offline-banner');
  DOM.patientList = document.getElementById('patient-list');
  DOM.queueCount = document.getElementById('queue-count');
  DOM.prioritySummary = document.getElementById('priority-summary');
  DOM.toggleAll = document.getElementById('toggle-all');
  DOM.toggleUnreviewed = document.getElementById('toggle-unreviewed');
  DOM.emptyState = document.getElementById('empty-state');
  DOM.detailView = document.getElementById('detail-view');
  DOM.loadingState = document.getElementById('loading-state');
  DOM.overdueBanner = document.getElementById('overdue-banner');
  DOM.patientIdLarge = document.getElementById('patient-id-large');
  DOM.demoPills = document.getElementById('demo-pills');
  DOM.arrivalInfo = document.getElementById('arrival-info');
  DOM.vitalsGrid = document.getElementById('vitals-grid');
  DOM.complaintText = document.getElementById('complaint-text');
  DOM.aiPriorityLevel = document.getElementById('ai-priority-level');
  DOM.aiPriorityLabel = document.getElementById('ai-priority-label');
  DOM.confidencePercent = document.getElementById('confidence-percent');
  DOM.confidenceFill = document.getElementById('confidence-fill');
  DOM.factorList = document.getElementById('factor-list');
  DOM.safetySummary = document.getElementById('safety-summary');
  DOM.refreshStatus = document.getElementById('refresh-status');
  DOM.traceExpander = document.getElementById('trace-expander');
  DOM.traceContent = document.getElementById('trace-content');
  DOM.footerMain = document.getElementById('footer-main');
  DOM.overridePanel = document.getElementById('override-panel');
  DOM.btnAccept = document.getElementById('btn-accept');
  DOM.btnOverride = document.getElementById('btn-override');
  DOM.btnReassessment = document.getElementById('btn-reassessment');
  DOM.btnAcknowledge = document.getElementById('btn-acknowledge');
  DOM.btnConfirmOverride = document.getElementById('btn-confirm-override');
  DOM.btnCancelOverride = document.getElementById('btn-cancel-override');
  DOM.overrideReasonSelect = document.getElementById('override-reason');
  DOM.overrideReasonCode = document.getElementById('override-reason-code');
  DOM.overrideClinicianReason = document.getElementById('override-clinician-reason');
  DOM.overrideContext = document.getElementById('override-context');
  // Modal
  DOM.modalOverlay = document.getElementById('modal-overlay');
  DOM.modalClose = document.getElementById('modal-close');
  DOM.btnAddPatient = document.getElementById('btn-add-patient');
  DOM.btnModalCancel = document.getElementById('btn-modal-cancel');
  DOM.btnModalSubmit = document.getElementById('btn-modal-submit');
  DOM.formName = document.getElementById('form-name');
  DOM.formAge = document.getElementById('form-age');
  DOM.formSex = document.getElementById('form-sex');
  DOM.formComplaint = document.getElementById('form-complaint');
  DOM.formHR = document.getElementById('form-hr');
  DOM.formBPSys = document.getElementById('form-bp-sys');
  DOM.formBPDia = document.getElementById('form-bp-dia');
  DOM.formSpo2 = document.getElementById('form-spo2');
  DOM.formRR = document.getElementById('form-rr');
  DOM.formTemp = document.getElementById('form-temp');
  DOM.formGCS = document.getElementById('form-gcs');
  // Audit
  DOM.btnAuditLog = document.getElementById('btn-audit-log');
  DOM.auditModalOverlay = document.getElementById('audit-modal-overlay');
  DOM.auditModalClose = document.getElementById('audit-modal-close');
  DOM.auditStats = document.getElementById('audit-stats');
  DOM.auditTableBody = document.getElementById('audit-table-body');
  DOM.auditEmpty = document.getElementById('audit-empty');
  DOM.btnClearQueue = document.getElementById('btn-clear-queue');
  DOM.btnReloadDemo = document.getElementById('btn-reload-demo');
}

// ---- API Helpers ----

async function apiGet(path) {
  const resp = await fetch(`${API_BASE}${path}`);
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

async function apiPost(path, body = null) {
  const opts = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== null) opts.body = JSON.stringify(body);
  const resp = await fetch(`${API_BASE}${path}`, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `API error: ${resp.status}`);
  }
  return resp.json();
}

// ---- Utility Functions ----

function minutesSince(isoString) {
  return Math.floor((Date.now() - new Date(isoString).getTime()) / 60000);
}

function formatWaitTime(mins) {
  if (mins < 1) return '<1m';
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function formatArrivalTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function getPriorityColor(level) {
  const colors = { 1: 'var(--p1-red)', 2: 'var(--p2-orange)', 3: 'var(--p3-yellow)', 4: 'var(--p4-green)', 5: 'var(--p5-blue)' };
  return colors[level] || 'var(--text-tertiary)';
}

/** Get the EFFECTIVE priority — uses nurse override if set, else AI suggestion. */
function getEffectivePriority(patient) {
  return patient.nurseAssignedPriority != null ? patient.nurseAssignedPriority : patient.aiSuggestedPriority;
}

/**
 * Check if a patient is overdue (ESI 3 waiting > 60 min).
 * If so, visually upgrade to ESI 2.
 */
function checkDeteriorationStatus(patient) {
  if (patient.status !== 'AWAITING_REVIEW') return { isOverdue: false };

  const waitMins = minutesSince(patient.arrivalTime);
  const effectivePriority = getEffectivePriority(patient);

  if (effectivePriority === 3 && waitMins >= OVERDUE_THRESHOLD_MINUTES) {
    return { isOverdue: true, originalPriority: 3, escalatedPriority: 2, waitMinutes: waitMins };
  }
  return { isOverdue: false };
}

/** Get display priority (accounting for deterioration escalation) */
function getDisplayPriority(patient) {
  const deterioration = checkDeteriorationStatus(patient);
  if (deterioration.isOverdue) return 2;
  return patient.lastQueuePriority || getEffectivePriority(patient);
}

function highlightComplaintText(text, factors) {
  let result = escapeHtml(text);
  const sorted = [...factors]
    .filter((f) => f.highlightTarget)
    .sort((a, b) => b.highlightTarget.length - a.highlightTarget.length);

  for (const factor of sorted) {
    const escaped = escapeHtml(factor.highlightTarget);
    const cssClass = factor.severityIndicator === 'CRITICAL' ? 'critical-flag' : 'warning-flag';
    const regex = new RegExp(`(${escapeRegex(escaped)})`, 'gi');
    result = result.replace(regex, `<mark class="${cssClass}">$1</mark>`);
  }
  return result;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ---- Loading State ----

function showLoading() {
  appState.isLoading = true;
  if (DOM.loadingState) DOM.loadingState.classList.add('visible');
  if (DOM.emptyState) DOM.emptyState.style.display = 'none';
  if (DOM.detailView) DOM.detailView.classList.remove('visible');
}

function hideLoading() {
  appState.isLoading = false;
  if (DOM.loadingState) DOM.loadingState.classList.remove('visible');
}

// ---- Rendering ----

function renderQueue() {
  const patients = getSortedFilteredPatients();
  const awaitingCount = appState.patients.filter((p) => p.status === 'AWAITING_REVIEW').length;
  const totalCount = appState.patients.length;

  DOM.queueCount.innerHTML = `<strong>${awaitingCount}</strong> of ${totalCount} awaiting review`;

  renderPrioritySummary();

  DOM.patientList.innerHTML = '';
  patients.forEach((patient) => {
    DOM.patientList.appendChild(createPatientCard(patient));
  });
}

/** Renders the P1–P5 patient-count pills above the buttons. */
function renderPrioritySummary() {
  if (!DOM.prioritySummary) return;
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  appState.patients.forEach((patient) => {
    counts[getDisplayPriority(patient)]++;
  });
  DOM.prioritySummary.innerHTML = [1, 2, 3, 4, 5]
    .map((level) => `
      <span class="priority-pill p${level}">
        <span class="pill-dot"></span>
        <span class="pill-count">${counts[level]}</span>
        <span class="pill-label">P${level}</span>
      </span>
    `)
    .join('');
}

/**
 * Sort by DISPLAY priority (nurse override > deterioration upgrade > AI suggestion).
 */
function getSortedFilteredPatients() {
  let list = [...appState.patients];
  if (appState.showOnlyUnreviewed) {
    list = list.filter((p) => p.status === 'AWAITING_REVIEW');
  }
  // The API returns the authoritative safety-first order; only filtering is local.
  return list;
}

function createPatientCard(patient) {
  const card = document.createElement('div');
  card.className = 'patient-card';
  card.dataset.patientId = patient.patientId;

  const displayPriority = getDisplayPriority(patient);
  const pClass = PRIORITY_CSS_CLASS[displayPriority];
  const waitMins = patient.waitingMinutes ?? minutesSince(patient.arrivalTime);
  const deterioration = checkDeteriorationStatus(patient);

  if (appState.selectedPatientId === patient.patientId) card.classList.add('selected');
  if (patient.status !== 'AWAITING_REVIEW') card.classList.add('reviewed');

  let statusBadgeHTML = '';
  if (patient.reassessmentRequired) {
    statusBadgeHTML = '<span class="overdue-badge">⚠ REASSESSMENT REQUIRED</span>';
  } else if (deterioration.isOverdue) {
    statusBadgeHTML = '<span class="overdue-badge">⏰ OVERDUE</span>';
  } else if (patient.status === 'REVIEWED_ACCEPTED') {
    statusBadgeHTML = '<span class="status-badge accepted">Confirmed</span>';
  } else if (patient.status === 'REVIEWED_OVERRIDDEN') {
    statusBadgeHTML = `<span class="status-badge overridden">P${patient.aiSuggestedPriority}→P${patient.nurseAssignedPriority}</span>`;
  }

  // Confidence indicator
  const confPercent = Math.round((patient.aiConfidenceScore || 0) * 100);
  const confClass = confPercent < 60 ? 'low-conf' : '';

  card.innerHTML = `
    <div class="priority-bar ${pClass}"></div>
    <div class="card-content">
      <div class="card-top-row">
        <span class="patient-id">${patient.patientId}</span>
        ${statusBadgeHTML}
      </div>
      <div class="card-mid-row">
        <span class="age-sex">${patient.age} ${patient.biologicalSex === 'M' ? 'Male' : patient.biologicalSex === 'F' ? 'Female' : patient.biologicalSex}</span>
        <span class="priority-badge ${pClass}">P${displayPriority} · ${PRIORITY_LABELS[displayPriority]}</span>
      </div>
      <div class="chief-complaint-preview">${escapeHtml(patient.chiefComplaint)}</div>
      <div class="card-safety-meta">
        <span>Risk: ${escapeHtml(patient.riskCategory || '--')}</span>
        <span>Age profile: ${escapeHtml(patient.ageGroup || '--')}</span>
        <span>Confidence: ${confPercent}% ${escapeHtml(patient.confidenceLabel || '')}</span>
        <span>History: ${escapeHtml(patient.historyAvailability || '--')}</span>
      </div>
      <div class="card-alert-meta">${patient.reassessmentRequired ? '⚠ Clinical reassessment required' : (patient.safetyFlags || []).length ? `⚠ Safety flags: ${(patient.safetyFlags || []).length}` : '✓ No active safety alert'}</div>
    </div>
  `;

  card.addEventListener('click', () => selectPatient(patient.patientId));
  return card;
}

function selectPatient(patientId) {
  const isDifferent = appState.selectedPatientId !== patientId;
  appState.selectedPatientId = patientId;
  renderQueue();
  renderDetailView(isDifferent);
}

function renderDetailView(forceResetFooter = false) {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);

  if (!patient) {
    DOM.emptyState.style.display = 'flex';
    DOM.detailView.classList.remove('visible');
    if (DOM.overdueBanner) DOM.overdueBanner.classList.remove('visible');
    resetFooterState(null);
    return;
  }

  DOM.emptyState.style.display = 'none';
  DOM.detailView.classList.add('visible');

  // Deterioration check
  const deterioration = checkDeteriorationStatus(patient);
  if (DOM.overdueBanner) {
    DOM.overdueBanner.classList.toggle('visible', deterioration.isOverdue);
  }

  // 2A
  DOM.patientIdLarge.textContent = patient.patientId;
  DOM.demoPills.innerHTML = `
    <span class="demo-pill">${patient.age} yrs</span>
    <span class="demo-pill">${patient.biologicalSex === 'M' ? 'Male' : patient.biologicalSex === 'F' ? 'Female' : patient.biologicalSex}</span>
  `;
  DOM.arrivalInfo.innerHTML = `
    <span class="arrival-label">Arrived:</span> ${formatArrivalTime(patient.arrivalTime)} 
    <span style="margin-left:8px" class="arrival-label">Waiting:</span> ${formatWaitTime(minutesSince(patient.arrivalTime))}
  `;

  renderVitals(patient);

  // 2B
  DOM.complaintText.innerHTML = highlightComplaintText(patient.chiefComplaint, patient.explainability || []);

  const displayPriority = getDisplayPriority(patient);
  const pColor = getPriorityColor(patient.aiSuggestedPriority);
  DOM.aiPriorityLevel.textContent = `P${patient.aiSuggestedPriority}`;
  DOM.aiPriorityLevel.style.color = pColor;
  DOM.aiPriorityLabel.textContent = PRIORITY_LABELS[patient.aiSuggestedPriority];

  const confPercent = Math.round((patient.aiConfidenceScore || 0) * 100);
  DOM.confidencePercent.textContent = `${confPercent}%`;
  DOM.confidenceFill.style.width = `${confPercent}%`;

  // Color confidence bar based on confidence level
  if (confPercent < 60) {
    DOM.confidenceFill.style.background = `linear-gradient(90deg, var(--p1-red), var(--p2-orange))`;
  } else if (confPercent < 80) {
    DOM.confidenceFill.style.background = `linear-gradient(90deg, var(--p2-orange), var(--p3-yellow))`;
  } else {
    DOM.confidenceFill.style.background = `linear-gradient(90deg, ${pColor}, ${pColor}88)`;
  }

  renderFactorList(patient);
  renderSafetySummary(patient);
  renderPipelineTrace(patient);

  // Only reset footer/override panel if explicitly forced (e.g. switching patients or confirmed)
  // Otherwise, if the clinician is in the middle of overriding, do not close or wipe it!
  const isOverrideOpen = DOM.overridePanel && DOM.overridePanel.classList.contains('visible');
  if (forceResetFooter || !isOverrideOpen) {
    resetFooterState(patient);
  }
}

function renderVitals(patient) {
  const vitals = patient.vitals;
  const alertTargets = (patient.explainability || [])
    .filter((f) => f.category === 'VITAL_ALERT')
    .map((f) => f.highlightTarget.toLowerCase());

  const vitalDefs = [
    { label: 'Heart Rate', unit: 'bpm', value: vitals.heartRateBpm, alertKeywords: ['heart rate'] },
    { label: 'Blood Pressure', unit: 'mmHg', value: vitals.bloodPressureSys != null ? `${vitals.bloodPressureSys}/${vitals.bloodPressureDia}` : null, alertKeywords: ['blood pressure', 'bp'] },
    { label: 'SpO₂', unit: '%', value: vitals.o2SaturationPercent, alertKeywords: ['o2 saturation', 'spo2', 'oxygen'] },
    { label: 'Resp Rate', unit: '/min', value: vitals.respiratoryRate, alertKeywords: ['respiratory', 'resp rate'] },
    { label: 'Temp', unit: '°C', value: vitals.temperatureCelsius, alertKeywords: ['temperature', 'temp', 'fever'] },
    { label: 'GCS', unit: '/15', value: vitals.gcsScore, alertKeywords: ['gcs', 'glasgow'] },
  ];

  DOM.vitalsGrid.innerHTML = '';
  vitalDefs.forEach((v) => {
    const card = document.createElement('div');
    card.className = 'vital-card';

    const isAlerted = alertTargets.some((t) => v.alertKeywords.some((kw) => t.includes(kw)));
    const matchedFactor = (patient.explainability || []).find(
      (f) => f.category === 'VITAL_ALERT' && v.alertKeywords.some((kw) => f.highlightTarget.toLowerCase().includes(kw))
    );

    if (isAlerted && matchedFactor) {
      card.classList.add(matchedFactor.severityIndicator === 'CRITICAL' ? 'alert-critical' : 'alert-warning');
    }

    const displayVal = v.value != null ? v.value : '--';
    const alertIcon = isAlerted ? `<span class="vital-alert-icon">${matchedFactor && matchedFactor.severityIndicator === 'CRITICAL' ? '🔴' : '⚠️'}</span>` : '';

    card.innerHTML = `
      ${alertIcon}
      <div class="vital-label">${v.label}</div>
      <div class="vital-value">${displayVal}</div>
      <div class="vital-unit">${v.value != null ? v.unit : 'Pending'}</div>
    `;
    DOM.vitalsGrid.appendChild(card);
  });
}

function renderFactorList(patient) {
  DOM.factorList.innerHTML = '';
  (patient.explainability || []).forEach((factor) => {
    const li = document.createElement('li');
    const severityClass = factor.severityIndicator === 'CRITICAL' ? 'critical' : 'warning';
    const icon = factor.severityIndicator === 'CRITICAL' ? '🔴' : '⚠️';
    const categoryLabels = { NLP_KEYWORD: 'Keyword Alert', VITAL_ALERT: 'Vital Alert', HISTORICAL_RISK: 'Risk Factor' };

    li.className = `factor-item ${severityClass}`;
    li.innerHTML = `
      <span class="factor-icon">${icon}</span>
      <div><span class="factor-category">${categoryLabels[factor.category] || factor.category}:</span> ${escapeHtml(factor.aiReasoning)}</div>
    `;
    DOM.factorList.appendChild(li);
  });
}

function renderSafetySummary(patient) {
  if (!DOM.safetySummary) return;

  const flags = patient.safetyFlags || patient.safety_flags || [];
  const missing = patient.missingFields || patient.missing_fields || [];
  const history = patient.historyAvailability || (patient.history_available ? 'available' : 'unavailable');
  const confidenceLabel = patient.confidenceLabel || patient.confidence_label || 'Unclassified';
  const safetyAlert = flags.length > 0 && flags.some((flag) => flag !== 'history_available' && flag !== 'adult_safety_profile');
  const statusClass = safetyAlert || missing.length > 0 || history === 'none' ? 'safety-alert' : 'safety-clear';
  const statusText = safetyAlert ? 'Clinical review required' : 'Safety checks passed';
  const missingText = missing.length > 0 ? ` Missing: ${missing.join(', ')}.` : '';
  const timingText = ` Last vitals: ${patient.lastVitalsAt ? formatArrivalTime(patient.lastVitalsAt) : '--'} · Last assessment: ${patient.lastAssessmentAt ? formatArrivalTime(patient.lastAssessmentAt) : '--'}.`;
  const recommendationText = ` Recommended action: ${patient.recommendedAction || 'Clinical review required'}.`;

  DOM.safetySummary.className = `safety-summary ${statusClass}`;
  const flagsText = flags.length ? ` Safety flags: ${flags.join(', ')}.` : '';
  DOM.safetySummary.innerHTML = `
    <strong>${escapeHtml(statusText)}</strong>
    <span>Age group: ${escapeHtml(patient.ageGroup || 'unknown')} · History: ${escapeHtml(history)} · Confidence in recommendation: ${escapeHtml(confidenceLabel)}.${escapeHtml(missingText + flagsText + timingText + recommendationText)}</span>
  `;
}

function renderPipelineTrace(patient) {
  if (!DOM.traceContent) return;

  const trace = patient._trace || [];
  if (trace.length === 0) {
    DOM.traceExpander.style.display = 'none';
    return;
  }

  DOM.traceExpander.style.display = 'block';
  DOM.traceContent.innerHTML = trace.map((step) => `
    <div class="trace-step">
      <span class="trace-node">${escapeHtml(step.node)}</span>
      <span class="trace-time">${step.elapsed_ms}ms</span>
      <span class="trace-detail">${escapeHtml(step.detail)}</span>
    </div>
  `).join('');
}

function resetFooterState(patient) {
  DOM.footerMain.style.display = 'flex';
  DOM.overridePanel.classList.remove('visible');

  if (!patient) return;

  const isReviewed = patient.status !== 'AWAITING_REVIEW';

  if (isReviewed) {
    if (patient.status === 'REVIEWED_ACCEPTED') {
      DOM.btnAccept.textContent = `✓ Confirmed as P${patient.nurseAssignedPriority}`;
    } else {
      DOM.btnAccept.textContent = `✓ Overridden to P${patient.nurseAssignedPriority}`;
    }
    DOM.btnAccept.disabled = true;
    DOM.btnAccept.style.opacity = '0.45';
    DOM.btnAccept.style.cursor = 'default';
  } else {
    DOM.btnAccept.textContent = `Confirm AI Assessment (P${patient.aiSuggestedPriority})`;
    DOM.btnAccept.disabled = false;
    DOM.btnAccept.style.opacity = '1';
    DOM.btnAccept.style.cursor = 'pointer';
  }

  DOM.btnOverride.disabled = isReviewed;
  DOM.btnOverride.style.opacity = isReviewed ? '0.45' : '1';

  document.querySelectorAll('.priority-radio').forEach((r) => (r.checked = false));
  DOM.overrideReasonSelect.value = '';
  if (DOM.overrideReasonCode) DOM.overrideReasonCode.value = '';
  if (DOM.overrideClinicianReason) DOM.overrideClinicianReason.value = '';
  DOM.btnConfirmOverride.disabled = true;
  if (DOM.btnReassessment) {
    DOM.btnReassessment.style.display = patient.status === 'AWAITING_REVIEW' ? 'inline-flex' : 'none';
  }
  if (DOM.btnAcknowledge) {
    DOM.btnAcknowledge.style.display = patient.reassessmentRequired ? 'inline-flex' : 'none';
  }
}

// ---- Interaction Handlers ----

async function acceptRecommendation() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (!patient || patient.status !== 'AWAITING_REVIEW') return;

  try {
    // Call backend to log the acceptance
    await apiPost(`/api/accept/${patient.patientId}`);

    patient.status = 'REVIEWED_ACCEPTED';
    patient.nurseAssignedPriority = patient.aiSuggestedPriority;

    renderQueue();
    renderDetailView();
    showToast(`P${patient.aiSuggestedPriority} confirmed for ${patient.patientId}`, 'success');
  } catch (err) {
    console.error('Accept failed:', err);
    showToast('Acceptance was not recorded. Please retry while connected.', 'info');
  }
}

function openOverridePanel() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (!patient || patient.status !== 'AWAITING_REVIEW') return;

  DOM.footerMain.style.display = 'none';
  if (DOM.overrideContext) {
    DOM.overrideContext.textContent = `Current AI recommendation: P${patient.aiSuggestedPriority} · ${Math.round((patient.aiConfidenceScore || 0) * 100)}% confidence · ${patient.safetyFlags?.length || 0} safety flags`;
  }
  DOM.overridePanel.classList.add('visible');
}

function cancelOverride() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (patient) resetFooterState(patient);
}

function validateOverrideForm() {
  const selectedPriority = document.querySelector('.priority-radio:checked');
  const reasonCode = DOM.overrideReasonCode ? DOM.overrideReasonCode.value : DOM.overrideReasonSelect.value;
  const clinicianReason = DOM.overrideClinicianReason ? DOM.overrideClinicianReason.value.trim() : DOM.overrideReasonSelect.value;
  DOM.btnConfirmOverride.disabled = !(selectedPriority && reasonCode && clinicianReason);
}

async function confirmOverride() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (!patient) return;

  const selectedPriority = document.querySelector('.priority-radio:checked');
  const reasonCode = DOM.overrideReasonCode ? DOM.overrideReasonCode.value : DOM.overrideReasonSelect.value;
  const clinicianReason = DOM.overrideClinicianReason ? DOM.overrideClinicianReason.value.trim() : DOM.overrideReasonSelect.value;
  if (!selectedPriority || !reasonCode || !clinicianReason) return;

  const newPriority = parseInt(selectedPriority.value, 10);

  try {
    // Call backend to log the override
    await apiPost(`/api/override/${patient.patientId}`, {
      nurse_esi: newPriority,
      override_reason: clinicianReason.slice(0, 100),
      reason_code: reasonCode,
      clinician_reason: clinicianReason,
    });

    patient.status = 'REVIEWED_OVERRIDDEN';
    patient.nurseAssignedPriority = newPriority;

    renderQueue();
    renderDetailView(true);
    showToast(`${patient.patientId} overridden: P${patient.aiSuggestedPriority} → P${newPriority}`, 'info');
  } catch (err) {
    console.error('Override failed:', err);
    showToast('Override was not recorded. Please retry while connected.', 'info');
  }
}

function setQueueFilter(showUnreviewed) {
  appState.showOnlyUnreviewed = showUnreviewed;
  DOM.toggleAll.classList.toggle('active', !showUnreviewed);
  DOM.toggleUnreviewed.classList.toggle('active', showUnreviewed);
  renderQueue();
}

async function requestReassessment() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (!patient) return;
  try {
    const updated = await apiPost(`/api/patients/${patient.patientId}/reassessment`);
    Object.assign(patient, updated);
    renderQueue();
    renderDetailView();
    showToast('Reassessment requested and recorded', 'info');
  } catch (err) {
    showToast('Unable to request reassessment', 'info');
  }
}

async function acknowledgeAlert() {
  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
  if (!patient) return;
  try {
    const updated = await apiPost(`/api/patients/${patient.patientId}/reassessment/acknowledge`);
    Object.assign(patient, updated);
    renderQueue();
    renderDetailView();
    showToast('Reassessment alert acknowledged', 'success');
  } catch (err) {
    showToast('Unable to acknowledge alert', 'info');
  }
}

// ---- Add Patient Modal ----

function openAddPatientModal() {
  DOM.modalOverlay.classList.add('visible');
  DOM.formAge.focus();
}

function closeAddPatientModal() {
  DOM.modalOverlay.classList.remove('visible');
  // Reset form
  if (DOM.formName) DOM.formName.value = '';
  DOM.formAge.value = '';
  DOM.formSex.value = 'M';
  DOM.formComplaint.value = '';
  DOM.formHR.value = '';
  DOM.formBPSys.value = '';
  DOM.formBPDia.value = '';
  DOM.formSpo2.value = '';
  DOM.formRR.value = '';
  DOM.formTemp.value = '';
  DOM.formGCS.value = '';
}

async function submitNewPatient() {
  const age = parseInt(DOM.formAge.value, 10);
  const sex = DOM.formSex.value;
  const complaint = DOM.formComplaint.value.trim();
  const name = DOM.formName ? DOM.formName.value.trim() : '';

  if (isNaN(age) || age < 0 || !complaint) {
    showToast('Valid age and chief complaint are required', 'info');
    return;
  }

  // Disable submit button during processing
  DOM.btnModalSubmit.disabled = true;
  DOM.btnModalSubmit.textContent = 'Processing...';

  try {
    // Call backend to register patient and run AI triage
    const payload = {
      age: age,
      gender: sex,
      chief_complaint: complaint,
      heart_rate: parseFloat(DOM.formHR.value) || null,
      blood_pressure_sys: parseFloat(DOM.formBPSys.value) || null,
      blood_pressure_dia: parseFloat(DOM.formBPDia.value) || null,
      oxygen_saturation: parseFloat(DOM.formSpo2.value) || null,
      respiratory_rate: parseFloat(DOM.formRR.value) || null,
      temperature: parseFloat(DOM.formTemp.value) || null,
      gcs_score: parseFloat(DOM.formGCS.value) || null,
    };
    if (name) payload.name = name;

    const newPatient = await apiPost('/api/patients', payload);

    // Refresh full queue from backend to get authoritative priority ordering
    try {
      const refreshedQueue = await apiGet('/api/patients');
      appState.patients = refreshedQueue;
    } catch (refreshErr) {
      if (!appState.patients.some((p) => p.patientId === newPatient.patientId)) {
        appState.patients.unshift(newPatient);
      }
    }

    closeAddPatientModal();
    renderQueue();
    selectPatient(newPatient.patientId);

    // Smoothly scroll the new card into view
    setTimeout(() => {
      const card = document.querySelector(`.patient-card[data-patient-id="${newPatient.patientId}"]`);
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 50);

    showToast(`${newPatient.patientId} added — AI: P${newPatient.aiSuggestedPriority} (${Math.round(newPatient.aiConfidenceScore * 100)}% confidence)`, 'success');
  } catch (err) {
    console.error('Add patient failed:', err);
    showToast('Failed to add patient: ' + (err.message || 'Server error'), 'info');
  } finally {
    DOM.btnModalSubmit.disabled = false;
    DOM.btnModalSubmit.textContent = 'Add to Queue';
  }
}


// ---- Clear Queue & Reload Demo Data ----

async function clearQueue() {
  try {
    showLoading();
    await apiPost('/api/reset');
    appState.patients = [];
    appState.selectedPatientId = null;
    hideLoading();
    renderQueue();
    renderDetailView();
    showToast('Patient queue cleared successfully', 'success');
  } catch (err) {
    console.error('Clear queue failed:', err);
    hideLoading();
    showToast('Failed to clear queue', 'info');
  }
}

async function reloadDemoData() {
  try {
    showLoading();
    await apiPost('/api/seed');
    const patients = await apiGet('/api/patients');
    appState.patients = patients;
    hideLoading();
    renderQueue();
    renderDetailView();
    showToast('Loaded 20 demo patients from patients.json', 'success');
  } catch (err) {
    console.error('Reload demo failed:', err);
    hideLoading();
    showToast('Failed to reload demo data', 'info');
  }
}

// ---- Audit Log ----

async function openAuditLog() {
  DOM.auditModalOverlay.classList.add('visible');

  try {
    const data = await apiGet('/api/audit-log');

    // Render stats
    const stats = data.stats || {};
    DOM.auditStats.innerHTML = `
      <div class="audit-stat-card">
        <div class="audit-stat-value">${stats.total_decisions || 0}</div>
        <div class="audit-stat-label">Total Decisions</div>
      </div>
      <div class="audit-stat-card">
        <div class="audit-stat-value">${stats.accepts || 0}</div>
        <div class="audit-stat-label">Accepted</div>
      </div>
      <div class="audit-stat-card">
        <div class="audit-stat-value">${stats.overrides || 0}</div>
        <div class="audit-stat-label">Overridden</div>
      </div>
      <div class="audit-stat-card">
        <div class="audit-stat-value">${stats.override_rate || 0}%</div>
        <div class="audit-stat-label">Override Rate</div>
      </div>
    `;

    // Render log entries
    const logs = data.logs || [];
    const events = data.events || [];
    const eventRows = events.map((event) => ({
      timestamp: event.timestamp,
      patient_id: event.patient_id,
      patient_age: null,
      patient_gender: '',
      ai_esi: event.previous_triage_level,
      ai_confidence: event.confidence,
      action_type: event.event_type,
      nurse_esi: event.new_triage_level,
      override_reason: event.clinician_reason || event.trigger_reason || event.reason_code || '--',
    }));
    const auditRows = [...logs, ...eventRows].sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
    if (auditRows.length === 0) {
      DOM.auditTableBody.innerHTML = '';
      DOM.auditEmpty.classList.add('visible');
    } else {
      DOM.auditEmpty.classList.remove('visible');
      DOM.auditTableBody.innerHTML = auditRows.map((log) => {
        const ts = log.timestamp ? new Date(log.timestamp).toLocaleString() : '--';
        const actionClass = log.action_type.includes('OVERRIDDEN') || log.action_type === 'OVERRIDE' ? 'override' : 'accept';
        return `
          <tr>
            <td>${ts}</td>
            <td>${escapeHtml(log.patient_id || '')}</td>
            <td>${log.patient_age || '--'} ${log.patient_gender || ''}</td>
            <td>P${log.ai_esi}</td>
            <td>${log.ai_confidence ? Math.round(log.ai_confidence * 100) + '%' : '--'}</td>
            <td><span class="audit-action-badge ${actionClass}">${log.action_type}</span></td>
            <td>${log.nurse_esi ? 'P' + log.nurse_esi : '--'}</td>
            <td>${escapeHtml(log.override_reason || '--')}</td>
          </tr>
        `;
      }).join('');
    }
  } catch (err) {
    console.error('Failed to load audit log:', err);
    DOM.auditStats.innerHTML = '<p style="padding:12px; color: var(--text-tertiary);">Failed to load audit data</p>';
    DOM.auditTableBody.innerHTML = '';
    DOM.auditEmpty.classList.add('visible');
  }
}

function closeAuditLog() {
  DOM.auditModalOverlay.classList.remove('visible');
}

// ---- Toast ----

function showToast(message, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 2600);
}

// ---- Network Status ----

function updateOnlineStatus() {
  appState.isNetworkOnline = navigator.onLine;
  DOM.offlineBanner.classList.toggle('visible', !appState.isNetworkOnline);
}

// ---- Keyboard Shortcuts ----

function handleKeyboard(e) {
  // Don't capture when modals are open or typing in form
  if (DOM.modalOverlay.classList.contains('visible')) return;
  if (DOM.auditModalOverlay.classList.contains('visible')) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

  const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);

  if (e.key === 'Enter' && patient && patient.status === 'AWAITING_REVIEW') {
    if (!DOM.overridePanel.classList.contains('visible')) {
      e.preventDefault();
      acceptRecommendation();
    }
  }

  if (e.key === 'Escape') {
    if (DOM.overridePanel.classList.contains('visible')) {
      cancelOverride();
    } else {
      appState.selectedPatientId = null;
      renderQueue();
      renderDetailView();
    }
  }

  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    const list = getSortedFilteredPatients();
    if (list.length === 0) return;

    const currentIdx = list.findIndex((p) => p.patientId === appState.selectedPatientId);
    let nextIdx;
    if (e.key === 'ArrowDown') {
      nextIdx = currentIdx < list.length - 1 ? currentIdx + 1 : 0;
    } else {
      nextIdx = currentIdx > 0 ? currentIdx - 1 : list.length - 1;
    }
    selectPatient(list[nextIdx].patientId);
  }
}

// ---- Wait time auto-updater & deterioration check ----

function startWaitTimeUpdater() {
  setInterval(() => {
    refreshQueueFromBackend();
    if (appState.selectedPatientId) {
      const patient = appState.patients.find((p) => p.patientId === appState.selectedPatientId);
      if (patient) {
        DOM.arrivalInfo.innerHTML = `
          <span class="arrival-label">Arrived:</span> ${formatArrivalTime(patient.arrivalTime)} 
          <span style="margin-left:8px" class="arrival-label">Waiting:</span> ${formatWaitTime(patient.waitingMinutes ?? minutesSince(patient.arrivalTime))}
        `;
        // Update overdue banner
        const deterioration = checkDeteriorationStatus(patient);
        if (DOM.overdueBanner) {
          DOM.overdueBanner.classList.toggle('visible', deterioration.isOverdue);
        }
      }
    }
  }, 30000);
}

// ---- Initialization ----

async function init() {
  cacheDOMReferences();

  // Toggles
  DOM.toggleAll.addEventListener('click', () => setQueueFilter(false));
  DOM.toggleUnreviewed.addEventListener('click', () => setQueueFilter(true));

  // Action buttons
  DOM.btnAccept.addEventListener('click', acceptRecommendation);
  DOM.btnOverride.addEventListener('click', openOverridePanel);
  DOM.btnCancelOverride.addEventListener('click', cancelOverride);
  DOM.btnConfirmOverride.addEventListener('click', confirmOverride);
  if (DOM.btnReassessment) DOM.btnReassessment.addEventListener('click', requestReassessment);
  if (DOM.btnAcknowledge) DOM.btnAcknowledge.addEventListener('click', acknowledgeAlert);

  // Override form
  document.querySelectorAll('.priority-radio').forEach((r) => r.addEventListener('change', validateOverrideForm));
  DOM.overrideReasonSelect.addEventListener('change', validateOverrideForm);
  if (DOM.overrideReasonCode) DOM.overrideReasonCode.addEventListener('change', validateOverrideForm);
  if (DOM.overrideClinicianReason) DOM.overrideClinicianReason.addEventListener('input', validateOverrideForm);

  // Add Patient modal
  DOM.btnAddPatient.addEventListener('click', openAddPatientModal);
  DOM.modalClose.addEventListener('click', closeAddPatientModal);
  DOM.btnModalCancel.addEventListener('click', closeAddPatientModal);
  DOM.btnModalSubmit.addEventListener('click', submitNewPatient);
  DOM.modalOverlay.addEventListener('click', (e) => {
    if (e.target === DOM.modalOverlay) closeAddPatientModal();
  });


  // Clear & Reload Demo buttons
  if (DOM.btnClearQueue) {
    DOM.btnClearQueue.addEventListener('click', clearQueue);
  }
  if (DOM.btnReloadDemo) {
    DOM.btnReloadDemo.addEventListener('click', reloadDemoData);
  }

  // Audit log
  if (DOM.btnAuditLog) {
    DOM.btnAuditLog.addEventListener('click', openAuditLog);
  }
  if (DOM.auditModalClose) {
    DOM.auditModalClose.addEventListener('click', closeAuditLog);
  }
  if (DOM.auditModalOverlay) {
    DOM.auditModalOverlay.addEventListener('click', (e) => {
      if (e.target === DOM.auditModalOverlay) closeAuditLog();
    });
  }

  // Network
  window.addEventListener('online', updateOnlineStatus);
  window.addEventListener('offline', updateOnlineStatus);
  updateOnlineStatus();

  // Keyboard
  document.addEventListener('keydown', handleKeyboard);

  // Load patients from backend
  await loadPatientsFromBackend();

  // Start auto-updater
  startWaitTimeUpdater();
}

async function loadPatientsFromBackend() {
  showLoading();

  try {
    const patients = await apiGet('/api/patients');
    appState.patients = patients;
    appState.backendConnected = true;
    hideLoading();
    renderQueue();
    renderDetailView();
    console.log(`[PatientTriage.ai] Loaded ${patients.length} patients from SQLite database`);
  } catch (err) {
    console.error('Backend connection failed:', err);
    appState.backendConnected = false;
    appState.patients = [];
    hideLoading();
    renderQueue();
    renderDetailView();
    showToast('Failed to connect to SQLite backend. Please run `python server.py`.', 'info');
  }
}

async function refreshQueueFromBackend() {
  if (appState.isLoading) return;
  try {
    if (DOM.refreshStatus) DOM.refreshStatus.textContent = 'Refreshing queue...';
    const patients = await apiGet('/api/queue');
    appState.patients = patients;
    appState.backendConnected = true;
    renderQueue();
    const isOverrideOpen = DOM.overridePanel && DOM.overridePanel.classList.contains('visible');
    if (!isOverrideOpen) {
      renderDetailView(false);
    }
    if (DOM.refreshStatus) DOM.refreshStatus.textContent = `Live Monitoring · Updated ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  } catch (err) {
    if (DOM.refreshStatus) DOM.refreshStatus.textContent = 'Queue refresh unavailable';
  }
}

document.addEventListener('DOMContentLoaded', init); 