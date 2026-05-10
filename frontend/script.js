/* ══════════════════════════════════════════════════════════════════
   ANTIDOTE AI — Frontend Logic
   Handles: Upload → Train → Predict workflow
   ══════════════════════════════════════════════════════════════════ */

const API = '';  // same-origin; Flask serves both API and frontend

// ── DOM refs ─────────────────────────────────────────────────────
const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const fileInput      = $('#file-input');
const uploadZone     = $('#upload-zone');
const uploadSpinner  = $('#upload-spinner');
const uploadStats    = $('#upload-stats');
const statTotal      = $('#stat-total');
const statSuspicious = $('#stat-suspicious');
const statCleaned    = $('#stat-cleaned');
const downloadRow    = $('#download-row');

const trainBtn     = $('#train-btn');
const trainSpinner = $('#train-spinner');
const trainStatus  = $('#train-status');
const trainStats   = $('#train-stats');
const statAccuracy = $('#stat-accuracy');
const statSamples  = $('#stat-samples');
const statFeatures = $('#stat-features');
const individualStats = $('#individual-stats');
const statRf       = $('#stat-rf');
const statLr       = $('#stat-lr');
const statGb       = $('#stat-gb');

const predictBtn     = $('#predict-btn');
const predictSpinner = $('#predict-spinner');
const extraFields    = $('#extra-fields');

const outputSection  = $('#output-section');
const outPoisoning   = $('#out-poisoning');
const outEvasion     = $('#out-evasion');
const outPrediction  = $('#out-prediction');
const outDecision    = $('#out-decision');
const outRiskScore   = $('#out-riskscore');
const riskBarFill    = $('#risk-bar-fill');
const outDetails     = $('#out-details');
const outSeverity    = $('#out-severity');
const outDrift       = $('#out-drift');
const outConfidence  = $('#out-confidence');
const outExplanation = $('#out-explanation');

const toastEl = $('#toast');

// Voting panel refs
const voteSvmCard      = $('#vote-svm');
const voteSvmStatus    = $('#vote-svm-status');
const voteSvmResult    = $('#vote-svm-result');
const voteIforestCard  = $('#vote-iforest');
const voteIforestStatus= $('#vote-iforest-status');
const voteIforestResult= $('#vote-iforest-result');
const voteZscoreCard   = $('#vote-zscore');
const voteZscoreStatus = $('#vote-zscore-status');
const voteZscoreResult = $('#vote-zscore-result');
const votingSummary    = $('#voting-summary');
const voteCount        = $('#vote-count');
const voteConfidence   = $('#vote-confidence');
const voteConfBar      = $('#vote-conf-bar');
const voteBadge        = $('#vote-badge');

// ── State ────────────────────────────────────────────────────────
let nFeatures = 3;  // updated after training

// ══════════════════════════════════════════════════════════════════
//  TOAST
// ══════════════════════════════════════════════════════════════════

function showToast(message, type = 'info') {
    toastEl.textContent = message;
    toastEl.className = `toast toast--${type} toast--visible`;
    setTimeout(() => toastEl.classList.remove('toast--visible'), 3500);
}

// ══════════════════════════════════════════════════════════════════
//  INTERSECTION OBSERVER — fade-in sections
// ══════════════════════════════════════════════════════════════════

const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('section--visible');
            }
        });
    },
    { threshold: 0.12 }
);

document.addEventListener('DOMContentLoaded', () => {
    $$('.section').forEach((sec) => observer.observe(sec));
});

// ══════════════════════════════════════════════════════════════════
//  DRAG & DROP
// ══════════════════════════════════════════════════════════════════

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleUpload(fileInput.files[0]);
});

// ══════════════════════════════════════════════════════════════════
//  UPLOAD
// ══════════════════════════════════════════════════════════════════

async function handleUpload(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Only CSV files are supported.', 'error');
        return;
    }

    uploadSpinner.classList.add('spinner--visible');
    uploadStats.style.display = 'none';
    downloadRow.style.display = 'none';

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch(`${API}/upload`, { method: 'POST', body: form });
        const data = await res.json();

        if (data.error) {
            showToast(data.error, 'error');
            return;
        }

        // Show stats
        statTotal.textContent      = data.total_rows;
        statSuspicious.textContent = data.suspicious_rows;
        statCleaned.textContent    = data.cleaned_rows;
        uploadStats.style.display  = 'grid';

        // Animate numbers
        animateNumber(statTotal, data.total_rows);
        animateNumber(statSuspicious, data.suspicious_rows);
        animateNumber(statCleaned, data.cleaned_rows);

        // Show download button
        downloadRow.style.display = 'flex';

        // Enable training
        trainBtn.disabled = false;
        trainStatus.textContent = 'Ready to train ensemble.';
        trainStatus.className = 'train-status';

        showToast(`Dataset uploaded — ${data.suspicious_rows} suspicious rows removed.`, 'success');
    } catch (err) {
        showToast('Upload failed. Check server.', 'error');
        console.error(err);
    } finally {
        uploadSpinner.classList.remove('spinner--visible');
    }
}

// ══════════════════════════════════════════════════════════════════
//  TRAIN
// ══════════════════════════════════════════════════════════════════

trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    trainSpinner.classList.add('spinner--visible');
    trainStatus.textContent = 'Training ensemble models…';
    trainStatus.className = 'train-status';
    trainStats.style.display = 'none';
    individualStats.style.display = 'none';

    try {
        const res = await fetch(`${API}/train`, { method: 'POST' });
        const data = await res.json();

        if (data.error) {
            trainStatus.textContent = data.error;
            trainStatus.className = 'train-status train-status--error';
            showToast(data.error, 'error');
            trainBtn.disabled = false;
            return;
        }

        trainStatus.textContent = `Ensemble trained — ${(data.accuracy * 100).toFixed(1)}% accuracy`;
        trainStatus.className = 'train-status train-status--success';

        statAccuracy.textContent = (data.accuracy * 100).toFixed(1) + '%';
        statSamples.textContent  = data.n_samples;
        statFeatures.textContent = data.n_features;
        trainStats.style.display = 'grid';

        // Individual model accuracies
        if (data.individual) {
            statRf.textContent = data.individual.rf ? (data.individual.rf * 100).toFixed(1) + '%' : '—';
            statLr.textContent = data.individual.lr ? (data.individual.lr * 100).toFixed(1) + '%' : '—';
            statGb.textContent = data.individual.gb ? (data.individual.gb * 100).toFixed(1) + '%' : '—';
            individualStats.style.display = 'grid';
        }

        nFeatures = data.n_features;
        buildFeatureInputs(nFeatures);

        predictBtn.disabled = false;
        showToast('Ensemble models trained successfully!', 'success');
    } catch (err) {
        trainStatus.textContent = 'Training failed.';
        trainStatus.className = 'train-status train-status--error';
        showToast('Training error — check server.', 'error');
        trainBtn.disabled = false;
        console.error(err);
    } finally {
        trainSpinner.classList.remove('spinner--visible');
    }
});

// ══════════════════════════════════════════════════════════════════
//  DYNAMIC FEATURE INPUTS
// ══════════════════════════════════════════════════════════════════

function buildFeatureInputs(count) {
    // Keep default 3, add extra if needed
    const existing = 3;
    extraFields.innerHTML = '';

    if (count > existing) {
        for (let i = existing + 1; i <= count; i++) {
            const group = document.createElement('div');
            group.className = 'form-group';
            group.innerHTML = `
                <label class="form-label" for="input-f${i}">Feature ${i}</label>
                <input class="form-input" id="input-f${i}" type="number" step="any" placeholder="e.g. 0.0">
            `;
            extraFields.appendChild(group);
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  PREDICT
// ══════════════════════════════════════════════════════════════════

predictBtn.addEventListener('click', async () => {
    // Collect features
    const features = [];
    for (let i = 1; i <= nFeatures; i++) {
        const el = $(`#input-f${i}`);
        if (!el) { showToast(`Missing input field ${i}`, 'error'); return; }
        const val = parseFloat(el.value);
        if (isNaN(val)) {
            showToast(`Feature ${i} must be a number.`, 'error');
            el.focus();
            return;
        }
        features.push(val);
    }

    predictBtn.disabled = true;
    predictSpinner.classList.add('spinner--visible');
    resetPipeline();
    resetVotingPanel();
    activatePipeStep('pipe-input');

    try {
        const res = await fetch(`${API}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features }),
        });
        const data = await res.json();

        if (data.error) {
            showToast(data.error, 'error');
            predictBtn.disabled = false;
            return;
        }

        // Animate pipeline steps
        await animatePipeline(data);

        // Fill dashboard
        renderDashboard(data);
        outputSection.style.display = 'block';
        outputSection.classList.add('section--visible');

        // Scroll to output
        setTimeout(() => {
            outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 200);

        showToast(`Decision: ${data.decision}`, data.decision === 'ALLOW' ? 'success' : 'error');
    } catch (err) {
        showToast('Prediction failed — check server.', 'error');
        console.error(err);
    } finally {
        predictBtn.disabled = false;
        predictSpinner.classList.remove('spinner--visible');
    }
});

// ══════════════════════════════════════════════════════════════════
//  PIPELINE ANIMATION
// ══════════════════════════════════════════════════════════════════

function resetPipeline() {
    $$('.pipeline__step').forEach((s) => {
        s.className = 'pipeline__step';
    });
}

function activatePipeStep(id) {
    const el = $(`#${id}`);
    if (el) el.classList.add('pipeline__step--active');
}

function passPipeStep(id) {
    const el = $(`#${id}`);
    if (el) {
        el.classList.remove('pipeline__step--active');
        el.classList.add('pipeline__step--pass');
    }
}

function failPipeStep(id) {
    const el = $(`#${id}`);
    if (el) {
        el.classList.remove('pipeline__step--active');
        el.classList.add('pipeline__step--fail');
    }
}

async function animatePipeline(data) {
    const delay = (ms) => new Promise((r) => setTimeout(r, ms));

    // Input
    passPipeStep('pipe-input');
    await delay(200);

    // Validator
    activatePipeStep('pipe-validator');
    await delay(300);
    passPipeStep('pipe-validator');
    await delay(200);

    // Evasion
    activatePipeStep('pipe-evasion');
    await delay(400);
    data.evasion_risk ? failPipeStep('pipe-evasion') : passPipeStep('pipe-evasion');
    await delay(200);

    // Drift
    activatePipeStep('pipe-drift');
    await delay(400);
    data.drift_flag ? failPipeStep('pipe-drift') : passPipeStep('pipe-drift');
    await delay(200);

    // Ensemble Model
    activatePipeStep('pipe-ensemble');
    await delay(400);
    passPipeStep('pipe-ensemble');
    await delay(200);

    // Risk Scoring
    activatePipeStep('pipe-risk');
    await delay(300);
    data.severity === 'HIGH' ? failPipeStep('pipe-risk') : passPipeStep('pipe-risk');
    await delay(200);

    // Output
    activatePipeStep('pipe-output');
    await delay(200);
    data.decision === 'ALLOW' ? passPipeStep('pipe-output') : failPipeStep('pipe-output');
}

// ══════════════════════════════════════════════════════════════════
//  RENDER DASHBOARD
// ══════════════════════════════════════════════════════════════════

function renderDashboard(data) {
    // Decision
    outDecision.textContent = data.decision;
    outDecision.className = 'dash-card__value decision--' + data.decision.toLowerCase();

    // Risk Score
    const risk = data.risk_score;
    outRiskScore.textContent = `${risk} / 100`;
    riskBarFill.style.width = `${risk}%`;
    riskBarFill.className = 'risk-bar__fill '
        + (risk >= 66 ? 'risk-bar__fill--high' : risk >= 33 ? 'risk-bar__fill--medium' : 'risk-bar__fill--low');

    // Severity Level
    outSeverity.textContent = data.severity || '—';
    outSeverity.className = 'dash-card__value severity--' + (data.severity || 'low').toLowerCase();

    // Drift Status
    const driftDetected = data.drift_flag;
    outDrift.innerHTML = boolIndicator(driftDetected);

    // Poisoning
    outPoisoning.innerHTML = boolIndicator(data.poisoning_risk);

    // Evasion
    outEvasion.innerHTML = boolIndicator(data.evasion_risk);

    // Model Prediction
    outPrediction.textContent = `Class ${data.model_prediction}`;
    outPrediction.style.color = data.model_prediction === 1 ? '#D02020' : '#1a8a3a';

    // Model Confidence
    const conf = data.model_confidence || 0;
    outConfidence.textContent = (conf * 100).toFixed(1) + '%';
    outConfidence.style.color = conf >= 0.7 ? '#1a8a3a' : conf >= 0.4 ? '#b89000' : '#D02020';

    // Explanation
    if (data.explanation && Array.isArray(data.explanation)) {
        outExplanation.innerHTML = data.explanation.map(e => {
            const isNormal = e === 'All features within expected range';
            const cls = isNormal ? 'explanation-item--normal' : 'explanation-item--alert';
            return `<div class="explanation-item ${cls}">${isNormal ? '✓' : '⚠'} ${e}</div>`;
        }).join('');
    } else {
        outExplanation.textContent = '—';
    }

    // Details
    outDetails.textContent = data.details || '—';

    // Voting panel — sequential animated reveal
    if (data.evasion_votes) {
        renderVotingPanel(data.evasion_votes, data.evasion_confidence);
    }
}

// ══════════════════════════════════════════════════════════════════
//  VOTING PANEL — Reset & Animated Reveal
// ══════════════════════════════════════════════════════════════════

function resetVotingPanel() {
    votingSummary.style.display = 'none';

    [voteSvmCard, voteIforestCard, voteZscoreCard].forEach((card) => {
        card.className = 'vote-card vote-card--loading';
    });

    [voteSvmStatus, voteIforestStatus, voteZscoreStatus].forEach((st) => {
        st.className = 'vote-card__status vote-card__status--loading';
        st.innerHTML = '<div class="vote-shimmer"></div>Analyzing\u2026';
    });

    [voteSvmResult, voteIforestResult, voteZscoreResult].forEach((r) => {
        r.textContent = '—';
    });

    voteCount.textContent = '— / 3';
    voteConfidence.textContent = '—%';
    voteConfBar.style.width = '0%';
    voteConfBar.className = 'voting-summary__bar-fill';
    voteBadge.textContent = '—';
    voteBadge.className = 'voting-summary__badge';
}

function revealVoteCard(card, statusEl, resultEl, result, scoreText) {
    const isAnomaly = result === 'anomaly';
    const stateClass = isAnomaly ? 'anomaly' : 'normal';
    const label = isAnomaly ? '⬛ ANOMALY' : '✓ NORMAL';

    // Card colour
    card.className = `vote-card vote-card--${stateClass} vote-card--revealed`;

    // Status chip — remove shimmer, apply colour
    statusEl.className = `vote-card__status vote-card__status--${stateClass}`;
    statusEl.textContent = label;

    // Detail score
    resultEl.textContent = scoreText || '';
}

async function renderVotingPanel(votes, confidence) {
    const delay = (ms) => new Promise((r) => setTimeout(r, ms));

    // Staggered sequential reveal
    await delay(300);
    revealVoteCard(
        voteSvmCard, voteSvmStatus, voteSvmResult,
        votes.svm,
        votes.svm === 'anomaly' ? 'Decision boundary exceeded' : 'Within normal boundary'
    );

    await delay(300);
    revealVoteCard(
        voteIforestCard, voteIforestStatus, voteIforestResult,
        votes.isolation_forest,
        votes.isolation_forest === 'anomaly' ? 'Isolated outlier detected' : 'Within forest distribution'
    );

    await delay(300);
    revealVoteCard(
        voteZscoreCard, voteZscoreStatus, voteZscoreResult,
        votes.zscore,
        votes.zscore === 'anomaly' ? '|Z| > threshold detected' : 'Z-score within bounds'
    );

    // Slight pause before summary
    await delay(200);

    // Populate summary
    const pct = Math.round((confidence || 0) * 100);
    voteCount.textContent = `${votes.votes_for} / ${votes.votes_total}`;
    voteConfidence.textContent = `${pct}%`;

    // Confidence bar colour
    const barClass = pct >= 67 ? 'high' : pct >= 34 ? 'medium' : 'low';
    voteConfBar.className = `voting-summary__bar-fill voting-summary__bar-fill--${barClass}`;
    // Trigger transition after a frame
    requestAnimationFrame(() => { voteConfBar.style.width = `${pct}%`; });

    // Badge
    const isEvasion = (votes.votes_for || 0) >= 2;
    voteBadge.textContent = isEvasion ? '⚠ EVASION DETECTED' : '✓ SAFE INPUT';
    voteBadge.className = `voting-summary__badge voting-summary__badge--${isEvasion ? 'evasion' : 'safe'}`;

    votingSummary.style.display = 'flex';
}

function boolIndicator(value) {
    const cls = value ? 'indicator--true' : 'indicator--false';
    const label = value ? 'DETECTED' : 'CLEAR';
    const color = value ? '#D02020' : '#1a8a3a';
    return `<span class="indicator ${cls}"></span><span style="color:${color};font-size:1rem;">${label}</span>`;
}

// ══════════════════════════════════════════════════════════════════
//  ANIMATED NUMBER COUNTER
// ══════════════════════════════════════════════════════════════════

function animateNumber(el, target) {
    const duration = 600;
    const start = 0;
    const startTime = performance.now();

    function step(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // ease-out quad
        const eased = 1 - (1 - progress) * (1 - progress);
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}
