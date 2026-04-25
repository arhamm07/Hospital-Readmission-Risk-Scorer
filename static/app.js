const API_BASE = 'http://localhost:8000';

document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupForm();
});

async function checkHealth() {
    const statusDot = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();

        if (data.status === 'ok' && data.model_loaded) {
            statusDot.classList.add('connected');
            statusText.textContent = `API Connected | Model v${data.model_version}`;
        } else {
            statusDot.classList.add('error');
            statusText.textContent = 'Model not loaded';
        }
    } catch (e) {
        statusDot.classList.add('error');
        statusText.textContent = 'API not reachable';
    }
}

function setupForm() {
    const form = document.getElementById('prediction-form');
    const results = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const submitBtn = form.querySelector('.submit-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        results.classList.add('hidden');
        submitBtn.disabled = true;
        submitBtn.textContent = 'Predicting...';

        const patientId = document.getElementById('patient_id').value || 'unknown';
        const patientData = buildPatientPayload();

        try {
            const resp = await fetch(`${API_BASE}/predict-risk-debug?patient_id=${encodeURIComponent(patientId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientData)
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || `Error ${resp.status}: ${JSON.stringify(err)}`);
            }

            const result = await resp.json();
            displayResults(result);
        } catch (e) {
            showError(e.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Predict Risk';
        }
    });
}

function buildPatientPayload() {
    const fieldMappings = {
        'age_risk_bucket': { 'Young': 0, 'Middle': 1, 'Elderly': 2, 'Unknown': 3 },
        'diag_1_category': { 'Other': 0, 'Circulatory': 1, 'Respiratory': 2, 'Digestive': 3, 'Diabetes': 4, 'Infectious': 5 },
        'diag_2_category': { 'Other': 0, 'Circulatory': 1, 'Respiratory': 2, 'Digestive': 3, 'Diabetes': 4, 'Infectious': 5 },
        'diag_3_category': { 'Other': 0, 'Circulatory': 1, 'Respiratory': 2, 'Digestive': 3, 'Diabetes': 4, 'Infectious': 5 }
    };

    const payload = {};

    const ageNumeric = parseFloat(document.getElementById('age_numeric')?.value) || 0;
    const nDiagnoses = parseFloat(document.getElementById('number_diagnoses')?.value) || 0;
    const nMeds = parseFloat(document.getElementById('num_medications')?.value) || 0;
    const ageBucket = parseFloat(document.getElementById('age_risk_bucket')?.value) || 0;

    // Demographics
    payload.age_numeric = ageNumeric;
    payload.gender = parseFloat(document.getElementById('gender')?.value) || 0;
    payload.age_risk_bucket = ageBucket;

    // Visit history
    payload.total_prior_visits = parseFloat(document.getElementById('total_prior_visits')?.value) || 0;
    payload.high_utilizer = parseFloat(document.getElementById('high_utilizer')?.value) || 0;
    payload.time_in_hospital = parseFloat(document.getElementById('time_in_hospital')?.value) || 0;
    payload.number_inpatient = 0;
    payload.number_emergency = 0;
    payload.number_outpatient = 0;
    payload.has_number_inpatient = 0;
    payload.has_number_emergency = 0;
    payload.has_number_outpatient = 0;

    // Clinical
    payload.number_diagnoses = nDiagnoses;
    payload.num_medications = nMeds;
    payload.num_lab_procedures = parseFloat(document.getElementById('num_lab_procedures')?.value) || 0;
    payload.num_procedures = parseFloat(document.getElementById('num_procedures')?.value) || 0;
    payload.clinical_complexity = 0;
    payload.days_per_diagnosis = 0;
    payload.procedure_intensity = 0;
    payload.lab_intensity = 0;

    // Lab
    payload.a1c_tested = parseFloat(document.getElementById('a1c_tested')?.value) || 0;
    payload.a1c_high = parseFloat(document.getElementById('a1c_high')?.value) || 0;
    payload.a1c_very_high = 0;
    payload.glucose_tested = parseFloat(document.getElementById('glucose_tested')?.value) || 0;
    payload.glucose_high = 0;
    payload.both_labs_done = (payload.a1c_tested && payload.glucose_tested) ? 1 : 0;

    // Medications
    payload.n_active_medications = parseFloat(document.getElementById('n_active_medications')?.value) || 0;
    payload.any_insulin = parseFloat(document.getElementById('any_insulin')?.value) || 0;
    payload.insulin_increased = 0;
    payload.insulin_decreased = 0;
    payload.any_med_change = 0;
    payload.on_diabetes_med = payload.any_insulin;

    // Medication columns
    ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'glimepiride', 
     'repaglinide', 'nateglinide', 'chlorpropamide', 'tolbutamide', 'miglitol'].forEach(m => payload[m] = 0);
    payload.insulin = payload.any_insulin;
    payload['glyburide-metformin'] = 0;

    // Diagnosis categories
    payload.diag_1_category = parseFloat(document.getElementById('diag_1_category')?.value) || 0;
    payload.diag_2_category = 0;
    payload.diag_3_category = 0;
    payload.primary_diag_high_risk = 0;

    // Interactions
    payload.age_x_ndiagnoses = ageNumeric * nDiagnoses;
    payload.n_meds_x_ndiagnoses = nMeds * nDiagnoses;
    payload.inpatient_x_emergency = 0;
    payload.elderly_high_complexity = (ageBucket === 2 && nDiagnoses > 5) ? 1 : 0;

    // IDs
    payload.admission_type_id = 0;
    payload.discharge_disposition_id = 0;
    payload.admission_source_id = 0;

    // Other features
    payload.race = 0;
    payload.change = 0;
    payload.diabetesMed = payload.any_insulin;

    return payload;
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const riskScore = document.getElementById('result-risk-score');
    const riskTier = document.getElementById('result-risk-tier');
    const factorsList = document.getElementById('factors-list');
    const modelVersion = document.getElementById('result-model-version');
    const baseRate = document.getElementById('result-base-rate');

    if (data.risk_score !== undefined) {
        riskScore.textContent = (data.risk_score * 100).toFixed(1) + '%';
    } else {
        riskScore.textContent = (data.risk_score * 100).toFixed(1) + '%';
    }

    const tier = data.risk_tier?.toLowerCase() || 'low';
    riskTier.textContent = data.risk_tier || 'Unknown';
    riskTier.className = 'tier-badge ' + tier;

    factorsList.innerHTML = '<p style="color: var(--text-muted)">SHAP not available in debug mode</p>';

    modelVersion.textContent = data.model_version || '1.0.0';
    baseRate.textContent = '--';

    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function showError(msg) {
    const errorDiv = document.getElementById('error');
    const errorMsg = document.getElementById('error-message');
    errorMsg.textContent = msg;
    errorDiv.classList.remove('hidden');
    errorDiv.scrollIntoView({ behavior: 'smooth' });
}

function hideError() {
    document.getElementById('error').classList.add('hidden');
}

function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}