"""
Phase 6 — Security, Privacy & Adversarial Validation Test Suite

Validates:
  1. PII / Privacy validation (names, phones, emails, SSN, MRN, dates, addresses, zero leak in traces/logs)
  2. Adversarial / Malformed input resilience (missing, negative, impossible, extreme, weird types, long text, Unicode)
  3. Safety-rule bypass resistance (critical symptoms, dangerous vitals, pediatric, geriatric, zero history)
  4. Triage boundary validation (P1-P5 strict clamping, invalid model outputs)
  5. Confidence safety & boundaries (0.0-1.0 range, LOW/MEDIUM/HIGH bands, low-conf auto-escalation)
  6. API security & input sanitization (invalid patient IDs, SQL injection strings, schema validation)
  7. Database safety & rollback integrity
  8. Zero data leakage / evaluation integrity
"""

from __future__ import annotations

import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from backend.database import (
    db_clear_all_patients,
    db_get_patient,
    db_save_patient,
    db_get_all_patients,
    init_db,
)
from backend.queue_monitor import detect_vital_deterioration
from ml_engine import (
    extract_nlp,
    predict_triage,
    scrub_phi,
    _regex_phi_scrub,
    train_model,
)
from server import app
from workflow import build_and_run_pipeline

@pytest.fixture(scope="module")
def model():
    return train_model()

@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c

                                                                             
                             
                                                                             

def test_phi_scrubber_masks_patient_names():
    """Verify patient names with titles and labels are masked."""
    texts = [
        "Patient: John Smith presents with headache",
        "Mr. Robert Johnson reports dizziness",
        "Dr. Alice Walker evaluated the patient",
        "Name: Jane Doe complains of chest tightness",
        "Prof. Charles Xavier feeling unwell",
    ]
    for text in texts:
        scrubbed = scrub_phi(text)
        assert "John Smith" not in scrubbed
        assert "Robert Johnson" not in scrubbed
        assert "Alice Walker" not in scrubbed
        assert "Jane Doe" not in scrubbed
        assert "Charles Xavier" not in scrubbed
        assert any(tag in scrubbed for tag in ["[REDACTED_NAME]", "[REDACTED", "<PERSON>"])

def test_phi_scrubber_masks_phone_numbers():
    """Verify various phone number formats are masked."""
    texts = [
        "Call relative at 555-123-4567 immediately",
        "Contact: (800) 555-0199 for updates",
        "Patient phone +1-555-867-5309",
        "Alternate line 555.234.5678",
    ]
    for text in texts:
        scrubbed = scrub_phi(text)
        assert "555-123-4567" not in scrubbed
        assert "555-0199" not in scrubbed
        assert "867-5309" not in scrubbed
        assert "234.5678" not in scrubbed
        assert any(tag in scrubbed for tag in ["[REDACTED_PHONE]", "[REDACTED", "<PHONE_NUMBER>"])

def test_phi_scrubber_masks_email_addresses():
    """Verify email addresses are masked."""
    text = "Send lab results to patient.doe123@healthcare-system.org or emergency_contact@gmail.com"
    scrubbed = scrub_phi(text)
    assert "patient.doe123@healthcare-system.org" not in scrubbed
    assert "emergency_contact@gmail.com" not in scrubbed
    assert any(tag in scrubbed for tag in ["[REDACTED_EMAIL]", "[REDACTED", "<EMAIL_ADDRESS>"])

def test_phi_scrubber_masks_ssn_and_mrn():
    """Verify SSN and MRN are masked."""
    text = "Patient SSN is 000-12-3456 and hospital record is MRN 9876543."
    scrubbed = scrub_phi(text)
    assert "000-12-3456" not in scrubbed
    assert "9876543" not in scrubbed
    assert any(tag in scrubbed for tag in ["[REDACTED_SSN]", "[REDACTED", "<US_SSN>"])
    assert any(tag in scrubbed for tag in ["[REDACTED_MRN]", "[REDACTED", "<MEDICAL_LICENSE>"])

def test_phi_scrubber_masks_dates_and_addresses():
    """Verify DOB and residential addresses are masked."""
    text = "DOB: 05/12/1978 residing at 742 Evergreen Terrace with fever"
    scrubbed = scrub_phi(text)
    assert "05/12/1978" not in scrubbed
    assert "742 Evergreen Terrace" not in scrubbed
    assert any(tag in scrubbed for tag in ["[REDACTED_DATE]", "[REDACTED_ADDRESS]", "[REDACTED", "<DATE_TIME>", "<LOCATION>"])

def test_raw_phi_never_leaks_into_explainability_or_trace(model):
    """Verify that explainability factors and trace logs never expose unscrubbed PII."""
    raw_patient = {
        "id": "PT-SEC-01",
        "age": 45,
        "chief_complaint": "Patient: Johnathan Doe, SSN 123-45-6789, phone 555-432-1098, reports sudden crushing chest pain.",
        "heart_rate": 95,
        "oxygen_saturation": 97,
        "blood_pressure": "130/85",
        "temperature": 37.0,
        "respiratory_rate": 18,
        "gcs_score": 15,
        "history_available": True,
    }
    result = build_and_run_pipeline(
        patient=raw_patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )

                                                            
    assert "Johnathan Doe" not in result["scrubbed_complaint"]
    assert "123-45-6789" not in result["scrubbed_complaint"]
    assert "555-432-1098" not in result["scrubbed_complaint"]

                                                             
    for factor in result.get("explainability", []):
        reasoning = factor.get("aiReasoning", "")
        target = factor.get("highlightTarget", "")
        assert "Johnathan Doe" not in reasoning
        assert "123-45-6789" not in reasoning
        assert "555-432-1098" not in reasoning
        assert "Johnathan Doe" not in target

                                                    
    for step in result.get("trace", []):
        detail = step.get("detail", "")
        assert "Johnathan Doe" not in detail
        assert "123-45-6789" not in detail
        assert "555-432-1098" not in detail

                                                                             
                                          
                                                                             

def test_pipeline_handles_all_missing_vitals_safely(model):
    """Patient with completely missing vitals must not crash and must assign safe ESI with missing data flags."""
    patient = {
        "id": "PT-SEC-MISSING",
        "age": 30,
        "chief_complaint": "Unspecified malaise and generalized body aches",
        "heart_rate": None,
        "oxygen_saturation": None,
        "blood_pressure": None,
        "temperature": None,
        "respiratory_rate": None,
        "gcs_score": None,
        "history_available": False,
    }
    result = build_and_run_pipeline(
        patient=patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert 1 <= result["final_esi"] <= 5
    assert "MISSING_DATA" in result["safety_flags"]
    assert result["missing_data_penalty"] > 0
    assert 0.0 <= result["final_confidence"] <= 1.0

def test_pipeline_handles_extreme_and_negative_vitals(model):
    """Vitals with extreme and negative values must fail safely without crash or NaN."""
    adversarial_patient = {
        "id": "PT-SEC-EXTREME",
        "age": -5,
        "chief_complaint": "Severe weakness",
        "heart_rate": 99999,
        "oxygen_saturation": -50,
        "blood_pressure": "500/300",
        "temperature": 150.0,
        "respiratory_rate": -10,
        "gcs_score": 99,
        "history_available": True,
    }
    result = build_and_run_pipeline(
        patient=adversarial_patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert 1 <= result["final_esi"] <= 5
    assert not np.isnan(result["final_confidence"])
    assert 0.0 <= result["final_confidence"] <= 1.0

def test_pipeline_handles_malformed_blood_pressure(model):
    """Malformed blood pressure strings must be handled gracefully."""
    bad_bps = ["invalid", "120/", "/80", "abc/def", "120-80", 12345, None, "", "///"]
    for bp in bad_bps:
        patient = {
            "id": "PT-SEC-BP",
            "age": 40,
            "chief_complaint": "Routine checkup",
            "heart_rate": 75,
            "blood_pressure": bp,
            "oxygen_saturation": 98,
            "temperature": 37.0,
            "respiratory_rate": 16,
            "gcs_score": 15,
        }
        result = build_and_run_pipeline(
            patient=patient,
            model=model,
            phi_scrubber=scrub_phi,
            predictor=predict_triage,
            nlp_extractor=extract_nlp,
        )
        assert 1 <= result["final_esi"] <= 5
        assert 0.0 <= result["final_confidence"] <= 1.0

def test_pipeline_handles_extremely_long_chief_complaint(model):
    """50,000 character complaint should be processed without hanging or memory explosion."""
    long_complaint = "Patient reports mild headache. " * 1600              
    patient = {
        "id": "PT-SEC-LONG",
        "age": 35,
        "chief_complaint": long_complaint,
        "heart_rate": 78,
        "oxygen_saturation": 99,
        "blood_pressure": "120/80",
        "temperature": 36.8,
        "respiratory_rate": 16,
        "gcs_score": 15,
    }
    result = build_and_run_pipeline(
        patient=patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert 1 <= result["final_esi"] <= 5

def test_pipeline_handles_unusual_unicode_and_special_chars(model):
    """Emojis, null characters, RTL text, and special symbols should not crash the pipeline."""
    unicode_complaints = [
        "Patient has chest pain 🚑💥❤️ with severe distress",
        "المريض يعاني من ألم شديد في الصدر وضيق في التنفس",
        "Complaint with null \x00 bytes and \t tabs and \r\n newlines",
        "Zalgo text: H̶e̶a̶d̶a̶c̶h̶e̶ with n̶a̶u̶s̶e̶a̶",
    ]
    for comp in unicode_complaints:
        patient = {
            "id": "PT-SEC-UNICODE",
            "age": 45,
            "chief_complaint": comp,
            "heart_rate": 88,
            "oxygen_saturation": 96,
            "blood_pressure": "125/82",
            "temperature": 37.1,
            "respiratory_rate": 18,
            "gcs_score": 15,
        }
        result = build_and_run_pipeline(
            patient=patient,
            model=model,
            phi_scrubber=scrub_phi,
            predictor=predict_triage,
            nlp_extractor=extract_nlp,
        )
        assert 1 <= result["final_esi"] <= 5

                                                                             
                                  
                                                                             

def test_critical_symptoms_cannot_be_bypassed_by_normal_vitals(model):
    """Critical symptoms ('crushing chest pain', 'slurred speech') force high priority even with pristine vitals."""
    patient = {
        "id": "PT-SEC-SAFE-01",
        "age": 50,
        "chief_complaint": "Sudden onset crushing chest pain radiating to jaw and left arm, diaphoresis",
        "heart_rate": 72,
        "oxygen_saturation": 99,
        "blood_pressure": "120/80",
        "temperature": 37.0,
        "respiratory_rate": 16,
        "gcs_score": 15,
        "history_available": True,
    }
    result = build_and_run_pipeline(
        patient=patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
                                                       
    assert result["final_esi"] <= 2
    assert result["nlp_analysis"]["risk_level"] in {"CRITICAL", "HIGH"}

def test_dangerous_vitals_cannot_be_bypassed_by_low_risk_complaint(model):
    """Profound hypoxia (O2 80%) or coma (GCS 5) must trigger ESI 1 even if complaint sounds trivial."""
    patient = {
        "id": "PT-SEC-SAFE-02",
        "age": 40,
        "chief_complaint": "Patient says they just have a minor papercut on index finger",
        "heart_rate": 145,
        "oxygen_saturation": 80,
        "blood_pressure": "70/40",
        "temperature": 35.0,
        "respiratory_rate": 34,
        "gcs_score": 6,
        "history_available": True,
    }
    result = build_and_run_pipeline(
        patient=patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert result["final_esi"] == 1
    assert "CRITICAL_SAFETY_RULE" in result["safety_flags"]

def test_pediatric_safety_rules_cannot_be_bypassed(model):
    """Pediatric infant with high fever and lethargy must trigger age-adjusted high priority."""
    ped_patient = {
        "id": "PT-SEC-PED",
        "age": 2,
        "chief_complaint": "Toddler is lethargic, high fever, decreased oral intake",
        "heart_rate": 165,
        "oxygen_saturation": 93,
        "blood_pressure": None,
        "temperature": 39.8,
        "respiratory_rate": 42,
        "gcs_score": 13,
        "history_available": False,
    }
    result = build_and_run_pipeline(
        patient=ped_patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert result["age_group"] == "pediatric"
    assert "PEDIATRIC_AGE_ADJUSTED" in result["safety_flags"]
    assert result["final_esi"] <= 2

def test_geriatric_fall_on_anticoagulants_cannot_be_bypassed(model):
    """Geriatric patient with fall and anticoagulant mention must trigger age safety override."""
    ger_patient = {
        "id": "PT-SEC-GER",
        "age": 82,
        "chief_complaint": "Mechanical fall from standing height, on warfarin anticoagulant, mild hip pain",
        "heart_rate": 78,
        "oxygen_saturation": 96,
        "blood_pressure": "135/80",
        "temperature": 36.6,
        "respiratory_rate": 18,
        "gcs_score": 15,
        "history_available": True,
    }
    result = build_and_run_pipeline(
        patient=ger_patient,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )
    assert result["age_group"] == "geriatric"
    assert "GERIATRIC_AGE_ADJUSTED" in result["safety_flags"]
    assert result["final_esi"] <= 2

                                                                             
                               
                                                                             

def test_final_triage_level_strictly_clamped_p1_to_p5(model):
    """Verify that final_esi is always in {1, 2, 3, 4, 5} regardless of predictor output."""
    def mock_bad_predictor_high(m, p):
        return {"predicted_esi": 99, "probabilities": [0,0,0,0,1], "confidence_margin": 1.0, "method": "mock"}

    def mock_bad_predictor_low(m, p):
        return {"predicted_esi": -5, "probabilities": [1,0,0,0,0], "confidence_margin": 1.0, "method": "mock"}

    def mock_bad_predictor_zero(m, p):
        return {"predicted_esi": 0, "probabilities": [0,0,0,0,0], "confidence_margin": 0.0, "method": "mock"}

    base_patient = {
        "id": "PT-BOUND",
        "age": 30,
        "chief_complaint": "Mild rash",
        "heart_rate": 75,
        "oxygen_saturation": 99,
        "blood_pressure": "120/80",
        "temperature": 37.0,
        "respiratory_rate": 16,
        "gcs_score": 15,
    }

    res_high = build_and_run_pipeline(base_patient, model, scrub_phi, mock_bad_predictor_high, extract_nlp)
    assert res_high["final_esi"] == 5

    res_low = build_and_run_pipeline(base_patient, model, scrub_phi, mock_bad_predictor_low, extract_nlp)
    assert res_low["final_esi"] == 1

    res_zero = build_and_run_pipeline(base_patient, model, scrub_phi, mock_bad_predictor_zero, extract_nlp)
    assert res_zero["final_esi"] == 1

                                                                             
                                       
                                                                             

def test_confidence_score_strictly_within_0_and_1(model):
    """Confidence score must never exceed 1.0 or fall below 0.0."""
    patient = {
        "id": "PT-SEC-CONF",
        "age": 70,
        "chief_complaint": "Dizziness and fatigue",
        "heart_rate": None,
        "oxygen_saturation": None,
        "blood_pressure": None,
        "temperature": None,
        "respiratory_rate": None,
        "gcs_score": None,
        "history_available": False,
    }
    result = build_and_run_pipeline(patient, model, scrub_phi, predict_triage, extract_nlp)
    conf = result["final_confidence"]
    assert 0.0 <= conf <= 1.0

def test_confidence_band_classification():
    """Verify confidence band mapping helper and boundaries."""
    from evaluation.evaluate_golden_dataset import _confidence_band
    assert _confidence_band(0.80) == "HIGH"
    assert _confidence_band(0.95) == "HIGH"
    assert _confidence_band(0.799) == "MEDIUM"
    assert _confidence_band(0.55) == "MEDIUM"
    assert _confidence_band(0.549) == "LOW"
    assert _confidence_band(0.0) == "LOW"

                                                                             
                                      
                                                                             

def test_api_invalid_patient_id_safe_404(client):
    """Non-existent patient IDs and SQL injection attempts return 404 cleanly."""
    bad_ids = [
        "PT-NONEXISTENT-99999",
        "' OR '1'='1",
        "../../etc/passwd",
        "<script>alert(1)</script>",
        "PT-0000; DROP TABLE patients;--",
    ]
    for pid in bad_ids:
        resp = client.get(f"/api/patients/{pid}/status")
        assert resp.status_code == 404
        assert "detail" in resp.json()

def test_api_vitals_update_rejects_malformed_types(client):
    """Sending non-numeric values in vitals update payload must return 422 Unprocessable Entity."""
    resp = client.post(
        "/api/patients/PT-5001/vitals",
        json={"heart_rate": "super_fast", "oxygen_saturation": "ninety_nine"}
    )
    assert resp.status_code == 422

def test_api_override_requires_valid_esi_boundary(client):
    """Override with ESI 0 or ESI 6 must be rejected with 422."""
    for bad_esi in [0, 6, -1, 10]:
        resp = client.post(
            "/api/override/PT-5001",
            json={"nurse_esi": bad_esi, "override_reason": "Clinical intuition"}
        )
        assert resp.status_code == 422

def test_api_queue_endpoint_returns_sanitized_records(client):
    """GET /api/queue must return 200 with standard schema."""
    resp = client.get("/api/queue")
    assert resp.status_code == 200
    queue = resp.json()
    assert isinstance(queue, list)
    for p in queue:
        assert "patientId" in p or "id" in p
        assert "aiSuggestedPriority" in p
        assert "aiConfidenceScore" in p

                                                                             
                                
                                                                             

def test_database_integrity_after_failed_operations(client):
    """Failed API calls or invalid queries must not corrupt SQLite database state."""
                           
    before_patients = db_get_all_patients()
    before_count = len(before_patients)

                                         
    client.get("/api/patients/' OR '1'='1/status")
    client.post("/api/override/PT-NONEXISTENT", json={"nurse_esi": 2, "override_reason": "test"})
    client.post("/api/patients/PT-NONEXISTENT/vitals", json={"heart_rate": 80})

                                              
    after_patients = db_get_all_patients()
    assert len(after_patients) == before_count

def test_db_get_patient_nonexistent_returns_none():
    """db_get_patient with non-existent or invalid ID safely returns None without exception."""
    assert db_get_patient("DOES_NOT_EXIST") is None
    assert db_get_patient("") is None
    assert db_get_patient(None) is None

                                                                             
                                        
                                                                             

def test_pipeline_never_reads_or_relies_on_evaluation_labels(model):
    """
    CRITICAL: Confirm that passing ground-truth evaluation labels into
    to_pipeline_patient() strips them and that build_and_run_pipeline()
    operates entirely without knowledge of expected outputs.
    """
    records = load_golden_dataset()
    assert len(records) > 0
    golden_sample = records[0]

    eval_fields = {
        "expected_triage_level",
        "expected_risk_category",
        "expected_confidence_band",
        "expected_escalation",
        "expected_primary_reason",
    }

                                                             
    for field in eval_fields:
        assert field in golden_sample

    pipeline_input = to_pipeline_patient(golden_sample)

                                                                                
    for field in eval_fields:
        assert field not in pipeline_input, f"Evaluation field {field} leaked into pipeline input!"

                      
    result = build_and_run_pipeline(
        patient=pipeline_input,
        model=model,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )

    assert 1 <= result["final_esi"] <= 5
    for field in eval_fields:
        assert field not in result
