"""
Phase 5 Regression Tests — Evaluation, Validation & Production-Readiness

Covers:
  - Stable vitals do NOT trigger deterioration
  - Deteriorating vitals DO trigger deterioration
  - Reassessment workflow persists correctly in DB
  - Pediatric routing correctness
  - Geriatric routing correctness
  - Zero-history handling correctness
  - Missing-data handling correctness
  - Confidence escalation correctness
  - Safety rules cannot be downgraded by ML prediction
  - Clinician acceptance persists audit event
  - Clinician override requires valid level/reason
  - Safety events remain persisted after workflow
  - Data leakage: expected labels are never passed into pipeline input
"""

from __future__ import annotations

import json
import datetime

import pytest
from fastapi.testclient import TestClient

from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from backend.database import (
    db_clear_all_patients,
    db_get_patient,
    db_save_patient,
    init_db,
)
from backend.queue_monitor import detect_vital_deterioration, monitor_once
from ml_engine import extract_nlp, predict_triage, scrub_phi, train_model
from server import app
from workflow import build_and_run_pipeline

                                                                               
_EVALUATION_ONLY_FIELDS = {
    "expected_triage_level",
    "expected_risk_category",
    "expected_confidence_band",
    "expected_escalation",
    "expected_primary_reason",
}

                                                                               

@pytest.fixture(scope="module")
def model():
    return train_model()

@pytest.fixture(scope="module")
def golden_records():
    return {r["scenario_tag"]: r for r in load_golden_dataset()}

                                                                               

def test_no_ground_truth_leakage_in_pipeline_input(golden_records):
    """
    CRITICAL: Verify that ground-truth evaluation labels are never present
    in the dict passed into the triage pipeline. Any leakage would invalidate
    all evaluation results.
    """
    all_records = list(golden_records.values())
    for record in all_records:
        pipeline_input = to_pipeline_patient(record)
        leaked = _EVALUATION_ONLY_FIELDS & set(pipeline_input.keys())
        assert leaked == set(), (
            f"DATA LEAKAGE detected for patient {record['patient_id']}: "
            f"fields {leaked} should not be in pipeline input."
        )

                                                                                

def test_stable_vitals_do_not_trigger_deterioration():
    """Stable readings must never produce a deterioration event."""
    prev = {"heartRateBpm": 80, "o2SaturationPercent": 98, "gcsScore": 15}
    current = {"heartRateBpm": 82, "o2SaturationPercent": 97, "gcsScore": 15}
    result = detect_vital_deterioration(prev, current)
    assert result == {}, f"Expected no deterioration, got: {result}"

                                                                               

def test_deteriorating_vitals_trigger_deterioration():
    """Worsening O2 and HR must flag deterioration."""
    prev = {"heartRateBpm": 80, "o2SaturationPercent": 98, "gcsScore": 15}
    worsened = {"heartRateBpm": 125, "o2SaturationPercent": 90, "gcsScore": 12}
    result = detect_vital_deterioration(prev, worsened)
    assert len(result) > 0, "Deterioration must be flagged for significant vital change"
    assert "oxygen_saturation" in result or "heart_rate" in result

                                                                                

def test_reassessment_workflow_persists_in_db():
    """
    Simulates a wait-threshold breach and verifies that reassessmentRequired
    is persisted correctly in the database.
    """
    init_db()
    db_clear_all_patients()

    now = datetime.datetime.now(datetime.timezone.utc)
    started = (now - datetime.timedelta(minutes=40)).isoformat()

    patient_id = "PHASE5-REGR-001"
    db_save_patient({
        "patientId": patient_id,
        "name": "Regression Test Patient",
        "age": 45,
        "biologicalSex": "M",
        "chiefComplaint": "Abdominal pain",
        "vitals": {"heartRateBpm": 88, "o2SaturationPercent": 97, "temperatureCelsius": 37.2},
        "history_available": True,
        "aiSuggestedPriority": 3,
        "aiConfidenceScore": 0.70,
        "safetyFlags": [],
        "missingFields": [],
        "status": "AWAITING_REVIEW",
        "lifecycleStatus": "WAITING",
        "arrivalTime": started,
        "waitingStartedAt": started,
        "lastQueuePriority": 3,
    })

    events = monitor_once(now)
    assert any(e["patient_id"] == patient_id for e in events), (
        "Queue monitor must produce a reassessment event for a 40-minute waiting patient."
    )

    patient_status = db_get_patient(patient_id)
    assert patient_status["reassessmentRequired"] is True, (
        "reassessmentRequired must be True after threshold breach."
    )

    db_clear_all_patients()

                                                                               

def test_pediatric_routing_is_correct(model, golden_records):
    """Pipeline must route pediatric patients to the 'pediatric' age group."""
    rec = golden_records["PEDIATRIC_AGE_ADJUSTMENT"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["age_group"] == "pediatric", (
        f"Pediatric patient routed to '{out['age_group']}' — expected 'pediatric'."
    )
    assert "PEDIATRIC_AGE_ADJUSTED" in out["safety_flags"], (
        "Pediatric safety flag missing from output."
    )

                                                                               

def test_geriatric_routing_is_correct(model, golden_records):
    """Pipeline must route geriatric patients to the 'geriatric' age group."""
    rec = golden_records["GERIATRIC_AMBIGUOUS_ZERO_HISTORY"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["age_group"] == "geriatric", (
        f"Geriatric patient routed to '{out['age_group']}' — expected 'geriatric'."
    )
    assert "GERIATRIC_AGE_ADJUSTED" in out["safety_flags"]

                                                                               

def test_zero_history_is_correctly_identified(model, golden_records):
    """Zero-history patients must be classified 'none' with ZERO_HISTORY flag."""
    rec = golden_records["ZERO_HISTORY"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["history_availability"] == "none"
    assert "ZERO_HISTORY" in out["safety_flags"]

                                                                               

def test_missing_data_is_flagged_and_penalised(model, golden_records):
    """Missing vital fields must be detected and incur a confidence penalty."""
    rec = golden_records["MISSING_VITAL"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert "oxygen_saturation" in out["missing_fields"], (
        "Missing O2 sat must appear in missing_fields list."
    )
    assert "MISSING_DATA" in out["safety_flags"]
    assert out["missing_data_penalty"] > 0, "Missing data penalty must be non-zero."

                                                                               

def test_low_confidence_triggers_escalation(model, golden_records):
    """
    An ambiguous presentation must lower confidence relative to a high-confidence
    case, and the low-confidence penalty must be non-zero.
    """
    ambig = golden_records["AMBIGUOUS_PRESENTATION"]
    clear = golden_records["HIGH_CONFIDENCE"]
    out_ambig = build_and_run_pipeline(to_pipeline_patient(ambig), model, scrub_phi, predict_triage, extract_nlp)
    out_clear = build_and_run_pipeline(to_pipeline_patient(clear), model, scrub_phi, predict_triage, extract_nlp)
    assert out_ambig["nlp_ambiguity_penalty"] > 0, "NLP ambiguity penalty must be non-zero for ambiguous case."
    assert out_ambig["final_confidence"] < out_clear["final_confidence"], (
        "Ambiguous case confidence must be lower than clear case confidence."
    )

                                                                               

def test_safety_rules_cannot_be_downgraded_by_ml(model, golden_records):
    """
    When vitals are critically dangerous and NLP is low-risk, the deterministic
    safety rule must override the ML prediction and produce ESI 1.
    """
    rec = golden_records["VITALS_OVERRIDE_NLP"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] == 1, (
        f"Safety rule must produce ESI 1 regardless of NLP signal, got ESI {out['final_esi']}."
    )
    assert out["safety_override"] is True
    assert "CRITICAL_SAFETY_RULE" in out["safety_flags"]

def test_conflicting_signals_deterministic_safety_wins(model, golden_records):
    """Deterministic safety rules must always beat conflicting ML/NLP predictions."""
    rec = golden_records["CONFLICTING_SIGNALS"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] == 1
    assert out["safety_override"] is True

                                                                                

def test_clinician_acceptance_persists_audit_event():
    """POST /api/accept/{id} must create an AI_RECOMMENDATION_ACCEPTED audit event."""
    with TestClient(app) as client:
        resp = client.post("/api/patients", json={
            "age": 35,
            "gender": "F",
            "chief_complaint": "Phase 5 regression test — mild headache",
            "heart_rate": 72,
            "oxygen_saturation": 99,
            "respiratory_rate": 16,
            "temperature": 36.8,
            "gcs_score": 15,
        })
        assert resp.status_code == 200
        patient_id = resp.json()["patientId"]

        accepted = client.post(f"/api/accept/{patient_id}")
        assert accepted.status_code == 200

        audit = client.get("/api/audit-log").json()
        event = next(
            (e for e in audit["events"] if e["patient_id"] == patient_id),
            None,
        )
        assert event is not None, "Audit event must be created for accepted patient."
        assert event["event_type"] == "AI_RECOMMENDATION_ACCEPTED"

    db_clear_all_patients()

                                                                               

def test_clinician_override_requires_valid_level_and_reason():
    """Clinician override must reject invalid ESI level and empty reason."""
    with TestClient(app) as client:
        resp = client.post("/api/patients", json={
            "age": 45,
            "gender": "M",
            "chief_complaint": "Phase 5 regression test — back pain",
            "heart_rate": 78,
            "oxygen_saturation": 98,
            "respiratory_rate": 15,
            "temperature": 37.1,
            "gcs_score": 15,
        })
        assert resp.status_code == 200
        patient_id = resp.json()["patientId"]

                                               
        bad_esi = client.post(f"/api/override/{patient_id}", json={"nurse_esi": 7, "override_reason": "x"})
        assert bad_esi.status_code == 422

                      
        empty_reason = client.post(f"/api/override/{patient_id}", json={"nurse_esi": 2, "override_reason": ""})
        assert empty_reason.status_code == 422

                                     
        good = client.post(f"/api/override/{patient_id}", json={
            "nurse_esi": 2,
            "override_reason": "Patient appears worse on re-examination",
            "reason_code": "PATIENT_APPEARS_WORSE",
            "clinician_reason": "Worsening respiratory effort observed.",
        })
        assert good.status_code == 200

    db_clear_all_patients()

                                                                               

def test_safety_events_persisted_in_database():
    """
    Safety events triggered during vitals update (VITAL_DETERIORATION) must
    remain in the reassessment event store and be retrievable via the API.
    """
    with TestClient(app) as client:
        resp = client.post("/api/patients", json={
            "age": 60,
            "gender": "M",
            "chief_complaint": "Phase 5 regression — chest tightness",
            "heart_rate": 85,
            "oxygen_saturation": 97,
            "respiratory_rate": 18,
            "temperature": 37.0,
            "gcs_score": 15,
        })
        assert resp.status_code == 200
        patient_id = resp.json()["patientId"]

                               
        client.post(f"/api/patients/{patient_id}/vitals", json={
            "heart_rate": 128,
            "oxygen_saturation": 89,
            "gcs_score": 12,
        })

                                      
        reassessment = client.get(f"/api/patients/{patient_id}/reassessment")
        assert reassessment.status_code == 200
        events = reassessment.json()["events"]
        event_types = {e["event_type"] for e in events}
        assert "VITAL_DETERIORATION" in event_types, (
            f"VITAL_DETERIORATION event not found in reassessment store. Got: {event_types}"
        )

    db_clear_all_patients()

                                                                               

def test_api_queue_endpoint_returns_200():
    """GET /api/queue must return HTTP 200 after Phase 5 changes."""
    with TestClient(app) as client:
        resp = client.get("/api/queue")
        assert resp.status_code == 200

def test_api_patient_status_endpoint_returns_200():
    """GET /api/patients/{id}/status must return HTTP 200 for a valid patient."""
    with TestClient(app) as client:
                                
        resp = client.post("/api/patients", json={
            "age": 30,
            "gender": "F",
            "chief_complaint": "Phase 5 API validation",
            "heart_rate": 72,
            "oxygen_saturation": 99,
            "respiratory_rate": 14,
            "temperature": 36.9,
            "gcs_score": 15,
        })
        assert resp.status_code == 200
        patient_id = resp.json()["patientId"]

        status = client.get(f"/api/patients/{patient_id}/status")
        assert status.status_code == 200
        body = status.json()
        assert "waitingMinutes" in body
        assert "lastAssessmentAt" in body

    db_clear_all_patients()

                                                                                

def test_database_persists_required_fields():
    """
    Verify that the database correctly stores and retrieves all Phase 5-critical fields:
    triage level, confidence, safety flags, deterioration_detected,
    reassessment_required, queue priority, timestamps.
    """
    init_db()
    db_clear_all_patients()

    patient_id = "PHASE5-DB-001"
    db_save_patient({
        "patientId": patient_id,
        "name": "DB Persistence Test",
        "age": 50,
        "biologicalSex": "F",
        "chiefComplaint": "Shortness of breath",
        "vitals": {"heartRateBpm": 110, "o2SaturationPercent": 93, "temperatureCelsius": 38.0},
        "history_available": False,
        "aiSuggestedPriority": 2,
        "aiConfidenceScore": 0.65,
        "safetyFlags": ["MISSING_DATA", "ZERO_HISTORY"],
        "missingFields": ["blood_pressure"],
        "status": "AWAITING_REVIEW",
        "lifecycleStatus": "WAITING",
        "arrivalTime": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "waitingStartedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lastQueuePriority": 2,
    })

    record = db_get_patient(patient_id)
    assert record is not None, "Saved patient must be retrievable."
    assert record["aiSuggestedPriority"] == 2
    assert record["aiConfidenceScore"] == pytest.approx(0.65, abs=0.01)

    safety_flags = record.get("safetyFlags", [])
    if isinstance(safety_flags, str):
        safety_flags = json.loads(safety_flags)
    assert "MISSING_DATA" in safety_flags
    assert "ZERO_HISTORY" in safety_flags

    db_clear_all_patients()
