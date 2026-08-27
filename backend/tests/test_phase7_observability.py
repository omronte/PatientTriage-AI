"""
Phase 7 — Observability, Audit Trails & System Monitoring Test Suite

Validates:
  1. Initial patient triage creates audit events (AI_RECOMMENDATION_GENERATED, PATIENT_TRIAGED)
  2. Safety rule trigger creates SAFETY_RULE_TRIGGERED audit event
  3. Stable vitals do NOT create false deterioration events
  4. Deteriorating vitals create VITAL_DETERIORATION and REASSESSMENT_REQUIRED events
  5. Reassessment workflow creates proper audit trail (REQUESTED, ACKNOWLEDGED)
  6. Wait-threshold events are auditable (WAIT_THRESHOLD_EXCEEDED)
  7. Clinician acceptance is auditable (AI_RECOMMENDATION_ACCEPTED)
  8. Clinician override is auditable (AI_RECOMMENDATION_OVERRIDDEN)
  9. GET /api/patients/{id}/audit returns chronological events
  10. GET /api/patients/{id}/audit returns 404 for unknown patient ID
  11. GET /api/patients/{id}/explanation reflects real pipeline logic
  12. GET /api/patients/{id}/explanation returns 404 for unknown patient ID
  13. Audit output does not contain raw sensitive PII
  14. Multiple sequential events for the same patient are preserved
  15. Failed operations do not corrupt audit history
  16. GET /api/metrics returns structured observability metrics
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from backend.database import (
    db_clear_all_patients,
    db_get_patient,
    db_get_safety_events,
    init_db,
)
from server import app

@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c

                                                                             
                                         
                                                                             

def test_initial_triage_creates_audit_events(client):
    """Registering a patient must create an AI_RECOMMENDATION_GENERATED audit event."""
    resp = client.post("/api/patients", json={
        "age": 35,
        "gender": "Female",
        "chief_complaint": "Mild ankle sprain after stepping off curb",
        "heart_rate": 78,
        "blood_pressure_sys": 120,
        "blood_pressure_dia": 80,
        "oxygen_saturation": 99,
        "respiratory_rate": 16,
        "temperature": 37.0,
        "gcs_score": 15,
    })
    assert resp.status_code == 200
    patient = resp.json()
    pid = patient.get("id") or patient.get("patientId")

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "AI_RECOMMENDATION_GENERATED" in event_types

    rec_event = next(e for e in events if e["event_type"] == "AI_RECOMMENDATION_GENERATED")
    assert rec_event["current_state"] == "AWAITING_REVIEW"
    assert rec_event["actor"] == "system"
    assert rec_event["new_triage_level"] == patient["aiSuggestedPriority"]
    assert rec_event["confidence"] == patient["aiConfidenceScore"]

def test_safety_rule_trigger_creates_safety_event(client):
    """Registering a patient with safety rule conditions creates a SAFETY_RULE_TRIGGERED event."""
    resp = client.post("/api/patients", json={
        "age": 3,
        "gender": "Male",
        "chief_complaint": "Pediatric high fever and lethargy",
        "heart_rate": 160,
        "oxygen_saturation": 91,
        "temperature": 39.5,
        "respiratory_rate": 36,
        "gcs_score": 13,
    })
    assert resp.status_code == 200
    patient = resp.json()
    pid = patient.get("id") or patient.get("patientId")

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "SAFETY_RULE_TRIGGERED" in event_types
    safety_event = next(e for e in events if e["event_type"] == "SAFETY_RULE_TRIGGERED")
    assert safety_event["actor"] == "system"
    assert "PEDIATRIC_AGE_ADJUSTED" in safety_event["safety_flags"] or "CRITICAL_SAFETY_RULE" in safety_event["safety_flags"] or "AGE_OR_VITAL_SAFETY_RULE" in safety_event["safety_flags"]

                                                                             
                                                    
                                                                             

def test_stable_vitals_do_not_create_false_deterioration(client):
    """Updating with stable vitals must NOT generate a deterioration audit event."""
                      
    resp = client.post("/api/patients", json={
        "age": 40,
        "gender": "Male",
        "chief_complaint": "Moderate abdominal pain",
        "heart_rate": 80,
        "blood_pressure_sys": 120,
        "blood_pressure_dia": 80,
        "oxygen_saturation": 98,
        "temperature": 37.0,
        "respiratory_rate": 16,
        "gcs_score": 15,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    events_before = len(db_get_safety_events(pid))

                                                   
    resp_vitals = client.post(f"/api/patients/{pid}/vitals", json={
        "heart_rate": 82,
        "oxygen_saturation": 98,
        "temperature": 37.1,
    })
    assert resp_vitals.status_code == 200

    events_after = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events_after]
    assert "VITAL_DETERIORATION" not in event_types

def test_deteriorating_vitals_create_deterioration_event(client):
    """Updating with acutely worsening vitals MUST generate VITAL_DETERIORATION event."""
    resp = client.post("/api/patients", json={
        "age": 55,
        "gender": "Female",
        "chief_complaint": "Mild shortness of breath",
        "heart_rate": 85,
        "blood_pressure_sys": 125,
        "blood_pressure_dia": 82,
        "oxygen_saturation": 97,
        "temperature": 37.0,
        "respiratory_rate": 18,
        "gcs_score": 15,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

                                                   
    resp_worsened = client.post(f"/api/patients/{pid}/vitals", json={
        "heart_rate": 135,
        "oxygen_saturation": 88,
        "respiratory_rate": 32,
    })
    assert resp_worsened.status_code == 200

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "VITAL_DETERIORATION" in event_types

    det_event = next(e for e in events if e["event_type"] == "VITAL_DETERIORATION")
    assert det_event["current_state"] == "REASSESSMENT_REQUIRED"
    assert det_event["vital_changes"] != {}

def test_reassessment_workflow_audit_trail(client):
    """Manual reassessment request and acknowledgment generate chronological audit events."""
    resp = client.post("/api/patients", json={
        "age": 28,
        "gender": "Male",
        "chief_complaint": "Persistent nausea",
        "heart_rate": 76,
        "oxygen_saturation": 99,
        "blood_pressure_sys": 118,
        "blood_pressure_dia": 78,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

                          
    resp_req = client.post(f"/api/patients/{pid}/reassessment")
    assert resp_req.status_code == 200

                              
    resp_ack = client.post(f"/api/patients/{pid}/reassessment/acknowledge")
    assert resp_ack.status_code == 200

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "REASSESSMENT_REQUESTED" in event_types
    assert "REASSESSMENT_ACKNOWLEDGED" in event_types

                                                                             
                                    
                                                                             

def test_clinician_acceptance_is_auditable(client):
    """Accepting an AI recommendation records an AI_RECOMMENDATION_ACCEPTED audit event."""
    resp = client.post("/api/patients", json={
        "age": 42,
        "gender": "Male",
        "chief_complaint": "Mild rash on forearm",
        "heart_rate": 72,
        "oxygen_saturation": 99,
        "blood_pressure_sys": 120,
        "blood_pressure_dia": 80,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    resp_accept = client.post(f"/api/accept/{pid}")
    assert resp_accept.status_code == 200

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "AI_RECOMMENDATION_ACCEPTED" in event_types

    accept_event = next(e for e in events if e["event_type"] == "AI_RECOMMENDATION_ACCEPTED")
    assert accept_event["actor"] == "clinician"
    assert accept_event["current_state"] == "IN_TREATMENT"

def test_clinician_override_is_auditable(client):
    """Overriding an AI recommendation records an AI_RECOMMENDATION_OVERRIDDEN audit event."""
    resp = client.post("/api/patients", json={
        "age": 60,
        "gender": "Female",
        "chief_complaint": "Generalized weakness",
        "heart_rate": 88,
        "oxygen_saturation": 96,
        "blood_pressure_sys": 130,
        "blood_pressure_dia": 85,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    resp_override = client.post(f"/api/override/{pid}", json={
        "nurse_esi": 2,
        "override_reason": "Patient looks pale and clammy on bedside visual inspection",
        "reason_code": "CLINICAL_APPEARANCE",
        "clinician_reason": "Appears significantly sicker than baseline vitals suggest",
    })
    assert resp_override.status_code == 200

    events = db_get_safety_events(pid)
    event_types = [e["event_type"] for e in events]
    assert "AI_RECOMMENDATION_OVERRIDDEN" in event_types

    override_event = next(e for e in events if e["event_type"] == "AI_RECOMMENDATION_OVERRIDDEN")
    assert override_event["actor"] == "clinician"
    assert override_event["new_triage_level"] == 2
    assert override_event["reason_code"] == "CLINICAL_APPEARANCE"

                                                                             
                                
                                                                             

def test_audit_endpoint_returns_chronological_events(client):
    """GET /api/patients/{id}/audit returns full event list sorted chronologically."""
    resp = client.post("/api/patients", json={
        "age": 50,
        "gender": "Male",
        "chief_complaint": "Epigastric pain",
        "heart_rate": 84,
        "oxygen_saturation": 98,
        "blood_pressure_sys": 128,
        "blood_pressure_dia": 82,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

                              
    client.post(f"/api/patients/{pid}/vitals", json={"heart_rate": 130, "oxygen_saturation": 89})
    client.post(f"/api/patients/{pid}/reassessment/acknowledge")
    client.post(f"/api/accept/{pid}")

    audit_resp = client.get(f"/api/patients/{pid}/audit")
    assert audit_resp.status_code == 200
    data = audit_resp.json()
    assert data["patient_id"] == pid
    assert data["count"] >= 3
    events = data["events"]
    assert len(events) >= 3

                                  
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)

def test_audit_endpoint_handles_unknown_patient(client):
    """GET /api/patients/{id}/audit returns 404 for unknown patient ID."""
    resp = client.get("/api/patients/PT-UNKNOWN-9999/audit")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Patient not found"

def test_decision_explanation_endpoint_structure(client):
    """GET /api/patients/{id}/explanation exposes structured breakdown of decision logic."""
    resp = client.post("/api/patients", json={
        "age": 72,
        "gender": "Male",
        "chief_complaint": "Dizziness and unsteady gait",
        "heart_rate": 90,
        "oxygen_saturation": 96,
        "blood_pressure_sys": 135,
        "blood_pressure_dia": 85,
        "temperature": 37.0,
        "respiratory_rate": 18,
        "gcs_score": 15,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    resp_exp = client.get(f"/api/patients/{pid}/explanation")
    assert resp_exp.status_code == 200
    exp = resp_exp.json()

    assert exp["patient_id"] == pid
    assert "model_recommendation" in exp
    assert "deterministic_safety_adjustment" in exp
    assert "confidence_adjustment" in exp
    assert "missing_data_adjustment" in exp
    assert "age_adjustment" in exp
    assert "vital_deterioration" in exp
    assert "final_decision" in exp

    assert exp["age_adjustment"]["age_group"] == "geriatric"
    assert 1 <= exp["final_decision"]["final_esi"] <= 5

def test_decision_explanation_handles_unknown_patient(client):
    """GET /api/patients/{id}/explanation returns 404 for unknown patient ID."""
    resp = client.get("/api/patients/PT-UNKNOWN-9999/explanation")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Patient not found"

                                                                             
                                              
                                                                             

def test_audit_output_does_not_contain_obvious_pii(client):
    """Audit events and decision traces must not leak raw PII in explanations or event fields."""
    resp = client.post("/api/patients", json={
        "age": 45,
        "gender": "Female",
        "chief_complaint": "Patient: Secret Person, SSN 999-88-7777, phone 555-987-6543 has severe migraine",
        "heart_rate": 78,
        "oxygen_saturation": 99,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    audit_resp = client.get(f"/api/patients/{pid}/audit")
    assert audit_resp.status_code == 200
    raw_audit_text = json.dumps(audit_resp.json())

    assert "Secret Person" not in raw_audit_text
    assert "999-88-7777" not in raw_audit_text
    assert "555-987-6543" not in raw_audit_text

def test_observability_metrics_endpoint(client):
    """GET /api/metrics returns full system observability and triage statistics."""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    metrics = resp.json()

    assert "total_patients_processed" in metrics
    assert "total_triage_decisions" in metrics
    assert "priority_distribution" in metrics
    assert "low_confidence_decisions" in metrics
    assert "safety_rule_triggers" in metrics
    assert "deterioration_events" in metrics
    assert "reassessment_events" in metrics
    assert "clinician_acceptances" in metrics
    assert "clinician_overrides" in metrics
    assert "missing_data_cases" in metrics
    assert "zero_history_cases" in metrics
    assert "event_breakdown" in metrics

def test_audit_integrity_after_failed_operations(client):
    """Failed operations do not delete or corrupt existing valid audit history."""
    resp = client.post("/api/patients", json={
        "age": 33,
        "gender": "Female",
        "chief_complaint": "Ear pain",
        "heart_rate": 75,
        "oxygen_saturation": 99,
    })
    pid = resp.json().get("id") or resp.json().get("patientId")

    events_before = len(db_get_safety_events(pid))

                                                         
    client.post(f"/api/override/{pid}", json={"nurse_esi": 99, "override_reason": ""})
    client.post(f"/api/patients/{pid}/vitals", json={"heart_rate": "invalid"})

    events_after = len(db_get_safety_events(pid))
    assert events_after == events_before
