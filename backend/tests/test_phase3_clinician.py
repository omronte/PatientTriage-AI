from fastapi.testclient import TestClient

from backend.database import db_clear_all_patients
from server import app

def _new_patient(client, complaint="Mild ankle pain"):
    response = client.post("/api/patients", json={
        "age": 30,
        "gender": "F",
        "chief_complaint": complaint,
        "heart_rate": 72,
        "oxygen_saturation": 99,
        "respiratory_rate": 16,
        "temperature": 36.8,
        "gcs_score": 15,
    })
    assert response.status_code == 200
    return response.json()

def test_override_requires_valid_level_and_reason_and_records_audit_event():
    with TestClient(app) as client:
        patient = _new_patient(client)
        patient_id = patient["patientId"]

        invalid = client.post(f"/api/override/{patient_id}", json={"nurse_esi": 7, "override_reason": "x"})
        assert invalid.status_code == 422

        missing_reason = client.post(f"/api/override/{patient_id}", json={"nurse_esi": 2, "override_reason": ""})
        assert missing_reason.status_code == 422

        overridden = client.post(f"/api/override/{patient_id}", json={
            "nurse_esi": 2,
            "override_reason": "Patient appears worse",
            "reason_code": "PATIENT_APPEARS_WORSE",
            "clinician_reason": "New work of breathing observed at bedside.",
        })
        assert overridden.status_code == 200
        assert overridden.json()["patient"]["lifecycleStatus"] == "IN_TREATMENT"

        audit = client.get("/api/audit-log")
        assert audit.status_code == 200
        event = next(event for event in audit.json()["events"] if event["patient_id"] == patient_id)
        assert event["event_type"] == "AI_RECOMMENDATION_OVERRIDDEN"
        assert event["previous_triage_level"] == patient["aiSuggestedPriority"]
        assert event["new_triage_level"] == 2
        assert event["reason_code"] == "PATIENT_APPEARS_WORSE"
        assert event["actor"] == "clinician"

    db_clear_all_patients()

def test_accept_records_clinician_event_without_raw_complaint():
    with TestClient(app) as client:
        patient = _new_patient(client, "Chest discomfort for synthetic demo")
        patient_id = patient["patientId"]
        accepted = client.post(f"/api/accept/{patient_id}")
        assert accepted.status_code == 200

        audit = client.get("/api/audit-log").json()
        event = next(event for event in audit["events"] if event["patient_id"] == patient_id)
        assert event["event_type"] == "AI_RECOMMENDATION_ACCEPTED"
        assert "complaint" not in event

    db_clear_all_patients()
