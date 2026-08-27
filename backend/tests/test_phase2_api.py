from fastapi.testclient import TestClient

from backend.database import db_clear_all_patients
from server import app

def test_reassessment_and_vitals_api_workflow():
    with TestClient(app) as client:
        queue = client.get("/api/queue")
        assert queue.status_code == 200
        assert queue.json()
        patient_id = queue.json()[0]["patientId"]

        status = client.get(f"/api/patients/{patient_id}/status")
        assert status.status_code == 200
        assert "waitingMinutes" in status.json()
        assert "lastAssessmentAt" in status.json()

        stable = client.post(f"/api/patients/{patient_id}/vitals", json={
            "heart_rate": 80,
            "oxygen_saturation": 98,
            "gcs_score": 15,
        })
        assert stable.status_code == 200
        assert stable.json()["deteriorationDetected"] is False

        worsening = client.post(f"/api/patients/{patient_id}/vitals", json={
            "heart_rate": 125,
            "oxygen_saturation": 90,
            "gcs_score": 12,
        })
        assert worsening.status_code == 200
        assert worsening.json()["reassessmentRequired"] is True
        assert worsening.json()["deteriorationDetected"] is True

        reassessment = client.get(f"/api/patients/{patient_id}/reassessment")
        assert reassessment.status_code == 200
        assert any(event["event_type"] == "VITAL_DETERIORATION" for event in reassessment.json()["events"])

        acknowledged = client.post(f"/api/patients/{patient_id}/reassessment/acknowledge")
        assert acknowledged.status_code == 200
        assert acknowledged.json()["reassessmentStatus"] == "ACKNOWLEDGED"
        assert acknowledged.json()["reassessmentRequired"] is False

    db_clear_all_patients()
