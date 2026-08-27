"""
test_api.py — Comprehensive automated test suite for PatientTriage.ai FastAPI backend
"""
import sys
from fastapi.testclient import TestClient
from server import app

def test_full_pipeline():
    with TestClient(app) as client:
        print("1. Testing GET / (Frontend serving)...")
        res = client.get("/")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "PatientTriage" in res.text
        print("[PASS] Frontend served properly")

        print("\n2. Testing GET /api/patients...")
        res = client.get("/api/patients")
        assert res.status_code == 200
        patients = res.json()
        assert len(patients) >= 20, f"Expected at least 20 patients, got {len(patients)}"
        sample_pt = patients[0]
        assert "patientId" in sample_pt
        assert "aiSuggestedPriority" in sample_pt
        assert "aiConfidenceScore" in sample_pt
        assert "explainability" in sample_pt
        print(f"[PASS] Retrieved {len(patients)} patients from queue")

        print("\n3. Testing POST /api/patients (Registration & LangGraph Triage)...")
        new_pt_payload = {
            "age": 72,
            "gender": "M",
            "chief_complaint": "Sudden onset chest tightness and shortness of breath while gardening.",
            "heart_rate": 115,
            "blood_pressure_sys": 160,
            "blood_pressure_dia": 95,
            "oxygen_saturation": 91,
            "respiratory_rate": 24,
            "temperature": 37.2,
            "gcs_score": 15
        }
        res = client.post("/api/patients", json=new_pt_payload)
        assert res.status_code == 200
        new_pt = res.json()
        new_pt_id = new_pt["patientId"]
        assert new_pt["aiSuggestedPriority"] in [1, 2, 3, 4, 5]
        assert len(new_pt["explainability"]) > 0
        print(f"[PASS] Registered {new_pt_id}: AI ESI {new_pt['aiSuggestedPriority']}, Confidence: {new_pt['aiConfidenceScore']:.0%}")

        print("\n4. Testing POST /api/accept/{patient_id}...")
        res = client.post(f"/api/accept/{new_pt_id}")
        assert res.status_code == 200
        accept_data = res.json()
        assert accept_data["status"] == "accepted"
        print(f"[PASS] Accepted AI decision for {new_pt_id}")

        print("\n5. Testing POST /api/override/{patient_id}...")
                                                   
        res = client.post("/api/patients", json={
            "age": 30,
            "gender": "F",
            "chief_complaint": "Mild ankle strain while walking",
            "heart_rate": 72,
            "oxygen_saturation": 99,
            "respiratory_rate": 16,
            "temperature": 36.8,
            "gcs_score": 15
        })
        override_pt_id = res.json()["patientId"]
        res = client.post(f"/api/override/{override_pt_id}", json={
            "nurse_esi": 4,
            "override_reason": "clinical"
        })
        assert res.status_code == 200
        override_data = res.json()
        assert override_data["status"] == "overridden"
        assert override_data["nurse_esi"] == 4
        print(f"[PASS] Overridden {override_pt_id} to ESI 4")

        print("\n6. Testing POST /api/surge (Inject 15 patients)...")
        res = client.post("/api/surge")
        assert res.status_code == 200
        surge_data = res.json()
        assert surge_data["injected_count"] == 15
        print(f"[PASS] Surge simulated: injected {surge_data['injected_count']} patients, total queue: {surge_data['total_queue_size']}")

        print("\n7. Testing GET /api/audit-log...")
        res = client.get("/api/audit-log")
        assert res.status_code == 200
        audit_data = res.json()
        assert len(audit_data["logs"]) >= 2
        assert audit_data["stats"]["total_decisions"] >= 2
        print(f"[PASS] Audit log verified: {len(audit_data['logs'])} records found, total decisions: {audit_data['stats']['total_decisions']}")

        print("\n8. Testing GET /api/stats...")
        res = client.get("/api/stats")
        assert res.status_code == 200
        stats = res.json()
        assert "total_patients" in stats
        assert "priority_counts" in stats
        print(f"[PASS] Stats verified: {stats['total_patients']} total patients, breakdown: {stats['priority_counts']}")

        print("\n9. Testing GET /api/trace/{patient_id}...")
        res = client.get(f"/api/trace/{new_pt_id}")
        assert res.status_code == 200
        trace_data = res.json()
        assert "trace" in trace_data
        print(f"[PASS] LangGraph trace verified for {new_pt_id}: {len(trace_data['trace'])} execution steps")

        print("\n==========================================")
        print("   ALL 9 ENDPOINT & WORKFLOW TESTS PASSED!")
        print("==========================================")

if __name__ == "__main__":
    test_full_pipeline()
