import datetime as dt
import uuid

from backend.database import db_clear_all_patients, db_save_patient, init_db
from backend.queue_monitor import calculate_queue_priority, detect_vital_deterioration, get_queue_projection, monitor_once

def _patient(age=40, priority=3, waiting_minutes=0):
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "patientId": f"PHASE2-{uuid.uuid4().hex[:8]}",
        "name": "Synthetic Queue Patient",
        "age": age,
        "biologicalSex": "U",
        "chiefComplaint": "Monitoring complaint",
        "vitals": {"heartRateBpm": 80, "o2SaturationPercent": 98, "temperatureCelsius": 37},
        "history_available": True,
        "aiSuggestedPriority": priority,
        "aiConfidenceScore": 0.85,
        "safetyFlags": [],
        "missingFields": [],
        "status": "AWAITING_REVIEW",
        "arrivalTime": (now - dt.timedelta(minutes=waiting_minutes)).isoformat(),
        "waitingStartedAt": (now - dt.timedelta(minutes=waiting_minutes)).isoformat(),
        "lastQueuePriority": priority,
    }

def test_wait_threshold_generates_one_reassessment_event_and_reorders_queue():
    init_db()
    db_clear_all_patients()
    patient = _patient(priority=3, waiting_minutes=31)
    db_save_patient(patient)

    first = monitor_once()
    second = monitor_once()
    projection = get_queue_projection()

    assert len(first) == 1
    assert second == []
    assert projection[0]["reassessmentRequired"] is True
    assert projection[0]["waitThresholdExceeded"] is True
    assert projection[0]["reassessmentStatus"] == "REQUIRED"

    db_clear_all_patients()

def test_priority_keeps_safety_ahead_of_waiting_time():
    critical = calculate_queue_priority(2, safety_flags=["critical_vital_signal"], waiting_minutes=1)
    old_but_less_safe = calculate_queue_priority(4, waiting_minutes=500)
    assert critical < old_but_less_safe

def test_configured_threshold_is_used(monkeypatch):
    monkeypatch.setenv("WAIT_THRESHOLDS_JSON", '{"3": 90}')
    init_db()
    db_clear_all_patients()
    patient = _patient(priority=3, waiting_minutes=31)
    db_save_patient(patient)

    events = monitor_once()
    assert events == []
    projection = get_queue_projection()
    assert projection[0]["waitThresholdMinutes"] == 90
    assert projection[0]["reassessmentRequired"] is False

    db_clear_all_patients()

def test_stable_vitals_do_not_trigger_deterioration():
    vitals = {"heartRateBpm": 80, "o2SaturationPercent": 98, "gcsScore": 15}
    assert detect_vital_deterioration(vitals, vitals) == {}

def test_worsening_vitals_trigger_reassessment_signal():
    previous = {"heartRateBpm": 80, "o2SaturationPercent": 98, "gcsScore": 15}
    current = {"heartRateBpm": 112, "o2SaturationPercent": 93, "gcsScore": 13}
    changes = detect_vital_deterioration(previous, current)
    assert set(changes) == {"heart_rate", "oxygen_saturation", "gcs_score"}
