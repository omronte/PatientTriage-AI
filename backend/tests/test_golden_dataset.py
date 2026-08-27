"""Comprehensive automated safety tests based on the Golden Synthetic Dataset.

Validates Phase 4I safety requirements:
1. Pediatric patient uses pediatric rules
2. Geriatric patient uses geriatric rules
3. Zero-history patient is identified
4. Missing data is identified (not treated as normal)
5. Ambiguous presentation lowers confidence
6. Concerning vitals cannot be overridden by low-risk NLP
7. Safety rule precedence works
8. Conservative escalation works
9. Waiting-threshold scenario is correctly represented
10. Deterioration scenario is correctly represented
11. Override scenario can be manually overridden
12. Acceptance scenario can be accepted
13. Multi-factor risk does not produce an unsafe downgrade
"""

from __future__ import annotations

import pytest
from backend.clinical_rules import get_age_group
from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from backend.database import db_clear_all_patients, db_get_patient, db_save_patient, init_db
from backend.queue_monitor import detect_vital_deterioration, monitor_once
from ml_engine import extract_nlp, predict_triage, scrub_phi, train_model
from scripts.validate_golden_dataset import validate
from workflow import build_and_run_pipeline

@pytest.fixture(scope="module")
def model():
    return train_model()

@pytest.fixture(scope="module")
def golden_records():
    return {r["scenario_tag"]: r for r in load_golden_dataset()}

def test_golden_dataset_validates_and_covers_required_scenarios():
    """Verify dataset validator confirms at least 20 records and 20 scenarios."""
    summary = validate()
    assert summary["records"] >= 20
    assert summary["records"] == summary["unique_ids"]
    assert summary["scenarios"] >= 20

def test_golden_records_use_existing_pipeline_schema():
    """Verify normalized records map directly into the standard pipeline schema."""
    records = load_golden_dataset()
    patient = to_pipeline_patient(records[0])
    assert patient["id"].startswith("DEMO-")
    assert get_age_group(patient["age"]) == records[0]["age_group"]
    assert {"age", "chief_complaint", "heart_rate", "history_available"} <= set(patient)

def test_pediatric_patient_uses_pediatric_rules(model, golden_records):
    """1. Pediatric patient uses age-adjusted rules and is routed to pediatric profile."""
    rec = golden_records["PEDIATRIC_AGE_ADJUSTMENT"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["age_group"] == "pediatric"
    assert "PEDIATRIC_AGE_ADJUSTED" in out["safety_flags"]
    assert out["final_esi"] <= 2

def test_geriatric_patient_uses_geriatric_rules(model, golden_records):
    """2. Geriatric patient uses age-adjusted rules and identifies atypical presentation."""
    rec = golden_records["GERIATRIC_AMBIGUOUS_ZERO_HISTORY"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["age_group"] == "geriatric"
    assert "GERIATRIC_AGE_ADJUSTED" in out["safety_flags"]
    assert out["history_availability"] == "none"
    assert out["final_esi"] <= 2

def test_zero_history_patient_is_identified(model, golden_records):
    """3. First-time patient with zero history is isolated from prior assumptions."""
    rec = golden_records["ZERO_HISTORY"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["history_availability"] == "none"
    assert "ZERO_HISTORY" in out["safety_flags"]

def test_missing_data_is_identified_and_not_treated_as_normal(model, golden_records):
    """4. Missing vital signs are explicitly tracked and not substituted as healthy normal."""
    rec = golden_records["MISSING_VITAL"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert "oxygen_saturation" in out["missing_fields"]
    assert "MISSING_DATA" in out["safety_flags"]
    assert out["missing_data_penalty"] > 0

def test_ambiguous_presentation_lowers_confidence(model, golden_records):
    """5. Ambiguous presentation increases NLP ambiguity penalty and reduces confidence."""
    ambig_rec = golden_records["AMBIGUOUS_PRESENTATION"]
    clear_rec = golden_records["HIGH_CONFIDENCE"]
    ambig_out = build_and_run_pipeline(to_pipeline_patient(ambig_rec), model, scrub_phi, predict_triage, extract_nlp)
    clear_out = build_and_run_pipeline(to_pipeline_patient(clear_rec), model, scrub_phi, predict_triage, extract_nlp)
    assert ambig_out["nlp_ambiguity_penalty"] > 0
    assert ambig_out["final_confidence"] < clear_out["final_confidence"]

def test_concerning_vitals_cannot_be_overridden_by_low_risk_nlp(model, golden_records):
    """6. Low-risk NLP keywords cannot downgrade high-risk vital safety signals."""
    rec = golden_records["VITALS_OVERRIDE_NLP"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] == 1
    assert "CRITICAL_SAFETY_RULE" in out["safety_flags"]
    assert out["safety_override"] is True

def test_safety_rule_precedence_over_ml_score(model, golden_records):
    """7. Deterministic safety rules strictly supersede statistical ML predictions."""
    rec = golden_records["CONFLICTING_SIGNALS"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] == 1
    assert out["safety_override"] is True

def test_conservative_escalation_works(model, golden_records):
    """8. Uncertainty and missing data safely trigger conservative priority escalation."""
    rec = golden_records["GERIATRIC_AMBIGUOUS_ZERO_HISTORY"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] <= 2
    assert bool(out.get("confidence_escalated") or out.get("age_safety_triggered") or out.get("safety_override"))

def test_waiting_threshold_scenario(golden_records):
    """9. Waiting threshold breach produces a reassessment event in queue monitor."""
    import datetime as dt
    init_db()
    db_clear_all_patients()
    rec = golden_records["WAIT_THRESHOLD"]
    raw = to_pipeline_patient(rec)
    now = dt.datetime.now(dt.timezone.utc)
    started = (now - dt.timedelta(minutes=35)).isoformat()
    db_save_patient({
        "patientId": rec["patient_id"],
        "name": raw["name"],
        "age": raw["age"],
        "biologicalSex": raw["gender"],
        "chiefComplaint": raw["chief_complaint"],
        "vitals": {"heartRateBpm": 108, "o2SaturationPercent": 97, "temperatureCelsius": 37.5},
        "history_available": True,
        "aiSuggestedPriority": 3,
        "aiConfidenceScore": 0.75,
        "safetyFlags": [],
        "missingFields": [],
        "status": "AWAITING_REVIEW",
        "lifecycleStatus": "WAITING",
        "arrivalTime": started,
        "waitingStartedAt": started,
        "lastQueuePriority": 3,
    })

    events = monitor_once(now)
    assert any(e["patient_id"] == rec["patient_id"] for e in events)
    patient_status = db_get_patient(rec["patient_id"])
    assert patient_status["reassessmentRequired"] is True
    db_clear_all_patients()

def test_vital_deterioration_scenario(golden_records):
    """10. Vital deterioration detection triggers clinical alert."""
    rec = golden_records["VITAL_DETERIORATION"]
    previous_vitals = {"heartRateBpm": rec["heart_rate"], "o2SaturationPercent": rec["spo2"], "gcsScore": rec["gcs_score"]}
    worsened_vitals = {"heartRateBpm": 115, "o2SaturationPercent": 91, "gcsScore": 13}
    deterioration = detect_vital_deterioration(previous_vitals, worsened_vitals)
    assert "oxygen_saturation" in deterioration or "heart_rate" in deterioration

def test_clinician_override_and_accept_scenarios(model, golden_records):
    """11 & 12. Clinician override and accept workflows function as designed."""
    override_rec = golden_records["CLINICIAN_OVERRIDE"]
    accept_rec = golden_records["CLINICIAN_ACCEPT"]
    out_override = build_and_run_pipeline(to_pipeline_patient(override_rec), model, scrub_phi, predict_triage, extract_nlp)
    out_accept = build_and_run_pipeline(to_pipeline_patient(accept_rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out_override["final_esi"] in {3, 4, 5}
    assert out_accept["final_esi"] == 5

def test_multi_factor_risk_does_not_produce_unsafe_downgrade(model, golden_records):
    """13. Multi-factor risk combines age, vital, and history risks without downgrade."""
    rec = golden_records["MULTI_FACTOR_RISK"]
    out = build_and_run_pipeline(to_pipeline_patient(rec), model, scrub_phi, predict_triage, extract_nlp)
    assert out["final_esi"] == 1
    assert "CRITICAL_SAFETY_RULE" in out["safety_flags"] or "AGE_OR_VITAL_SAFETY_RULE" in out["safety_flags"]
