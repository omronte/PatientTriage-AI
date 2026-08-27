"""Idempotently seed synthetic Golden Dataset records into the database for live demonstration."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from backend.database import PatientRecord, SessionLocal, db_save_patient, init_db
from ml_engine import extract_nlp, predict_triage, scrub_phi, train_model
from workflow import build_and_run_pipeline

def seed(reset: bool = False, dataset_path: Optional[str] = None) -> int:
    """Seed synthetic golden dataset patients into SQLite database.

    Args:
        reset: If True, selectively removes existing DEMO-* patient records before seeding.
               Does NOT delete unrelated clinical or surge records.
        dataset_path: Optional path to custom CSV/JSON golden dataset.

    Returns:
        Number of seeded synthetic patient records.
    """
    init_db()
    records = load_golden_dataset(dataset_path) if dataset_path else load_golden_dataset()
    ids = [record["patient_id"] for record in records]

    session = SessionLocal()
    try:
        if reset:
            deleted = session.query(PatientRecord).filter(PatientRecord.patient_id.in_(ids)).delete(synchronize_session=False)
            session.commit()
            print(f"[RESET] Cleared {deleted} previous synthetic Golden Dataset records from database.")
    finally:
        session.close()

    model = train_model()
    now = dt.datetime.now(dt.timezone.utc)

    for idx, record in enumerate(records):
        raw = to_pipeline_patient(record)
        result = build_and_run_pipeline(raw, model, scrub_phi, predict_triage, extract_nlp)

                                                                                
        wait_offset_minutes = 5 * (idx % 12) + (35 if record["scenario_tag"] == "WAIT_THRESHOLD" else 5)
        arrival_iso = (now - dt.timedelta(minutes=wait_offset_minutes)).isoformat()

        confidence = float(result.get("final_confidence", 0.5))
        confidence_label = "HIGH" if confidence >= 0.80 else "MEDIUM" if confidence >= 0.55 else "LOW"

        bp_sys = None
        bp_dia = None
        if raw.get("blood_pressure") and "/" in str(raw["blood_pressure"]):
            try:
                parts = raw["blood_pressure"].split("/")
                bp_sys = int(parts[0])
                bp_dia = int(parts[1])
            except (ValueError, IndexError):
                pass

        safety_flags = list(result.get("safety_flags", []))
        if record["scenario_tag"] not in safety_flags:
            safety_flags.append(f"SCENARIO:{record['scenario_tag']}")

        assessed = {
            "patientId": record["patient_id"],
            "name": f"Synthetic Patient {record['patient_id']}",
            "age": raw["age"],
            "ageGroup": record.get("age_group", "adult"),
            "biologicalSex": raw["gender"],
            "chiefComplaint": raw["chief_complaint"],
            "arrivalTime": arrival_iso,
            "waitingMinutes": wait_offset_minutes,
            "status": "AWAITING_REVIEW",
            "lifecycleStatus": "WAITING",
            "history_available": raw["history_available"],
            "historyAvailability": record.get("history_availability", "none"),
            "riskCategory": record.get("expected_risk_category", "moderate"),
            "aiSuggestedPriority": result.get("final_esi", 3),
            "aiConfidenceScore": confidence,
            "confidenceLabel": confidence_label,
            "safetyFlags": safety_flags,
            "missingFields": result.get("missing_fields", []),
            "explanation": result.get("explainability", result.get("explanation", [])),
            "explainability": result.get("explainability", result.get("explanation", [])),
            "_trace": result.get("trace", []),
            "_mlPrediction": result.get("ml_prediction", {}),
            "_nlpAnalysis": result.get("nlp_analysis", {}),
            "_ageSafetyTriggered": result.get("age_safety_triggered", False),
            "_ageSafetyReason": result.get("age_safety_reason"),
            "_confidenceEscalated": result.get("confidence_escalated", False),
            "_missingDataPenalty": result.get("missing_data_penalty", 0),
            "_nlpAmbiguityPenalty": result.get("nlp_ambiguity_penalty", 0),
            "vitals": {
                "heartRateBpm": raw["heart_rate"],
                "bloodPressureSys": bp_sys,
                "bloodPressureDia": bp_dia,
                "o2SaturationPercent": raw["oxygen_saturation"],
                "respiratoryRate": raw["respiratory_rate"],
                "temperatureCelsius": raw["temperature"],
                "gcsScore": raw["gcs_score"] if raw["gcs_score"] is not None else 15,
            },
        }
        db_save_patient(assessed)

    return len(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed synthetic Golden Dataset into SQLite database.")
    parser.add_argument("--reset-golden", action="store_true", help="Selectively reset only Golden Dataset records.")
    parser.add_argument("--path", type=str, default=None, help="Custom dataset CSV/JSON path.")
    args = parser.parse_args()

    count = seed(reset=args.reset_golden, dataset_path=args.path)
    print(f"[OK] Successfully seeded {count} synthetic Golden Dataset records (idempotent demo mode).")
