"""Canonical loader for the synthetic Golden Dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "golden" / "golden_patients.csv"
DATASET_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "golden" / "golden_patients.json"

REQUIRED_COLUMNS = {
    "patient_id", "age", "age_group", "sex", "history_availability", "chief_complaint",
    "symptoms", "heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature",
    "respiratory_rate", "gcs", "pain_score", "prior_conditions", "current_medications",
    "missing_fields", "expected_triage_level", "expected_risk_category",
    "expected_confidence_band", "expected_escalation", "expected_primary_reason", "scenario_tag",
}
CONFIDENCE_BANDS = {"LOW", "MEDIUM", "HIGH"}
HISTORY_VALUES = {"none", "partial", "rich"}

def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "n/a", "na", "null"}:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def load_golden_dataset(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load and normalize the synthetic golden dataset from CSV or JSON."""
    target = Path(path) if path else DATASET_CSV_PATH
    if not target.exists() and target == DATASET_CSV_PATH and DATASET_JSON_PATH.exists():
        target = DATASET_JSON_PATH

    if target.suffix.lower() == ".json":
        with target.open(encoding="utf-8") as handle:
            raw_data = json.load(handle)
        return [normalize_json_record(item) for item in raw_data]

    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED_COLUMNS - set(rows[0]) if rows else REQUIRED_COLUMNS
    if missing:
        raise ValueError(f"Golden Dataset missing columns: {sorted(missing)}")
    return [normalize_record(row) for row in rows]

def normalize_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a flat CSV row."""
    raw_missing = str(row.get("missing_fields", "") or "")
    missing_fields = [item.strip() for item in raw_missing.split("|") if item.strip()]
    history = str(row.get("history_availability", "none")).strip().lower()
    age = int(row["age"])
    spo2 = _optional_float(row.get("spo2"))

    systolic = row.get("systolic_bp")
    diastolic = row.get("diastolic_bp")
    bp = f"{systolic}/{diastolic}" if systolic and diastolic and str(systolic).strip() and str(diastolic).strip() else None

    return {
        **row,
        "patient_id": str(row["patient_id"]).strip(),
        "age": age,
        "age_group": str(row.get("age_group", "")).strip().lower(),
        "gender": str(row.get("sex", "U")).strip(),
        "sex": str(row.get("sex", "U")).strip(),
        "history_available": history != "none",
        "history_availability": history,
        "chief_complaint": str(row.get("chief_complaint", "")).strip(),
        "symptoms": str(row.get("symptoms", "")).strip(),
        "heart_rate": _optional_float(row.get("heart_rate")),
        "blood_pressure": bp,
        "systolic_bp": _optional_float(systolic),
        "diastolic_bp": _optional_float(diastolic),
        "spo2": spo2,
        "oxygen_saturation": spo2,
        "temperature": _optional_float(row.get("temperature")),
        "respiratory_rate": _optional_float(row.get("respiratory_rate")),
        "gcs_score": _optional_float(row.get("gcs")),
        "pain_level": _optional_float(row.get("pain_score")),
        "prior_conditions": str(row.get("prior_conditions", "")).strip(),
        "current_medications": str(row.get("current_medications", "")).strip(),
        "missing_fields": missing_fields,
        "expected_triage_level": int(row["expected_triage_level"]),
        "expected_risk_category": str(row.get("expected_risk_category", "moderate")).strip().lower(),
        "expected_confidence_band": str(row.get("expected_confidence_band", "MEDIUM")).strip().upper(),
        "expected_escalation": str(row.get("expected_escalation", "no")).strip().lower() in {"yes", "true", "1"},
        "expected_primary_reason": str(row.get("expected_primary_reason", "")).strip(),
        "scenario_tag": str(row["scenario_tag"]).strip(),
    }

def normalize_json_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a structured JSON object."""
    vitals = item.get("vitals", {})
    history = str(item.get("history_availability", "none")).strip().lower()
    spo2 = _optional_float(vitals.get("spo2", vitals.get("oxygen_saturation")))
    sys_bp = _optional_float(vitals.get("systolic_bp"))
    dia_bp = _optional_float(vitals.get("diastolic_bp"))
    bp = f"{int(sys_bp)}/{int(dia_bp)}" if sys_bp is not None and dia_bp is not None else None

    symptoms_val = item.get("symptoms", [])
    if isinstance(symptoms_val, list):
        symptoms_str = "|".join(str(s) for s in symptoms_val)
    else:
        symptoms_str = str(symptoms_val)

    missing_val = item.get("missing_fields", [])
    if isinstance(missing_val, str):
        missing_fields = [m.strip() for m in missing_val.split("|") if m.strip()]
    else:
        missing_fields = list(missing_val)

    return {
        "patient_id": str(item["patient_id"]).strip(),
        "age": int(item["age"]),
        "age_group": str(item.get("age_group", "")).strip().lower(),
        "gender": str(item.get("sex", item.get("gender", "U"))).strip(),
        "sex": str(item.get("sex", item.get("gender", "U"))).strip(),
        "history_available": history != "none",
        "history_availability": history,
        "chief_complaint": str(item.get("chief_complaint", "")).strip(),
        "symptoms": symptoms_str,
        "heart_rate": _optional_float(vitals.get("heart_rate")),
        "blood_pressure": bp,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "spo2": spo2,
        "oxygen_saturation": spo2,
        "temperature": _optional_float(vitals.get("temperature")),
        "respiratory_rate": _optional_float(vitals.get("respiratory_rate")),
        "gcs_score": _optional_float(vitals.get("gcs", vitals.get("gcs_score"))),
        "pain_level": _optional_float(vitals.get("pain_score", vitals.get("pain_level"))),
        "prior_conditions": str(item.get("prior_conditions", "")).strip(),
        "current_medications": str(item.get("current_medications", "")).strip(),
        "missing_fields": missing_fields,
        "expected_triage_level": int(item["expected_triage_level"]),
        "expected_risk_category": str(item.get("expected_risk_category", "moderate")).strip().lower(),
        "expected_confidence_band": str(item.get("expected_confidence_band", "MEDIUM")).strip().upper(),
        "expected_escalation": bool(item.get("expected_escalation", False)),
        "expected_primary_reason": str(item.get("expected_primary_reason", "")).strip(),
        "scenario_tag": str(item["scenario_tag"]).strip(),
    }

def to_pipeline_patient(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a normalized row to the existing workflow/server patient schema."""
    return {
        "id": record["patient_id"],
        "name": f"Synthetic Demo {record['patient_id']}",
        "age": record["age"],
        "gender": record["gender"],
        "chief_complaint": record["chief_complaint"],
        "heart_rate": record["heart_rate"],
        "blood_pressure": record["blood_pressure"],
        "oxygen_saturation": record["oxygen_saturation"],
        "temperature": record["temperature"],
        "respiratory_rate": record["respiratory_rate"],
        "gcs_score": record["gcs_score"],
        "history_available": record["history_available"],
        "history_availability": record["history_availability"],
        "past_medical_history": record["prior_conditions"],
        "medications": record["current_medications"],
        "pain_level": record["pain_level"],
        "symptoms": record["symptoms"],
        "missing_fields": record.get("missing_fields", []),
    }
