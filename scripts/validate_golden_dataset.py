"""Validate the canonical synthetic Golden Dataset (CSV and JSON)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data.golden_dataset import (
    CONFIDENCE_BANDS,
    DATASET_CSV_PATH,
    DATASET_JSON_PATH,
    HISTORY_VALUES,
    REQUIRED_COLUMNS,
    load_golden_dataset,
)

REQUIRED_SCENARIOS = {
    "PEDIATRIC_AGE_ADJUSTMENT",
    "GERIATRIC_AMBIGUOUS_ZERO_HISTORY",
    "VITALS_OVERRIDE_NLP",
    "ZERO_HISTORY",
    "RICH_HISTORY",
    "PARTIAL_HISTORY",
    "AMBIGUOUS_PRESENTATION",
    "MISSING_VITAL",
    "CONFLICTING_SIGNALS",
    "HIGH_CONFIDENCE",
    "LOW_CONFIDENCE",
    "WAIT_THRESHOLD",
    "VITAL_DETERIORATION",
    "CRITICAL_SAFETY_RULE",
    "PEDIATRIC_MISSING_DATA",
    "GERIATRIC_RICH_HISTORY",
    "STABLE_LOW_RISK",
    "CLINICIAN_OVERRIDE",
    "CLINICIAN_ACCEPT",
    "MULTI_FACTOR_RISK",
}

                                                                                       
PHI_PATTERN = re.compile(
    r"(\b\d{3}-\d{2}-\d{4}\b|\b\d{3}-\d{3}-\d{4}\b|\bMRN\d{6,}\b|\bSSN\b)",
    re.IGNORECASE,
)

def validate_records(records: List[Dict[str, Any]], source_label: str = "dataset") -> Dict[str, Any]:
    if len(records) < 20:
        raise ValueError(f"[{source_label}] Expected at least 20 records, found {len(records)}")

    ids = [record["patient_id"] for record in records]
    if len(ids) != len(set(ids)):
        duplicates = [pid for pid in ids if ids.count(pid) > 1]
        raise ValueError(f"[{source_label}] Duplicate patient IDs found: {set(duplicates)}")

    for record in records:
        pid = record["patient_id"]
                                
        if not pid.startswith("DEMO-"):
            raise ValueError(f"[{source_label}] Non-synthetic patient ID: '{pid}' (must start with DEMO-)")

                    
        age = record["age"]
        if not isinstance(age, int) or not (0 <= age <= 120):
            raise ValueError(f"[{source_label}] Invalid age for {pid}: {age}")

                            
        expected_group = "pediatric" if age < 18 else "geriatric" if age > 65 else "adult"
        if record.get("age_group") and record["age_group"] != expected_group:
            raise ValueError(f"[{source_label}] Age group mismatch for {pid}: got {record['age_group']}, expected {expected_group}")

                                     
        if record["history_availability"] not in HISTORY_VALUES:
            raise ValueError(f"[{source_label}] Invalid history availability for {pid}: {record['history_availability']}")

                         
        if record["expected_confidence_band"] not in CONFIDENCE_BANDS:
            raise ValueError(f"[{source_label}] Invalid expected confidence band for {pid}: {record['expected_confidence_band']}")

                      
        if record["expected_triage_level"] not in range(1, 6):
            raise ValueError(f"[{source_label}] Invalid expected triage level for {pid}: {record['expected_triage_level']}")

                                                                        
        complaint = record.get("chief_complaint", "")
        if PHI_PATTERN.search(complaint):
            raise ValueError(f"[{source_label}] Potential real identifier pattern detected in chief complaint for {pid}")

    tags = {record["scenario_tag"] for record in records}
    missing_tags = REQUIRED_SCENARIOS - tags
    if missing_tags:
        raise ValueError(f"[{source_label}] Missing required scenario tags: {sorted(missing_tags)}")

                                     
    pediatric = [r for r in records if r["scenario_tag"] == "PEDIATRIC_AGE_ADJUSTMENT" and r["age"] == 8]
    if not pediatric:
        raise ValueError(f"[{source_label}] Required 8yo pediatric anchor case (PEDIATRIC_AGE_ADJUSTMENT) missing")

    geriatric = [r for r in records if r["scenario_tag"] == "GERIATRIC_AMBIGUOUS_ZERO_HISTORY" and r["age"] == 75]
    if not geriatric:
        raise ValueError(f"[{source_label}] Required 75yo geriatric anchor case (GERIATRIC_AMBIGUOUS_ZERO_HISTORY) missing")

    zero_hist = [r for r in records if r["scenario_tag"] == "ZERO_HISTORY" and r["history_availability"] == "none"]
    if not zero_hist:
        raise ValueError(f"[{source_label}] Required zero-history case missing")

    missing_vital = [r for r in records if r["scenario_tag"] == "MISSING_VITAL" and r["spo2"] is None]
    if not missing_vital:
        raise ValueError(f"[{source_label}] Required missing vital case (MISSING_VITAL with spo2=None) missing")

    override_demo = [r for r in records if r["scenario_tag"] == "CLINICIAN_OVERRIDE"]
    if not override_demo:
        raise ValueError(f"[{source_label}] Required clinician override demo scenario missing")

    accept_demo = [r for r in records if r["scenario_tag"] == "CLINICIAN_ACCEPT"]
    if not accept_demo:
        raise ValueError(f"[{source_label}] Required clinician accept demo scenario missing")

    multi_risk = [r for r in records if r["scenario_tag"] == "MULTI_FACTOR_RISK"]
    if not multi_risk:
        raise ValueError(f"[{source_label}] Required multi-factor risk scenario missing")

    deterioration = [r for r in records if r["scenario_tag"] == "VITAL_DETERIORATION"]
    if not deterioration:
        raise ValueError(f"[{source_label}] Required vital deterioration scenario missing")

    return {
        "records": len(records),
        "unique_ids": len(set(ids)),
        "scenarios": len(tags),
        "source": source_label,
    }

def validate(path: str | Path | None = None) -> Dict[str, Any]:
    """Validate canonical CSV (and JSON if available)."""
    if path:
        records = load_golden_dataset(path)
        return validate_records(records, str(path))

    csv_records = load_golden_dataset(DATASET_CSV_PATH)
    csv_summary = validate_records(csv_records, "golden_patients.csv")

    if DATASET_JSON_PATH.exists():
        json_records = load_golden_dataset(DATASET_JSON_PATH)
        json_summary = validate_records(json_records, "golden_patients.json")
        if json_summary["records"] != csv_summary["records"]:
            raise ValueError(f"CSV and JSON record count mismatch: {csv_summary['records']} vs {json_summary['records']}")

    return csv_summary

if __name__ == "__main__":
    try:
        summary = validate()
        print(f"Golden Dataset valid: {summary['records']} synthetic records, {summary['scenarios']} scenarios checked ({summary['source']})")
    except ValueError as error:
        print(f"Golden Dataset validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
