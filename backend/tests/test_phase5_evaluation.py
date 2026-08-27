"""
Phase 5 Evaluation Pipeline Tests

Validates that the evaluate_golden_dataset.py output:
  - Contains all required metric fields
  - Produces a valid 5×5 confusion matrix
  - Covers all 20 required scenario tags
  - Is deterministic across two consecutive runs
  - Reports safety-critical fields
  - Has no data-leakage violations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_golden_dataset import evaluate

                                                                        
@pytest.fixture(scope="module")
def evaluation_payload():
    return evaluate()

                                                                                

def test_evaluation_has_required_top_level_keys(evaluation_payload):
    required = {"metrics", "safety_critical", "confusion_matrix", "scenario_breakdown", "results"}
    assert required <= set(evaluation_payload.keys()), (
        f"Missing keys: {required - set(evaluation_payload.keys())}"
    )

                                                                               

def test_metrics_contain_all_required_fields(evaluation_payload):
    metrics = evaluation_payload["metrics"]
    required_fields = {
        "dataset_size",
        "triage_agreement_count",
        "triage_agreement_rate",
        "under_triage_count",
        "under_triage_rate",
        "over_triage_count",
        "over_triage_rate",
        "safety_rule_trigger_rate",
        "low_confidence_case_rate",
        "conservative_escalation_rate",
        "missing_data_detection_rate",
        "pediatric_routing_correct",
        "geriatric_routing_correct",
        "zero_history_handling_correct",
        "confidence_distribution",
        "scenario_coverage_count",
        "data_leakage_violations",
        "data_leakage_clean",
    }
    missing = required_fields - set(metrics.keys())
    assert missing == set(), f"Missing metric fields: {missing}"

                                                                               

def test_safety_critical_fields_present(evaluation_payload):
    sc = evaluation_payload["safety_critical"]
    required = {
        "critical_under_triage_count",
        "critical_rule_miss_count",
        "pediatric_safety_miss_count",
        "geriatric_safety_miss_count",
        "missing_data_safety_miss_count",
        "zero_history_safety_miss_count",
        "vital_deterioration_miss_count",
        "low_confidence_escalation_miss_count",
        "critical_miss_patient_ids",
        "critical_rule_miss_patient_ids",
    }
    missing = required - set(sc.keys())
    assert missing == set(), f"Missing safety-critical fields: {missing}"

                                                                               

def test_confusion_matrix_is_5x5(evaluation_payload):
    matrix = evaluation_payload["confusion_matrix"]
    assert len(matrix) == 5, f"Confusion matrix must have 5 rows, got {len(matrix)}"
    for row_key, row in matrix.items():
        assert len(row) == 5, (
            f"Confusion matrix row {row_key} must have 5 columns, got {len(row)}"
        )

def test_confusion_matrix_total_equals_dataset_size(evaluation_payload):
    matrix = evaluation_payload["confusion_matrix"]
    total = sum(v for row in matrix.values() for v in row.values())
    assert total == evaluation_payload["metrics"]["dataset_size"], (
        f"Confusion matrix total ({total}) must equal dataset size "
        f"({evaluation_payload['metrics']['dataset_size']})"
    )

                                                                               

def test_scenario_coverage_is_at_least_20(evaluation_payload):
    count = evaluation_payload["metrics"]["scenario_coverage_count"]
    assert count >= 20, f"Must cover >= 20 scenario tags, got {count}"

def test_scenario_breakdown_covers_all_results(evaluation_payload):
    result_tags = {r["scenario_tag"] for r in evaluation_payload["results"]}
    breakdown_tags = set(evaluation_payload["scenario_breakdown"].keys())
    assert result_tags == breakdown_tags, (
        f"Scenario breakdown must match result tags. "
        f"Missing from breakdown: {result_tags - breakdown_tags}"
    )

                                                                               

def test_per_patient_table_has_required_columns(evaluation_payload):
    required_cols = {
        "patient_id",
        "scenario_tag",
        "expected_triage_level",
        "predicted_triage_level",
        "expected_confidence_band",
        "predicted_confidence_band",
        "confidence_score",
        "expected_escalation",
        "predicted_escalation",
        "safety_flags",
        "primary_reason",
        "age_group",
        "history_availability",
    }
    for row in evaluation_payload["results"]:
        missing = required_cols - set(row.keys())
        assert missing == set(), (
            f"Patient {row['patient_id']} result row missing columns: {missing}"
        )

                                                                               

def test_dataset_size_is_at_least_20(evaluation_payload):
    assert evaluation_payload["metrics"]["dataset_size"] >= 20

                                                                               

def test_metric_rates_are_valid(evaluation_payload):
    metrics = evaluation_payload["metrics"]
    for key in [
        "triage_agreement_rate",
        "under_triage_rate",
        "over_triage_rate",
        "safety_rule_trigger_rate",
        "low_confidence_case_rate",
        "conservative_escalation_rate",
        "missing_data_detection_rate",
    ]:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"Rate {key}={val} out of [0, 1]"

                                                                                

def test_triage_rates_sum_to_dataset_size(evaluation_payload):
    m = evaluation_payload["metrics"]
    total = m["triage_agreement_count"] + m["under_triage_count"] + m["over_triage_count"]
    assert total == m["dataset_size"], (
        f"Agreement + under + over ({total}) must equal dataset_size ({m['dataset_size']})"
    )

                                                                               

def test_no_data_leakage(evaluation_payload):
    violations = evaluation_payload["metrics"]["data_leakage_violations"]
    assert violations == [], (
        f"DATA LEAKAGE DETECTED in {len(violations)} patient(s):\n"
        + "\n".join(violations)
    )

                                                                                

def test_evaluation_is_deterministic():
    """Run evaluate() twice; metrics must be bit-for-bit identical."""
    run1 = evaluate()["metrics"]
    run2 = evaluate()["metrics"]

                                                         
    assert json.dumps(run1, sort_keys=True) == json.dumps(run2, sort_keys=True), (
        "Evaluation is non-deterministic: metrics changed between consecutive runs."
    )

                                                                              

def test_result_files_are_written(evaluation_payload):
    result_dir = ROOT / "evaluation" / "results"
    assert (result_dir / "golden_dataset_results.json").exists(), (
        "golden_dataset_results.json must be written by evaluate()"
    )
    assert (result_dir / "golden_dataset_report.md").exists(), (
        "golden_dataset_report.md must be written by evaluate()"
    )
    assert (result_dir / "golden_dataset_report.json").exists(), (
        "golden_dataset_report.json must be written by evaluate()"
    )
