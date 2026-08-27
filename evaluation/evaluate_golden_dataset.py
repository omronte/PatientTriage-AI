"""
Phase 5 — Enhanced Golden Dataset Evaluation Pipeline.

Runs the real PatientTriage.ai pipeline against every golden record and reports:
  - exact triage agreement / under-triage / over-triage rates
  - safety-critical failure metrics
  - triage-level confusion matrix (P1-P5)
  - per-scenario breakdown
  - per-patient detail table
  - data-leakage audit (expected labels never enter pipeline)
  - confidence-band distribution

Usage:
    python evaluation/evaluate_golden_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from ml_engine import extract_nlp, predict_triage, scrub_phi, train_model
from workflow import build_and_run_pipeline

RESULT_PATH = ROOT / "evaluation" / "results" / "golden_dataset_results.json"
REPORT_PATH = ROOT / "evaluation" / "results" / "golden_dataset_report.md"
REPORT_JSON_PATH = ROOT / "evaluation" / "results" / "golden_dataset_report.json"

                                                                              
_EVALUATION_ONLY_FIELDS = {
    "expected_triage_level",
    "expected_risk_category",
    "expected_confidence_band",
    "expected_escalation",
    "expected_primary_reason",
}

def _audit_leakage(pipeline_input: dict) -> list[str]:
    """Return any evaluation-only field keys found in the pipeline input dict."""
    return [k for k in _EVALUATION_ONLY_FIELDS if k in pipeline_input]

def _confidence_band(confidence: float) -> str:
    if confidence >= 0.80:
        return "HIGH"
    if confidence >= 0.55:
        return "MEDIUM"
    return "LOW"

def evaluate() -> Dict[str, Any]:
    """Execute real pipeline on all golden records and compute Phase 5 metrics."""
    records = load_golden_dataset()
    model = train_model()

    results: List[Dict[str, Any]] = []
    leakage_violations: List[str] = []

    for record in records:
        pipeline_input = to_pipeline_patient(record)

                                                                            
        violations = _audit_leakage(pipeline_input)
        if violations:
            leakage_violations.append(
                f"{record['patient_id']}: leaked fields {violations}"
            )

        output = build_and_run_pipeline(
            patient=pipeline_input,
            model=model,
            phi_scrubber=scrub_phi,
            predictor=predict_triage,
            nlp_extractor=extract_nlp,
        )

        confidence = float(output.get("final_confidence", 0.0))
        predicted_esi = int(output.get("final_esi", 3))
        expected_esi = record["expected_triage_level"]
        band = _confidence_band(confidence)

        is_escalated = bool(
            output.get("confidence_escalated")
            or output.get("age_safety_triggered")
            or output.get("safety_override")
            or output.get("history_escalated")
        )

        results.append({
            "patient_id": record["patient_id"],
            "scenario_tag": record["scenario_tag"],
            "expected_triage_level": expected_esi,
            "predicted_triage_level": predicted_esi,
            "expected_confidence_band": record["expected_confidence_band"],
            "predicted_confidence_band": band,
            "confidence_score": round(confidence, 4),
            "expected_escalation": record["expected_escalation"],
            "predicted_escalation": is_escalated,
            "safety_flags": output.get("safety_flags", []),
            "primary_reason": output.get("age_safety_reason") or "",
            "age_group": output.get("age_group", "unknown"),
            "history_availability": output.get("history_availability", "unknown"),
            "missing_fields": output.get("missing_fields", []),
            "missing_data_penalty": output.get("missing_data_penalty", 0.0),
            "nlp_ambiguity_penalty": output.get("nlp_ambiguity_penalty", 0.0),
        })

    total = len(results)

                                                                               
    under_count = sum(
        r["predicted_triage_level"] > r["expected_triage_level"] for r in results
    )
    over_count = sum(
        r["predicted_triage_level"] < r["expected_triage_level"] for r in results
    )
    agreement_count = sum(
        r["predicted_triage_level"] == r["expected_triage_level"] for r in results
    )
    safety_count = sum(bool(r["safety_flags"]) for r in results)
    low_conf_count = sum(r["predicted_confidence_band"] == "LOW" for r in results)
    escalated_count = sum(r["predicted_escalation"] for r in results)
    missing_count = sum(bool(r["missing_fields"]) for r in results)

                                                                               
    pediatric_correct = all(
        r["age_group"] == "pediatric"
        for r in results
        if r["scenario_tag"].startswith("PEDIATRIC")
    )
    geriatric_correct = all(
        r["age_group"] == "geriatric"
        for r in results
        if r["scenario_tag"].startswith("GERIATRIC")
    )
    zero_hist_correct = all(
        r["history_availability"] == "none"
        for r in results
        if r["scenario_tag"] == "ZERO_HISTORY"
    )

    conf_dist = dict(Counter(r["predicted_confidence_band"] for r in results))

                                                                               
    def _is_critical_expected(r: dict) -> bool:
        return r["expected_triage_level"] == 1

    def _is_safety_miss(r: dict) -> bool:
        """A safety miss: expected ESI 1 but predicted > 1."""
        return r["expected_triage_level"] == 1 and r["predicted_triage_level"] > 1

    def _is_critical_rule_miss(r: dict) -> bool:
        return (
            _is_safety_miss(r)
            and "CRITICAL_SAFETY_RULE" not in r["safety_flags"]
        )

    critical_misses = [r for r in results if _is_safety_miss(r)]
    critical_rule_misses = [r for r in results if _is_critical_rule_miss(r)]

    ped_misses = [
        r for r in results
        if r["scenario_tag"].startswith("PEDIATRIC") and r["predicted_triage_level"] > r["expected_triage_level"]
    ]
    ger_misses = [
        r for r in results
        if r["scenario_tag"].startswith("GERIATRIC") and r["predicted_triage_level"] > r["expected_triage_level"]
    ]
    missing_data_misses = [
        r for r in results
        if r["scenario_tag"] in {"MISSING_VITAL", "PEDIATRIC_MISSING_DATA"}
        and r["predicted_triage_level"] > r["expected_triage_level"]
    ]
    zero_hist_misses = [
        r for r in results
        if r["scenario_tag"] == "ZERO_HISTORY"
        and r["predicted_triage_level"] > r["expected_triage_level"]
    ]
    deterioration_misses = [
        r for r in results
        if r["scenario_tag"] == "VITAL_DETERIORATION"
        and r["predicted_triage_level"] > r["expected_triage_level"]
    ]
    low_conf_escalation_misses = [
        r for r in results
        if r["predicted_confidence_band"] == "LOW"
        and not r["predicted_escalation"]
    ]

    safety_critical = {
        "critical_under_triage_count": len(critical_misses),
        "critical_rule_miss_count": len(critical_rule_misses),
        "pediatric_safety_miss_count": len(ped_misses),
        "geriatric_safety_miss_count": len(ger_misses),
        "missing_data_safety_miss_count": len(missing_data_misses),
        "zero_history_safety_miss_count": len(zero_hist_misses),
        "vital_deterioration_miss_count": len(deterioration_misses),
        "low_confidence_escalation_miss_count": len(low_conf_escalation_misses),
        "critical_miss_patient_ids": [r["patient_id"] for r in critical_misses],
        "critical_rule_miss_patient_ids": [r["patient_id"] for r in critical_rule_misses],
    }

                                                                               
    confusion: Dict[int, Dict[int, int]] = {
        e: {p: 0 for p in range(1, 6)} for e in range(1, 6)
    }
    for r in results:
        exp = max(1, min(5, r["expected_triage_level"]))
        pred = max(1, min(5, r["predicted_triage_level"]))
        confusion[exp][pred] += 1

                                                                               
    scenario_groups: Dict[str, list] = defaultdict(list)
    for r in results:
        scenario_groups[r["scenario_tag"]].append(r)

    scenario_breakdown = {}
    for tag, group in scenario_groups.items():
        n = len(group)
        sc_under = sum(r["predicted_triage_level"] > r["expected_triage_level"] for r in group)
        sc_over = sum(r["predicted_triage_level"] < r["expected_triage_level"] for r in group)
        sc_agree = sum(r["predicted_triage_level"] == r["expected_triage_level"] for r in group)
        sc_safety = sum(bool(r["safety_flags"]) for r in group)
        scenario_breakdown[tag] = {
            "count": n,
            "agreement": sc_agree,
            "under_triage": sc_under,
            "over_triage": sc_over,
            "safety_triggered": sc_safety,
            "expected_esi": [r["expected_triage_level"] for r in group],
            "predicted_esi": [r["predicted_triage_level"] for r in group],
        }

                                                                               
    metrics = {
        "dataset_size": total,
        "triage_agreement_count": agreement_count,
        "triage_agreement_rate": round(agreement_count / total, 4),
        "under_triage_count": under_count,
        "under_triage_rate": round(under_count / total, 4),
        "over_triage_count": over_count,
        "over_triage_rate": round(over_count / total, 4),
        "safety_rule_trigger_rate": round(safety_count / total, 4),
        "low_confidence_case_rate": round(low_conf_count / total, 4),
        "conservative_escalation_rate": round(escalated_count / total, 4),
        "missing_data_detection_rate": round(missing_count / total, 4),
        "pediatric_routing_correct": pediatric_correct,
        "geriatric_routing_correct": geriatric_correct,
        "zero_history_handling_correct": zero_hist_correct,
        "confidence_distribution": conf_dist,
        "scenario_coverage_count": len({r["scenario_tag"] for r in results}),
        "data_leakage_violations": leakage_violations,
        "data_leakage_clean": len(leakage_violations) == 0,
    }

    payload = {
        "metrics": metrics,
        "safety_critical": safety_critical,
        "confusion_matrix": {str(k): v for k, v in confusion.items()},
        "scenario_breakdown": scenario_breakdown,
        "results": results,
        "synthetic_data_notice": (
            "All dataset records are synthetic. Expected outputs represent simulated "
            "prototype safety scenarios, not medically validated ground truth."
        ),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                                                                              
    report_md = render_report(metrics, safety_critical, confusion, scenario_breakdown, results)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

                                                                               
    report_json = {
        "metrics": metrics,
        "safety_critical": safety_critical,
        "confusion_matrix": {str(k): v for k, v in confusion.items()},
        "scenario_breakdown": scenario_breakdown,
    }
    REPORT_JSON_PATH.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    return payload

def render_report(
    metrics: Dict[str, Any],
    safety_critical: Dict[str, Any],
    confusion: Dict[int, Dict[int, int]],
    scenario_breakdown: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> str:
    """Generate human-readable evaluation report in markdown format."""
    under_rate_pct = f"{metrics['under_triage_rate']:.1%}"
    agree_rate_pct = f"{metrics['triage_agreement_rate']:.1%}"
    over_rate_pct = f"{metrics['over_triage_rate']:.1%}"
    leakage_status = "✅ CLEAN" if metrics["data_leakage_clean"] else "❌ VIOLATIONS FOUND"

    lines = [
        "# PatientTriage.ai — Golden Dataset Safety Evaluation Report (Phase 5)",
        "",
        "> [!IMPORTANT]",
        "> **SIMULATED SAFETY SCENARIO NOTICE**: All patient records in this dataset are entirely synthetic.",
        "> Expected outputs represent simulated prototype safety behaviors and do NOT constitute medically validated clinical ground truth.",
        "",
        "## Executive Summary",
        "",
        "| Metric | Count / Status | Rate |",
        "|---|---:|---:|",
        f"| Dataset Size | {metrics['dataset_size']} records | 100% |",
        f"| Scenario Coverage | {metrics['scenario_coverage_count']} / 20 tags | 100% |",
        f"| **Under-Triage Rate** | **{metrics['under_triage_count']} cases** | **{under_rate_pct}** |",
        f"| Over-Triage Rate | {metrics['over_triage_count']} cases | {over_rate_pct} |",
        f"| Exact Triage Agreement | {metrics['triage_agreement_count']} cases | {agree_rate_pct} |",
        f"| Safety Rule Trigger Rate | {metrics['safety_rule_trigger_rate'] * metrics['dataset_size']:.0f} cases | {metrics['safety_rule_trigger_rate']:.1%} |",
        f"| Low Confidence Rate | {metrics['low_confidence_case_rate'] * metrics['dataset_size']:.0f} cases | {metrics['low_confidence_case_rate']:.1%} |",
        f"| Conservative Escalation Rate | {metrics['conservative_escalation_rate'] * metrics['dataset_size']:.0f} cases | {metrics['conservative_escalation_rate']:.1%} |",
        f"| Missing Data Detection Rate | {metrics['missing_data_detection_rate'] * metrics['dataset_size']:.0f} cases | {metrics['missing_data_detection_rate']:.1%} |",
        f"| Pediatric Routing | {'✅ PASS' if metrics['pediatric_routing_correct'] else '❌ FAIL'} | — |",
        f"| Geriatric Routing | {'✅ PASS' if metrics['geriatric_routing_correct'] else '❌ FAIL'} | — |",
        f"| Zero-History Routing | {'✅ PASS' if metrics['zero_history_handling_correct'] else '❌ FAIL'} | — |",
        f"| Data Leakage | {leakage_status} | — |",
        "",
        "## Safety-Critical Failures",
        "",
        "| Safety Check | Count |",
        "|---|---:|",
        f"| Critical under-triage (ESI 1 expected, higher predicted) | {safety_critical['critical_under_triage_count']} |",
        f"| Critical safety rule misses | {safety_critical['critical_rule_miss_count']} |",
        f"| Pediatric safety misses | {safety_critical['pediatric_safety_miss_count']} |",
        f"| Geriatric safety misses | {safety_critical['geriatric_safety_miss_count']} |",
        f"| Missing-data safety misses | {safety_critical['missing_data_safety_miss_count']} |",
        f"| Zero-history safety misses | {safety_critical['zero_history_safety_miss_count']} |",
        f"| Vital deterioration misses | {safety_critical['vital_deterioration_miss_count']} |",
        f"| Low-confidence escalation misses | {safety_critical['low_confidence_escalation_miss_count']} |",
        "",
    ]

    if safety_critical["critical_miss_patient_ids"]:
        lines.append(f"> [!WARNING]")
        lines.append(f"> Critical under-triage patient IDs: {safety_critical['critical_miss_patient_ids']}")
        lines.append("")

                                                                               
    lines += [
        "## Triage Confusion Matrix",
        "",
        "Rows = Expected ESI | Columns = Predicted ESI",
        "",
        "| Expected \\ Predicted | P1 | P2 | P3 | P4 | P5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for exp in range(1, 6):
        row = confusion.get(exp, {})
        cells = " | ".join(str(row.get(pred, 0)) for pred in range(1, 6))
        lines.append(f"| **P{exp}** | {cells} |")
    lines.append("")

                                                                               
    lines += [
        "## Scenario-Level Breakdown",
        "",
        "| Scenario Tag | Cases | Agreement | Under-Triage | Over-Triage | Safety Triggered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tag in sorted(scenario_breakdown):
        s = scenario_breakdown[tag]
        lines.append(
            f"| `{tag}` | {s['count']} | {s['agreement']} | {s['under_triage']} | {s['over_triage']} | {s['safety_triggered']} |"
        )
    lines.append("")

                                                                                
    lines += [
        "## Per-Patient Detail Table",
        "",
        "| Patient ID | Scenario Tag | Exp ESI | Pred ESI | Exp Band | Pred Band | Conf | Exp Escalation | Pred Escalation | Age Group | History |",
        "|---|---|---:|---:|---|---|---:|:---:|:---:|---|---|",
    ]
    for r in results:
        under_marker = " ⚠️" if r["predicted_triage_level"] > r["expected_triage_level"] else ""
        lines.append(
            f"| `{r['patient_id']}` | `{r['scenario_tag']}` "
            f"| {r['expected_triage_level']} | {r['predicted_triage_level']}{under_marker} "
            f"| {r['expected_confidence_band']} | {r['predicted_confidence_band']} "
            f"| {r['confidence_score']:.0%} "
            f"| {'Yes' if r['expected_escalation'] else 'No'} "
            f"| {'Yes' if r['predicted_escalation'] else 'No'} "
            f"| {r['age_group']} | {r['history_availability']} |"
        )
    lines.append("")

                                                                                
    dist = metrics["confidence_distribution"]
    lines += [
        "## Confidence Distribution",
        "",
        f"- **HIGH** (≥80%): {dist.get('HIGH', 0)} cases",
        f"- **MEDIUM** (55–79%): {dist.get('MEDIUM', 0)} cases",
        f"- **LOW** (<55%): {dist.get('LOW', 0)} cases",
        "",
    ]

    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    payload = evaluate()
    print(json.dumps(payload["metrics"], indent=2))
    safety = payload["safety_critical"]
    print(f"\n--- Safety-Critical Failures ---")
    print(f"  Critical under-triage:       {safety['critical_under_triage_count']}")
    print(f"  Critical rule misses:        {safety['critical_rule_miss_count']}")
    print(f"  Pediatric misses:            {safety['pediatric_safety_miss_count']}")
    print(f"  Geriatric misses:            {safety['geriatric_safety_miss_count']}")
    print(f"  Data leakage:                {'CLEAN' if payload['metrics']['data_leakage_clean'] else 'VIOLATIONS'}")
