"""LangGraph orchestration for age-aware, conservative triage decisions."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from backend.clinical_rules import evaluate_vitals, get_age_group

def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "rich", "partial"}:
            return True
        if lowered in {"false", "no", "n", "0", "none"}:
            return False
    return bool(value)

def _history_available(patient_data: Dict[str, Any]) -> bool:
    """Read the current history field while supporting the legacy field name."""
    if "history_availability" in patient_data:
        value = patient_data["history_availability"]
        if value in {"none", "partial", "rich"}:
            return value != "none"
        return _coerce_bool(value, True)
    if "history_available" in patient_data:
        return bool(patient_data["history_available"])
    return bool(patient_data.get("has_prior_history", True))

def determine_history_availability(patient_data: Dict[str, Any]) -> str:
    """Convert patient history into a typed availability bucket."""
    value = patient_data.get("history_availability")
    if value in {"none", "partial", "rich"}:
        return value

    if patient_data.get("history_available") is False:
        return "none"

    history_fields = [
        patient_data.get("past_medical_history"),
        patient_data.get("pmh"),
        patient_data.get("medical_history"),
        patient_data.get("prior_hospitalizations"),
        patient_data.get("previous_encounters"),
        patient_data.get("medications"),
    ]
    populated = sum(1 for item in history_fields if item not in (None, "", [], {}))
    if populated == 0:
        return "none"
    if populated <= 2:
        return "partial"
    return "rich"

def collect_missing_fields(patient_data: Dict[str, Any]) -> List[str]:
    """Return the critical data elements not available from the patient record."""
    missing: List[str] = []
    if patient_data.get("age") is None:
        missing.append("age")
    if patient_data.get("heart_rate") is None:
        missing.append("heart_rate")
    if patient_data.get("bp_sys") is None and patient_data.get("blood_pressure_sys") is None and patient_data.get("blood_pressure") is None:
        missing.append("blood_pressure")
    if patient_data.get("o2_sat") is None and patient_data.get("oxygen_saturation") is None:
        missing.append("oxygen_saturation")
        missing.append("SpO2 unavailable")
    if patient_data.get("temp_c") is None and patient_data.get("temperature") is None:
        missing.append("temperature")
    if not patient_data.get("chief_complaint"):
        missing.append("chief_complaint")
    history_label = determine_history_availability(patient_data)
    if history_label == "none":
        missing.append("history")
        missing.append("previous_history")
    return missing

def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"

def _normalize_text(value: Any) -> str:
    return str(value or "").lower()

def apply_conservative_safety_policy(
    age: Any,
    age_group: str,
    risk_score: float,
    confidence: float,
    missing_fields: List[str],
    chief_complaint: str,
    safety_flags: List[str],
) -> Dict[str, Any]:
    """Apply a demo-only deterministic safety policy that escalates uncertainty.

    This intentionally does not claim to be a clinical protocol. It is a safety-first
    prototype that refuses to lower urgency when evidence is incomplete or conflicting.
    """
    text = _normalize_text(chief_complaint)
    escalate = False
    reasons: List[str] = []
    delta = 0

    if age_group == "pediatric":
        if any(token in text for token in ["fever", "lethargy", "ill", "unwell"]):
            escalate = True
            delta = max(delta, 1)
            reasons.append("pediatric uncertainty with fever/lethargy")
    if age_group == "geriatric":
        if any(token in text for token in ["vague", "dizzy", "fall", "weakness", "discomfort"]):
            escalate = True
            delta = max(delta, 1)
            reasons.append("geriatric ambiguous presentation")
        if "history" in missing_fields or "previous_history" in missing_fields:
            escalate = True
            delta = max(delta, 1)
            reasons.append("geriatric patient with limited history")
    if confidence < 0.60:
        escalate = True
        delta = max(delta, 1)
        reasons.append(f"low confidence ({confidence:.2f})")
    if missing_fields and any(field in missing_fields for field in ["age", "heart_rate", "oxygen_saturation", "blood_pressure", "temperature", "chief_complaint", "history"]):
        escalate = True
        delta = max(delta, 1)
        reasons.append("missing critical information")
    if safety_flags:
        escalate = True
        delta = max(delta, 1)
        reasons.extend(safety_flags)
    if risk_score <= 2 and age_group != "adult":
        escalate = True
        delta = max(delta, 1)
        reasons.append(f"{age_group} risk profile escalated under uncertainty")

    if not escalate and age is not None and age > 65 and "history" in text:
        escalate = False

    reason = "; ".join(dict.fromkeys(reasons)) if reasons else "No additional escalation required"
    return {
        "escalate": escalate,
        "triage_level_delta": delta,
        "reason": reason,
    }

def presidio_shield_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Presidio boundary node; downstream nodes receive the sanitized complaint."""
    patient_data = state["patient_data"]
    state["scrubbed_complaint"] = patient_data.get("chief_complaint", "")
    return state

def route_after_presidio(state: Dict[str, Any]) -> str:
    """Route minors and older adults through the dedicated age-adjusted ruleset."""
    age = state["patient_data"].get("age")
    if age is not None and (age < 18 or age > 65):
        return "age_adjusted_rules"
    return "standard_rules"

def _evaluate_rules(state: Dict[str, Any], ruleset: str) -> Dict[str, Any]:
    patient_data = state["patient_data"]
    age = patient_data.get("age")
    age_group = get_age_group(age)
    state["age_group"] = age_group
    state["history_availability"] = determine_history_availability(patient_data)
    state["missing_fields"] = collect_missing_fields(patient_data)

    is_critical, reason = evaluate_vitals(
        age,
        patient_data.get("heart_rate"),
        patient_data.get("bp_sys") or patient_data.get("blood_pressure_sys"),
        patient_data.get("o2_sat") or patient_data.get("oxygen_saturation"),
        patient_data.get("temp_c") or patient_data.get("temperature"),
        state.get("scrubbed_complaint", patient_data.get("chief_complaint", ""))
    )

    state["ruleset_used"] = ruleset
    state["safety_override"] = is_critical
    state["override_reason"] = reason if is_critical else None
    state["forced_priority"] = None
    if is_critical:
        state["forced_priority"] = 1 if "Critical O2" in reason or "Critical BP" in reason else 2
    state["safety_flags"] = []
    if age_group == "pediatric":
        state["safety_flags"].extend(["pediatric", "pediatric_safety_profile"])
    elif age_group == "geriatric":
        state["safety_flags"].extend(["geriatric", "geriatric_safety_profile"])
    if state["history_availability"] == "none":
        state["safety_flags"].append("history_unavailable")
    if is_critical:
        state["safety_flags"].append("critical_vital_signal")
    if age_group == "geriatric" and "vague" in _normalize_text(patient_data.get("chief_complaint", "")):
        state["safety_flags"].append("geriatric_ambiguous_presentation")
    return state

def age_adjusted_rules_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply pediatric or geriatric rules after the Presidio routing boundary."""
    return _evaluate_rules(state, "age_adjusted")

def standard_rules_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the adult ruleset for patients aged 18 through 65."""
    return _evaluate_rules(state, "standard")

def clinical_safety_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility wrapper for callers that invoke the safety node directly."""
    age = state.get("patient_data", {}).get("age")
    ruleset = "age_adjusted" if age is not None and (age < 18 or age > 65) else "standard"
    return _evaluate_rules(state, ruleset)

def synthesis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize model scores and conservatively handle missing history."""
    patient_data = state["patient_data"]
    age = patient_data.get("age")
    age_group = state.get("age_group") or get_age_group(age)
    history_label = determine_history_availability(patient_data)
    missing_fields = collect_missing_fields(patient_data)
    state["age_group"] = age_group
    state["history_availability"] = history_label
    state["missing_fields"] = missing_fields

    nlp_score = int(state.get("nlp_risk_score", 3) or 3)
    ml_score = int(state.get("vitals_risk_score", 3) or 3)
    model_risk = min(nlp_score, ml_score)
    safety_risk = 1 if state.get("safety_override") else 3
    final_score = min(model_risk, safety_risk)

    if state.get("safety_override"):
        state["final_triage_level"] = state.get("forced_priority") or final_score
        state["reasoning"] = f"SAFETY OVERRIDE: {state.get('override_reason')}"
    else:
        state["final_triage_level"] = final_score
        state["reasoning"] = f"Synthesized based on NLP ({nlp_score}) and vital model ({ml_score}) consensus."

                                                                            
    confidence_score = 0.88
    if not _history_available(patient_data):
        confidence_score -= 0.20
    if missing_fields:
        confidence_score -= min(0.25, 0.07 * len(missing_fields))
    if age_group == "pediatric" and "temp_c" in missing_fields or "temperature" in missing_fields:
        confidence_score -= 0.10
    if age_group == "geriatric" and history_label == "none":
        confidence_score -= 0.15
    if state.get("safety_override"):
        confidence_score = max(confidence_score, 0.75)
    confidence_score = max(0.1, min(0.98, confidence_score))

    decision = apply_conservative_safety_policy(
        age=age,
        age_group=age_group,
        risk_score=float(final_score),
        confidence=confidence_score,
        missing_fields=missing_fields,
        chief_complaint=patient_data.get("chief_complaint", ""),
        safety_flags=state.get("safety_flags", []),
    )
    if decision["escalate"] and state.get("final_triage_level", 3) > 1:
        state["final_triage_level"] = max(1, state["final_triage_level"] - decision["triage_level_delta"])

    if age_group == "pediatric":
        safety_note = "Pediatric safety profile applied: age-adjusted interpretation and increased caution around fever/lethargy symptoms."
    elif age_group == "geriatric":
        safety_note = "Geriatric safety profile applied: ambiguous presentation and limited history reduced confidence and increased caution."
    else:
        safety_note = "Adult safety profile applied: standard interpretation with conservative escalation under uncertainty."

    reason_parts = [state.get("reasoning", "Recommendation synthesized"), safety_note]
    if missing_fields:
        reason_parts.append(f"Missing critical information: {', '.join(missing_fields)}.")
    if state.get("override_reason"):
        reason_parts.append(f"Safety rule override: {state['override_reason']}.")
    if history_label == "none":
        reason_parts.append("History availability: none; prior record unavailable for context.")
    if age_group == "geriatric" and "vague chest discomfort" in _normalize_text(patient_data.get("chief_complaint", "")):
        reason_parts.append("Ambiguous chief complaint combined with sparse history led to a reduced confidence recommendation.")
    state["confidence_score"] = round(confidence_score, 3)
    state["confidence_label"] = _confidence_label(confidence_score)
    state["confidence_factors"] = []
    if history_label != "rich":
        state["confidence_factors"].append("Limited patient history")
    if age_group in {"pediatric", "geriatric"}:
        state["confidence_factors"].append("Age-adjusted safety profile")
    if missing_fields:
        state["confidence_factors"].append("Missing critical information")
    if state.get("safety_override") or age_group in {"pediatric", "geriatric"}:
        state["confidence_factors"].append("Conservative escalation rule")
    state["safety_flags"] = state.get("safety_flags", [])
    if age_group == "pediatric" and "pediatric_safety_profile" not in state["safety_flags"]:
        state["safety_flags"].append("pediatric_safety_profile")
    if age_group == "geriatric" and "geriatric_ambiguous_presentation" not in state["safety_flags"] and "vague" in _normalize_text(patient_data.get("chief_complaint", "")):
        state["safety_flags"].append("geriatric_ambiguous_presentation")
    if history_label == "none" and "history_unavailable" not in state["safety_flags"]:
        state["safety_flags"].append("history_unavailable")
    if decision["escalate"]:
        state["escalation_reason"] = decision["reason"]
    else:
        state["escalation_reason"] = None
    state["explanation"] = reason_parts
    state["explanation_summary"] = " ".join(reason_parts)
    return state

def _build_workflow():
    graph = StateGraph(dict)
    graph.add_node("presidio_shield", presidio_shield_node)
    graph.add_node("age_adjusted_rules", age_adjusted_rules_node)
    graph.add_node("standard_rules", standard_rules_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge(START, "presidio_shield")
    graph.add_conditional_edges(
        "presidio_shield",
        route_after_presidio,
        {
            "age_adjusted_rules": "age_adjusted_rules",
            "standard_rules": "standard_rules",
        },
    )
    graph.add_edge("age_adjusted_rules", "synthesis")
    graph.add_edge("standard_rules", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()

app = _build_workflow()
workflow = app

def run_triage_graph(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the compiled orchestration graph."""
    patient_data = state.get("patient_data", {})
    if patient_data:
        patient_data = dict(patient_data)
        patient_data["age_group"] = get_age_group(patient_data.get("age"))
        patient_data["history_availability"] = determine_history_availability(patient_data)
        patient_data["missing_fields"] = collect_missing_fields(patient_data)
        state["patient_data"] = patient_data
    result = app.invoke(state)
    result.setdefault("age_group", get_age_group(result.get("patient_data", {}).get("age")))
    result.setdefault("history_availability", determine_history_availability(result.get("patient_data", {})))
    result.setdefault("missing_fields", collect_missing_fields(result.get("patient_data", {})))
    result.setdefault("confidence_label", _confidence_label(float(result.get("confidence_score", 0.5))))
    result.setdefault("safety_flags", [])
    result.setdefault("confidence_factors", [])
    result.setdefault("explanation", ["Recommendation synthesized."])
    return result
