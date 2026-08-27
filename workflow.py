"""
workflow.py — LangGraph Orchestrator for PatientTriage.ai

Defines a StateGraph with nodes:
  1. scrub_phi       — Anonymize chief complaint
  2. score_vitals    — XGBoost prediction on vitals
  3. extract_nlp     — NLP analysis of chief complaint
  4. age_safety_check — Deterministic age+vitals safety override
  5. calculate_confidence — Composite confidence scoring
  6. final_synthesis  — Merge all signals into triage recommendation

Uses conditional edges for routing. Returns complete triage state
with priority, confidence, explainability factors, and audit trace.
"""

import logging
import time
from typing import TypedDict

logger = logging.getLogger("workflow")

                                                                      
try:
                                      
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not installed -- using sequential pipeline fallback")

                                                                             
              
                                                                             

class TriageState(TypedDict, total=False):
    patient: dict

    scrubbed_complaint: str

    ml_prediction: dict
    nlp_analysis: dict

    age_group: str
    history_availability: str
    missing_fields: list
    safety_flags: list

    age_safety_triggered: bool
    age_safety_override_esi: int | None
    age_safety_reason: str | None
    safety_override: bool

    raw_confidence: float
    missing_data_penalty: float
    nlp_ambiguity_penalty: float
    final_confidence: float
    confidence_escalated: bool
    history_escalated: bool

    final_esi: int
    explainability: list
    trace: list

                                                                             
                
                                                                             

def collect_missing_fields(patient: dict) -> list:
    missing = []

    if patient.get("age") is None:
        missing.append("age")

    if patient.get("heart_rate") is None:
        missing.append("heart_rate")

    if not patient.get("blood_pressure"):
        missing.append("blood_pressure")

    if patient.get("oxygen_saturation") is None:
        missing.append("oxygen_saturation")

    if patient.get("temperature") is None:
        missing.append("temperature")

    if patient.get("respiratory_rate") is None:
        missing.append("respiratory_rate")

    if patient.get("gcs_score") is None:
        missing.append("gcs_score")

    if not patient.get("chief_complaint"):
        missing.append("chief_complaint")

    if determine_history_availability(patient) == "none":
        missing.append("history")

    return missing

def determine_history_availability(patient: dict) -> str:
    value = patient.get("history_availability")

    if value in {"none", "partial", "rich"}:
        return value

    if patient.get("history_available") is False:
        return "none"

    fields = [
        patient.get("past_medical_history"),
        patient.get("pmh"),
        patient.get("medical_history"),
        patient.get("prior_hospitalizations"),
        patient.get("previous_encounters"),
        patient.get("medications"),
    ]

    populated = sum(
        1 for item in fields
        if item not in (None, "", [], {})
    )

    if populated == 0:
        return "none"
    elif populated <= 2:
        return "partial"
    else:
        return "rich"

def node_scrub_phi(state: dict, phi_scrubber) -> dict:
    """Node 1: Scrub PHI from chief complaint."""
    t0 = time.time()
    complaint = state["patient"].get("chief_complaint", "")
    scrubbed = phi_scrubber(complaint)
    elapsed = round(time.time() - t0, 3)

    state["scrubbed_complaint"] = scrubbed
    state.setdefault("trace", []).append({
        "node": "scrub_phi",
        "elapsed_ms": int(elapsed * 1000),
        "detail": f"PHI scrubbed: {len(complaint)} → {len(scrubbed)} chars",
    })
    return state

def route_after_scrub_phi(state: dict) -> str:
    age = state["patient"].get("age")
    try:
        if age is not None:
            numeric_age = float(age)
            if numeric_age < 18 or numeric_age > 65:
                return "age_adjusted_rules"
    except (ValueError, TypeError):
        pass

    return "standard_rules"

def node_age_adjusted_rules(state: dict) -> dict:
    state["ruleset_used"] = "age_adjusted"

    state.setdefault("trace", []).append({
        "node": "age_adjusted_rules",
        "elapsed_ms": 0,
        "detail": f"Age-adjusted ruleset selected for age {state['patient'].get('age')}",
    })

    return state

def node_standard_rules(state: dict) -> dict:
    state["ruleset_used"] = "standard"

    state.setdefault("trace", []).append({
        "node": "standard_rules",
        "elapsed_ms": 0,
        "detail": "Standard ruleset selected",
    })

    return state

def node_score_vitals(state: dict, model, predictor) -> dict:
    """Node 2A: Run XGBoost prediction on patient vitals."""
    t0 = time.time()
    patient = state["patient"]
    prediction = predictor(model, patient)
    elapsed = round(time.time() - t0, 3)

    state["ml_prediction"] = prediction
    state.setdefault("trace", []).append({
        "node": "score_vitals",
        "elapsed_ms": int(elapsed * 1000),
        "detail": f"Predicted ESI {prediction['predicted_esi']} via {prediction['method']} "
                  f"(margin: {prediction['confidence_margin']:.2f})",
    })
    return state

def node_extract_nlp(state: dict, nlp_extractor) -> dict:
    """Node 2B: NLP analysis of scrubbed chief complaint."""
    t0 = time.time()
    complaint = state.get("scrubbed_complaint", state["patient"].get("chief_complaint", ""))
    analysis = nlp_extractor(complaint)
    elapsed = round(time.time() - t0, 3)

    state["nlp_analysis"] = analysis
    state.setdefault("trace", []).append({
        "node": "extract_nlp",
        "elapsed_ms": int(elapsed * 1000),
        "detail": f"NLP via {analysis.get('method', 'unknown')}: "
                  f"{len(analysis.get('red_flags', []))} red flags, "
                  f"risk={analysis.get('risk_level', 'UNKNOWN')}",
    })
    return state

def node_age_safety_check(state: dict) -> dict:
    """Node 3: deterministic, age-aware safety layer.

    Safety rules are authoritative over ML/NLP. Pediatric (<18) and
    geriatric (>65) patients receive explicit age-profile flags. Missing
    vitals are never treated as normal; they reduce confidence and request
    conservative review. Critical deterministic findings force ESI 1;
    significant age-specific findings force ESI 2.
    """
    t0 = time.time()
    patient = state["patient"]
    age_raw = patient.get("age")
    try:
        age = float(age_raw) if age_raw is not None else None
    except (ValueError, TypeError):
        age = None
    ml_esi = int(state.get("ml_prediction", {}).get("predicted_esi", 3))
    history = determine_history_availability(patient)

    flags = list(state.get("safety_flags", []))
    missing = list(state.get("missing_fields", []))
    triggered = False
    override_esi = None
    reasons = []

    if age is None:
        age_group = "unknown"
    elif age < 18:
        age_group = "pediatric"
    elif age > 65:
        age_group = "geriatric"
    else:
        age_group = "adult"
    state["age_group"] = age_group
    state["history_availability"] = history

    if age_group == "pediatric":
        flags.append("PEDIATRIC_AGE_ADJUSTED")
    elif age_group == "geriatric":
        flags.append("GERIATRIC_AGE_ADJUSTED")

    if history == "none":
        flags.append("ZERO_HISTORY")
    elif history == "partial":
        flags.append("PARTIAL_HISTORY")

    if missing:
        flags.append("MISSING_DATA")
        reasons.append("Incomplete intake data requires conservative clinical review")

    def num(key, default=None):
        value = patient.get(key, default)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return default

    o2 = num("oxygen_saturation")
    hr = num("heart_rate")
    temp = num("temperature")
    gcs = num("gcs_score")
    rr = num("respiratory_rate")

    systolic = None
    bp = patient.get("blood_pressure")
    if isinstance(bp, str) and "/" in bp:
        try:
            systolic = float(bp.split("/")[0])
        except (ValueError, IndexError):
            pass

    critical = []
    urgent = []

                                                           
    if o2 is not None and o2 <= 88:
        critical.append(f"O2 saturation critically low ({o2:g}%)")
    if gcs is not None and gcs <= 8:
        critical.append(f"GCS critically depressed ({gcs:g})")
    if hr is not None and (hr < 40 or hr > 160 or (hr >= 140 and systolic is not None and systolic <= 90)):
        critical.append(f"Severe hemodynamic instability (HR {hr:g} bpm)")

    if o2 is not None and o2 < 92 and o2 > 88:
        urgent.append(f"O2 saturation low ({o2:g}%)")
    if hr is not None and (hr < 50 or hr > 130) and not critical:
        urgent.append(f"Heart rate abnormal ({hr:g} bpm)")
    if gcs is not None and gcs <= 12 and gcs > 8:
        urgent.append(f"Altered consciousness (GCS {gcs:g})")
    if rr is not None and (rr > 28 or rr < 10):
        urgent.append(f"Respiratory rate abnormal ({rr:g}/min)")
    if systolic is not None and (systolic < 85 or systolic > 180):
        urgent.append(f"Blood pressure abnormal ({systolic:g} mmHg)")

    complaint_lower = str(patient.get("chief_complaint", "")).lower()

                                                 
    if o2 is None and any(term in complaint_lower for term in ("shortness of breath", "dyspnea", "breathing", "respiratory", "asthma", "wheezing")):
        urgent.append("Missing oxygen saturation in respiratory complaint requires clinical review")

                                                                      
    if age_group == "pediatric":
        if temp is not None and temp >= 38.5:
            urgent.append(f"Pediatric fever ({temp:g}C)")
        if hr is not None and hr >= 140:
            urgent.append(f"Pediatric tachycardia ({hr:g} bpm)")
        if rr is not None and rr >= 30:
            urgent.append(f"Pediatric respiratory rate elevated ({rr:g}/min)")
    elif age_group == "geriatric":
        if temp is not None and temp >= 38.0:
            urgent.append(f"Geriatric fever ({temp:g}C)")
        if hr is not None and (hr >= 110 or hr <= 50):
            urgent.append(f"Geriatric heart-rate abnormality ({hr:g} bpm)")
                                                                              
        if any(term in complaint_lower for term in ("vague", "feeling off", "confusion", "weakness", "fall", "fell", "anticoagulant", "warfarin")):
            urgent.append("Geriatric atypical presentation or fall risk requires review")

                                                                            
                                                                         
                                                                            
                                                                             
                                                                           
                                                                           
    nlp = state.get("nlp_analysis", {})
    nlp_red_flags = nlp.get("red_flags", [])
    nlp_risk_level = nlp.get("risk_level", "MODERATE")

    if nlp_risk_level == "CRITICAL" or len(nlp_red_flags) >= 2:
        critical.append(f"NLP critical finding(s): {', '.join(nlp_red_flags) or nlp_risk_level}")
    elif nlp_risk_level == "HIGH" or len(nlp_red_flags) >= 1:
        urgent.append(f"NLP high-risk finding: {', '.join(nlp_red_flags) or nlp_risk_level}")

                                                                       
                                                                           
                                                                          
                                                                          
                                                        
    if len(urgent) >= 3 and not critical:
        critical.append(f"Multiple concurrent urgent findings ({len(urgent)}): " + "; ".join(urgent))

    if critical:
        override_esi = 1
        triggered = True
        flags.append("CRITICAL_SAFETY_RULE")
        reasons.extend(critical)
    elif urgent:
        override_esi = min(ml_esi, 2)
        triggered = True
        flags.append("AGE_OR_VITAL_SAFETY_RULE")
        reasons.extend(urgent)

                                                                           
                                                
    if triggered and override_esi is not None:
        override_esi = min(override_esi, ml_esi)
        state["safety_override"] = True
    else:
        state["safety_override"] = False

                                          
    state["safety_flags"] = list(dict.fromkeys(flags))
    state["missing_fields"] = missing
    state["age_safety_triggered"] = triggered
    state["age_safety_override_esi"] = override_esi
    state["age_safety_reason"] = "; ".join(reasons) if reasons else None

    elapsed = round(time.time() - t0, 3)
    state.setdefault("trace", []).append({
        "node": "age_safety_check",
        "elapsed_ms": int(elapsed * 1000),
        "detail": (
            f"Age group={age_group}, history={history}, missing={len(missing)}, "
            f"flags={len(state['safety_flags'])}, "
            f"safety override={override_esi if override_esi is not None else 'none'}"
        ),
    })
    return state

def node_calculate_confidence(state: dict) -> dict:
    """
    Node 4: Calculate composite confidence score.

    Components:
      - ML prediction margin (35% weight)
      - Missing data and history penalty (40% weight)
      - NLP ambiguity penalty (25% weight)

    If confidence < 60%, auto-escalate ESI by 1 level.
    """
    t0 = time.time()
    patient = state["patient"]
    ml_pred = state.get("ml_prediction", {})
    nlp = state.get("nlp_analysis", {})
    age_group = state.get("age_group", "adult")
    history_label = state.get("history_availability") or determine_history_availability(patient)

                                                  
    ml_margin = ml_pred.get("confidence_margin", 0.5)
    ml_confidence = min(ml_margin * 1.25, 1.0)                     

                             
    missing_penalty = 0.0
    bp = patient.get("blood_pressure")
    if bp is None or bp == "" or bp == "null":
        missing_penalty += 0.15
    if patient.get("heart_rate") is None:
        missing_penalty += 0.15
    if patient.get("oxygen_saturation") is None:
        missing_penalty += 0.20
    if patient.get("temperature") is None:
        missing_penalty += 0.15
    if patient.get("respiratory_rate") is None:
        missing_penalty += 0.10
    if patient.get("gcs_score") is None:
        missing_penalty += 0.10

                                 
    if history_label == "none" or not patient.get("history_available", True):
        missing_penalty += 0.25
    elif history_label == "partial":
        missing_penalty += 0.12

                                         
    if age_group == "geriatric" and history_label == "none":
        missing_penalty += 0.15
    if age_group == "pediatric" and (patient.get("temperature") is None or patient.get("oxygen_saturation") is None):
        missing_penalty += 0.15

    missing_penalty = min(missing_penalty, 0.65)

                                                                                  
    llm_ambiguity = nlp.get("nlp_ambiguity_score")
    if llm_ambiguity is not None and isinstance(llm_ambiguity, (int, float)):
        nlp_ambiguity = float(llm_ambiguity) * 0.60
    else:
        nlp_ambiguity = 0.0
        risk_level = nlp.get("risk_level", "MODERATE")
        red_flags = nlp.get("red_flags", [])
        symptoms = nlp.get("symptoms", [])

        if risk_level == "LOW" and ml_pred.get("predicted_esi", 3) <= 2:
            nlp_ambiguity += 0.35                                          
        elif risk_level == "CRITICAL" and ml_pred.get("predicted_esi", 3) >= 4:
            nlp_ambiguity += 0.30                                     
        if len(red_flags) == 0 and len(symptoms) == 0:
            nlp_ambiguity += 0.25                          

    nlp_ambiguity = min(nlp_ambiguity, 0.60)

                                                                                                           
    effective_ml_conf = ml_confidence * (1.0 - nlp_ambiguity * 0.5)

                                                                                         
    raw_confidence = (
        effective_ml_conf * 0.35 +
        (1.0 - missing_penalty) * 0.40 +
        (1.0 - nlp_ambiguity) * 0.25
    )
    final_confidence = max(0.0, min(1.0, raw_confidence))

                                         
    confidence_escalated = False
    history_escalated = False
    current_esi = state.get("age_safety_override_esi") or ml_pred.get("predicted_esi", 3)

    if final_confidence < 0.60 and current_esi > 1:
        current_esi = current_esi - 1
        confidence_escalated = True

    if not patient.get("history_available", True) and not state.get("age_safety_override_esi") and current_esi > 1 and not confidence_escalated:
        current_esi = current_esi - 1
        history_escalated = True

    elapsed = round(time.time() - t0, 3)

    state["raw_confidence"] = round(raw_confidence, 4)
    state["missing_data_penalty"] = round(missing_penalty, 4)
    state["nlp_ambiguity_penalty"] = round(nlp_ambiguity, 4)
    state["final_confidence"] = round(final_confidence, 4)
    state["confidence_escalated"] = confidence_escalated
    state["history_escalated"] = history_escalated

                                         
    if confidence_escalated or history_escalated:
        if state.get("age_safety_override_esi"):
            state["age_safety_override_esi"] = current_esi
        else:
            state["ml_prediction"]["predicted_esi"] = current_esi

    state.setdefault("trace", []).append({
        "node": "calculate_confidence",
        "elapsed_ms": int(elapsed * 1000),
        "detail": f"Confidence: {final_confidence:.1%} "
                  f"(ML margin: {ml_confidence:.2f}, missing penalty: {missing_penalty:.2f}, "
                  f"LLM ambiguity: {nlp_ambiguity:.2f})"
                  + (f" → AUTO-ESCALATED to ESI {current_esi}" if confidence_escalated or history_escalated else ""),
    })
    return state

def node_final_synthesis(state: dict) -> dict:
    """
    Node 5: Merge all signals into final triage recommendation
    with explainability factors.
    """
    t0 = time.time()
    patient = state["patient"]
    ml_pred = state.get("ml_prediction", {})
    nlp = state.get("nlp_analysis", {})
    age_triggered = state.get("age_safety_triggered", False)

                         
    ml_esi = int(ml_pred.get("predicted_esi", 3))
    safety_esi = state.get("age_safety_override_esi")
    if state.get("safety_override") and safety_esi is not None:
                                                                        
        final_esi = min(int(safety_esi), ml_esi)
    else:
        final_esi = ml_esi

                                                                           
                                                                           
                                                                          
                                                                          

    final_esi = max(1, min(5, int(final_esi)))

                                  
    factors = []
    complaint = patient.get("chief_complaint", "")

                                                                
    for rf in nlp.get("red_flags", []):
        factors.append({
            "id": f"ef-rf-{len(factors)}",
            "category": "NLP_KEYWORD",
            "highlightTarget": rf,
            "severityIndicator": "CRITICAL",
            "aiReasoning": _generate_reasoning("red_flag", rf, nlp),
        })

    for symptom in nlp.get("symptoms", []):
        factors.append({
            "id": f"ef-sx-{len(factors)}",
            "category": "NLP_KEYWORD",
            "highlightTarget": symptom,
            "severityIndicator": "WARNING",
            "aiReasoning": _generate_reasoning("symptom", symptom, nlp),
        })

                         
    vital_factors = _generate_vital_factors(patient)
    factors.extend(vital_factors)

                       
    if age_triggered:
        factors.append({
            "id": f"ef-age-{len(factors)}",
            "category": "HISTORICAL_RISK",
            "highlightTarget": f"Age {patient.get('age')}",
            "severityIndicator": "CRITICAL",
            "aiReasoning": state.get("age_safety_reason", "Age safety override triggered"),
        })

                                                                        
    for flag in state.get("safety_flags", []):
        if flag not in {"PEDIATRIC_AGE_ADJUSTED", "GERIATRIC_AGE_ADJUSTED", "ZERO_HISTORY", "PARTIAL_HISTORY", "MISSING_DATA", "CRITICAL_SAFETY_RULE", "AGE_OR_VITAL_SAFETY_RULE"}:
            continue
        severity = "CRITICAL" if flag in {"CRITICAL_SAFETY_RULE", "AGE_OR_VITAL_SAFETY_RULE"} else "WARNING"
        factors.append({
            "id": f"ef-safety-{len(factors)}",
            "category": "SAFETY_RULE",
            "highlightTarget": flag,
            "severityIndicator": severity,
            "aiReasoning": state.get("age_safety_reason") or f"Deterministic safety state: {flag}",
        })

                                  
    if state.get("confidence_escalated"):
        factors.append({
            "id": f"ef-conf-{len(factors)}",
            "category": "HISTORICAL_RISK",
            "highlightTarget": f"Low confidence ({state['final_confidence']:.0%})",
            "severityIndicator": "WARNING",
            "aiReasoning": f"AI confidence is below 60% threshold ({state['final_confidence']:.0%}). "
                          f"ESI automatically escalated by 1 level as a safety precaution. "
                          f"Missing data penalty: {state.get('missing_data_penalty', 0):.0%}, "
                          f"LLM ambiguity: {state.get('nlp_ambiguity_penalty', 0):.0%}.",
        })

                       
    bp = patient.get("blood_pressure")
    if bp is None:
        factors.append({
            "id": f"ef-bp-{len(factors)}",
            "category": "VITAL_ALERT",
            "highlightTarget": "Blood pressure missing",
            "severityIndicator": "WARNING",
            "aiReasoning": "Blood pressure data is unavailable. This introduces a 20% confidence penalty. "
                          "Hemodynamic status cannot be assessed — consider manual BP measurement urgently.",
        })

                                 
    conditions = nlp.get("suspected_conditions", [])
    if conditions:
        factors.append({
            "id": f"ef-dx-{len(factors)}",
            "category": "NLP_KEYWORD",
            "highlightTarget": conditions[0],
            "severityIndicator": "CRITICAL" if final_esi <= 2 else "WARNING",
            "aiReasoning": f"Suspected condition(s): {', '.join(conditions)}. "
                          f"Based on combined NLP and vitals analysis.",
        })

                                                     
    if len(factors) > 8:
        critical = [f for f in factors if f["severityIndicator"] == "CRITICAL"]
        warning = [f for f in factors if f["severityIndicator"] == "WARNING"]
        factors = critical[:5] + warning[:3]

    elapsed = round(time.time() - t0, 3)

    state["final_esi"] = final_esi
    state["explainability"] = factors
    state.setdefault("trace", []).append({
        "node": "final_synthesis",
        "elapsed_ms": int(elapsed * 1000),
        "detail": f"Final ESI: {final_esi}, Confidence: {state.get('final_confidence', 0):.1%}, "
                  f"Factors: {len(factors)}",
    })
    return state

def _generate_reasoning(factor_type: str, keyword: str, nlp: dict) -> str:
    """Generate clinical reasoning text, prioritizing LLM reasoning if present."""
                                                                        
    alert_map = nlp.get("alert_reasoning", {})
    if isinstance(alert_map, dict):
        for k, v in alert_map.items():
            if k.lower() == keyword.lower() or keyword.lower() in k.lower():
                return v

                                     
    if factor_type == "red_flag":
        return f"Critical clinical finding: \"{keyword}\" detected. Indicative of acute high-acuity pathology requiring emergent evaluation."
    else:
        return f"Clinical symptom: \"{keyword}\" identified during presentation. Correlated with active physiological distress."

def _generate_vital_factors(patient: dict) -> list:
    """Generate explainability factors from vital signs."""
    factors = []

    o2 = patient.get("oxygen_saturation")
    if o2 is not None:
        if o2 < 88:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "O2 saturation",
                "severityIndicator": "CRITICAL",
                "aiReasoning": f"O₂ saturation at {o2}% is critically low, indicating possible hypoxia. "
                              f"Immediate oxygen supplementation required.",
            })
        elif o2 < 92:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "O2 saturation",
                "severityIndicator": "WARNING",
                "aiReasoning": f"O₂ saturation at {o2}% is below normal range. "
                              f"Supplemental oxygen and monitoring required.",
            })

    hr = patient.get("heart_rate")
    if hr is not None:
        if hr > 130 or hr < 50:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "Heart rate",
                "severityIndicator": "CRITICAL",
                "aiReasoning": f"Heart rate of {hr} bpm is significantly abnormal. "
                              f"May indicate shock, arrhythmia, or severe physiological stress.",
            })
        elif hr > 110:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "Heart rate",
                "severityIndicator": "WARNING",
                "aiReasoning": f"Tachycardia at {hr} bpm. May be compensatory or indicate underlying pathology.",
            })

    bp = patient.get("blood_pressure")
    if bp and isinstance(bp, str) and "/" in bp:
        try:
            sys_val = int(bp.split("/")[0])
            dia_val = int(bp.split("/")[1])
            if sys_val > 180 or sys_val < 85:
                factors.append({
                    "id": f"ef-v-{len(factors)}",
                    "category": "VITAL_ALERT",
                    "highlightTarget": "Blood pressure",
                    "severityIndicator": "CRITICAL",
                    "aiReasoning": f"Blood pressure {bp} mmHg is critically abnormal. "
                                  f"{'Hypertensive emergency' if sys_val > 180 else 'Hypotension'} "
                                  f"requires urgent intervention.",
                })
            elif sys_val > 160 or sys_val < 90:
                factors.append({
                    "id": f"ef-v-{len(factors)}",
                    "category": "VITAL_ALERT",
                    "highlightTarget": "Blood pressure",
                    "severityIndicator": "WARNING",
                    "aiReasoning": f"Blood pressure {bp} mmHg is outside normal range. Monitor closely.",
                })
        except (ValueError, IndexError):
            pass

    gcs = patient.get("gcs_score")
    if gcs is not None and gcs < 15:
        severity = "CRITICAL" if gcs <= 12 else "WARNING"
        factors.append({
            "id": f"ef-v-{len(factors)}",
            "category": "VITAL_ALERT",
            "highlightTarget": "GCS Score",
            "severityIndicator": severity,
            "aiReasoning": f"GCS of {gcs} indicates {'significantly ' if gcs <= 12 else ''}altered "
                          f"level of consciousness. Requires close neurological monitoring.",
        })

    temp = patient.get("temperature")
    if temp is not None:
        if temp > 39.5:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "Temperature",
                "severityIndicator": "WARNING",
                "aiReasoning": f"High fever at {temp}°C. Indicates significant inflammatory or infectious process.",
            })
        elif temp > 38.5:
            factors.append({
                "id": f"ef-v-{len(factors)}",
                "category": "VITAL_ALERT",
                "highlightTarget": "Temperature",
                "severityIndicator": "WARNING",
                "aiReasoning": f"Elevated temperature at {temp}°C suggests possible infection or inflammation.",
            })

    rr = patient.get("respiratory_rate")
    if rr is not None and (rr > 28 or rr < 10):
        factors.append({
            "id": f"ef-v-{len(factors)}",
            "category": "VITAL_ALERT",
            "highlightTarget": "Respiratory rate",
            "severityIndicator": "CRITICAL",
            "aiReasoning": f"Respiratory rate of {rr}/min is critically abnormal. "
                          f"{'Tachypnea' if rr > 28 else 'Bradypnea'} suggests respiratory compromise.",
        })

    return factors

                                                                             
                 
                                                                             

def build_and_run_pipeline(patient: dict, model, phi_scrubber, predictor, nlp_extractor) -> dict:
    """
    Build and run the full triage pipeline.

    If LangGraph is available, uses a proper StateGraph.
    Otherwise, runs nodes sequentially.
    """
    initial_state: dict = {
        "patient": patient,
        "trace": [],
        "safety_flags": [],
        "missing_fields": collect_missing_fields(patient),
        "history_availability": determine_history_availability(patient),
        "safety_override": False,
    }

    if LANGGRAPH_AVAILABLE:
        return _run_langgraph_pipeline(initial_state, model, phi_scrubber, predictor, nlp_extractor)
    else:
        return _run_sequential_pipeline(initial_state, model, phi_scrubber, predictor, nlp_extractor)

def _run_langgraph_pipeline(state: dict, model, phi_scrubber, predictor, nlp_extractor) -> dict:
    """Run the triage pipeline using LangGraph StateGraph."""

    def _scrub(s):
        return node_scrub_phi(s, phi_scrubber)

    def _vitals(s):
        return node_score_vitals(s, model, predictor)

    def _nlp(s):
        return node_extract_nlp(s, nlp_extractor)

    def _age_safety(s):
        return node_age_safety_check(s)

    def _confidence(s):
        return node_calculate_confidence(s)

    def _synthesis(s):
        return node_final_synthesis(s)

                     
    graph = StateGraph(dict)

    graph.add_node("scrub_phi", _scrub)
    graph.add_node("age_adjusted_rules", node_age_adjusted_rules)
    graph.add_node("standard_rules", node_standard_rules)
    graph.add_node("score_vitals", _vitals)
    graph.add_node("extract_nlp", _nlp)
    graph.add_node("age_safety_check", _age_safety)
    graph.add_node("calculate_confidence", _confidence)
    graph.add_node("final_synthesis", _synthesis)

                  
    graph.set_entry_point("scrub_phi")
    graph.add_conditional_edges(
        "scrub_phi",
        route_after_scrub_phi,
        {
            "age_adjusted_rules": "age_adjusted_rules",
            "standard_rules": "standard_rules",
        },
    )
    graph.add_edge("age_adjusted_rules", "score_vitals")
    graph.add_edge("standard_rules", "score_vitals")
    graph.add_edge("score_vitals", "extract_nlp")
    graph.add_edge("extract_nlp", "age_safety_check")
    graph.add_edge("age_safety_check", "calculate_confidence")
    graph.add_edge("calculate_confidence", "final_synthesis")
    graph.add_edge("final_synthesis", END)

                     
    app = graph.compile()
    result = app.invoke(state)
    return result

def _run_sequential_pipeline(state: dict, model, phi_scrubber, predictor, nlp_extractor) -> dict:
    """Fallback: run nodes sequentially without LangGraph."""
    state = node_scrub_phi(state, phi_scrubber)
    state = node_age_adjusted_rules(state) if route_after_scrub_phi(state) == "age_adjusted_rules" else node_standard_rules(state)
    state = node_score_vitals(state, model, predictor)
    state = node_extract_nlp(state, nlp_extractor)
    state = node_age_safety_check(state)
    state = node_calculate_confidence(state)
    state = node_final_synthesis(state)
    return state