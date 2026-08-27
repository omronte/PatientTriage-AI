"""
streamlit_app.py — Streamlit Frontend Alternative for PatientTriage.ai
Implements all master prompt requirements using the shared backend modules:
  - Vitals & Age Safety LangGraph orchestration
  - XGBoost inline model & NLP extraction (Ollama / fallback)
  - SQLite audit trail
  - Surge simulation (3x volume / 15 patients)
  - Deterioration tracking (ESI 3 > 60m -> ESI 2 overdue)
  - Clinician Accept / Override with audit logging
"""

import json
import os
import time
from datetime import datetime, timedelta
import streamlit as st

# Import shared backend modules
from backend.database import init_db, log_decision, get_recent_logs, get_audit_stats, db_add_safety_event
from backend.data.golden_dataset import load_golden_dataset, to_pipeline_patient
from backend.queue_monitor import calculate_queue_priority, get_wait_thresholds
from ml_engine import train_model, predict_triage, extract_nlp, scrub_phi
from workflow import build_and_run_pipeline
from surge_simulator import generate_surge_patients

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PatientTriage.ai — ED Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.info("🔬 **SYNTHETIC DEMO DATASET** — All patient profiles and simulated triage outcomes are purely synthetic scenarios for prototype safety evaluation. No real patient data is used.")

# ---------------------------------------------------------------------------
# Initialization & Caching
# ---------------------------------------------------------------------------
@st.cache_resource
def get_ml_model():
    init_db()
    return train_model()

model = get_ml_model()

if "patients" not in st.session_state:
    patients_file = os.path.join(os.path.dirname(__file__), "patients.json")
    if os.path.exists(patients_file):
        with open(patients_file, "r", encoding="utf-8") as f:
            raw_patients = json.load(f)
    else:
        raw_patients = []

    st.session_state.patients = []
    for raw in raw_patients:
        res = build_and_run_pipeline(
            patient=raw,
            model=model,
            phi_scrubber=scrub_phi,
            predictor=predict_triage,
            nlp_extractor=extract_nlp,
        )
        wait_mins = raw.get("wait_time_minutes", 0)
        arrival_time = datetime.now() - timedelta(minutes=wait_mins)

        st.session_state.patients.append({
            "id": raw.get("id"),
            "name": raw.get("name"),
            "age": raw.get("age"),
            "gender": raw.get("gender"),
            "chief_complaint": raw.get("chief_complaint"),
            "vitals": {
                "heart_rate": raw.get("heart_rate"),
                "blood_pressure": raw.get("blood_pressure"),
                "oxygen_saturation": raw.get("oxygen_saturation"),
                "respiratory_rate": raw.get("respiratory_rate"),
                "temperature": raw.get("temperature"),
                "gcs_score": raw.get("gcs_score", 15),
            },
            "ai_esi": res.get("final_esi", 3),
            "ai_confidence": res.get("final_confidence", 0.5),
            "confidence_label": res.get("confidence_label", "Unclassified"),
            "age_group": res.get("age_group", "unknown"),
            "history_availability": res.get("history_availability", "unavailable"),
            "missing_fields": res.get("missing_fields", []),
            "safety_flags": res.get("safety_flags", []),
            "safety_escalated": res.get("safety_escalated", False),
            "explainability": res.get("explainability", []),
            "trace": res.get("trace", []),
            "arrival_time": arrival_time,
            "status": "AWAITING_REVIEW",
            "nurse_esi": None,
            "override_reason": None,
        })

if "surge_counter" not in st.session_state:
    st.session_state.surge_counter = 7001


def refresh_streamlit_queue():
    """Refresh the in-memory alternative dashboard using the shared queue policy."""
    now = datetime.now()
    thresholds = get_wait_thresholds()
    for patient in st.session_state.patients:
        patient.setdefault("waiting_started_at", patient["arrival_time"])
        patient.setdefault("last_assessment_at", patient["arrival_time"])
        patient.setdefault("last_vitals_at", patient["arrival_time"])
        patient.setdefault("reassessment_required", False)
        patient.setdefault("reassessment_reason", None)
        patient.setdefault("deterioration_detected", False)
        level = patient["nurse_esi"] if patient["nurse_esi"] is not None else patient["ai_esi"]
        waiting_minutes = int((now - patient["waiting_started_at"]).total_seconds() / 60)
        threshold = thresholds.get(level, thresholds[5])
        patient["waiting_minutes"] = waiting_minutes
        patient["wait_threshold_minutes"] = threshold
        if patient["status"] == "AWAITING_REVIEW" and waiting_minutes >= threshold:
            patient["reassessment_required"] = True
            patient["reassessment_reason"] = f"Safe waiting threshold exceeded for triage level {level}"
        patient["queue_priority_key"] = calculate_queue_priority(
            level, safety_flags=patient.get("safety_flags", []),
            reassessment_required=patient["reassessment_required"],
            deterioration_detected=patient["deterioration_detected"],
            confidence_score=patient["ai_confidence"], waiting_minutes=waiting_minutes)


refresh_streamlit_queue()

# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 PatientTriage.ai")
    st.caption("Emergency Department · Decision Support")
    st.divider()

    st.subheader("⚡ Surge & Queue Controls")
    if st.button("🚨 Simulate Surge (3x Volume)", use_container_width=True):
        surge_batch = generate_surge_patients(count=15, start_id=st.session_state.surge_counter)
        st.session_state.surge_counter += 15
        for raw in surge_batch:
            res = build_and_run_pipeline(
                patient=raw,
                model=model,
                phi_scrubber=scrub_phi,
                predictor=predict_triage,
                nlp_extractor=extract_nlp,
            )
            st.session_state.patients.append({
                "id": raw.get("id"),
                "name": raw.get("name"),
                "age": raw.get("age"),
                "gender": raw.get("gender"),
                "chief_complaint": raw.get("chief_complaint"),
                "vitals": {
                    "heart_rate": raw.get("heart_rate"),
                    "blood_pressure": raw.get("blood_pressure"),
                    "oxygen_saturation": raw.get("oxygen_saturation"),
                    "respiratory_rate": raw.get("respiratory_rate"),
                    "temperature": raw.get("temperature"),
                    "gcs_score": raw.get("gcs_score", 15),
                },
                "ai_esi": res.get("final_esi", 3),
                "ai_confidence": res.get("final_confidence", 0.5),
                "confidence_label": res.get("confidence_label", "Unclassified"),
                "age_group": res.get("age_group", "unknown"),
                "history_availability": res.get("history_availability", "unavailable"),
                "missing_fields": res.get("missing_fields", []),
                "safety_flags": res.get("safety_flags", []),
                "safety_escalated": res.get("safety_escalated", False),
                "explainability": res.get("explainability", []),
                "trace": res.get("trace", []),
                "arrival_time": datetime.now() - timedelta(minutes=raw.get("wait_time_minutes", 0)),
                "status": "AWAITING_REVIEW",
                "nurse_esi": None,
                "override_reason": None,
            })
        st.success("Injected 15 surge patients into the queue!")
        st.rerun()

    if st.button("🧪 Load Golden Dataset (22 Synthetic Cases)"):
        records = load_golden_dataset()
        st.session_state.patients = []
        for rec in records:
            raw = to_pipeline_patient(rec)
            res = build_and_run_pipeline(patient=raw, model=model, phi_scrubber=scrub_phi, predictor=predict_triage, nlp_extractor=extract_nlp)
            st.session_state.patients.append({
                "id": rec["patient_id"],
                "name": raw["name"],
                "age": raw["age"],
                "gender": raw["gender"],
                "chief_complaint": raw["chief_complaint"],
                "vitals": {
                    "heart_rate": raw.get("heart_rate"),
                    "blood_pressure": raw.get("blood_pressure"),
                    "oxygen_saturation": raw.get("oxygen_saturation"),
                    "respiratory_rate": raw.get("respiratory_rate"),
                    "temperature": raw.get("temperature"),
                    "gcs_score": raw.get("gcs_score", 15),
                },
                "ai_esi": res.get("final_esi", 3),
                "ai_confidence": res.get("final_confidence", 0.5),
                "confidence_label": res.get("confidence_label", "Unclassified"),
                "age_group": res.get("age_group", "unknown"),
                "history_availability": res.get("history_availability", "none"),
                "missing_fields": res.get("missing_fields", []),
                "safety_flags": res.get("safety_flags", []),
                "safety_escalated": res.get("safety_escalated", False),
                "explainability": res.get("explainability", []),
                "trace": res.get("trace", []),
                "arrival_time": datetime.now() - timedelta(minutes=15),
                "status": "AWAITING_REVIEW",
                "nurse_esi": None,
                "override_reason": None,
            })
        st.success("Loaded 22 synthetic Golden Dataset patients!")
        st.rerun()

    filter_unreviewed = st.checkbox("Show only unreviewed patients", value=False)

    st.divider()
    st.subheader("📊 Audit Trail Summary")
    stats = get_audit_stats()
    col1, col2 = st.columns(2)
    col1.metric("Decisions", stats["total_decisions"])
    col2.metric("Override %", f"{stats['override_rate']}%")

# ---------------------------------------------------------------------------
# Main Layout: Queue & Patient Cards
# ---------------------------------------------------------------------------
st.header("📋 Live Emergency Triage Queue")

# Priority & Deterioration calculations
active_patients = [p for p in st.session_state.patients if not filter_unreviewed or p["status"] == "AWAITING_REVIEW"]

def get_effective_sort_priority(p):
    return p["queue_priority_key"]

active_patients.sort(key=get_effective_sort_priority)

# Header stats
p_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
for p in st.session_state.patients:
    eff = p["nurse_esi"] if p["nurse_esi"] is not None else p["ai_esi"]
    if eff in p_counts:
        p_counts[eff] += 1

stat_cols = st.columns(5)
colors = ["#E5303F", "#D97A0A", "#B08F00", "#159A63", "#1C7FE0"]
labels = ["P1 · Resuscitation", "P2 · Emergent", "P3 · Urgent", "P4 · Less Urgent", "P5 · Non-Urgent"]
for i in range(5):
    stat_cols[i].metric(labels[i], p_counts[i + 1])

st.divider()

reassessment_patients = [p for p in active_patients if p["reassessment_required"]]
if reassessment_patients:
    st.subheader("Patients Requiring Reassessment")
    for patient in reassessment_patients:
        st.error(f"{patient['id']} · {patient['reassessment_reason']} · Waiting {patient['waiting_minutes']} min / threshold {patient['wait_threshold_minutes']} min")
    st.divider()

# Render Patient Cards
for patient in active_patients:
    wait_mins = patient["waiting_minutes"]
    is_overdue = patient["reassessment_required"]
    display_esi = patient["nurse_esi"] if patient["nurse_esi"] is not None else patient["ai_esi"]

    # Color card border
    border_color = colors[display_esi - 1]
    conf_pct = int(patient["ai_confidence"] * 100)

    card_container = st.container()
    with card_container:
        c1, c2, c3 = st.columns([3, 2, 2])

        with c1:
            overdue_tag = " 🚨 **[REASSESSMENT REQUIRED]**" if is_overdue else ""
            st.markdown(f"### {patient['id']} — {patient['name']} ({patient['age']}{patient['gender']}){overdue_tag}")
            st.write(f"**Chief Complaint:** {patient['chief_complaint']}")
            v = patient["vitals"]
            st.caption(f"**Vitals:** HR: {v.get('heart_rate', '--')} bpm | BP: {v.get('blood_pressure', '--')} | SpO₂: {v.get('oxygen_saturation', '--')}% | RR: {v.get('respiratory_rate', '--')}/min | Temp: {v.get('temperature', '--')}°C | GCS: {v.get('gcs_score', '--')}")
            st.caption(f"Last assessment: {patient['last_assessment_at'].strftime('%H:%M')} | Last vitals: {patient['last_vitals_at'].strftime('%H:%M')}")

        with c2:
            st.markdown(f"**AI Recommendation:** <span style='color:{border_color}; font-size: 20px; font-weight:bold;'>ESI {patient['ai_esi']}</span>", unsafe_allow_html=True)
            st.progress(patient["ai_confidence"], text=f"AI Confidence: {conf_pct}%")
            safety_alert = patient["safety_escalated"] or patient["history_availability"] != "available" or patient["missing_fields"]
            if safety_alert:
                st.error("Clinical review required: safety checks escalated this case.")
            else:
                st.success("Safety checks passed")
            st.caption(f"Age group: {patient['age_group']} | History: {patient['history_availability']} | Confidence: {patient['confidence_label']}")
            if patient["missing_fields"]:
                st.warning(f"Missing data: {', '.join(patient['missing_fields'])}")
            if conf_pct < 60:
                st.warning("⚠️ Low confidence (<60%) — Auto-escalated by 1 level for patient safety.")
            st.caption(f"⏱ Wait Time: **{wait_mins} mins** / Safe threshold: **{patient['wait_threshold_minutes']} mins**")

        with c3:
            if patient["status"] == "AWAITING_REVIEW":
                col_acc, col_over = st.columns(2)
                with col_acc:
                    if st.button("✅ Accept", key=f"acc_{patient['id']}", use_container_width=True):
                        patient["status"] = "REVIEWED_ACCEPTED"
                        patient["nurse_esi"] = patient["ai_esi"]
                        log_decision(
                            patient_id=patient["id"],
                            ai_esi=patient["ai_esi"],
                            ai_confidence=patient["ai_confidence"],
                            nurse_esi=patient["ai_esi"],
                            override_reason=None,
                            action_type="ACCEPT",
                            patient_age=patient["age"],
                            patient_gender=patient["gender"],
                        )
                        db_add_safety_event(
                            patient["id"], "AI_RECOMMENDATION_ACCEPTED",
                            previous_state="WAITING", current_state="IN_TREATMENT",
                            previous_triage_level=patient["ai_esi"], new_triage_level=patient["ai_esi"],
                            confidence=patient["ai_confidence"], reason_code="ACCEPTED",
                            clinician_reason="Clinician accepted the AI recommendation", actor="clinician",
                            safety_flags_json=json.dumps(patient.get("safety_flags", [])),
                            missing_fields_json=json.dumps(patient.get("missing_fields", [])),
                            recommendation="Clinician decision recorded",
                        )
                        st.success(f"Accepted ESI {patient['ai_esi']}")
                        st.rerun()

                with col_over:
                    with st.popover("⚠️ Override", use_container_width=True):
                        new_esi = st.selectbox("New ESI Level", [1, 2, 3, 4, 5], index=max(0, display_esi - 1), key=f"sel_{patient['id']}")
                        reason = st.selectbox("Reason", [
                            "Patient appearance stable",
                            "Vitals artifact / sensor error",
                            "Known chronic condition",
                            "Protocol requires lower acuity",
                            "Clinical judgment differs",
                            "Other (documented in chart)"
                        ], key=f"reas_{patient['id']}")
                        if st.button("Confirm Override", key=f"conf_over_{patient['id']}"):
                            patient["status"] = "REVIEWED_OVERRIDDEN"
                            patient["nurse_esi"] = new_esi
                            patient["override_reason"] = reason
                            log_decision(
                                patient_id=patient["id"],
                                ai_esi=patient["ai_esi"],
                                ai_confidence=patient["ai_confidence"],
                                nurse_esi=new_esi,
                                override_reason=reason,
                                action_type="OVERRIDE",
                                patient_age=patient["age"],
                                patient_gender=patient["gender"],
                            )
                            db_add_safety_event(
                                patient["id"], "AI_RECOMMENDATION_OVERRIDDEN",
                                previous_state="WAITING", current_state="IN_TREATMENT",
                                previous_triage_level=patient["ai_esi"], new_triage_level=new_esi,
                                confidence=patient["ai_confidence"], reason_code="CLINICAL_JUDGMENT",
                                clinician_reason=reason, actor="clinician",
                                safety_flags_json=json.dumps(patient.get("safety_flags", [])),
                                missing_fields_json=json.dumps(patient.get("missing_fields", [])),
                                recommendation="Clinician decision recorded",
                            )
                            st.info(f"Overridden to ESI {new_esi}")
                            st.rerun()
            else:
                if patient["status"] == "REVIEWED_ACCEPTED":
                    st.success(f"✓ Confirmed ESI {patient['nurse_esi']}")
                else:
                    st.warning(f"✓ Overridden: ESI {patient['ai_esi']} → ESI {patient['nurse_esi']} ({patient['override_reason']})")

        with st.expander("🔍 AI Clinical Reasoning & LangGraph Pipeline Trace"):
            st.markdown("**Explainability Factors:**")
            for f in patient.get("explainability", []):
                sev_icon = "🔴" if f.get("severityIndicator") == "CRITICAL" else "⚠️"
                st.write(f"{sev_icon} **{f.get('category')}:** {f.get('aiReasoning')}")
            st.markdown("**LangGraph State Trace:**")
            for step in patient.get("trace", []):
                st.caption(f"• **[{step.get('node')}]** ({step.get('elapsed_ms')}ms) — {step.get('detail')}")

        st.divider()

# ---------------------------------------------------------------------------
# Audit Trail Table Tab
# ---------------------------------------------------------------------------
with st.expander("📜 View Full SQLite Audit Trail (Compliance & Governance)"):
    logs = get_recent_logs(limit=50)
    if logs:
        st.dataframe(logs, use_container_width=True)
    else:
        st.write("No decisions logged yet.")
