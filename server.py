"""
server.py — FastAPI Backend for PatientTriage.ai
Fully Database-Driven: All patients, registrations, overrides, and surge data
are persistently stored in SQLite via SQLAlchemy.
"""

import json
import os
import time
import uuid
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

               
from backend.database import (
    init_db, log_decision, get_recent_logs, get_audit_stats,
    db_save_patient, db_get_all_patients, db_get_patient,
    db_update_patient_review, db_count_patients, db_clear_all_patients, db_delete_patient,
    db_update_queue_state, db_add_safety_event, db_get_safety_events, db_get_all_safety_events
)
from backend.queue_monitor import get_queue_projection, monitor_once, acknowledge_reassessment, detect_vital_deterioration
from ml_engine import train_model, predict_triage, extract_nlp, scrub_phi
from workflow import build_and_run_pipeline
from surge_simulator import generate_surge_patients

                                                                             
         
                                                                             
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("server")

                                                                             
                   
                                                                             
app = FastAPI(
    title="PatientTriage.ai API",
    description="AI-powered Emergency Department Triage Decision Support",
    version="1.0.0",
)

_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
                                                                            
                                                                    
CORS_ALLOW_CREDENTIALS = CORS_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

                                                                             
              
                                                                             
ML_MODEL = None
SURGE_COUNTER = 6001
PATIENT_COUNTER = 5021
QUEUE_MONITOR_TASK = None

                                                                             
                         
                                                                             

class NewPatientRequest(BaseModel):
    age: int
    gender: str
    chief_complaint: str
    heart_rate: Optional[float] = None
    blood_pressure_sys: Optional[float] = None
    blood_pressure_dia: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    respiratory_rate: Optional[float] = None
    temperature: Optional[float] = None
    gcs_score: Optional[float] = None

class OverrideRequest(BaseModel):
    nurse_esi: int
    override_reason: str = Field(min_length=1, max_length=100)
    reason_code: Optional[str] = Field(default=None, max_length=60)
    clinician_reason: Optional[str] = Field(default=None, max_length=500)

    @field_validator("nurse_esi")
    @classmethod
    def validate_esi(cls, value):
        if not 1 <= value <= 5:
            raise ValueError("ESI must be between 1 and 5")
        return value

class VitalsUpdateRequest(BaseModel):
    heart_rate: Optional[float] = None
    blood_pressure_sys: Optional[float] = None
    blood_pressure_dia: Optional[float] = None
    oxygen_saturation: Optional[float] = None
    respiratory_rate: Optional[float] = None
    temperature: Optional[float] = None
    gcs_score: Optional[float] = None

                                                                             
                                                          
                                                                             

@app.on_event("startup")
async def startup():
    global ML_MODEL, PATIENT_COUNTER, SURGE_COUNTER, QUEUE_MONITOR_TASK

    logger.info("===========================================")
    logger.info("  PatientTriage.ai -- Starting up...")
    logger.info("===========================================")

                                   
    init_db()
    logger.info("[OK] SQLite database initialized")

                              
    logger.info("Training XGBoost model on synthetic data...")
    t0 = time.time()
    ML_MODEL = train_model()
    elapsed = time.time() - t0
    if ML_MODEL:
        logger.info("[OK] XGBoost model trained in %.2fs", elapsed)
    else:
        logger.warning("[WARN] XGBoost unavailable -- using rule-based fallback")

                                               
    existing_count = db_count_patients()
    if existing_count == 0:
        logger.info("Database empty -- seeding initial patients from patients.json...")
        patients_file = os.path.join(os.path.dirname(__file__), "patients.json")
        if os.path.exists(patients_file):
            with open(patients_file, "r", encoding="utf-8") as f:
                raw_patients = json.load(f)
            logger.info("Running LangGraph triage on %d initial patients...", len(raw_patients))
            for raw in raw_patients:
                assessed = _run_triage_on_raw_patient(raw)
                db_save_patient(assessed)
            logger.info("[OK] %d patients triaged and saved to SQLite database", len(raw_patients))
    else:
        logger.info("[OK] Database already contains %d persistent patient records", existing_count)

    PATIENT_COUNTER = 5001 + db_count_patients()
    SURGE_COUNTER = 6001 + db_count_patients()
    QUEUE_MONITOR_TASK = asyncio.create_task(_run_queue_monitor())

    logger.info("  PatientTriage.ai -- Ready!")
    logger.info("  Open http://localhost:8000 in your browser")
    logger.info("===========================================")

async def _run_queue_monitor():
    from backend.queue_monitor import check_waiting_queue
    await check_waiting_queue(interval_seconds=int(os.getenv("QUEUE_MONITOR_INTERVAL_SECONDS", "60")))

@app.on_event("shutdown")
async def shutdown():
    if QUEUE_MONITOR_TASK:
        QUEUE_MONITOR_TASK.cancel()
        try:
            await QUEUE_MONITOR_TASK
        except asyncio.CancelledError:
            pass
    logger.info("  PatientTriage.ai -- Shutting down...")
    logger.info("===========================================")

                                                                             
                      
                                                                             

def _run_triage_on_raw_patient(raw: dict) -> dict:
    """
    Run the LangGraph triage pipeline on a raw patient dictionary
    and transform into the standard schema.
    """
    result = build_and_run_pipeline(
        patient=raw,
        model=ML_MODEL,
        phi_scrubber=scrub_phi,
        predictor=predict_triage,
        nlp_extractor=extract_nlp,
    )

    wait_mins = raw.get("wait_time_minutes", 0)
    arrival_time = (datetime.now() - timedelta(minutes=wait_mins)).isoformat()

    age = raw.get("age")
    age_group = "pediatric" if age is not None and age < 18 else "adult" if age is None or age <= 65 else "geriatric"
    history_value = raw.get("history_availability") or raw.get("history_available") or "none"
    if history_value is True:
        history_value = "rich"
    elif history_value is False:
        history_value = "none"
    elif history_value not in {"none", "partial", "rich"}:
        history_value = "partial" if raw.get("history_available") is not None else "none"

    assessed = {
        "patientId": raw.get("id") or f"PT-{uuid.uuid4().hex[:8].upper()}",
        "name": raw.get("name", "Registered Patient"),
        "arrivalTime": arrival_time,
        "age": age,
        "ageGroup": age_group,
        "biologicalSex": raw.get("gender", "Unknown"),
        "chiefComplaint": raw.get("chief_complaint", "No complaint recorded"),
        "vitals": {
            "heartRateBpm": raw.get("heart_rate"),
            "bloodPressureSys": None,
            "bloodPressureDia": None,
            "o2SaturationPercent": raw.get("oxygen_saturation"),
            "respiratoryRate": raw.get("respiratory_rate"),
            "temperatureCelsius": raw.get("temperature"),
            "gcsScore": raw.get("gcs_score", 15),
        },
        "history_available": raw.get("history_available", True),
        "historyAvailability": history_value,
        "riskCategory": "critical" if result.get("final_esi", 3) <= 2 else "moderate" if result.get("final_esi", 3) == 3 else "low",
        "aiSuggestedPriority": result.get("final_esi", 3),
        "aiConfidenceScore": result.get("final_confidence", 0.5),
        "confidenceLabel": "HIGH" if result.get("final_confidence", 0.5) >= 0.8 else "MEDIUM" if result.get("final_confidence", 0.5) >= 0.55 else "LOW",
        "safetyFlags": result.get("safety_flags", []),
        "missingFields": result.get("missing_fields", []),
        "explanation": result.get("explanation") or result.get("explainability", []),
        "explainability": result.get("explanation") or result.get("explainability", []),
        "recommendedAction": "Clinical review required" if result.get("final_confidence", 0.5) < 0.7 else "Continue monitoring",
        "status": "AWAITING_REVIEW",
        "_mlPrediction": result.get("ml_prediction", {}),
        "_nlpAnalysis": result.get("nlp_analysis", {}),
        "_ageSafetyTriggered": result.get("age_safety_triggered", False),
        "_ageSafetyReason": result.get("age_safety_reason"),
        "_confidenceEscalated": result.get("confidence_escalated", False),
        "_missingDataPenalty": result.get("missing_data_penalty", 0),
        "_nlpAmbiguityPenalty": result.get("nlp_ambiguity_penalty", 0),
        "_trace": result.get("trace", []),
    }

    bp = raw.get("blood_pressure")
    if bp and isinstance(bp, str) and "/" in bp:
        try:
            parts = bp.split("/")
            assessed["vitals"]["bloodPressureSys"] = int(parts[0])
            assessed["vitals"]["bloodPressureDia"] = int(parts[1])
        except (ValueError, IndexError):
            pass

    return assessed

                                                                             
                                  
                                                                             

@app.get("/api/patients")
async def get_patients():
    """Retrieve the backend-owned active waiting queue in authoritative order."""
    return get_queue_projection()

@app.get("/api/queue")
async def get_active_queue():
    return get_queue_projection()

@app.get("/api/patients/{patient_id}/status")
async def get_patient_status(patient_id: str):
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    monitor_once()
    return db_get_patient(patient_id)

@app.get("/api/patients/{patient_id}/reassessment")
async def get_reassessment_status(patient_id: str):
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"patient_id": patient_id, "reassessment_required": patient["reassessmentRequired"],
            "reassessment_status": patient["reassessmentStatus"],
            "reason": patient["reassessmentReason"], "events": db_get_safety_events(patient_id)}

@app.post("/api/patients/{patient_id}/vitals")
async def update_patient_vitals(patient_id: str, req: VitalsUpdateRequest):
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    old = patient["vitals"]
    new = {**old, "heartRateBpm": req.heart_rate if req.heart_rate is not None else old.get("heartRateBpm"),
           "bloodPressureSys": req.blood_pressure_sys if req.blood_pressure_sys is not None else old.get("bloodPressureSys"),
           "bloodPressureDia": req.blood_pressure_dia if req.blood_pressure_dia is not None else old.get("bloodPressureDia"),
           "o2SaturationPercent": req.oxygen_saturation if req.oxygen_saturation is not None else old.get("o2SaturationPercent"),
           "respiratoryRate": req.respiratory_rate if req.respiratory_rate is not None else old.get("respiratoryRate"),
           "temperatureCelsius": req.temperature if req.temperature is not None else old.get("temperatureCelsius"),
           "gcsScore": req.gcs_score if req.gcs_score is not None else old.get("gcsScore")}
    raw = {"id": patient_id, "name": patient["name"], "age": patient["age"], "gender": patient["biologicalSex"],
           "chief_complaint": patient["chiefComplaint"], "heart_rate": new["heartRateBpm"],
           "blood_pressure": f'{new["bloodPressureSys"]}/{new["bloodPressureDia"]}' if new["bloodPressureSys"] is not None and new["bloodPressureDia"] is not None else None,
           "oxygen_saturation": new["o2SaturationPercent"], "respiratory_rate": new["respiratoryRate"],
           "temperature": new["temperatureCelsius"], "gcs_score": new["gcsScore"],
           "history_available": patient.get("history_available", True)}
    previous_level = patient["aiSuggestedPriority"]

    assessed = _run_triage_on_raw_patient(raw)

    assessed["arrivalTime"] = patient["arrivalTime"]
    assessed["lifecycleStatus"] = patient["lifecycleStatus"]
    assessed["lastVitalsAt"] = datetime.now(timezone.utc).isoformat()

    changes = detect_vital_deterioration(old, new)

    age_safety_triggered = bool(
        assessed.get("_ageSafetyTriggered", False)
        or assessed.get("age_safety_triggered", False)
    )

    new_priority = assessed.get("aiSuggestedPriority", 3)

                                         
                                        
                                                
                                                   
     
                                                                
    deteriorated = bool(changes)
    assessed["deteriorationDetected"] = deteriorated

    updated = db_save_patient(assessed)
    logger.debug("Vitals update for %s -- changes: %s, deteriorated: %s", patient_id, changes, deteriorated)
    if deteriorated:
        reason = "New vitals indicate deterioration or a safety-rule trigger"

        updated = db_update_queue_state(
            patient_id,
            lifecycle_status="REASSESSMENT_REQUIRED",
            reassessment_status="REQUIRED",
            reassessment_required=True,
            reassessment_reason=reason,
            deterioration_detected=True,
            last_vitals_at=datetime.now(timezone.utc),
            last_safety_event_type="VITAL_DETERIORATION",
            last_safety_event_at=datetime.now(timezone.utc),
            last_queue_priority=assessed["aiSuggestedPriority"],
        )

        db_add_safety_event(
            patient_id,
            "VITAL_DETERIORATION",
            previous_state=patient["lifecycleStatus"],
            current_state="REASSESSMENT_REQUIRED",
            trigger_reason=reason,
            vital_changes_json=json.dumps(changes),
            recommendation="Clinical reassessment required",
        )

    return updated or db_get_patient(patient_id)

@app.post("/api/patients/{patient_id}/reassessment")
async def request_reassessment(patient_id: str):
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    reason = patient.get("reassessmentReason") or "Clinician requested reassessment"
    updated = db_update_queue_state(patient_id, lifecycle_status="REASSESSMENT_REQUIRED", reassessment_status="REQUIRED",
                                    reassessment_required=True, reassessment_reason=reason)
    db_add_safety_event(patient_id, "REASSESSMENT_REQUESTED", previous_state=patient["lifecycleStatus"],
                        current_state="REASSESSMENT_REQUIRED", trigger_reason=reason,
                        recommendation="Clinical reassessment required", actor="clinician",
                        confidence=patient.get("aiConfidenceScore"),
                        safety_flags_json=json.dumps(patient.get("safetyFlags", [])),
                        missing_fields_json=json.dumps(patient.get("missingFields", [])))
    return updated

@app.post("/api/patients/{patient_id}/reassessment/acknowledge")
async def acknowledge_patient_reassessment(patient_id: str):
    updated = acknowledge_reassessment(patient_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Patient not found")
    return updated

@app.post("/api/patients")
async def add_patient(req: NewPatientRequest):
    """
    Register a new patient:
      1. Runs LangGraph triage pipeline (XGBoost + NLP + Age Safety + Confidence).
      2. Persists new patient record in SQLite database.
    """
    global PATIENT_COUNTER

    patient_id = f"PT-{PATIENT_COUNTER}"
    PATIENT_COUNTER += 1

    bp = None
    if req.blood_pressure_sys and req.blood_pressure_dia:
        bp = f"{int(req.blood_pressure_sys)}/{int(req.blood_pressure_dia)}"

    raw_patient = {
        "id": patient_id,
        "name": "Walk-in Patient",
        "age": req.age,
        "gender": req.gender,
        "temperature": req.temperature,
        "heart_rate": req.heart_rate,
        "respiratory_rate": req.respiratory_rate,
        "oxygen_saturation": req.oxygen_saturation,
        "blood_pressure": bp,
        "chief_complaint": req.chief_complaint,
        "history_available": False,
        "wait_time_minutes": 0,
        "gcs_score": req.gcs_score,
    }

                                   
    assessed = _run_triage_on_raw_patient(raw_patient)

                             
    saved_patient = db_save_patient(assessed)

                                                 
    db_add_safety_event(
        patient_id,
        "AI_RECOMMENDATION_GENERATED",
        previous_state=None,
        current_state="AWAITING_REVIEW",
        previous_triage_level=None,
        new_triage_level=saved_patient["aiSuggestedPriority"],
        confidence=saved_patient["aiConfidenceScore"],
        trigger_reason="Initial AI triage assessment generated",
        recommendation="Awaiting clinician review",
        actor="system",
        safety_flags_json=json.dumps(saved_patient.get("safetyFlags", [])),
        missing_fields_json=json.dumps(saved_patient.get("missingFields", [])),
    )

    if saved_patient.get("_ageSafetyTriggered") or "CRITICAL_SAFETY_RULE" in saved_patient.get("safetyFlags", []) or "AGE_OR_VITAL_SAFETY_RULE" in saved_patient.get("safetyFlags", []):
        db_add_safety_event(
            patient_id,
            "SAFETY_RULE_TRIGGERED",
            previous_state="AWAITING_REVIEW",
            current_state="AWAITING_REVIEW",
            previous_triage_level=None,
            new_triage_level=saved_patient["aiSuggestedPriority"],
            confidence=saved_patient["aiConfidenceScore"],
            trigger_reason=saved_patient.get("_ageSafetyReason") or "Deterministic safety rule override",
            recommendation="Safety rule applied to priority level",
            actor="system",
            safety_flags_json=json.dumps(saved_patient.get("safetyFlags", [])),
            missing_fields_json=json.dumps(saved_patient.get("missingFields", [])),
        )

    logger.info("New patient %s registered & saved to DB -- AI ESI: %d, Confidence: %.0f%%",
                patient_id, saved_patient["aiSuggestedPriority"],
                saved_patient["aiConfidenceScore"] * 100)

    return saved_patient

@app.post("/api/accept/{patient_id}")
async def accept_recommendation(patient_id: str):
    """Accept the AI recommendation -- updates SQLite patient & creates immutable audit log."""
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found in database")

    if patient["status"] != "AWAITING_REVIEW":
        raise HTTPException(status_code=400, detail="Patient already reviewed")

                                     
    updated = db_update_patient_review(
        patient_id=patient_id,
        status="REVIEWED_ACCEPTED",
        nurse_esi=patient["aiSuggestedPriority"],
        override_reason=None
    )

                                                             
    audit_record = log_decision(
        patient_id=patient_id,
        ai_esi=patient["aiSuggestedPriority"],
        ai_confidence=patient["aiConfidenceScore"],
        nurse_esi=patient["aiSuggestedPriority"],
        override_reason=None,
        action_type="ACCEPT",
        patient_age=patient.get("age"),
        patient_gender=patient.get("biologicalSex"),
        chief_complaint_scrubbed=patient.get("_scrubbed_complaint"),
        escalation_flags=json.dumps({
            "age_safety": patient.get("_ageSafetyTriggered", False),
            "confidence_escalated": patient.get("_confidenceEscalated", False),
        }),
    )
    db_add_safety_event(
        patient_id, "AI_RECOMMENDATION_ACCEPTED",
        previous_state=patient.get("lifecycleStatus"), current_state="IN_TREATMENT",
        previous_triage_level=patient["aiSuggestedPriority"], new_triage_level=patient["aiSuggestedPriority"],
        confidence=patient["aiConfidenceScore"], reason_code="ACCEPTED",
        clinician_reason="Clinician accepted the AI recommendation", actor="clinician",
        safety_flags_json=json.dumps(patient.get("safetyFlags", [])),
        missing_fields_json=json.dumps(patient.get("missingFields", [])),
        recommendation="Clinician decision recorded",
    )

    logger.info("ACCEPT: %s -- ESI %d (confidence: %.0f%%) logged in SQLite",
                patient_id, patient["aiSuggestedPriority"],
                patient["aiConfidenceScore"] * 100)

    return {
        "status": "accepted",
        "patient_id": patient_id,
        "esi": patient["aiSuggestedPriority"],
        "audit_record": audit_record,
        "patient": updated,
    }

@app.post("/api/override/{patient_id}")
async def override_recommendation(patient_id: str, req: OverrideRequest):
    """Override the AI recommendation -- updates SQLite patient & creates immutable audit log with reason."""
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found in database")

    if patient["status"] != "AWAITING_REVIEW":
        raise HTTPException(status_code=400, detail="Patient already reviewed")

    if not 1 <= req.nurse_esi <= 5:
        raise HTTPException(status_code=400, detail="ESI must be between 1 and 5")

                                     
    updated = db_update_patient_review(
        patient_id=patient_id,
        status="REVIEWED_OVERRIDDEN",
        nurse_esi=req.nurse_esi,
        override_reason=req.override_reason
    )

                                                             
    audit_record = log_decision(
        patient_id=patient_id,
        ai_esi=patient["aiSuggestedPriority"],
        ai_confidence=patient["aiConfidenceScore"],
        nurse_esi=req.nurse_esi,
        override_reason=req.override_reason,
        action_type="OVERRIDE",
        patient_age=patient.get("age"),
        patient_gender=patient.get("biologicalSex"),
        chief_complaint_scrubbed=patient.get("_scrubbed_complaint"),
        escalation_flags=json.dumps({
            "age_safety": patient.get("_ageSafetyTriggered", False),
            "confidence_escalated": patient.get("_confidenceEscalated", False),
        }),
    )
    db_add_safety_event(
        patient_id, "AI_RECOMMENDATION_OVERRIDDEN",
        previous_state=patient.get("lifecycleStatus"), current_state="IN_TREATMENT",
        previous_triage_level=patient["aiSuggestedPriority"], new_triage_level=req.nurse_esi,
        confidence=patient["aiConfidenceScore"], reason_code=req.reason_code or req.override_reason,
        clinician_reason=req.clinician_reason or req.override_reason, actor="clinician",
        safety_flags_json=json.dumps(patient.get("safetyFlags", [])),
        missing_fields_json=json.dumps(patient.get("missingFields", [])),
        recommendation="Clinician decision recorded",
    )

    logger.info("OVERRIDE: %s -- AI ESI %d -> Nurse ESI %d (reason: %s) logged in SQLite",
                patient_id, patient["aiSuggestedPriority"],
                req.nurse_esi, req.override_reason)

    return {
        "status": "overridden",
        "patient_id": patient_id,
        "ai_esi": patient["aiSuggestedPriority"],
        "nurse_esi": req.nurse_esi,
        "reason": req.override_reason,
        "audit_record": audit_record,
        "patient": updated,
    }

@app.post("/api/surge")
async def simulate_surge():
    """Simulate a 3x volume surge by generating and saving 15 triaged patients to SQLite."""
    global SURGE_COUNTER

    surge_patients = generate_surge_patients(count=15, start_id=SURGE_COUNTER)
    SURGE_COUNTER += 15

    saved_surge_patients = []
    for raw in surge_patients:
        assessed = _run_triage_on_raw_patient(raw)
        saved = db_save_patient(assessed)
        saved_surge_patients.append(saved)

    total_in_db = db_count_patients()
    logger.info("SURGE: Saved %d new surge patients to SQLite (Total queue: %d)",
                len(saved_surge_patients), total_in_db)

    return {
        "status": "surge_complete",
        "injected_count": len(saved_surge_patients),
        "total_queue_size": total_in_db,
        "patients": saved_surge_patients,
    }

@app.get("/api/audit-log")
async def get_audit_log(limit: int = 50):
    """Retrieve recent compliance audit log entries from SQLite."""
    logs = get_recent_logs(limit=limit)
    stats = get_audit_stats()
    safety_events = db_get_all_safety_events(limit=limit)
    return {
        "logs": logs,
        "events": safety_events,
        "stats": stats,
    }

@app.get("/api/stats")
async def get_queue_stats():
    """Get queue summary statistics directly from SQLite."""
    patients_list = db_get_all_patients()
    total = len(patients_list)
    awaiting = sum(1 for p in patients_list if p.get("status") == "AWAITING_REVIEW")

    priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for p in patients_list:
        esi = p.get("nurseAssignedPriority") if p.get("nurseAssignedPriority") is not None else p.get("aiSuggestedPriority", 3)
        if esi in priority_counts:
            priority_counts[esi] += 1

    wait_times = []
    for p in patients_list:
        if p.get("status") == "AWAITING_REVIEW":
            try:
                arrival = datetime.fromisoformat(p["arrivalTime"])
                wait = (datetime.now() - arrival).total_seconds() / 60
                wait_times.append(wait)
            except Exception:
                pass

    avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0

    return {
        "total_patients": total,
        "awaiting_review": awaiting,
        "reviewed": total - awaiting,
        "priority_counts": priority_counts,
        "avg_wait_minutes": round(avg_wait, 1),
    }

@app.get("/api/trace/{patient_id}")
async def get_patient_trace(patient_id: str):
    """Get the LangGraph pipeline trace for a patient from SQLite."""
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "patient_id": patient_id,
        "trace": patient.get("_trace", []),
        "ml_prediction": patient.get("_mlPrediction", {}),
        "nlp_analysis": patient.get("_nlpAnalysis", {}),
        "age_safety_triggered": patient.get("_ageSafetyTriggered", False),
        "age_safety_reason": patient.get("_ageSafetyReason"),
        "confidence_escalated": patient.get("_confidenceEscalated", False),
        "missing_data_penalty": patient.get("_missingDataPenalty", 0),
        "nlp_ambiguity_penalty": patient.get("_nlpAmbiguityPenalty", 0),
    }

@app.get("/api/patients/{patient_id}/audit")
async def get_patient_audit_history(patient_id: str):
    """Retrieve full chronological audit and safety event history for a specific patient."""
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    events = db_get_safety_events(patient_id, limit=100)
                                                    
    events_chronological = sorted(events, key=lambda e: (e.get("timestamp", ""), e.get("id", 0)))

    return {
        "patient_id": patient_id,
        "count": len(events_chronological),
        "events": events_chronological,
    }

@app.get("/api/patients/{patient_id}/explanation")
async def get_patient_decision_explanation(patient_id: str):
    """Expose structured breakdown explaining why the final triage level was assigned."""
    patient = db_get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    ml_pred = patient.get("_mlPrediction", {})
    model_esi = ml_pred.get("predicted_esi", 3)
    safety_triggered = bool(patient.get("_ageSafetyTriggered", False))
    safety_reason = patient.get("_ageSafetyReason")
    confidence_score = float(patient.get("aiConfidenceScore", 0.5))
    confidence_escalated = bool(patient.get("_confidenceEscalated", False))
    missing_penalty = float(patient.get("_missingDataPenalty", 0.0))
    nlp_ambiguity = float(patient.get("_nlpAmbiguityPenalty", 0.0))
    safety_flags = patient.get("safetyFlags", [])
    missing_fields = patient.get("missingFields", [])
    deterioration = bool(patient.get("deteriorationDetected", False))
    final_esi = int(patient.get("aiSuggestedPriority", 3))

    return {
        "patient_id": patient_id,
        "model_recommendation": {
            "predicted_esi": model_esi,
            "confidence_margin": ml_pred.get("confidence_margin", 0.0),
            "probabilities": ml_pred.get("probabilities", []),
            "method": ml_pred.get("method", "xgboost"),
        },
        "deterministic_safety_adjustment": {
            "triggered": safety_triggered or "CRITICAL_SAFETY_RULE" in safety_flags or "AGE_OR_VITAL_SAFETY_RULE" in safety_flags,
            "reason": safety_reason,
            "safety_flags": safety_flags,
        },
        "confidence_adjustment": {
            "score": round(confidence_score, 4),
            "confidence_band": "HIGH" if confidence_score >= 0.80 else "MEDIUM" if confidence_score >= 0.55 else "LOW",
            "escalated": confidence_escalated,
            "nlp_ambiguity_penalty": round(nlp_ambiguity, 4),
        },
        "missing_data_adjustment": {
            "missing_fields": missing_fields,
            "penalty": round(missing_penalty, 4),
            "detected": len(missing_fields) > 0 or "MISSING_DATA" in safety_flags,
        },
        "age_adjustment": {
            "age": patient.get("age"),
            "age_group": patient.get("ageGroup", "adult"),
            "adjusted_ruleset_used": "PEDIATRIC_AGE_ADJUSTED" in safety_flags or "GERIATRIC_AGE_ADJUSTED" in safety_flags,
        },
        "vital_deterioration": {
            "detected": deterioration,
            "reassessment_required": bool(patient.get("reassessmentRequired", False)),
            "reassessment_reason": patient.get("reassessmentReason"),
        },
        "final_decision": {
            "final_esi": final_esi,
            "status": patient.get("status"),
            "nurse_assigned_priority": patient.get("nurseAssignedPriority"),
            "override_reason": patient.get("overrideReason"),
        },
    }

@app.get("/api/metrics")
async def get_system_metrics():
    """Observability and system health monitoring metrics."""
    patients_list = db_get_all_patients()
    safety_events = db_get_all_safety_events(limit=500)
    audit_stats = get_audit_stats()

    total_patients = len(patients_list)
    total_decisions = audit_stats.get("total_decisions", 0)

    priority_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    low_conf_count = 0
    safety_rule_count = 0
    missing_data_count = 0
    zero_history_count = 0

    for p in patients_list:
        esi = p.get("aiSuggestedPriority", 3)
        if esi in priority_counts:
            priority_counts[esi] += 1
        conf = p.get("aiConfidenceScore", 1.0)
        if conf < 0.60:
            low_conf_count += 1
        if p.get("_ageSafetyTriggered") or "CRITICAL_SAFETY_RULE" in p.get("safetyFlags", []) or "AGE_OR_VITAL_SAFETY_RULE" in p.get("safetyFlags", []):
            safety_rule_count += 1
        if p.get("missingFields") or "MISSING_DATA" in p.get("safetyFlags", []):
            missing_data_count += 1
        if p.get("history_available") is False or "ZERO_HISTORY" in p.get("safetyFlags", []):
            zero_history_count += 1

    event_type_counts = {}
    for ev in safety_events:
        etype = ev.get("event_type", "UNKNOWN")
        event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

    return {
        "total_patients_processed": total_patients,
        "total_triage_decisions": total_decisions,
        "priority_distribution": priority_counts,
        "low_confidence_decisions": low_conf_count,
        "safety_rule_triggers": safety_rule_count,
        "deterioration_events": event_type_counts.get("VITAL_DETERIORATION", 0),
        "reassessment_events": (
            event_type_counts.get("REASSESSMENT_REQUIRED", 0) +
            event_type_counts.get("WAIT_THRESHOLD_EXCEEDED", 0) +
            event_type_counts.get("REASSESSMENT_REQUESTED", 0)
        ),
        "clinician_acceptances": audit_stats.get("accepts", 0),
        "clinician_overrides": audit_stats.get("overrides", 0),
        "missing_data_cases": missing_data_count,
        "zero_history_cases": zero_history_count,
        "event_breakdown": event_type_counts,
    }

@app.post("/api/reset")
@app.delete("/api/patients")
async def reset_patient_queue():
    """Clear all patients from SQLite database to start fresh with an empty queue."""
    global PATIENT_COUNTER, SURGE_COUNTER
    db_clear_all_patients()
    PATIENT_COUNTER = 5001
    SURGE_COUNTER = 6001
    logger.info("QUEUE RESET: All patients cleared from SQLite database")
    return {
        "status": "cleared",
        "message": "Patient queue cleared successfully. Starting with empty database.",
        "total_patients": 0
    }

@app.post("/api/seed")
async def seed_demo_patients():
    """Seed the 20 realistic mock patients from patients.json into SQLite database."""
    patients_file = os.path.join(os.path.dirname(__file__), "patients.json")
    if os.path.exists(patients_file):
        with open(patients_file, "r", encoding="utf-8") as f:
            raw_patients = json.load(f)
        for raw in raw_patients:
            assessed = _run_triage_on_raw_patient(raw)
            db_save_patient(assessed)
        return {
            "status": "seeded",
            "count": len(raw_patients),
            "total_patients": db_count_patients()
        }
    raise HTTPException(status_code=404, detail="patients.json not found")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)