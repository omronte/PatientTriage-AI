"""
database.py — SQLite Database & Audit Trail for PatientTriage.ai
Uses SQLAlchemy with SQLite for:
  1. Persistent Patient Records (all registered, seeded, & surge patients)
  2. Immutable Compliance Audit Logs (every accept/override decision)
"""

import os
import json
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, Boolean, inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker

                                                                             
                
                                                                             
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "triage_audit.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def utc_now():
    """Return timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

                                                                             
        
                                                                             

class PatientRecord(Base):
    """Persistent database record for every patient in the ED triage queue."""
    __tablename__ = "patients"

    patient_id = Column(String(20), primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    arrival_time = Column(String(40), nullable=False)
    chief_complaint = Column(Text, nullable=False)

            
    heart_rate = Column(Float, nullable=True)
    blood_pressure_sys = Column(Integer, nullable=True)
    blood_pressure_dia = Column(Integer, nullable=True)
    oxygen_saturation = Column(Float, nullable=True)
    respiratory_rate = Column(Float, nullable=True)
    temperature = Column(Float, nullable=True)
    gcs_score = Column(Float, default=15)
    history_available = Column(Boolean, default=True)

                                   
    ai_suggested_priority = Column(Integer, nullable=False)
    ai_confidence_score = Column(Float, nullable=False)
    explainability_json = Column(Text, nullable=True)
    trace_json = Column(Text, nullable=True)
    ml_prediction_json = Column(Text, nullable=True)
    nlp_analysis_json = Column(Text, nullable=True)
    age_safety_triggered = Column(Boolean, default=False)
    age_safety_reason = Column(String(255), nullable=True)
    confidence_escalated = Column(Boolean, default=False)
    missing_data_penalty = Column(Float, default=0.0)
    nlp_ambiguity_penalty = Column(Float, default=0.0)
    safety_flags_json = Column(Text, nullable=True)
    missing_fields_json = Column(Text, nullable=True)

                   
    status = Column(String(30), default="AWAITING_REVIEW", nullable=False)
    nurse_assigned_priority = Column(Integer, nullable=True)
    override_reason = Column(String(100), nullable=True)

                                                      
    lifecycle_status = Column(String(30), default="WAITING", nullable=False)
    waiting_started_at = Column(DateTime, nullable=True)
    initial_triage_at = Column(DateTime, nullable=True)
    last_assessment_at = Column(DateTime, nullable=True)
    last_vitals_at = Column(DateTime, nullable=True)
    reassessment_status = Column(String(30), default="NOT_REQUIRED", nullable=False)
    reassessment_required = Column(Boolean, default=False, nullable=False)
    reassessment_reason = Column(String(255), nullable=True)
    reassessment_count = Column(Integer, default=0, nullable=False)
    wait_threshold_minutes = Column(Integer, nullable=True)
    wait_threshold_exceeded = Column(Boolean, default=False, nullable=False)
    deterioration_detected = Column(Boolean, default=False, nullable=False)
    last_queue_priority = Column(Integer, nullable=True)
    last_safety_event_type = Column(String(50), nullable=True)
    last_safety_event_at = Column(DateTime, nullable=True)

                
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

class TriageLog(Base):
    """Immutable audit record for every clinician accept/override decision."""
    __tablename__ = "triage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False)
    patient_id = Column(String(20), nullable=False, index=True)
    patient_age = Column(Integer, nullable=True)
    patient_gender = Column(String(10), nullable=True)
    chief_complaint_scrubbed = Column(Text, nullable=True)
    ai_esi = Column(Integer, nullable=False)
    ai_confidence = Column(Float, nullable=False)
    nurse_esi = Column(Integer, nullable=True)
    override_reason = Column(String(100), nullable=True)
    action_type = Column(String(20), nullable=False)                          
    escalation_flags = Column(Text, nullable=True)                        

class SafetyEvent(Base):
    """Structured, idempotent record of automatic queue safety events."""
    __tablename__ = "safety_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(20), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False)
    previous_state = Column(String(30), nullable=True)
    current_state = Column(String(30), nullable=True)
    trigger_reason = Column(String(255), nullable=False)
    waiting_minutes = Column(Float, nullable=True)
    threshold_minutes = Column(Integer, nullable=True)
    vital_changes_json = Column(Text, nullable=True)
    recommendation = Column(String(255), nullable=False)
    previous_triage_level = Column(Integer, nullable=True)
    new_triage_level = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    reason_code = Column(String(60), nullable=True)
    clinician_reason = Column(String(500), nullable=True)
    actor = Column(String(100), nullable=True)
    safety_flags_json = Column(Text, nullable=True)
    missing_fields_json = Column(Text, nullable=True)

                                                                             
                         
                                                                             

def init_db():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)
                                                                             
                                                                        
    columns = {
        "lifecycle_status": "VARCHAR(30) NOT NULL DEFAULT 'WAITING'",
        "waiting_started_at": "DATETIME",
        "initial_triage_at": "DATETIME",
        "last_assessment_at": "DATETIME",
        "last_vitals_at": "DATETIME",
        "reassessment_status": "VARCHAR(30) NOT NULL DEFAULT 'NOT_REQUIRED'",
        "reassessment_required": "BOOLEAN NOT NULL DEFAULT 0",
        "reassessment_reason": "VARCHAR(255)",
        "reassessment_count": "INTEGER NOT NULL DEFAULT 0",
        "wait_threshold_minutes": "INTEGER",
        "wait_threshold_exceeded": "BOOLEAN NOT NULL DEFAULT 0",
        "deterioration_detected": "BOOLEAN NOT NULL DEFAULT 0",
        "last_queue_priority": "INTEGER",
        "last_safety_event_type": "VARCHAR(50)",
        "last_safety_event_at": "DATETIME",
        "safety_flags_json": "TEXT",
        "missing_fields_json": "TEXT",
    }
    existing = {column["name"] for column in inspect(engine).get_columns("patients")}
    event_columns = {
        "previous_triage_level": "INTEGER",
        "new_triage_level": "INTEGER",
        "confidence": "FLOAT",
        "reason_code": "VARCHAR(60)",
        "clinician_reason": "VARCHAR(500)",
        "actor": "VARCHAR(100)",
        "safety_flags_json": "TEXT",
        "missing_fields_json": "TEXT",
    }
    existing_event_columns = {column["name"] for column in inspect(engine).get_columns("safety_events")}
    with engine.begin() as connection:
        for name, definition in columns.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE patients ADD COLUMN {name} {definition}"))
        for name, definition in event_columns.items():
            if name not in existing_event_columns:
                connection.execute(text(f"ALTER TABLE safety_events ADD COLUMN {name} {definition}"))

                                                                             
                                                
                                                                             

def patient_model_to_dict(p: PatientRecord) -> dict:
    """Convert database PatientRecord instance to frontend/API schema dictionary."""
    vitals = {
        "heartRateBpm": p.heart_rate,
        "bloodPressureSys": p.blood_pressure_sys,
        "bloodPressureDia": p.blood_pressure_dia,
        "o2SaturationPercent": p.oxygen_saturation,
        "respiratoryRate": p.respiratory_rate,
        "temperatureCelsius": p.temperature,
        "gcsScore": p.gcs_score if p.gcs_score is not None else 15,
    }

    try:
        explainability = json.loads(p.explainability_json) if p.explainability_json else []
    except Exception:
        explainability = []

    try:
        trace = json.loads(p.trace_json) if p.trace_json else []
    except Exception:
        trace = []

    try:
        ml_prediction = json.loads(p.ml_prediction_json) if p.ml_prediction_json else {}
    except Exception:
        ml_prediction = {}

    try:
        nlp_analysis = json.loads(p.nlp_analysis_json) if p.nlp_analysis_json else {}
    except Exception:
        nlp_analysis = {}

    if p.age is None:
        age_group = "adult"
    elif p.age < 18:
        age_group = "pediatric"
    elif p.age <= 65:
        age_group = "adult"
    else:
        age_group = "geriatric"

    history_availability = "rich" if p.history_available is True else "none"
    if p.history_available is None:
        history_availability = "partial"

    confidence_score = float(p.ai_confidence_score) if p.ai_confidence_score is not None else 0.5
    confidence_label = "HIGH" if confidence_score >= 0.8 else "MEDIUM" if confidence_score >= 0.55 else "LOW"
    try:
        safety_flags = json.loads(p.safety_flags_json) if p.safety_flags_json else []
    except Exception:
        safety_flags = []
    try:
        missing_fields = json.loads(p.missing_fields_json) if p.missing_fields_json else []
    except Exception:
        missing_fields = []
    waiting_started = p.waiting_started_at or p.arrival_time
    now = datetime.datetime.now(datetime.timezone.utc)
    if isinstance(waiting_started, str):
        try:
            waiting_started_dt = datetime.datetime.fromisoformat(waiting_started)
        except ValueError:
            waiting_started_dt = now
    else:
        waiting_started_dt = waiting_started or now
    if waiting_started_dt.tzinfo is None:
        waiting_started_dt = waiting_started_dt.replace(tzinfo=datetime.timezone.utc)
    waiting_minutes = max(0.0, (now - waiting_started_dt).total_seconds() / 60)
    lifecycle_status = p.lifecycle_status or ("WAITING" if p.status == "AWAITING_REVIEW" else p.status)
    queue_priority = p.last_queue_priority or p.nurse_assigned_priority or p.ai_suggested_priority

    fallback_flags = [
        "history_unavailable" if not p.history_available else "history_available",
        age_group + "_safety_profile" if age_group in {"pediatric", "geriatric"} else "adult_safety_profile",
    ]

    return {
        "id": p.patient_id,
        "patientId": p.patient_id,
        "name": p.name or "Unknown Patient",
        "arrivalTime": p.arrival_time,
        "age": p.age,
        "ageGroup": age_group,
        "biologicalSex": p.gender,
        "chiefComplaint": p.chief_complaint,
        "vitals": vitals,
        "historyAvailability": history_availability,
        "history_available": p.history_available if p.history_available is not None else False,
        "riskCategory": "critical" if p.ai_suggested_priority <= 2 else "moderate" if p.ai_suggested_priority == 3 else "low",
        "aiSuggestedPriority": p.ai_suggested_priority,
        "triageLevel": queue_priority,
        "priority": queue_priority,
        "aiConfidenceScore": p.ai_confidence_score,
        "confidenceLabel": confidence_label,
        "confidence_label": confidence_label,
        "safetyFlags": safety_flags or fallback_flags,
        "safety_flags": safety_flags or fallback_flags,
        "missingFields": missing_fields,
        "missing_fields": missing_fields,
        "explainability": explainability,
        "explanation": explainability,
        "status": p.status,
        "lifecycleStatus": lifecycle_status,
        "waitingStartedAt": waiting_started.isoformat() if hasattr(waiting_started, "isoformat") else waiting_started,
        "initialTriageAt": p.initial_triage_at.isoformat() if p.initial_triage_at else None,
        "lastAssessmentAt": p.last_assessment_at.isoformat() if p.last_assessment_at else None,
        "lastVitalsAt": p.last_vitals_at.isoformat() if p.last_vitals_at else None,
        "waitingMinutes": round(waiting_minutes, 1),
        "waitThresholdMinutes": p.wait_threshold_minutes,
        "waitThresholdExceeded": bool(p.wait_threshold_exceeded),
        "reassessmentStatus": p.reassessment_status or "NOT_REQUIRED",
        "reassessmentRequired": bool(p.reassessment_required),
        "reassessmentReason": p.reassessment_reason,
        "reassessmentCount": p.reassessment_count or 0,
        "deteriorationDetected": bool(p.deterioration_detected),
        "lastQueuePriority": queue_priority,
        "lastSafetyEventType": p.last_safety_event_type,
        "lastSafetyEventAt": p.last_safety_event_at.isoformat() if p.last_safety_event_at else None,
        "recommendedAction": "Clinical reassessment required" if p.reassessment_required else "Continue monitoring",
        "nurseAssignedPriority": p.nurse_assigned_priority,
        "overrideReason": p.override_reason,
        "_mlPrediction": ml_prediction,
        "_nlpAnalysis": nlp_analysis,
        "_ageSafetyTriggered": p.age_safety_triggered,
        "_ageSafetyReason": p.age_safety_reason,
        "_confidenceEscalated": p.confidence_escalated,
        "_missingDataPenalty": p.missing_data_penalty,
        "_nlpAmbiguityPenalty": p.nlp_ambiguity_penalty,
        "_trace": trace,
    }

def db_save_patient(patient_dict: dict) -> dict:
    """Insert or update a patient in SQLite database."""
    session = SessionLocal()
    try:
        pid = patient_dict.get("patientId") or patient_dict.get("id")
        existing = session.query(PatientRecord).filter_by(patient_id=pid).first()

        vitals = patient_dict.get("vitals", {})

        explainability = patient_dict.get("explainability") or patient_dict.get("explanation", [])
        trace = patient_dict.get("_trace", patient_dict.get("trace", []))
        ml_pred = patient_dict.get("_mlPrediction", {})
        nlp_an = patient_dict.get("_nlpAnalysis", {})

        if not existing:
            record = PatientRecord(
                patient_id=pid,
                name=patient_dict.get("name"),
                age=patient_dict.get("age", 40),
                gender=patient_dict.get("biologicalSex") or patient_dict.get("gender", "Unknown"),
                arrival_time=patient_dict.get("arrivalTime") or utc_now().isoformat(),
                chief_complaint=patient_dict.get("chiefComplaint") or patient_dict.get("chief_complaint", ""),
                heart_rate=vitals.get("heartRateBpm") or patient_dict.get("heart_rate"),
                blood_pressure_sys=vitals.get("bloodPressureSys"),
                blood_pressure_dia=vitals.get("bloodPressureDia"),
                oxygen_saturation=vitals.get("o2SaturationPercent") or patient_dict.get("oxygen_saturation"),
                respiratory_rate=vitals.get("respiratoryRate") or patient_dict.get("respiratory_rate"),
                temperature=vitals.get("temperatureCelsius") or patient_dict.get("temperature"),
                gcs_score=vitals.get("gcsScore") or patient_dict.get("gcs_score", 15),
                history_available=patient_dict.get("history_available", True),
                ai_suggested_priority=patient_dict.get("aiSuggestedPriority") or patient_dict.get("ai_esi", 3),
                ai_confidence_score=patient_dict.get("aiConfidenceScore") or patient_dict.get("ai_confidence", 0.5),
                explainability_json=json.dumps(explainability),
                trace_json=json.dumps(trace),
                ml_prediction_json=json.dumps(ml_pred),
                nlp_analysis_json=json.dumps(nlp_an),
                age_safety_triggered=patient_dict.get("_ageSafetyTriggered", False),
                age_safety_reason=patient_dict.get("_ageSafetyReason"),
                confidence_escalated=patient_dict.get("_confidenceEscalated", False),
                missing_data_penalty=patient_dict.get("_missingDataPenalty", 0.0),
                nlp_ambiguity_penalty=patient_dict.get("_nlpAmbiguityPenalty", 0.0),
                safety_flags_json=json.dumps(patient_dict.get("safetyFlags", patient_dict.get("safety_flags", []))),
                missing_fields_json=json.dumps(patient_dict.get("missingFields", patient_dict.get("missing_fields", []))),
                status=patient_dict.get("status", "AWAITING_REVIEW"),
                nurse_assigned_priority=patient_dict.get("nurseAssignedPriority"),
                override_reason=patient_dict.get("overrideReason"),
                lifecycle_status=patient_dict.get("lifecycleStatus", "WAITING"),
                waiting_started_at=_parse_datetime(patient_dict.get("waitingStartedAt")) or _parse_datetime(patient_dict.get("arrivalTime")),
                initial_triage_at=_parse_datetime(patient_dict.get("initialTriageAt")) or _parse_datetime(patient_dict.get("arrivalTime")),
                last_assessment_at=_parse_datetime(patient_dict.get("lastAssessmentAt")) or _parse_datetime(patient_dict.get("arrivalTime")),
                last_vitals_at=_parse_datetime(patient_dict.get("lastVitalsAt")) or _parse_datetime(patient_dict.get("arrivalTime")),
                wait_threshold_minutes=patient_dict.get("waitThresholdMinutes"),
                last_queue_priority=patient_dict.get("lastQueuePriority") or patient_dict.get("aiSuggestedPriority") or 3,
            )
            session.add(record)
            session.add_all([
                SafetyEvent(
                    patient_id=pid,
                    event_type="PATIENT_TRIAGED",
                    previous_state="ARRIVED",
                    current_state="WAITING",
                    trigger_reason="Initial triage completed",
                    recommendation="Clinical review required",
                    previous_triage_level=None,
                    new_triage_level=record.ai_suggested_priority,
                    confidence=record.ai_confidence_score,
                    safety_flags_json=json.dumps(patient_dict.get("safetyFlags", patient_dict.get("safety_flags", []))),
                    missing_fields_json=json.dumps(patient_dict.get("missingFields", patient_dict.get("missing_fields", []))),
                    actor="system",
                ),
                SafetyEvent(
                    patient_id=pid,
                    event_type="AI_RECOMMENDATION_GENERATED",
                    previous_state="ARRIVED",
                    current_state="WAITING",
                    trigger_reason="Phase 1 triage pipeline produced a recommendation",
                    recommendation="Clinician review required",
                    new_triage_level=record.ai_suggested_priority,
                    confidence=record.ai_confidence_score,
                    safety_flags_json=json.dumps(patient_dict.get("safetyFlags", patient_dict.get("safety_flags", []))),
                    missing_fields_json=json.dumps(patient_dict.get("missingFields", patient_dict.get("missing_fields", []))),
                    actor="system",
                ),
            ])
        else:
            record = existing
            record.status = patient_dict.get("status", record.status)
            if "nurseAssignedPriority" in patient_dict:
                record.nurse_assigned_priority = patient_dict["nurseAssignedPriority"]
            if "overrideReason" in patient_dict:
                record.override_reason = patient_dict["overrideReason"]
            if "aiSuggestedPriority" in patient_dict:
                record.ai_suggested_priority = patient_dict["aiSuggestedPriority"]
            if "aiConfidenceScore" in patient_dict:
                record.ai_confidence_score = patient_dict["aiConfidenceScore"]
            record.explainability_json = json.dumps(explainability)
            record.trace_json = json.dumps(trace)
            record.safety_flags_json = json.dumps(patient_dict.get("safetyFlags", patient_dict.get("safety_flags", [])))
            record.missing_fields_json = json.dumps(patient_dict.get("missingFields", patient_dict.get("missing_fields", [])))
            if "deteriorationDetected" in patient_dict:
                record.deterioration_detected = bool(patient_dict["deteriorationDetected"])
            if vitals:
                record.heart_rate = vitals.get("heartRateBpm", record.heart_rate)
                record.blood_pressure_sys = vitals.get("bloodPressureSys", record.blood_pressure_sys)
                record.blood_pressure_dia = vitals.get("bloodPressureDia", record.blood_pressure_dia)
                record.oxygen_saturation = vitals.get("o2SaturationPercent", record.oxygen_saturation)
                record.respiratory_rate = vitals.get("respiratoryRate", record.respiratory_rate)
                record.temperature = vitals.get("temperatureCelsius", record.temperature)
                record.gcs_score = vitals.get("gcsScore", record.gcs_score)
            record.updated_at = utc_now()
            for field in ("lifecycleStatus", "waitThresholdMinutes", "lastQueuePriority", "lastAssessmentAt", "lastVitalsAt"):
                if field in patient_dict:
                    target = {
                        "lifecycleStatus": "lifecycle_status",
                        "waitThresholdMinutes": "wait_threshold_minutes",
                        "lastQueuePriority": "last_queue_priority",
                        "lastAssessmentAt": "last_assessment_at",
                        "lastVitalsAt": "last_vitals_at",
                    }[field]
                    value = patient_dict[field]
                    setattr(record, target, _parse_datetime(value) if target in {"last_assessment_at", "last_vitals_at"} else value)

        session.commit()
        session.refresh(record)
        return patient_model_to_dict(record)
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def _parse_datetime(value):
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        return None

def db_get_all_patients() -> list[dict]:
    """Retrieve all patients from the SQLite database."""
    session = SessionLocal()
    try:
        records = session.query(PatientRecord).all()
        return [patient_model_to_dict(r) for r in records]
    finally:
        session.close()

def db_get_patient(patient_id: str) -> dict | None:
    """Retrieve a single patient by ID from the database."""
    session = SessionLocal()
    try:
        record = session.query(PatientRecord).filter_by(patient_id=patient_id).first()
        return patient_model_to_dict(record) if record else None
    finally:
        session.close()

def db_update_patient_review(patient_id: str, status: str, nurse_esi: int | None, override_reason: str | None) -> dict | None:
    """Update patient status in SQLite when reviewed/overridden."""
    session = SessionLocal()
    try:
        record = session.query(PatientRecord).filter_by(patient_id=patient_id).first()
        if not record:
            return None
        record.status = status
        if status in {"REVIEWED_ACCEPTED", "REVIEWED_OVERRIDDEN"}:
            record.lifecycle_status = "IN_TREATMENT"
        record.nurse_assigned_priority = nurse_esi
        record.override_reason = override_reason
        record.updated_at = utc_now()
        session.commit()
        session.refresh(record)
        return patient_model_to_dict(record)
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def db_count_patients() -> int:
    """Count total patients stored in database."""
    session = SessionLocal()
    try:
        return session.query(PatientRecord).count()
    finally:
        session.close()

def db_clear_all_patients():
    """Clear all patient records from the database so queue starts fresh."""
    session = SessionLocal()
    try:
        session.query(PatientRecord).delete()
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def db_delete_patient(patient_id: str) -> bool:
    """Delete a single patient from the database."""
    session = SessionLocal()
    try:
        record = session.query(PatientRecord).filter_by(patient_id=patient_id).first()
        if record:
            session.delete(record)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

                                                                             
                        
                                                                             

def log_decision(
    patient_id: str,
    ai_esi: int,
    ai_confidence: float,
    nurse_esi: int,
    override_reason: str | None,
    action_type: str,
    patient_age: int | None = None,
    patient_gender: str | None = None,
    chief_complaint_scrubbed: str | None = None,
    escalation_flags: str | None = None,
):
    """Insert an immutable compliance audit record in SQLite."""
    session = SessionLocal()
    try:
        record = TriageLog(
            patient_id=patient_id,
            ai_esi=ai_esi,
            ai_confidence=ai_confidence,
            nurse_esi=nurse_esi,
            override_reason=override_reason,
            action_type=action_type,
            patient_age=patient_age,
            patient_gender=patient_gender,
            chief_complaint_scrubbed=chief_complaint_scrubbed,
            escalation_flags=escalation_flags,
        )
        session.add(record)
        session.commit()
        return {
            "id": record.id,
            "timestamp": record.timestamp.isoformat(),
            "patient_id": record.patient_id,
            "action_type": record.action_type,
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_recent_logs(limit: int = 50):
    """Retrieve recent audit log entries."""
    session = SessionLocal()
    try:
        rows = (
            session.query(TriageLog)
            .order_by(TriageLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        results = []
        for r in rows:
            results.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "patient_id": r.patient_id,
                "patient_age": r.patient_age,
                "patient_gender": r.patient_gender,
                "chief_complaint_scrubbed": r.chief_complaint_scrubbed,
                "ai_esi": r.ai_esi,
                "ai_confidence": round(r.ai_confidence, 3) if r.ai_confidence else None,
                "nurse_esi": r.nurse_esi,
                "override_reason": r.override_reason,
                "action_type": r.action_type,
                "escalation_flags": r.escalation_flags,
            })
        return results
    finally:
        session.close()

def get_audit_stats():
    """Return summary statistics from the audit log."""
    session = SessionLocal()
    try:
        total = session.query(TriageLog).count()
        accepts = session.query(TriageLog).filter(TriageLog.action_type == "ACCEPT").count()
        overrides = session.query(TriageLog).filter(TriageLog.action_type == "OVERRIDE").count()
        return {
            "total_decisions": total,
            "accepts": accepts,
            "overrides": overrides,
            "override_rate": round(overrides / total * 100, 1) if total > 0 else 0,
        }
    finally:
        session.close()

def db_update_queue_state(patient_id: str, **values) -> dict | None:
    """Update backend-owned waiting state without changing clinician review status."""
    session = SessionLocal()
    try:
        record = session.query(PatientRecord).filter_by(patient_id=patient_id).first()
        if not record:
            return None
        allowed = {
            "lifecycle_status", "reassessment_status", "reassessment_required", "reassessment_reason",
            "reassessment_count", "wait_threshold_minutes", "wait_threshold_exceeded",
            "deterioration_detected", "last_queue_priority", "last_safety_event_type", "last_safety_event_at",
            "last_assessment_at", "last_vitals_at", "ai_suggested_priority", "ai_confidence_score",
            "explainability_json", "trace_json", "ml_prediction_json", "nlp_analysis_json",
            "safety_flags_json", "missing_fields_json",
        }
        for key, value in values.items():
            if key in allowed:
                setattr(record, key, value)
        record.updated_at = utc_now()
        session.commit()
        session.refresh(record)
        return patient_model_to_dict(record)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def db_add_safety_event(patient_id: str, event_type: str, **values) -> dict:
    """Write a structured safety event, de-duplicating active event types."""
    session = SessionLocal()
    try:
        values.setdefault("trigger_reason", event_type.replace("_", " ").title())
        values.setdefault("recommendation", "Clinician review required")
        event = SafetyEvent(patient_id=patient_id, event_type=event_type,
                            timestamp=values.pop("timestamp", utc_now()),
                            **values)
        session.add(event)
        session.commit()
        return {"id": event.id, "duplicate": False, "event_type": event_type,
                "timestamp": event.timestamp.isoformat()}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def db_get_safety_events(patient_id: str, limit: int = 50) -> list[dict]:
    session = SessionLocal()
    try:
        rows = session.query(SafetyEvent).filter_by(patient_id=patient_id).order_by(SafetyEvent.timestamp.desc(), SafetyEvent.id.desc()).limit(limit).all()
        return [{"id": row.id, "patient_id": row.patient_id, "event_type": row.event_type,
                 "timestamp": row.timestamp.isoformat(), "previous_state": row.previous_state,
                 "current_state": row.current_state, "trigger_reason": row.trigger_reason,
                 "waiting_minutes": row.waiting_minutes, "threshold_minutes": row.threshold_minutes,
                 "recommendation": row.recommendation, "previous_triage_level": row.previous_triage_level,
                 "new_triage_level": row.new_triage_level, "confidence": row.confidence,
                 "reason_code": row.reason_code, "clinician_reason": row.clinician_reason,
                 "actor": row.actor,
                 "safety_flags": json.loads(row.safety_flags_json) if row.safety_flags_json else [],
                 "missing_fields": json.loads(row.missing_fields_json) if row.missing_fields_json else [],
                 "vital_changes": json.loads(row.vital_changes_json) if row.vital_changes_json else {}}
                for row in rows]
    finally:
        session.close()

def db_get_all_safety_events(limit: int = 100) -> list[dict]:
    """Return a compact chronological audit stream without complaint text."""
    session = SessionLocal()
    try:
        rows = session.query(SafetyEvent).order_by(SafetyEvent.timestamp.desc(), SafetyEvent.id.desc()).limit(limit).all()
        return [{
            "id": row.id,
            "patient_id": row.patient_id,
            "event_type": row.event_type,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "previous_state": row.previous_state,
            "current_state": row.current_state,
            "previous_triage_level": row.previous_triage_level,
            "new_triage_level": row.new_triage_level,
            "confidence": row.confidence,
            "reason_code": row.reason_code,
            "clinician_reason": row.clinician_reason,
            "actor": row.actor or "system",
            "trigger_reason": row.trigger_reason,
            "recommendation": row.recommendation,
        } for row in rows]
    finally:
        session.close()
