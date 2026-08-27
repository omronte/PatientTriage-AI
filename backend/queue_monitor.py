"""Backend-owned waiting queue monitoring for the Phase 2 prototype."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
from typing import Any

from backend.database import PatientRecord, SafetyEvent, SessionLocal, patient_model_to_dict

logger = logging.getLogger("queue_monitor")
DEFAULT_WAIT_THRESHOLDS = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}
WAIT_TIME_THRESHOLDS = DEFAULT_WAIT_THRESHOLDS.copy()
ACTIVE_LIFECYCLE_STATES = {"WAITING", "REASSESSMENT_REQUIRED", "REASSESSED"}

def detect_vital_deterioration(previous: dict, current: dict) -> dict[str, list[Any]]:
    """Return clinically significant demo vital changes for reassessment."""
    changes = {}
    if previous.get("o2SaturationPercent") is not None and current.get("o2SaturationPercent") is not None and current["o2SaturationPercent"] <= previous["o2SaturationPercent"] - 3:
        changes["oxygen_saturation"] = [previous["o2SaturationPercent"], current["o2SaturationPercent"]]
    if previous.get("heartRateBpm") is not None and current.get("heartRateBpm") is not None and current["heartRateBpm"] - previous["heartRateBpm"] >= 25:
        changes["heart_rate"] = [previous["heartRateBpm"], current["heartRateBpm"]]
    if previous.get("gcsScore") is not None and current.get("gcsScore") is not None and current["gcsScore"] < previous["gcsScore"]:
        changes["gcs_score"] = [previous["gcsScore"], current["gcsScore"]]
    return changes

def get_wait_thresholds() -> dict[int, int]:
    """Read hospital-specific demo thresholds from WAIT_THRESHOLDS_JSON."""
    raw = os.getenv("WAIT_THRESHOLDS_JSON")
    if not raw:
        return DEFAULT_WAIT_THRESHOLDS.copy()
    try:
        parsed = json.loads(raw)
        return {**DEFAULT_WAIT_THRESHOLDS, **{int(k): max(0, int(v)) for k, v in parsed.items()}}
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Invalid WAIT_THRESHOLDS_JSON; using prototype defaults")
        return DEFAULT_WAIT_THRESHOLDS.copy()

def _effective_level(record: PatientRecord) -> int:
    return int(record.last_queue_priority or record.nurse_assigned_priority or record.ai_suggested_priority or 3)

def calculate_queue_priority(triage_level: int, *, safety_flags=None, reassessment_required=False,
                             deterioration_detected=False, confidence_score=1.0, waiting_minutes=0):
    """Return a sortable key where safety severity always beats waiting time."""
    safety_rank = 0 if triage_level <= 1 or deterioration_detected or safety_flags else 1
    return (safety_rank, triage_level, 0 if reassessment_required else 1,
            0 if confidence_score < 0.6 else 1, -waiting_minutes)

def _waiting_minutes(record: PatientRecord, now: dt.datetime) -> float:
    started = record.waiting_started_at or record.created_at
    if record.waiting_started_at is None and record.arrival_time:
        try:
            started = dt.datetime.fromisoformat(record.arrival_time.replace("Z", "+00:00"))
        except ValueError:
            pass
    if started is None:
        return 0.0
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    if started.tzinfo is not None:
        started = started.replace(tzinfo=None)
    return max(0.0, (now - started).total_seconds() / 60)

def monitor_once(now: dt.datetime | None = None) -> list[dict[str, Any]]:
    """Scan active waiting records and create one event per threshold breach."""
    now = now or dt.datetime.now(dt.timezone.utc)
    generated = []
    session = SessionLocal()
    try:
        records = session.query(PatientRecord).filter(
            PatientRecord.lifecycle_status.in_(ACTIVE_LIFECYCLE_STATES),
            PatientRecord.status == "AWAITING_REVIEW",
        ).all()
        thresholds = get_wait_thresholds()
        for record in records:
            level = _effective_level(record)
            threshold = thresholds.get(level, DEFAULT_WAIT_THRESHOLDS[5])
            waiting = _waiting_minutes(record, now)
            exceeded = waiting >= threshold
            record.wait_threshold_minutes = threshold
            record.wait_threshold_exceeded = exceeded
            record.last_queue_priority = level
            if exceeded and not record.reassessment_required:
                previous = record.lifecycle_status
                reason = f"Safe waiting threshold exceeded for triage level {level}"
                record.lifecycle_status = "REASSESSMENT_REQUIRED"
                record.reassessment_status = "REQUIRED"
                record.reassessment_required = True
                record.reassessment_reason = reason
                record.last_safety_event_type = "WAIT_THRESHOLD_EXCEEDED"
                record.last_safety_event_at = now
                session.add(SafetyEvent(
                    patient_id=record.patient_id, event_type="WAIT_THRESHOLD_EXCEEDED", timestamp=now,
                    previous_state=previous, current_state=record.lifecycle_status, trigger_reason=reason,
                    waiting_minutes=waiting, threshold_minutes=threshold,
                    recommendation="Clinical reassessment required",
                ))
                generated.append({"patient_id": record.patient_id, "event_type": "WAIT_THRESHOLD_EXCEEDED",
                                  "waiting_minutes": round(waiting, 1), "threshold_minutes": threshold,
                                  "recommendation": "Clinical reassessment required"})
        session.commit()
        return generated
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

async def check_waiting_queue(interval_seconds: int = 60):
    """Run lightweight monitoring until the application shuts down."""
    while True:
        try:
            monitor_once()
        except Exception:
            logger.exception("Waiting queue scan failed")
        await asyncio.sleep(interval_seconds)

monitor_loop = check_waiting_queue

def get_queue_projection() -> list[dict[str, Any]]:
    """Return the authoritative backend-sorted active waiting queue."""
    monitor_once()
    session = SessionLocal()
    try:
        records = session.query(PatientRecord).filter(
            PatientRecord.lifecycle_status.in_(ACTIVE_LIFECYCLE_STATES),
            PatientRecord.status == "AWAITING_REVIEW",
        ).all()
        projection = []
        for record in records:
            patient = patient_model_to_dict(record)
            patient["queuePriorityKey"] = list(calculate_queue_priority(
                patient["lastQueuePriority"], safety_flags=patient.get("safetyFlags"),
                reassessment_required=patient["reassessmentRequired"],
                deterioration_detected=patient["deteriorationDetected"],
                confidence_score=patient["aiConfidenceScore"], waiting_minutes=patient["waitingMinutes"]))
            projection.append(patient)
        projection.sort(key=lambda item: item["queuePriorityKey"])
        return projection
    finally:
        session.close()

def acknowledge_reassessment(patient_id: str) -> dict[str, Any] | None:
    session = SessionLocal()
    try:
        record = session.query(PatientRecord).filter_by(patient_id=patient_id).first()
        if not record:
            return None
        now = dt.datetime.now(dt.timezone.utc)
        record.reassessment_status = "ACKNOWLEDGED"
        record.reassessment_required = False
        record.lifecycle_status = "REASSESSED"
        record.reassessment_count = (record.reassessment_count or 0) + 1
        record.last_assessment_at = now
        session.add(SafetyEvent(
            patient_id=patient_id, event_type="REASSESSMENT_ACKNOWLEDGED", timestamp=now,
            previous_state="REASSESSMENT_REQUIRED", current_state="REASSESSED",
            trigger_reason=record.reassessment_reason or "Clinician acknowledged reassessment",
            recommendation="Continue clinician-led monitoring",
            actor="clinician",
        ))
        session.commit()
        session.refresh(record)
        return patient_model_to_dict(record)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()