from workflow import build_and_run_pipeline

def fake_predictor(model, patient):
    return {
        "predicted_esi": 4,
        "confidence_margin": 0.9,
        "method": "test",
    }

def fake_nlp(_complaint):
    return {
        "red_flags": [],
        "symptoms": [],
        "risk_level": "LOW",
        "nlp_ambiguity_score": 0.0,
    }

def run_case(patient):
    return build_and_run_pipeline(
        patient=patient,
        model=None,
        phi_scrubber=lambda complaint: complaint,
        predictor=fake_predictor,
        nlp_extractor=fake_nlp,
    )

def test_active_workflow_routes_age_adjusted_patients():
    result = run_case({
        "age": 8,
        "heart_rate": 80,
        "respiratory_rate": 16,
        "oxygen_saturation": 98,
        "temperature": 37.0,
        "blood_pressure": "100/60",
        "gcs_score": 15,
        "chief_complaint": "Feeling well",
        "history_available": True,
    })

    assert result["ruleset_used"] == "age_adjusted"

def test_active_workflow_escalates_without_history():
    result = run_case({
        "age": 30,
        "heart_rate": 80,
        "respiratory_rate": 16,
        "oxygen_saturation": 98,
        "temperature": 37.0,
        "blood_pressure": "120/80",
        "gcs_score": 15,
        "chief_complaint": "Sprained wrist",
        "history_available": False,
    })

    assert result["ruleset_used"] == "standard"
    assert result["final_confidence"] < 1.0
    assert result["history_escalated"] is True
    assert result["final_esi"] == 3
