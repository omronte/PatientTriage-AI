from backend.langgraph_orchestrator import run_triage_graph

def run_case(patient_data):
    return run_triage_graph({
        "patient_data": patient_data,
        "nlp_risk_score": 4,
        "vitals_risk_score": 4,
    })

def test_minor_uses_age_adjusted_ruleset_and_escalates():
    result = run_case({
        "age": 8,
        "heart_rate": 145,
        "bp_sys": 90,
        "o2_sat": 97,
        "temp_c": 39.5,
        "chief_complaint": "fever and lethargy",
        "history_availability": "rich",
    })

    assert result["ruleset_used"] == "age_adjusted"
    assert result["age_group"] == "pediatric"
    assert result["safety_override"] is True
    assert result["final_triage_level"] <= 2

def test_older_adult_uses_age_adjusted_ruleset_and_catches_fall():
    result = run_case({
        "age": 75,
        "heart_rate": 85,
        "bp_sys": 130,
        "o2_sat": 95,
        "temp_c": 37.0,
        "chief_complaint": "Had a fall at home",
        "history_availability": "partial",
    })

    assert result["ruleset_used"] == "age_adjusted"
    assert result["age_group"] == "geriatric"
    assert result["safety_override"] is True
    assert "fall" in result["override_reason"].lower()

def test_adult_without_history_lowers_confidence_and_escalates():
    result = run_case({
        "age": 30,
        "heart_rate": 80,
        "bp_sys": 120,
        "o2_sat": 99,
        "temp_c": 37.0,
        "chief_complaint": "Sprained wrist",
        "history_availability": "none",
    })

    assert result["ruleset_used"] == "standard"
    assert result["age_group"] == "adult"
    assert result["history_availability"] == "none"
    assert 0.0 <= result["confidence_score"] <= 0.85
    assert result["final_triage_level"] <= 3
    assert "history" in result["explanation_summary"].lower()

def test_missing_vitals_do_not_crash_age_routing():
    result = run_case({
        "age": 10,
        "chief_complaint": "Unknown complaint",
        "history_availability": "none",
    })

    assert result["ruleset_used"] == "age_adjusted"
    assert "history" in result["history_availability"] or result["history_availability"] == "none"
    assert result["missing_fields"]
    assert 0.0 <= result["confidence_score"] <= 1.0
