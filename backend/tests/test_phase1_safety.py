from backend.langgraph_orchestrator import run_triage_graph, apply_conservative_safety_policy

def test_pediatric_case_uses_age_adjusted_profile():
    result = run_triage_graph({
        "patient_data": {
            "age": 8,
            "heart_rate": 128,
            "bp_sys": 94,
            "o2_sat": 96,
            "temp_c": 38.5,
            "chief_complaint": "fever and lethargy",
            "history_available": True,
        },
        "nlp_risk_score": 3,
        "vitals_risk_score": 3,
    })

    assert result["age_group"] == "pediatric"
    assert "pediatric" in result["safety_flags"]
    assert result["confidence_label"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["final_triage_level"] in {1, 2, 3}

def test_geriatric_ambiguous_missing_history_is_conservative():
    result = run_triage_graph({
        "patient_data": {
            "age": 75,
            "heart_rate": 86,
            "bp_sys": 128,
            "o2_sat": 97,
            "temp_c": 36.9,
            "chief_complaint": "vague chest discomfort",
            "history_available": False,
        },
        "nlp_risk_score": 2,
        "vitals_risk_score": 2,
    })

    assert result["age_group"] == "geriatric"
    assert "geriatric_ambiguous_presentation" in result["safety_flags"]
    assert result["history_availability"] == "none"
    assert result["confidence_score"] <= 0.75
    assert result["final_triage_level"] <= 2

def test_missing_vitals_are_captured_and_not_treated_as_normal():
    result = run_triage_graph({
        "patient_data": {
            "age": 42,
            "chief_complaint": "shortness of breath",
            "history_available": "partial",
        },
        "nlp_risk_score": 4,
        "vitals_risk_score": 4,
    })

    assert "oxygen_saturation" in result["missing_fields"] or "o2_sat" in result["missing_fields"]
    assert "SpO2 unavailable" in " ".join(result["explanation"]).replace("  ", " ") or "missing" in " ".join(result["explanation"]).lower()

def test_conservative_policy_raises_critical_when_uncertain():
    decision = apply_conservative_safety_policy(
        age=12,
        age_group="pediatric",
        risk_score=2,
        confidence=0.45,
        missing_fields=["oxygen_saturation"],
        chief_complaint="fever and lethargy",
        safety_flags=[]
    )

    assert decision["escalate"] is True
    assert decision["triage_level_delta"] >= 1
    assert "pediatric" in decision["reason"].lower()
