                         

from backend.langgraph_orchestrator import clinical_safety_node, synthesis_node

def run_test(test_name, patient_data, expected_override, min_confidence):
    print(f"\n--- Running Test: {test_name} ---")
    state = {
        "patient_data": patient_data,
        "nlp_risk_score": 4,
        "vitals_risk_score": 4,
    }
    state = clinical_safety_node(state)
    state = synthesis_node(state)

    print(f"Safety Override Triggered: {state.get('safety_override')}")
    print(f"Final Triage Level: {state.get('final_triage_level')}")
    print(f"Confidence Score: {state.get('confidence_score')}")
    print(f"Synthesis Reasoning: {state.get('reasoning')}")

    assert state.get('safety_override') == expected_override, f"Failed override check for {test_name}"
    assert state.get('confidence_score') >= min_confidence, f"Failed confidence check for {test_name}"
    print(f"✅ {test_name} PASSED")

test_1_data = {
    "age": 8,
    "heart_rate": 145,
    "bp_sys": 90,
    "o2_sat": 97,
    "temp_c": 39.5,
    "chief_complaint": "fever and lethargy",
    "has_prior_history": True,
}
run_test("Pediatric Fever Override", test_1_data, expected_override=True, min_confidence=0.65)

test_2_data = {
    "age": 30,
    "heart_rate": 80,
    "bp_sys": 120,
    "o2_sat": 99,
    "temp_c": 37.0,
    "chief_complaint": "Sprained wrist",
    "has_prior_history": False,
}
run_test("Zero-History Escalation", test_2_data, expected_override=False, min_confidence=0.30)

test_3_data = {
    "age": 75,
    "heart_rate": 85,
    "bp_sys": 130,
    "o2_sat": 95,
    "temp_c": 37.0,
    "chief_complaint": "Had a fall at home",
    "has_prior_history": True,
}
run_test("Geriatric Keyword Override", test_3_data, expected_override=True, min_confidence=0.65)

print("\n🎉 All LangGraph routing logic verified successfully!")