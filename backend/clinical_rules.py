
SAFETY_THRESHOLDS = {
    "pediatric": {
        "age_range": (0, 17),
        "heart_rate": {"min": 60, "max": 160},
        "systolic_bp": {"min": 70, "max": 120},
        "o2_sat": {"min": 92},
        "temp_c": {"max": 39.0},
        "critical_flags": ["lethargic", "not eating", "crying constantly", "fever", "vomiting", "seizure"]
    },
    "adult": {
        "age_range": (18, 64),
        "heart_rate": {"min": 50, "max": 120},
        "systolic_bp": {"min": 90, "max": 180},
        "o2_sat": {"min": 92},
        "temp_c": {"max": 39.5},
        "critical_flags": ["chest pain", "shortness of breath", "slurred speech", "weakness"]
    },
    "geriatric": {
        "age_range": (65, 120),
        "heart_rate": {"min": 50, "max": 100},
        "systolic_bp": {"min": 100, "max": 160},
        "o2_sat": {"min": 90},
        "temp_c": {"max": 37.8},
        "critical_flags": ["fall", "blood thinners", "dizzy", "confusion", "vague pain", "chest discomfort", "weakness"]
    }
}


def get_age_group(age):
    """Return the demo safety group used by the prototype triage policy."""
    if age is None:
        return "adult"
    if age < 18:
        return "pediatric"
    if age <= 65:
        return "adult"
    return "geriatric"


def evaluate_vitals(age, hr, sys_bp, o2, temp, text_cues):
    """Return a conservative override signal when age-specific vitals or red flags are concerning."""
    if age is None:
        category = "adult"
    elif age < 18:
        category = "pediatric"
    elif age <= 65:
        category = "adult"
    else:
        category = "geriatric"

    rules = SAFETY_THRESHOLDS[category]
    text = (text_cues or "").lower()

    if hr is not None and (hr < rules["heart_rate"]["min"] or hr > rules["heart_rate"]["max"]):
        return True, f"Critical HR for {category}"
    if sys_bp is not None and (sys_bp < rules["systolic_bp"]["min"] or sys_bp > rules["systolic_bp"]["max"]):
        return True, f"Critical BP for {category}"
    if o2 is not None and o2 < rules["o2_sat"]["min"]:
        return True, f"Critical O2 for {category}"
    if temp is not None and temp > rules["temp_c"]["max"]:
        return True, f"Critical Temp for {category}"

    for flag in rules["critical_flags"]:
        if flag in text:
            return True, f"High-risk symptom flagged for {category}: {flag}"

    return False, "Vitals within age-adjusted safe limits"