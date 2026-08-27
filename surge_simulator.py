"""
surge_simulator.py — Surge Mode Patient Generator for PatientTriage.ai

Generates realistic random patients to simulate a 3× volume surge.
Each patient is assigned varied demographics, vitals, and complaints.
"""

import random
import string
from datetime import datetime, timedelta

                                                                             
                                                  
                                                                             

SURGE_COMPLAINTS = [
    {
        "complaint": "Chest tightness and shortness of breath after climbing stairs. No prior cardiac history. Currently mildly diaphoretic.",
        "acuity_hint": "high",
    },
    {
        "complaint": "Motor vehicle accident, low speed. Complains of neck pain and mild headache. Was wearing seatbelt. No loss of consciousness.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Severe headache described as the worst headache of life. Sudden onset 20 minutes ago. Photophobia present.",
        "acuity_hint": "critical",
    },
    {
        "complaint": "Laceration to left hand after broken glass. Bleeding controlled with pressure dressing. Wound approximately 3cm.",
        "acuity_hint": "low",
    },
    {
        "complaint": "Abdominal pain with cramping for past 12 hours. Two episodes of vomiting. Unable to keep food down.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Child fell from playground equipment onto outstretched hand. Swelling and pain in left forearm. No open wound.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Difficulty breathing for 2 hours. Known asthmatic. Used rescue inhaler 3 times with minimal relief. Wheezing audible.",
        "acuity_hint": "high",
    },
    {
        "complaint": "Diabetic patient found confused at home. Blood glucose reading of 42 mg/dL. Given orange juice by family en route.",
        "acuity_hint": "high",
    },
    {
        "complaint": "Sore throat and fever for 3 days. Difficulty swallowing solid food. No breathing difficulty. Taking paracetamol.",
        "acuity_hint": "low",
    },
    {
        "complaint": "Fell in bathroom. Pain in left hip. Unable to stand. No head injury. Alert and oriented.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Eye injury from metal grinding fragment. Sensation of foreign body. Tearing and redness. Wearing safety goggles at time.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Insect bite on right arm 6 hours ago. Progressively worsening swelling. Now redness extending up arm. Low grade fever.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Recurrent nosebleed for past 45 minutes. Unable to stop with pinching. On aspirin for cardiac condition.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Anxiety attack. Hyperventilating. Tingling in fingers. Reports feeling of impending doom. No chest pain.",
        "acuity_hint": "low",
    },
    {
        "complaint": "Severe back pain after lifting heavy object. Pain radiates to right leg. Numbness in foot reported.",
        "acuity_hint": "medium",
    },
    {
        "complaint": "Uncontrolled nosebleed with dizziness. Patient appears pale. History of liver cirrhosis and coagulopathy.",
        "acuity_hint": "high",
    },
    {
        "complaint": "Toddler ingested unknown quantity of grandmother's blood pressure medication approximately 30 minutes ago.",
        "acuity_hint": "critical",
    },
    {
        "complaint": "Pregnant, 32 weeks. Persistent headache, blurred vision, and right upper quadrant pain. Blood pressure elevated.",
        "acuity_hint": "critical",
    },
    {
        "complaint": "Burn injury to both forearms from hot oil splash while cooking. Partial thickness, approximately 10% TBSA.",
        "acuity_hint": "high",
    },
    {
        "complaint": "Ear pain for 2 days. Mild hearing reduction in left ear. No discharge. No fever. Taking over-the-counter pain relief.",
        "acuity_hint": "low",
    },
]

FIRST_NAMES_M = ["Amit", "Rohan", "Karan", "Nikhil", "Suraj", "Vivek", "Manish", "Rahul", "Dev", "Sameer"]
FIRST_NAMES_F = ["Nisha", "Pooja", "Ritu", "Divya", "Swati", "Anjali", "Komal", "Isha", "Tanya", "Megha"]
LAST_NAMES = ["Sharma", "Patel", "Singh", "Gupta", "Kumar", "Reddy", "Jain", "Verma", "Malhotra", "Bhat"]

def _generate_vitals(acuity_hint: str) -> dict:
    """Generate vitals appropriate to acuity level."""
    if acuity_hint == "critical":
        return {
            "heart_rate": random.choice(range(120, 165)),
            "respiratory_rate": random.choice(range(26, 40)),
            "oxygen_saturation": random.choice(range(82, 92)),
            "blood_pressure": f"{random.choice(range(75, 95))}/{random.choice(range(40, 60))}",
            "temperature": round(random.uniform(38.5, 40.5), 1),
            "gcs_score": random.choice(range(6, 14)),
        }
    elif acuity_hint == "high":
        return {
            "heart_rate": random.choice(range(100, 135)),
            "respiratory_rate": random.choice(range(22, 30)),
            "oxygen_saturation": random.choice(range(90, 96)),
            "blood_pressure": f"{random.choice(range(90, 165))}/{random.choice(range(55, 95))}",
            "temperature": round(random.uniform(37.5, 39.5), 1),
            "gcs_score": random.choice(range(12, 16)),
        }
    elif acuity_hint == "medium":
        return {
            "heart_rate": random.choice(range(78, 110)),
            "respiratory_rate": random.choice(range(16, 24)),
            "oxygen_saturation": random.choice(range(95, 100)),
            "blood_pressure": f"{random.choice(range(110, 150))}/{random.choice(range(65, 90))}",
            "temperature": round(random.uniform(36.5, 38.5), 1),
            "gcs_score": 15,
        }
    else:       
        return {
            "heart_rate": random.choice(range(65, 90)),
            "respiratory_rate": random.choice(range(14, 18)),
            "oxygen_saturation": random.choice(range(97, 100)),
            "blood_pressure": f"{random.choice(range(110, 135))}/{random.choice(range(65, 82))}",
            "temperature": round(random.uniform(36.4, 37.5), 1),
            "gcs_score": 15,
        }

def generate_surge_patients(count: int = 15, start_id: int = 6001) -> list:
    """
    Generate a batch of realistic random patients for surge simulation.

    Args:
        count: number of patients to generate (default 15 for 3× volume)
        start_id: starting patient ID number

    Returns:
        list of patient dicts in the standard schema
    """
    patients = []
    complaint_pool = list(SURGE_COMPLAINTS)

    for i in range(count):
                                                                          
        template = complaint_pool[i % len(complaint_pool)]

                             
        gender = random.choice(["M", "F"])
        if gender == "M":
            name = f"{random.choice(FIRST_NAMES_M)} {random.choice(LAST_NAMES)}"
        else:
            name = f"{random.choice(FIRST_NAMES_F)} {random.choice(LAST_NAMES)}"

                                                             
        age = random.choices(
            population=[random.randint(1, 10), random.randint(11, 17),
                        random.randint(18, 45), random.randint(46, 65),
                        random.randint(66, 90)],
            weights=[0.10, 0.08, 0.42, 0.25, 0.15],
            k=1
        )[0]

        vitals = _generate_vitals(template["acuity_hint"])

                                                       
        wait_time = random.randint(0, 20)

        patient = {
            "id": f"PT-{start_id + i}",
            "name": name,
            "age": age,
            "gender": gender,
            "temperature": vitals["temperature"],
            "heart_rate": vitals["heart_rate"],
            "respiratory_rate": vitals["respiratory_rate"],
            "oxygen_saturation": vitals["oxygen_saturation"],
            "blood_pressure": vitals["blood_pressure"],
            "chief_complaint": template["complaint"],
            "history_available": random.random() > 0.2,
            "wait_time_minutes": wait_time,
        }
        patients.append(patient)

    return patients
