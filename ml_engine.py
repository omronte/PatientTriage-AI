"""
ml_engine.py — ML Models, NLP Agent & PHI Scrubbing for PatientTriage.ai

Components:
  1. XGBoost classifier trained inline on synthetic data (no .pkl needed)
  2. NLP agent targeting Ollama/Llama-3 with graceful mock fallback
  3. PHI scrubber using Presidio with regex fallback
"""

import re
import json
import random
import logging
import numpy as np

logger = logging.getLogger("ml_engine")

_PRESIDIO_ANALYZER = None
_PRESIDIO_ANONYMIZER = None

                                                                             
                                                 
                                                                             

def generate_synthetic_data(n_samples: int = 500, seed: int = 42):
    """
    Generate realistic synthetic ED patient data for training.
    Features: age, heart_rate, respiratory_rate, o2_saturation,
              systolic_bp, temperature, gcs_score, has_bp (0/1)
    Target:   ESI level (1-5)
    """
    rng = np.random.RandomState(seed)

    ages = rng.randint(1, 95, size=n_samples)
    heart_rates = rng.normal(85, 20, size=n_samples).clip(40, 200).astype(int)
    resp_rates = rng.normal(18, 5, size=n_samples).clip(8, 45).astype(int)
    o2_sats = rng.normal(96, 3, size=n_samples).clip(70, 100).astype(int)
    systolic_bps = rng.normal(125, 20, size=n_samples).clip(60, 220).astype(int)
    temps = rng.normal(37.0, 0.8, size=n_samples).clip(34.0, 42.0).round(1)
    gcs_scores = rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                            size=n_samples,
                            p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.1, 0.15, 0.55])
    has_bp = rng.choice([0, 1], size=n_samples, p=[0.1, 0.9])

                                                 
    esi_labels = np.full(n_samples, 3, dtype=int)

    for i in range(n_samples):
        score = 0.0

                           
        if gcs_scores[i] <= 8:
            score += 4.0
        elif gcs_scores[i] <= 12:
            score += 2.5
        elif gcs_scores[i] <= 14:
            score += 1.0

                       
        if o2_sats[i] < 88:
            score += 3.5
        elif o2_sats[i] < 92:
            score += 2.0
        elif o2_sats[i] < 95:
            score += 1.0

                    
        if heart_rates[i] > 130 or heart_rates[i] < 50:
            score += 2.5
        elif heart_rates[i] > 110 or heart_rates[i] < 55:
            score += 1.5
        elif heart_rates[i] > 100:
            score += 0.5

                          
        if resp_rates[i] > 30 or resp_rates[i] < 10:
            score += 2.0
        elif resp_rates[i] > 24:
            score += 1.0

                        
        if systolic_bps[i] < 80 or systolic_bps[i] > 190:
            score += 2.5
        elif systolic_bps[i] < 90 or systolic_bps[i] > 170:
            score += 1.5

                     
        if temps[i] > 40.0 or temps[i] < 35.0:
            score += 2.0
        elif temps[i] > 39.0 or temps[i] < 35.5:
            score += 1.0
        elif temps[i] > 38.0:
            score += 0.5

                             
        if ages[i] < 5 or ages[i] > 75:
            score *= 1.3
        elif ages[i] < 12 or ages[i] > 65:
            score *= 1.15

                   
        score += rng.normal(0, 0.5)

                          
        if score >= 6.0:
            esi_labels[i] = 1
        elif score >= 4.0:
            esi_labels[i] = 2
        elif score >= 2.0:
            esi_labels[i] = 3
        elif score >= 1.0:
            esi_labels[i] = 4
        else:
            esi_labels[i] = 5

    X = np.column_stack([
        ages, heart_rates, resp_rates, o2_sats,
        systolic_bps, temps, gcs_scores, has_bp
    ])

    feature_names = [
        "age", "heart_rate", "respiratory_rate", "o2_saturation",
        "systolic_bp", "temperature", "gcs_score", "has_bp"
    ]

    return X, esi_labels, feature_names

def train_model():
    """
    Train an XGBoost classifier inline on synthetic data.
    Returns the trained model.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed — using rule-based fallback for predictions")
        return None

    X, y, feature_names = generate_synthetic_data()

                                       
    y_adjusted = y - 1                 

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y_adjusted)
    logger.info("XGBoost model trained on %d synthetic samples", len(X))
    return model

def predict_triage(model, patient: dict) -> dict:
    """
    Predict ESI level and return probabilities.

    Args:
        model: trained XGBoost model (or None for rule-based fallback)
        patient: dict with vitals keys

    Returns:
        {
            "predicted_esi": int (1-5),
            "probabilities": list of 5 floats,
            "confidence_margin": float (gap between top two classes),
            "method": "xgboost" or "rule_based"
        }
    """
                                                                       
                                                                         
                                                                       
                                                                        
                                                                        
                                                                        
                                                                           
                                                                          
                                         
    def _val(key, default, cast_fn=float):
        value = patient.get(key)
        if value is None:
            return default
        try:
            val = cast_fn(value)
            if np.isnan(val) or np.isinf(val):
                return default
            return val
        except (ValueError, TypeError):
            return default

                      
    age = _val("age", 40)
    heart_rate = _val("heart_rate", 80)
    respiratory_rate = _val("respiratory_rate", 16)
    o2_saturation = _val("oxygen_saturation", 98)
    temperature = _val("temperature", 37.0)
    gcs_score = _val("gcs_score", 15)

                          
    bp = patient.get("blood_pressure")
    if bp and isinstance(bp, str) and "/" in bp:
        try:
            systolic_bp = int(float(bp.split("/")[0]))
            has_bp = 1
        except (ValueError, IndexError):
            systolic_bp = 120
            has_bp = 0
    elif bp is None:
        systolic_bp = 120
        has_bp = 0
    else:
        systolic_bp = 120
        has_bp = 0

    features = np.array([[age, heart_rate, respiratory_rate, o2_saturation,
                          systolic_bp, temperature, gcs_score, has_bp]])

    if model is not None:
        try:
            probas = model.predict_proba(features)[0]
            predicted_class = int(np.argmax(probas))
            predicted_esi = predicted_class + 1               

            sorted_probs = sorted(probas, reverse=True)
            confidence_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

            return {
                "predicted_esi": predicted_esi,
                "probabilities": [round(float(p), 4) for p in probas],
                "confidence_margin": round(confidence_margin, 4),
                "method": "xgboost",
            }
        except Exception as e:
            logger.error("XGBoost prediction failed: %s — falling back to rules", e)

                         
    return _rule_based_prediction(patient, features[0])

def _rule_based_prediction(patient: dict, features: np.ndarray) -> dict:
    """Deterministic rule-based fallback if XGBoost fails."""
    age, hr, rr, o2, sbp, temp, gcs, has_bp_flag = features

    score = 0.0
    if gcs <= 8:
        score += 4.0
    elif gcs <= 12:
        score += 2.5
    if o2 < 88:
        score += 3.5
    elif o2 < 92:
        score += 2.0
    if hr > 130 or hr < 50:
        score += 2.5
    elif hr > 110:
        score += 1.5
    if rr > 30 or rr < 10:
        score += 2.0
    if sbp < 80 or sbp > 190:
        score += 2.5
    if temp > 40.0 or temp < 35.0:
        score += 2.0
    elif temp > 39.0:
        score += 1.0

    if score >= 6:
        esi = 1
    elif score >= 4:
        esi = 2
    elif score >= 2:
        esi = 3
    elif score >= 1:
        esi = 4
    else:
        esi = 5

                        
    probs = [0.05, 0.05, 0.05, 0.05, 0.05]
    probs[esi - 1] = 0.70
    remaining = 0.10
    for i in range(5):
        if i != esi - 1:
            probs[i] = remaining / 4

    return {
        "predicted_esi": esi,
        "probabilities": probs,
        "confidence_margin": 0.60,
        "method": "rule_based",
    }

                                                                             
                                                  
                                                                             

_OLLAMA_AVAILABLE = None
_LAST_OLLAMA_CHECK = 0.0

def _is_ollama_online() -> bool:
    """Fast non-blocking check if Ollama service is responsive."""
    global _OLLAMA_AVAILABLE, _LAST_OLLAMA_CHECK
    import time
    now = time.time()
                                                                        
    if _OLLAMA_AVAILABLE is not None and (now - _LAST_OLLAMA_CHECK) < 20.0:
        return _OLLAMA_AVAILABLE
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=0.3)
        _OLLAMA_AVAILABLE = (resp.status_code == 200)
    except Exception:
        _OLLAMA_AVAILABLE = False
    _LAST_OLLAMA_CHECK = now
    return _OLLAMA_AVAILABLE

CLINICAL_RED_FLAGS = {
    "chest pain": "CRITICAL",
    "crushing": "CRITICAL",
    "radiating": "CRITICAL",
    "jaw pain": "CRITICAL",
    "slurred speech": "CRITICAL",
    "weakness": "CRITICAL",
    "difficulty breathing": "CRITICAL",
    "stridor": "CRITICAL",
    "unresponsive": "CRITICAL",
    "seizure": "CRITICAL",
    "syncope": "CRITICAL",
    "hemoptysis": "CRITICAL",
    "blood-tinged sputum": "CRITICAL",
    "altered mental status": "CRITICAL",
    "confusion": "CRITICAL",
    "sudden onset": "CRITICAL",
    "allergic reaction": "CRITICAL",
    "anaphylaxis": "CRITICAL",
    "swollen tongue": "CRITICAL",
    "lips are swollen": "CRITICAL",
    "unable to breathe": "CRITICAL",
    "diaphoretic": "WARNING",
    "nausea": "WARNING",
    "vomiting": "WARNING",
    "fever": "WARNING",
    "high fever": "WARNING",
    "weight loss": "WARNING",
    "night sweats": "WARNING",
    "deformity": "WARNING",
    "swelling": "WARNING",
    "unable to bear weight": "WARNING",
    "blood": "WARNING",
    "bleeding": "WARNING",
    "palpitations": "WARNING",
    "lightheadedness": "WARNING",
    "dizziness": "WARNING",
    "headache": "WARNING",
    "rash": "WARNING",
    "decreased oral intake": "WARNING",
    "lethargy": "WARNING",
    "lethargic": "WARNING",
    "poor feeding": "WARNING",
    "shortness of breath": "WARNING",
    "worsening": "WARNING",
    "warfarin": "WARNING",
    "anticoagulant": "WARNING",
    "fall": "WARNING",
    "dark stools": "CRITICAL",
    "chest tightness": "WARNING",
    "ear pain": "WARNING",
    "abdominal": "WARNING",
    "dehydration": "WARNING",
    "wheezing": "WARNING",
    "asthma": "WARNING",
}

def extract_nlp(chief_complaint: str) -> dict:
    """
    Analyze the chief complaint using clinical LLM / NLP.

    Extracts red flags, symptoms, suspected conditions, dynamic clinical
    reasoning per alert, NLP ambiguity score, and clinical summary.
    """
    if _is_ollama_online():
        try:
            import requests
            prompt = f"""You are an advanced emergency department clinical AI triage specialist.
Analyze the following patient's chief complaint:
"{chief_complaint}"

Return a valid JSON object with the following fields:
1. "red_flags": list of high-risk keywords/phrases indicating life-threatening emergency
2. "symptoms": list of specific clinical symptoms
3. "urgency_cues": list of temporal or severity cues (e.g. "sudden onset", "worsening")
4. "suspected_conditions": list of probable clinical diagnoses
5. "risk_level": "CRITICAL", "HIGH", "MODERATE", or "LOW"
6. "alert_reasoning": object mapping each extracted red_flag or symptom to a context-rich clinical reasoning sentence explaining why it is significant
7. "nlp_ambiguity_score": float between 0.0 (very clear) and 1.0 (highly ambiguous/atypical)
8. "clinical_summary": a 2-sentence clinical triage report summarizing the findings

Respond ONLY with raw valid JSON."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=5,
            )

            if response.status_code == 200:
                result_text = response.json().get("response", "")
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info("NLP analysis via Ollama/Llama-3 succeeded")
                    return {
                        "red_flags": parsed.get("red_flags", []),
                        "symptoms": parsed.get("symptoms", []),
                        "urgency_cues": parsed.get("urgency_cues", []),
                        "suspected_conditions": parsed.get("suspected_conditions", []),
                        "risk_level": parsed.get("risk_level", "MODERATE"),
                        "alert_reasoning": parsed.get("alert_reasoning", {}),
                        "nlp_ambiguity_score": float(parsed.get("nlp_ambiguity_score", 0.0)),
                        "clinical_summary": parsed.get("clinical_summary", ""),
                        "method": "ollama_llama3",
                    }
        except Exception as e:
            logger.info("Ollama execution failed (%s) -- falling back to dynamic keyword NLP", e)

                                              
    return _keyword_nlp_fallback(chief_complaint)

def _keyword_nlp_fallback(chief_complaint: str) -> dict:
    """
    Keyword-based NLP extraction as fallback.
    Scans for known clinical red flags and symptoms.
    """
    text_lower = chief_complaint.lower()

    red_flags = []
    symptoms = []
    urgency_cues = []

    for keyword, severity in CLINICAL_RED_FLAGS.items():
        if keyword in text_lower:
            if severity == "CRITICAL":
                red_flags.append(keyword)
            else:
                symptoms.append(keyword)

                  
    urgency_phrases = [
        "sudden onset", "progressively worsening", "unable to",
        "difficulty", "severe", "acute", "worst", "uncontrolled",
        "increasing", "rapidly"
    ]
    for phrase in urgency_phrases:
        if phrase in text_lower:
            urgency_cues.append(phrase)

                          
    if len(red_flags) >= 2:
        risk_level = "CRITICAL"
    elif len(red_flags) >= 1:
        risk_level = "HIGH"
    elif len(symptoms) >= 3:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

                                                        
    suspected = _infer_conditions(text_lower)

                                                                 
    ambiguous_tokens = ["vague", "feeling off", "strange", "unwell", "unclear", "unsure", "incomplete", "scratch", "tiny", "mild", "fine"]
    if any(tok in text_lower for tok in ambiguous_tokens):
        nlp_ambiguity_score = 0.85
    elif "Undifferentiated presentation" in suspected and not red_flags and not symptoms:
        nlp_ambiguity_score = 0.70
    elif len(red_flags) == 0 and len(symptoms) == 0:
        nlp_ambiguity_score = 0.50
    elif len(red_flags) >= 1 and len(urgency_cues) >= 1:
        nlp_ambiguity_score = 0.05
    else:
        nlp_ambiguity_score = 0.20

    return {
        "red_flags": red_flags,
        "symptoms": symptoms,
        "urgency_cues": urgency_cues,
        "suspected_conditions": suspected,
        "risk_level": risk_level,
        "nlp_ambiguity_score": nlp_ambiguity_score,
        "method": "keyword_fallback",
    }

def _infer_conditions(text: str) -> list:
    """Infer suspected conditions from keyword patterns."""
    conditions = []

    if "chest pain" in text and ("jaw" in text or "arm" in text or "crushing" in text):
        conditions.append("Acute Coronary Syndrome (ACS)")
    if "slurred speech" in text or ("weakness" in text and "sudden" in text):
        conditions.append("Acute Stroke / CVA")
    if "right lower quadrant" in text and ("pain" in text or "nausea" in text):
        conditions.append("Acute Appendicitis")
    if "hip" in text and ("fall" in text or "deformity" in text):
        conditions.append("Hip Fracture")
    if "allergic" in text or "anaphylaxis" in text or ("swollen" in text and "tongue" in text):
        conditions.append("Anaphylaxis")
    if "shortness of breath" in text and ("ankle swelling" in text or "lie flat" in text):
        conditions.append("Acute Heart Failure Exacerbation")
    if "cough" in text and "blood" in text and "weight loss" in text:
        conditions.append("Suspected Lung Malignancy / TB")
    if ("fever" in text or "high fever" in text) and ("rash" in text or "lethargi" in text):
        conditions.append("Febrile Illness — rule out serious bacterial infection")
    if "palpitations" in text:
        conditions.append("Cardiac Arrhythmia")
    if "burning" in text and "urination" in text:
        conditions.append("Urinary Tract Infection")
    if "headache" in text or "migraine" in text:
        conditions.append("Migraine / Primary Headache")
    if "back pain" in text and "radiat" in text:
        conditions.append("Lumbar Radiculopathy")
    if "wrist" in text and ("deformity" in text or "fell" in text or "fall" in text):
        conditions.append("Distal Radius Fracture")
    if "ankle" in text and ("twist" in text or "sprain" in text):
        conditions.append("Ankle Sprain")
    if "dizz" in text and "faint" in text:
        conditions.append("Presyncope — etiology unclear")
    if "jaw pain" in text and "nausea" in text:
        conditions.append("Atypical ACS presentation (esp. in females)")

    if not conditions:
        conditions.append("Undifferentiated presentation — further workup needed")

    return conditions

                                                                             
                                                
                                                                             

def scrub_phi(text: str) -> str:
    """
    Remove Protected Health Information (names, phone numbers, etc.)
    from text before NLP processing.

    Uses regex scrubbing combined with Microsoft Presidio for defense in depth.
    """
    global _PRESIDIO_ANALYZER, _PRESIDIO_ANONYMIZER

    if not isinstance(text, str):
        return ""

                                           
    scrubbed = _regex_phi_scrub(text)

                                                                                   
    try:
                                          
        from presidio_analyzer import AnalyzerEngine
                                          
        from presidio_analyzer.nlp_engine import NlpEngineProvider
                                          
        from presidio_anonymizer import AnonymizerEngine

        if _PRESIDIO_ANALYZER is None:
            nlp_provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            })
            _PRESIDIO_ANALYZER = AnalyzerEngine(
                nlp_engine=nlp_provider.create_engine(),
                supported_languages=["en"],
            )
            _PRESIDIO_ANONYMIZER = AnonymizerEngine()

        results = _PRESIDIO_ANALYZER.analyze(
            text=scrubbed,
            entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "MEDICAL_LICENSE", "LOCATION", "DATE_TIME"],
            language="en",
        )

        anonymized = _PRESIDIO_ANONYMIZER.anonymize(text=scrubbed, analyzer_results=results)
        logger.info("PHI scrubbed: %d additional entities found via Presidio", len(results))
        return anonymized.text

    except ImportError:
        return scrubbed
    except Exception as e:
        logger.warning("Presidio error (%s) -- returning regex scrubbed text", e)
        return scrubbed

def _regex_phi_scrub(text: str) -> str:
    """Regex-based fallback for PHI removal."""
    if not isinstance(text, str):
        return ""

                                                                   
    text = re.sub(r'\b(?:Patient(?:\s+Name)?|Name)\s*:\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '[REDACTED_NAME]', text, flags=re.IGNORECASE)
                                                                                            
    text = re.sub(r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '[REDACTED_NAME]', text)
                                                                       
    text = re.sub(r'\b(?:DOB|Date of Birth)[:\s]+\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b', '[REDACTED_DATE]', text, flags=re.IGNORECASE)
                                                                                
    text = re.sub(r'(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', text)
         
    text = re.sub(r'\b(?:\d{3}-\d{2}-\d{4}|SSN[:\s]*\d{9})\b', '[REDACTED_SSN]', text, flags=re.IGNORECASE)
           
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', '[REDACTED_EMAIL]', text)
                                             
    text = re.sub(r'\bMRN[:\s#]*\d{5,}\b', '[REDACTED_MRN]', text, flags=re.IGNORECASE)
                                                                                      
    text = re.sub(r"\b\d+\s+[A-Za-z0-9\s,.'-]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Terrace|Ter|Place|Pl|Circle|Cir|Highway|Hwy|Parkway|Pkwy)\b", '[REDACTED_ADDRESS]', text, flags=re.IGNORECASE)

    return text