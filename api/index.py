from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from flask import Flask, jsonify, request

ROOT = Path("/var/task")
MODEL_PATH = ROOT / "model" / "disease_model.pkl"
DESC_PATH = ROOT / "data" / "symptom_Description.csv"
PREC_PATH = ROOT / "data" / "symptom_precaution.csv"

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
model = data["model"]
le = data["label_encoder"]
all_symptoms = data["all_symptoms"]

desc_df = pd.read_csv(DESC_PATH)
prec_df = pd.read_csv(PREC_PATH)
desc_df.columns = desc_df.columns.str.strip()
prec_df.columns = prec_df.columns.str.strip()

ALIASES = {'fever': ['high_fever', 'mild_fever'], 'high fever': ['high_fever'], 'mild fever': ['mild_fever'], 'low grade fever': ['mild_fever'], 'feverish': ['mild_fever'], 'feel hot': ['mild_fever'], 'temperature': ['mild_fever'], 'cough': ['cough'], 'coughing': ['cough'], 'dry cough': ['cough'], 'wet cough': ['cough', 'phlegm'], 'phlegm': ['phlegm'], 'cold': ['runny_nose', 'chills'], 'flu': ['high_fever', 'chills', 'cough', 'muscle_pain', 'headache'], 'common cold': ['runny_nose', 'chills', 'cough', 'continuous_sneezing'], 'runny nose': ['runny_nose'], 'blocked nose': ['congestion'], 'sneezing': ['continuous_sneezing'], 'sore throat': ['throat_irritation'], 'throat pain': ['throat_irritation'], 'headache': ['headache'], 'head pain': ['headache'], 'migraine': ['headache'], 'body ache': ['muscle_pain', 'joint_pain'], 'body pain': ['muscle_pain', 'joint_pain'], 'whole body hurts': ['muscle_pain', 'joint_pain', 'back_pain'], 'pain everywhere': ['muscle_pain', 'joint_pain', 'back_pain'], 'muscle pain': ['muscle_pain'], 'joint pain': ['joint_pain'], 'back pain': ['back_pain'], 'neck pain': ['neck_pain'], 'stiff neck': ['stiff_neck'], 'chest pain': ['chest_pain'], 'stomach pain': ['stomach_pain'], 'stomach ache': ['stomach_pain'], 'tummy ache': ['stomach_pain'], 'fatigue': ['fatigue'], 'tired': ['fatigue'], 'exhausted': ['fatigue'], 'no energy': ['fatigue', 'muscle_weakness'], 'weakness': ['muscle_weakness'], 'weak': ['muscle_weakness'], 'feel weak': ['muscle_weakness'], 'lethargy': ['lethargy'], 'lethargic': ['lethargy'], 'drowsy': ['lethargy'], 'dizziness': ['dizziness'], 'dizzy': ['dizziness'], 'lightheaded': ['dizziness'], 'vertigo': ['dizziness'], 'nausea': ['nausea'], 'nauseous': ['nausea'], 'feel sick': ['nausea'], 'feel like vomiting': ['nausea', 'vomiting'], 'throwing up': ['vomiting'], 'vomiting': ['vomiting'], 'vomit': ['vomiting'], 'puking': ['vomiting'], 'diarrhea': ['diarrhoea'], 'diarrhoea': ['diarrhoea'], 'loose stools': ['diarrhoea'], 'loose motions': ['diarrhoea'], 'loose motion': ['diarrhoea'], 'constipation': ['constipation'], 'bloating': ['distention_of_abdomen'], 'indigestion': ['indigestion'], 'acidity': ['acidity'], 'heartburn': ['acidity'], 'no appetite': ['loss_of_appetite'], 'loss of appetite': ['loss_of_appetite'], 'breathlessness': ['breathlessness'], 'shortness of breath': ['breathlessness'], 'cant breathe': ['breathlessness'], 'difficulty breathing': ['breathlessness'], 'chest tightness': ['chest_pain', 'breathlessness'], 'palpitations': ['palpitations'], 'heart racing': ['palpitations'], 'red eyes': ['redness_of_eyes'], 'watery eyes': ['watering_from_eyes'], 'blurred vision': ['blurred_and_distorted_vision'], 'yellow eyes': ['yellowing_of_eyes'], 'rash': ['skin_rash'], 'skin rash': ['skin_rash'], 'itching': ['itching'], 'itchy': ['itching'], 'yellow skin': ['yellowish_skin'], 'jaundice': ['yellowish_skin', 'yellow_urine', 'yellowing_of_eyes'], 'swelling': ['swelled_lymph_nodes'], 'confused': ['altered_sensorium'], 'confusion': ['altered_sensorium'], 'feel confused': ['altered_sensorium'], 'brain fog': ['altered_sensorium'], 'anxiety': ['anxiety'], 'depression': ['depression'], 'mood swings': ['mood_swings'], 'chills': ['chills'], 'shivering': ['shivering', 'chills'], 'sweating': ['sweating'], 'night sweats': ['sweating'], 'burning urination': ['burning_micturition'], 'frequent urination': ['polyuria'], 'dark urine': ['dark_urine'], 'weight loss': ['weight_loss'], 'weight gain': ['weight_gain'], 'thirsty': ['excessive_hunger'], 'excessive thirst': ['excessive_hunger'], 'dry mouth': ['drying_and_tingling_lips'], 'cold hands': ['cold_hands_and_feets'], 'cold feet': ['cold_hands_and_feets'], 'coma': ['altered_sensorium'], 'unconscious': ['altered_sensorium'], 'slurred speech': ['slurred_speech'], 'coughing blood': ['blood_in_sputum']}
SEVERE = ['chest pain', 'cant breathe', 'breathlessness', 'shortness of breath', 'difficulty breathing', 'unconscious', 'coma', 'coughing blood', 'blood in vomit', 'slurred speech', 'heart racing', 'palpitations', 'high fever']
PREDICTION_STAGES = {3: {'show': 1, 'min_conf': 20}, 5: {'show': 2, 'min_conf': 15}, 7: {'show': 3, 'min_conf': 10}}

def get_stage(count):
    if count >= 7:
        return PREDICTION_STAGES[7]
    if count >= 5:
        return PREDICTION_STAGES[5]
    return PREDICTION_STAGES[3]

def match_symptoms(user_input):
    user_lower = user_input.lower().strip()
    found = set()
    fillers = [
        "i have been having", "i have been feeling", "i have been",
        "i am having", "i am feeling", "i am experiencing",
        "i feel like i have", "i feel like", "i've been",
        "i'm having", "i'm feeling", "i got", "i get",
        "i have", "i feel", "i am", "i've", "i'm",
        "suffering from", "experiencing", "having", "getting",
        "since yesterday", "since today", "for a while",
        "really", "very", "quite", "slightly", "a bit of",
        "also", "too", "and", "with", "plus",
    ]
    cleaned = user_lower
    for f in sorted(fillers, key=len, reverse=True):
        cleaned = cleaned.replace(f, " ")
    cleaned = " ".join(cleaned.split())

    for phrase, mapped in sorted(ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if phrase in user_lower or phrase in cleaned:
            for s in mapped:
                if s in all_symptoms:
                    found.add(s)

    for symptom in all_symptoms:
        readable = symptom.replace("_", " ")
        if readable in user_lower or readable in cleaned:
            found.add(symptom)

    words = [w for w in cleaned.replace(",", " ").replace(".", " ").split() if len(w) > 4]
    for symptom in all_symptoms:
        for sw in symptom.replace("_", " ").split():
            if len(sw) > 4:
                for w in words:
                    if w == sw or (len(w) > 5 and len(sw) > 5 and (w in sw or sw in w)):
                        found.add(symptom)

    readable_map = {s.replace("_", " "): s for s in all_symptoms}
    words = cleaned.split()
    chunks = set()
    for i in range(len(words) - 1):
        chunks.add(f"{words[i]} {words[i+1]}")
    for i in range(len(words) - 2):
        chunks.add(f"{words[i]} {words[i+1]} {words[i+2]}")
    for chunk in chunks:
        match = process.extractOne(chunk, readable_map.keys(), scorer=fuzz.ratio)
        if match and match[1] >= 90:
            found.add(readable_map[match[0]])

    # Guardrails copied from the existing app so fuzzy matching does not invent symptoms.
    if "cold hands" not in user_lower and "cold feet" not in user_lower:
        found.discard("cold_hands_and_feets")
    if "chest" not in user_lower:
        found.discard("chest_pain")
    if "eye" not in user_lower and "vision" not in user_lower:
        found.discard("redness_of_eyes")
        found.discard("blurred_and_distorted_vision")
    if "neck" not in user_lower:
        found.discard("stiff_neck")
        found.discard("neck_pain")
    if "hair" not in user_lower:
        found.discard("hair_loss")
    if "weight" not in user_lower:
        found.discard("weight_loss")
        found.discard("weight_gain")
    return sorted(found)

def predict_disease(symptoms_list):
    vec = [1 if s in symptoms_list else 0 for s in all_symptoms]
    df_in = pd.DataFrame([vec], columns=all_symptoms)
    probs = model.predict_proba(df_in)[0]
    stage = get_stage(len(symptoms_list))
    top = np.argsort(probs)[-10:][::-1]
    results = []
    for idx in top:
        disease = le.inverse_transform([idx])[0]
        conf = round(float(probs[idx]) * 100, 1)
        if conf >= stage["min_conf"]:
            results.append((disease, conf))
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        idx = int(np.argmax(probs))
        results = [(le.inverse_transform([idx])[0], round(float(probs[idx]) * 100, 1))]
    return results[:stage["show"]]

def description(disease):
    row = desc_df[desc_df["Disease"].str.lower() == disease.lower()]
    return str(row.iloc[0]["Description"]) if not row.empty else "No description available."

def precautions(disease):
    row = prec_df[prec_df["Disease"].str.lower() == disease.lower()]
    if row.empty:
        return ["Consult a qualified healthcare professional."]
    values = [row.iloc[0][f"Precaution_{i}"] for i in range(1, 5) if f"Precaution_{i}" in row.columns]
    return [str(v) for v in values if pd.notna(v) and str(v).strip()]

def severe_matches(text):
    u = text.lower()
    return [s for s in SEVERE if s in u]

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": "loaded"})

@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    previous = payload.get("symptoms", [])
    if not isinstance(previous, list):
        previous = []
    matched = set(str(s) for s in previous if s in all_symptoms)
    if text:
        matched.update(match_symptoms(text))
    symptoms = sorted(matched)
    severe = severe_matches(text)
    if len(symptoms) < 3:
        return jsonify({
            "ok": True,
            "ready": False,
            "symptoms": symptoms,
            "symptom_count": len(symptoms),
            "severe": severe,
            "message": "Please provide at least 3 symptoms for a prediction."
        })
    predictions = []
    for disease, confidence in predict_disease(symptoms):
        predictions.append({
            "disease": disease,
            "confidence": confidence,
            "description": description(disease),
            "precautions": precautions(disease),
        })
    return jsonify({
        "ok": True, "ready": True, "symptoms": symptoms,
        "symptom_count": len(symptoms), "severe": severe,
        "predictions": predictions,
        "disclaimer": "Educational information only. This is not a medical diagnosis. Seek professional care for medical concerns."
    })

@app.errorhandler(Exception)
def handle_error(error):
    return jsonify({"ok": False, "error": "The prediction service encountered an error."}), 500
