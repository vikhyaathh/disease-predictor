import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from colorama import Fore, init
from rapidfuzz import process, fuzz

init(autoreset=True)

# ─────────────────────────────────────────
# Load model
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

print("Loading model...")
with open(BASE_DIR / 'model' / 'disease_model.pkl', 'rb') as f:
    data = pickle.load(f)

model         = data['model']
le            = data['label_encoder']
all_symptoms  = data['all_symptoms']
severity_dict = data['severity_dict']

desc_df = pd.read_csv(BASE_DIR / 'data' / 'symptom_Description.csv')
prec_df = pd.read_csv(BASE_DIR / 'data' / 'symptom_precaution.csv')
desc_df.columns = desc_df.columns.str.strip()
prec_df.columns = prec_df.columns.str.strip()
print("✅ Ready!\n")

# ─────────────────────────────────────────
# AI Response
# ─────────────────────────────────────────
# ─────────────────────────────────────────
# Symptom alias map
# Only maps what is CERTAIN and MEDICALLY CORRECT
# No fuzzy guessing — only verified mappings
# ─────────────────────────────────────────
ALIASES = {'fever': ['high_fever', 'mild_fever'],
 'high fever': ['high_fever'],
 'mild fever': ['mild_fever'],
 'low grade fever': ['mild_fever'],
 'feverish': ['mild_fever'],
 'feel hot': ['mild_fever'],
 'temperature': ['mild_fever'],
 'cough': ['cough'],
 'coughing': ['cough'],
 'dry cough': ['cough'],
 'wet cough': ['cough', 'phlegm'],
 'phlegm': ['phlegm'],
 'cold': ['runny_nose', 'chills'],
 'flu': ['high_fever', 'chills', 'cough', 'muscle_pain', 'headache'],
 'common cold': ['runny_nose', 'chills', 'cough', 'continuous_sneezing'],
 'runny nose': ['runny_nose'],
 'blocked nose': ['congestion'],
 'sneezing': ['continuous_sneezing'],
 'sore throat': ['throat_irritation'],
 'throat pain': ['throat_irritation'],
 'headache': ['headache'],
 'head pain': ['headache'],
 'migraine': ['headache'],
 'body ache': ['muscle_pain', 'joint_pain'],
 'body pain': ['muscle_pain', 'joint_pain'],
 'whole body hurts': ['muscle_pain', 'joint_pain', 'back_pain'],
 'pain everywhere': ['muscle_pain', 'joint_pain', 'back_pain'],
 'muscle pain': ['muscle_pain'],
 'joint pain': ['joint_pain'],
 'back pain': ['back_pain'],
 'neck pain': ['neck_pain'],
 'stiff neck': ['stiff_neck'],
 'chest pain': ['chest_pain'],
 'stomach pain': ['stomach_pain'],
 'stomach ache': ['stomach_pain'],
 'tummy ache': ['stomach_pain'],
 'fatigue': ['fatigue'],
 'tired': ['fatigue'],
 'exhausted': ['fatigue'],
 'no energy': ['fatigue', 'muscle_weakness'],
 'weakness': ['muscle_weakness'],
 'weak': ['muscle_weakness'],
 'feel weak': ['muscle_weakness'],
 'lethargy': ['lethargy'],
 'lethargic': ['lethargy'],
 'drowsy': ['lethargy'],
 'dizziness': ['dizziness'],
 'dizzy': ['dizziness'],
 'lightheaded': ['dizziness'],
 'vertigo': ['dizziness'],
 'nausea': ['nausea'],
 'nauseous': ['nausea'],
 'feel sick': ['nausea'],
 'feel like vomiting': ['nausea', 'vomiting'],
 'throwing up': ['vomiting'],
 'vomiting': ['vomiting'],
 'vomit': ['vomiting'],
 'puking': ['vomiting'],
 'diarrhea': ['diarrhoea'],
 'diarrhoea': ['diarrhoea'],
 'loose stools': ['diarrhoea'],
 'loose motions': ['diarrhoea'],
 'loose motion': ['diarrhoea'],
 'constipation': ['constipation'],
 'bloating': ['distention_of_abdomen'],
 'indigestion': ['indigestion'],
 'acidity': ['acidity'],
 'heartburn': ['acidity'],
 'no appetite': ['loss_of_appetite'],
 'loss of appetite': ['loss_of_appetite'],
 'breathlessness': ['breathlessness'],
 'shortness of breath': ['breathlessness'],
 'cant breathe': ['breathlessness'],
 'difficulty breathing': ['breathlessness'],
 'chest tightness': ['chest_pain', 'breathlessness'],
 'palpitations': ['palpitations'],
 'heart racing': ['palpitations'],
 'red eyes': ['redness_of_eyes'],
 'watery eyes': ['watering_from_eyes'],
 'blurred vision': ['blurred_and_distorted_vision'],
 'yellow eyes': ['yellowing_of_eyes'],
 'rash': ['skin_rash'],
 'skin rash': ['skin_rash'],
 'itching': ['itching'],
 'itchy': ['itching'],
 'yellow skin': ['yellowish_skin'],
 'jaundice': ['yellowish_skin', 'yellow_urine', 'yellowing_of_eyes'],
 'swelling': ['swelled_lymph_nodes'],
 'confused': ['altered_sensorium'],
 'confusion': ['altered_sensorium'],
 'feel confused': ['altered_sensorium'],
 'brain fog': ['altered_sensorium'],
 'anxiety': ['anxiety'],
 'depression': ['depression'],
 'mood swings': ['mood_swings'],
 'chills': ['chills'],
 'shivering': ['shivering', 'chills'],
 'sweating': ['sweating'],
 'night sweats': ['sweating'],
 'burning urination': ['burning_micturition'],
 'frequent urination': ['polyuria'],
 'dark urine': ['dark_urine'],
 'weight loss': ['weight_loss'],
 'weight gain': ['weight_gain'],
 'thirsty': ['excessive_hunger'],
 'excessive thirst': ['excessive_hunger'],
 'dry mouth': ['drying_and_tingling_lips'],
 'cold hands': ['cold_hands_and_feets'],
 'cold feet': ['cold_hands_and_feets'],
 'coma': ['altered_sensorium'],
 'unconscious': ['altered_sensorium'],
 'slurred speech': ['slurred_speech'],
 'coughing blood': ['blood_in_sputum']}

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
MIN_SYMPTOMS  = 3
HIGH_SEVERITY = 7

PREDICTION_STAGES = {
    3: {'show': 1, 'min_conf': 20},
    5: {'show': 2, 'min_conf': 15},
    7: {'show': 3, 'min_conf': 10},
}

def get_stage(count):
    if count >= 7:
        return PREDICTION_STAGES[7]
    elif count >= 5:
        return PREDICTION_STAGES[5]
    else:
        return PREDICTION_STAGES[3]

# ─────────────────────────────────────────
# Symptom matcher — clean and accurate
# ─────────────────────────────────────────
def match_symptoms(user_input):
    user_lower = user_input.lower().strip()
    found      = set()

    # Step 1: Strip filler words
    fillers = [
        "i have been having","i have been feeling",
        "i have been","i am having","i am feeling",
        "i am experiencing","i feel like i have",
        "i feel like","i've been","i've got",
        "i'm having","i'm feeling","i'm",
        "i keep getting","i got","i get",
        "i have","i feel","i am","i've",
        "suffering from","experiencing",
        "since yesterday","since today","since morning",
        "for a while","for days","since",
        "also","too","as well","and","with",
        "plus","along with","having","getting",
        "really","very","quite","slightly",
        "a bit of","a bit","some",
    ]
    cleaned = user_lower
    for f in sorted(fillers, key=len, reverse=True):
        cleaned = cleaned.replace(f, ' ')
    cleaned = ' '.join(cleaned.split())

    # Step 2: Alias map — longest phrase first
    # Check both original and cleaned
    for phrase, mapped in sorted(
        ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):
        if phrase in user_lower or phrase in cleaned:
            for s in mapped:
                if s in all_symptoms:
                    found.add(s)

    # Step 3: Direct dataset symptom name match
    for symptom in all_symptoms:
        readable = symptom.replace('_', ' ')
        if readable in user_lower or readable in cleaned:
            found.add(symptom)

    # Step 4: Keyword match — only strong matches
    words = [
        w for w in
        cleaned.replace(',', ' ').replace('.', ' ').split()
        if len(w) > 4
    ]
    for symptom in all_symptoms:
        sym_words = symptom.replace('_', ' ').split()
        for sw in sym_words:
            if len(sw) > 4:
                for w in words:
                    # Both must be long enough and very similar
                    if w == sw or (
                        len(w) > 5 and len(sw) > 5 and
                        (w in sw or sw in w)
                    ):
                        found.add(symptom)

    # Step 5: Strict fuzzy — only on 2+ word chunks
    readable_map = {
        s.replace('_', ' '): s for s in all_symptoms
    }
    all_words = cleaned.split()
    chunks    = set()
    for i in range(len(all_words) - 1):
        chunks.add(f"{all_words[i]} {all_words[i+1]}")
    for i in range(len(all_words) - 2):
        chunks.add(
            f"{all_words[i]} {all_words[i+1]} {all_words[i+2]}"
        )

    for chunk in chunks:
        match = process.extractOne(
            chunk,
            readable_map.keys(),
            scorer=fuzz.ratio   # strict ratio
        )
        if match and match[1] >= 90:  # very strict
            found.add(readable_map[match[0]])

    # Step 6: Strict cleanup
    # Only keep symptoms explicitly mentioned
    if 'cold hands' not in user_lower and \
       'cold feet'  not in user_lower:
        found.discard('cold_hands_and_feets')
    if 'chest' not in user_lower:
        found.discard('chest_pain')
    if 'eye'    not in user_lower and \
       'vision' not in user_lower:
        found.discard('redness_of_eyes')
        found.discard('pain_behind_the_eyes')
        found.discard('watering_from_eyes')
        found.discard('blurred_and_distorted_vision')
    if 'neck' not in user_lower:
        found.discard('stiff_neck')
        found.discard('neck_pain')
    if 'urin' not in user_lower and \
       'pee'  not in user_lower:
        found.discard('burning_micturition')
        found.discard('polyuria')
    if 'hair' not in user_lower:
        found.discard('hair_loss')
    if 'weight' not in user_lower:
        found.discard('weight_loss')
        found.discard('weight_gain')
    if 'back'   not in user_lower and \
       'spine'  not in user_lower:
        found.discard('back_pain')
    if 'joint'  not in user_lower and \
       'knee'   not in user_lower and \
       'joints' not in user_lower:
        found.discard('joint_pain')
        found.discard('swollen_joints')
    if 'black'  not in user_lower and \
       'head'   not in user_lower:
        found.discard('blackheads')
    if 'bleed'  not in user_lower and \
       'blood'  not in user_lower:
        found.discard('bleeding_from_gums')
        found.discard('blood_in_sputum')
        found.discard('stomach_bleeding')
    if 'bruise' not in user_lower and \
       'bruis'  not in user_lower:
        found.discard('bruising')
    if 'depression' not in user_lower and \
       'depress'    not in user_lower:
        found.discard('depression')
    if 'anxious'  not in user_lower and \
       'anxiety'  not in user_lower and \
       'panic'    not in user_lower:
        found.discard('anxiety')
    if 'peel' not in user_lower:
        found.discard('skin_peeling')
    if 'swollen' not in user_lower and \
       'swelling' not in user_lower and \
       'swell'    not in user_lower:
        found.discard('swelled_lymph_nodes')
        found.discard('swollen_joints')
    if 'spot'  not in user_lower and \
       'spots' not in user_lower:
        found.discard('red_spots_over_body')
    if 'abdom' not in user_lower:
        found.discard('abdominal_pain')
    if 'disten' not in user_lower and \
       'bloat'  not in user_lower:
        found.discard('distention_of_abdomen')

    return list(found)

# ─────────────────────────────────────────
# Severity check
# ─────────────────────────────────────────
SEVERE = [
    "chest pain","cant breathe","breathlessness",
    "shortness of breath","difficulty breathing",
    "unconscious","coma","coughing blood",
    "blood in vomit","slurred speech",
    "heart racing","palpitations","high fever",
]

def check_severity(user_input):
    u = user_input.lower()
    return [s for s in SEVERE if s in u]

# ─────────────────────────────────────────
# ML Prediction
# ─────────────────────────────────────────
def predict_disease(symptoms_list):
    vec   = [1 if s in symptoms_list else 0
             for s in all_symptoms]
    df_in = pd.DataFrame([vec], columns=all_symptoms)
    probs = model.predict_proba(df_in)[0]

    stage    = get_stage(len(symptoms_list))
    max_show = stage['show']
    min_conf = stage['min_conf']

    top = np.argsort(probs)[-10:][::-1]
    results = []
    for idx in top:
        disease = le.inverse_transform([idx])[0]
        conf    = round(probs[idx] * 100, 1)
        if conf >= min_conf:
            results.append((disease, conf))

    results.sort(key=lambda x: x[1], reverse=True)

    # Always show at least 1
    if not results:
        idx     = np.argmax(probs)
        disease = le.inverse_transform([idx])[0]
        conf    = round(probs[idx] * 100, 1)
        return [(disease, conf)]

    return results[:max_show]

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def get_description(disease):
    row = desc_df[
        desc_df['Disease'].str.lower() == disease.lower()
    ]
    if not row.empty:
        return row.iloc[0]['Description']
    return "No description available."

def get_precautions(disease):
    row = prec_df[
        prec_df['Disease'].str.lower() == disease.lower()
    ]
    if not row.empty:
        precs = [row.iloc[0][f'Precaution_{i}']
                 for i in range(1, 5)]
        return [p for p in precs if pd.notna(p)]
    return ["Consult a doctor immediately."]

# ─────────────────────────────────────────
# Local response formatter
# ─────────────────────────────────────────
def format_response(predictions, symptom_count):
    lines = []

    if symptom_count < 5:
        lines.append(
            "Based on your symptoms so far, here is "
            "my early assessment:"
        )
    elif symptom_count < 7:
        lines.append(
            "With these symptoms, here are the most "
            "likely conditions:"
        )
    else:
        lines.append(
            "Based on all your symptoms, here is "
            "my full assessment:"
        )

    lines.append("")

    for rank, (disease, conf) in enumerate(predictions, 1):
        label = "Most likely" if rank == 1 else "Also possible"
        desc  = get_description(disease)
        precs = get_precautions(disease)

        lines.append(f"{label}: {disease} ({conf}%)")
        lines.append(f"ℹ️  {desc}")
        lines.append("💊 Precautions:")
        for p in precs:
            lines.append(f"   • {p}")
        lines.append("")

    if symptom_count < 7:
        lines.append(
            "💡 Tip: Describe more symptoms for a "
            "more accurate diagnosis."
        )
        lines.append("")

    lines.append(
        "⚕️  These are possible conditions only. "
        "Please consult a real doctor for proper diagnosis."
    )

    return '\n'.join(lines)

# ─────────────────────────────────────────
# Memory
# ─────────────────────────────────────────
conversation_history = []
session_symptoms     = []

def save_to_history(role, message):
    conversation_history.append(
        {'role': role, 'message': message}
    )

def show_history():
    if not conversation_history:
        print(Fore.MAGENTA + "No history yet.")
        return
    print(Fore.MAGENTA + "\n📜 Conversation History:")
    print(Fore.MAGENTA + "-" * 50)
    for e in conversation_history:
        icon = "🧑 You" if e['role'] == 'user' else "🤖 Bot"
        print(Fore.MAGENTA + f"{icon}: {e['message']}")
    print(Fore.MAGENTA + "-" * 50)

# ─────────────────────────────────────────
# Display
# ─────────────────────────────────────────
def print_banner():
    print(Fore.CYAN + "=" * 60)
    print(Fore.CYAN + "          🏥 AI Disease Predictor Chatbot")
    print(Fore.CYAN + "   Commands: 'history' | 'clear' | 'help' | 'quit'")
    print(Fore.CYAN + "=" * 60)

def print_predictions(predictions, session_symptoms):
    count    = len(session_symptoms)
    readable = [s.replace('_', ' ') for s in session_symptoms]

    print(Fore.GREEN +
          f"\n🔍 Symptoms ({count}): {', '.join(readable)}")

    if count < 5:
        lbl = "🎯 Most Likely Condition:"
    elif count < 7:
        lbl = "🎯 Most Likely + Possible:"
    else:
        lbl = "🎯 Full Assessment:"

    print(Fore.GREEN + f"\n📋 {lbl}\n")

    for rank, (disease, conf) in enumerate(predictions, 1):
        bar   = "█" * int(conf/5) + "░" * (20 - int(conf/5))
        precs = get_precautions(disease)
        tag   = (Fore.GREEN  + "  ✅ MOST LIKELY"
                 if rank == 1 else
                 Fore.YELLOW + "  🔸 ALSO POSSIBLE")
        print(tag)
        print(Fore.WHITE  + f"     {disease}")
        print(Fore.BLUE   + f"     Confidence: [{bar}] {conf}%")
        print(Fore.YELLOW +
              f"     ⚠️  Precautions: {', '.join(precs)}")
        print()

    if count < 7:
        print(Fore.CYAN +
              "   💡 Add more symptoms to refine.\n")

# ─────────────────────────────────────────
# Main Chat
# ─────────────────────────────────────────
def chat():
    print_banner()

    greeting = (
        "Hello! I'm your AI Disease Predictor.\n"
        "      Describe your symptoms naturally!\n"
        "      💡 Example: 'I have nausea, vomiting\n"
        "         fever and headache'\n"
        "      Type 'help' for commands."
    )
    print(Fore.CYAN + f"\n🤖 Bot: {greeting}\n")
    save_to_history('bot', greeting)

    while True:
        try:
            user_input = input(Fore.WHITE + "🧑 You: ").strip()
        except KeyboardInterrupt:
            print(Fore.RED + "\n\nGoodbye! Stay healthy! 💊")
            break

        if not user_input:
            continue

        save_to_history('user', user_input)

        # ── Commands ──
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print(Fore.CYAN +
                  "\n🤖 Bot: Goodbye! Consult a doctor. "
                  "Stay safe! 💊")
            break

        if user_input.lower() == 'history':
            show_history()
            continue

        if user_input.lower() == 'clear':
            session_symptoms.clear()
            conversation_history.clear()
            print(Fore.CYAN + "\n🤖 Bot: Session cleared!\n")
            continue

        if user_input.lower() == 'help':
            print(Fore.CYAN + "\n🤖 Bot:\n" + (
                "  Commands:\n"
                "  - Type symptoms freely\n"
                "  - 'history' → View conversation\n"
                "  - 'clear'   → Reset session\n"
                "  - 'quit'    → Exit\n\n"
                "  Prediction stages:\n"
                "  - 3 symptoms → Most likely disease\n"
                "  - 5 symptoms → Top 2 conditions\n"
                "  - 7 symptoms → Full assessment\n"
            ))
            continue

        # ── Severity check ──
        warnings = check_severity(user_input)
        if warnings:
            print(Fore.RED +
                  f"\n   🚨 WARNING: {', '.join(warnings)}! "
                  f"Seek immediate medical attention.")
            save_to_history(
                'bot', f"WARNING: {', '.join(warnings)}"
            )

        # ── Match symptoms ──
        found = match_symptoms(user_input)

        if not found:
            msg = (
                "I couldn't identify any symptoms.\n"
                "      Please describe physical feelings like:\n"
                "      'I have fever, headache and feel weak'"
            )
            print(Fore.CYAN + f"\n🤖 Bot: {msg}\n")
            save_to_history('bot', msg)
            continue

        # ── Update session ──
        new = [s for s in found
               if s not in session_symptoms]
        session_symptoms.extend(new)

        if new:
            readable = [s.replace('_', ' ') for s in new]
            print(Fore.CYAN +
                  f"\n🤖 Bot: Understood! Noted: "
                  f"{Fore.WHITE}{', '.join(readable)}")
        else:
            print(Fore.CYAN +
                  "\n🤖 Bot: Already noted those symptoms.")

        # ── Need more ──
        if len(session_symptoms) < MIN_SYMPTOMS:
            needed = MIN_SYMPTOMS - len(session_symptoms)
            print(Fore.CYAN +
                  f"\n   Please describe {needed} more "
                  f"symptom(s) to begin prediction.\n")
            save_to_history(
                'bot', f"Need {needed} more symptoms."
            )
            continue

        # ── Predict ──
        predictions = predict_disease(session_symptoms)
        print_predictions(predictions, session_symptoms)

        # ── Local response ──
        response = format_response(
            predictions, len(session_symptoms)
        )
        print(Fore.CYAN + "\n🤖 Bot:")
        print(Fore.CYAN + "-" * 50)
        print(Fore.CYAN + response)
        print(Fore.CYAN + "-" * 50 + "\n")
        save_to_history('bot', response)

if __name__ == "__main__":
    chat()