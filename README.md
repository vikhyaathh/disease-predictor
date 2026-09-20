# 🏥 SympAI — Disease Predictor

SympAI is an educational machine-learning web application that accepts symptoms written in natural language, maps them to the symptom vocabulary used by the training data, and uses a Random Forest model to surface possible conditions.

## Vercel deployment

This repository is already structured for Vercel:

```text
.
├── index.html
├── style.css
├── script.js
├── api/
│   └── index.py
├── model/
│   └── disease_model.pkl
├── data/
│   ├── dataset.csv
│   ├── Symptom-severity.csv
│   ├── symptom_Description.csv
│   └── symptom_precaution.csv
├── train_model.py
├── chatbot.py
├── requirements.txt
└── LICENSE
```

Import the repository into Vercel and deploy with the default settings. The static frontend is served from the repository root and the Python prediction API is exposed at `/api/predict`.

## Local model training

```bash
pip install -r requirements.txt
python train_model.py
```

## Important

This is an educational/informational project, not a medical diagnostic tool. Model confidence is not a clinical probability and should not be used to make medical decisions. Seek qualified medical care for symptoms or concerns.
