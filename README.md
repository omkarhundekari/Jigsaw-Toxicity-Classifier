# Jigsaw Toxicity Classifier

Multi-label toxicity classification using the Jigsaw Toxic Comment dataset.

This repo explores:
- a baseline multi-label model with TF-IDF + Logistic Regression (One-vs-Rest)
- how simple obfuscation/noisy text affects predictions
- why character n-grams can be more robust under noise

## Labels
toxic, severe_toxic, obscene, threat, insult, identity_hate

## Project layout
- `notebooks/01_eda.ipynb` – data overview + label distribution
- `notebooks/02_baseline.ipynb` – baseline models + robustness test
- `src/train.py` – train and save word/char TF-IDF models
- `src/evaluate_noise.py` – evaluate clean vs noisy robustness
- `reports/results.md` – short summary of findings

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

