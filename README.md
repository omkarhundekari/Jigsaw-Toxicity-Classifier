# Jigsaw Toxicity Classifier

Multi-label toxicity classification using the Jigsaw Toxic Comment dataset.

This repo explores:
- a baseline multi-label model with TF-IDF + Logistic Regression (One-vs-Rest)
- how obfuscation/noisy text affects predictions
- why character n-grams can be more robust under noise

## Labels
toxic, severe_toxic, obscene, threat, insult, identity_hate

## Project layout
- `notebooks/01_eda.ipynb` – data overview + label distribution
- `notebooks/02_baseline.ipynb` – baseline models + robustness test
- `reports/results.md` – short summary of findings

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
