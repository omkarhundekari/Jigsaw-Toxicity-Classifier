import random
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

repo = Path(__file__).resolve().parent
models_dir = repo / "models"

def add_noise(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.translate(str.maketrans({"a":"4","e":"3","i":"1","o":"0","s":"5","t":"7"}))
    t = re.sub(r"([a-z])\1{1,}", r"\1\1\1", t)
    t = re.sub(r"([a-z])", r"\1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t + " !!!"
    return t

@st.cache_resource
def load_models():
    word_model = load(models_dir / "word_tfidf.joblib")
    char_model = load(models_dir / "char_tfidf.joblib")
    return word_model, char_model

st.set_page_config(page_title="Toxicity Demo", layout="centered")
st.title("Toxicity Classifier Demo")
st.caption("Multi-label predictions using TF-IDF + Logistic Regression (Jigsaw dataset).")

word_model, char_model = load_models()

model_choice = st.selectbox("Model", ["word_tfidf", "char_tfidf"])
use_noise = st.checkbox("Add noise/obfuscation", value=False)

text = st.text_area("Enter text", height=140, placeholder="Type a sentence here...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        x = text
        if use_noise:
            x = add_noise(x)

        model = word_model if model_choice == "word_tfidf" else char_model
        pred = model.predict(pd.Series([x]))[0]

        picked = [labels[i] for i, v in enumerate(pred) if v == 1]

        st.subheader("Prediction")
        if picked:
            st.write(", ".join(picked))
        else:
            st.write("no labels triggered")

        if use_noise:
            st.subheader("Noisy input used")
            st.code(x)
