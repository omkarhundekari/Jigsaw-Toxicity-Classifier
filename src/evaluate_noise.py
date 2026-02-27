import random
import re
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

repo = Path(__file__).resolve().parents[1]
data_path = repo / "data" / "train.csv"
models_dir = repo / "models"

word_model = load(models_dir / "word_tfidf.joblib")
char_model = load(models_dir / "char_tfidf.joblib")

df = pd.read_csv(data_path)
X = df["comment_text"].fillna("")
Y = df[labels]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

pred_word = word_model.predict(X_test)
pred_char = char_model.predict(X_test)

random.seed(42)

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
    if random.random() < 0.5:
        t = t + " !!!"
    return t

X_test_noisy = X_test.apply(add_noise)

pred_word_noisy = word_model.predict(X_test_noisy)
pred_char_noisy = char_model.predict(X_test_noisy)

out = pd.DataFrame({
    "model": ["word_tfidf", "char_tfidf"],
    "macro_f1_clean": [
        f1_score(y_test, pred_word, average="macro"),
        f1_score(y_test, pred_char, average="macro")
    ],
    "macro_f1_noisy": [
        f1_score(y_test, pred_word_noisy, average="macro"),
        f1_score(y_test, pred_char_noisy, average="macro")
    ],
})

out["drop"] = out["macro_f1_clean"] - out["macro_f1_noisy"]
out["retention"] = out["macro_f1_noisy"] / out["macro_f1_clean"]
print(out)
