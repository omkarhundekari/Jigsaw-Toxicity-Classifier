import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

repo = Path(__file__).resolve().parents[1]
data_path = repo / "data" / "train.csv"
models_dir = repo / "models"
models_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
X = df["comment_text"].fillna("")
Y = df[labels]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

word_clf = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
)

char_clf = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
)

word_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1,2),
        min_df=2,
        max_df=0.9
    )),
    ("clf", word_clf)
])

char_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3,5),
        min_df=2,
        max_df=0.9
    )),
    ("clf", char_clf)
])

word_model.fit(X_train, y_train)
char_model.fit(X_train, y_train)

dump(word_model, models_dir / "word_tfidf.joblib")
dump(char_model, models_dir / "char_tfidf.joblib")

print("saved:", models_dir / "word_tfidf.joblib")
print("saved:", models_dir / "char_tfidf.joblib")
