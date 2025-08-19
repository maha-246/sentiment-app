# src/train_ml.py
from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data import make_binary_corpus, clean_for_tfidf

def main():
    ds = make_binary_corpus()
    X_tr, y_tr = ds["train"]["text"], ds["train"]["label"]
    X_va, y_va = ds["validation"]["text"], ds["validation"]["label"]
    X_te, y_te = ds["test"]["text"], ds["test"]["label"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_for_tfidf,   # << no lambda; top-level function
            lowercase=False,                # already lowercased by cleaner
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=40000,
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced"
        )),
    ])

    pipe.fit(X_tr, y_tr)

    yhat_va = pipe.predict(X_va)
    va_acc  = accuracy_score(y_va, yhat_va)
    va_f1   = f1_score(y_va, yhat_va)
    print("Validation\n  acc:", va_acc, "  f1:", va_f1)
    print(classification_report(y_va, yhat_va, digits=3))

    yhat_te = pipe.predict(X_te)
    te_acc  = accuracy_score(y_te, yhat_te)
    te_f1   = f1_score(y_te, yhat_te)
    print("\nTest\n  acc:", te_acc, "  f1:", te_f1)
    print(classification_report(y_te, yhat_te, digits=3))

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, "models/tfidf_logreg.joblib")
    with open("models/tfidf_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val_acc": va_acc, "val_f1": va_f1, "test_acc": te_acc, "test_f1": te_f1}, f, indent=2)
    print("Saved -> models/tfidf_logreg.joblib")

if __name__ == "__main__":
    main()
