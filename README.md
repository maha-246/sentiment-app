# Sentiment Analysis (Streamlit)

A simple Streamlit web app that predicts sentiment for pasted text using one of two backends:

- **VADER (baseline)**: fast lexicon-based sentiment scoring (via NLTK).
- **TF-IDF + Logistic Regression**: a trained scikit-learn pipeline loaded from `models/tfidf_logreg.joblib`.

## Demo
👉 [Open the app](https://sentiment-app-ib94ysap6fmqysd7qrvst6.streamlit.app/)

## Features
- Paste any text and click **Analyze**
- Switch between **VADER** and **TF-IDF + Logistic**
- Shows **label + confidence** and a progress bar

## Expected project structure
```text
sentiment-app/
  app/
    app.py
  models/
    tfidf_logreg.joblib        # optional (only needed for TF-IDF backend)
  src/
    data.py                    # required if your joblib pipeline imports src.data
  requirements.txt
```

## Run locally

### 1) Create and activate a virtual environment

**Windows (PowerShell):**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Start the app
```bash
streamlit run app/app.py
```

## TF-IDF model notes [Not working for now]:

The TF-IDF backend expects this file to exist:

- models/tfidf_logreg.joblib

If it’s missing (or can’t be unpickled), the app will show an error and you’ll need to train the model locally or include the file in your deployment.

### Why ```src/data.py``` matters

The TF-IDF loader:

- Adds the repo root to ``sys.path``
- Imports ``src.data`` before calling ``joblib.load(...)``

This prevents unpickling errors when your saved pipeline references custom code (for example, a preprocessor defined in ``src/data.py``).

## Deploying (Streamlit Cloud)

- VADER works out of the box (it downloads the vader_lexicon on first run).
- For TF-IDF, include ``models/tfidf_logreg.joblib`` in the repo (or load it from remote storage) so the backend is available.

## Backends shown in the UI

- VADER (lexicon)
- TF-IDF + Logistic Regression (trained pipeline)

## Made with Streamlit 🧠

