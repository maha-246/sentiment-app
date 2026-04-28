import streamlit as st

st.set_page_config(page_title="Sentiment Demo", page_icon="🧠")
st.title("🧠 Sentiment Analysis")
st.write("Choose a backend and paste text below.")

backend = st.selectbox("Model", ["VADER (baseline)", "TF-IDF + Logistic"])
txt = st.text_area("Your text", height=180, placeholder="I loved the movie...")
go = st.button("🔍 Analyze")

@st.cache_resource
def load_vader():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_tfidf():
    # Make repo root importable so joblib can import `src.data`
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]   # .../sentiment-app
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Ensure the module is importable at unpickle time (preprocessor lives in src.data)
    import src.data  # noqa: F401

    import joblib
    model_path = ROOT / "models" / "tfidf_logreg.joblib"
    return joblib.load(model_path)

def show_result(label: str, score: float):
    if label == "POSITIVE":
        bg_color = "#d4edda"
        text_color = "#155724"
        border_color = "#28a745"
    elif label == "NEGATIVE":
        bg_color = "#f8d7da"
        text_color = "#721c24"
        border_color = "#dc3545"
    else:
        bg_color = "#e2e3e5"
        text_color = "#383d41"
        border_color = "#6c757d"

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            border-left: 6px solid {border_color};
            padding: 16px;
            border-radius: 8px;
            font-size: 18px;
            margin-top: 12px;
        ">
            <strong>{label}</strong><br>
            Score: {score:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(min(max(score, 0.0), 1.0))

if go:
    if not txt.strip():
        st.warning("Please enter some text.")
    else:
        if backend.startswith("VADER"):
            sia = load_vader()
            score = sia.polarity_scores(txt)["compound"]

            if score >= 0.05:
                label = "POSITIVE"
            elif score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            show_result(label, abs(score))
        else:
            # TF-IDF
            try:
                clf = load_tfidf()
            except Exception as e:
                st.error(
                    "TF-IDF model not available. Train it locally to create "
                    "`models/tfidf_logreg.joblib`, or deploy a hosted copy."
                )
                st.exception(e)
            else:
                proba = clf.predict_proba([txt])[0]
                idx = int(proba.argmax())
                label = "POSITIVE" if idx == 1 else "NEGATIVE"
                show_result(label, float(proba.max()))

st.caption("Backends: VADER (lexicon) • TF-IDF + Logistic (trained on IMDb + TweetEval binary).")
