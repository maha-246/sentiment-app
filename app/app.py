import streamlit as st

st.set_page_config(page_title="Sentiment Demo", page_icon="🧠")
st.title("🧠 Sentiment Analysis (VADER Baseline)")
st.write("Paste text below and click **Analyze**.")

@st.cache_resource
def get_analyzer():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

text = st.text_area("Your text", height=180, placeholder="I loved the movie...")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        sia = get_analyzer()
        score = sia.polarity_scores(text)["compound"]
        label = "POSITIVE" if score >= 0 else "NEGATIVE"
        st.success(f"**{label}** (compound: {score:.3f})")
        st.progress(min(max(abs(score), 0.0), 1.0))
