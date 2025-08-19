import streamlit as st

st.set_page_config(page_title="Sentiment Demo", page_icon="🧠")
st.title("🧠 Sentiment Analysis Demo")
st.write("Paste some text and click **Analyze**. We'll wire up models next.")

txt = st.text_area("Your text", height=180, placeholder="I loved the movie...")
if st.button("Analyze"):
    st.info("Prediction coming soon…")
