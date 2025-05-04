# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
import nltk
import traceback

nltk.download('punkt')

# Title
st.title("📄 Dual-Mode Text Summarizer")
st.markdown("Summarize text using **DistilBART** (abstractive) and **BERTSum** (extractive)")

# Load BERTSum model
@st.cache_resource
def load_extractive_model():
    return Summarizer()

# Input text area
text = st.text_area("Enter your text (up to 1000 words):", height=300)

if st.button("Generate Summaries"):
    if text:
        with st.spinner("Generating summaries..."):
            try:
                # Load models
              
                extractive_summarizer = load_extractive_model()

               
                # Extractive
                ext_summary = extractive_summarizer(text, num_sentences=5)

                # Display results
               

                st.subheader("🔶 Extractive Summary (BERTSum)")
                st.write(ext_summary)

            except Exception as e:
                st.error(f"❌ An error occurred:\n\n{e}")
                traceback.print_exc()
    else:
        st.warning("⚠️ Please enter some text.")
