# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
import nltk

nltk.download('punkt')

# Title
st.title("📄 Dual-Mode Text Summarizer")
st.markdown("Summarize text using **FLAN-T5-XL** (abstractive) and **BERTSum** (extractive)")

# Load FLAN-T5 summarizer
@st.cache_resource
def load_abstractive_model():
    model_name = "google/flan-t5-xl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Load BERTSum model
@st.cache_resource
def load_extractive_model():
    return Summarizer()

# Input text area
text = st.text_area("Enter your text (up to 1000 words):", height=300)

if st.button("Generate Summaries"):
    if text:
        with st.spinner("Generating summaries..."):
            # Load models
            abstractive_summarizer = load_abstractive_model()
            extractive_summarizer = load_extractive_model()

            # Abstractive
            abs_input = "Summarize this:\n" + text
            abs_summary = abstractive_summarizer(abs_input, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

            # Extractive
            ext_summary = extractive_summarizer(text, num_sentences=5)

        st.subheader("🔷 Abstractive Summary (FLAN-T5-XL)")
        st.write(abs_summary)

        st.subheader("🔶 Extractive Summary (BERTSum)")
        st.write(ext_summary)
    else:
        st.warning("Please enter some text.")