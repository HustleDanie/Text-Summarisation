# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
import nltk
import traceback

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Title
st.title("📄 Dual-Mode Text Summarizer")
st.markdown("Summarize text abstractive and extractively")

# Load T5-Small summarizer
@st.cache_resource
def load_abstractive_model():
    model_name = "sshleifer/distilbart-cnn-12-6"
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
            try:
                # Load models
                abstractive_summarizer = load_abstractive_model()
                extractive_summarizer = load_extractive_model()

                # Abstractive summarization
                abs_summary = abstractive_summarizer("summarize: " + text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

                # Extractive summarization
                ext_summary = extractive_summarizer(text, num_sentences=5)

                # Display results
                st.subheader("🔷 Abstractive Summary (T5-Small)")
                st.write(abs_summary)

                st.subheader("🔶 Extractive Summary (BERTSum)")
                st.write(ext_summary)

            except Exception as e:
                st.error(f"❌ An error occurred:\n\n{e}")
                traceback.print_exc()
    else:
        st.warning("⚠️ Please enter some text.")
