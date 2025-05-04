# streamlit_app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from summarizer import Summarizer
import nltk
import pandas as pd

# Download NLTK punkt tokenizer
nltk.download('punkt')

# Title
st.title("Dual-Mode Text Summarizer")
st.markdown("This app provides two types of text summarization using pretrained models:")
st.markdown(" **Abstractive Summarization** with `T5-Small` (generates new sentences)")
st.markdown(" **Extractive Summarization** with `BERTSum` (extracts key sentences from original text)")


# Load T5-Small summarizer (Abstractive)
@st.cache_resource
def load_abstractive_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

# Load BERTSum model (Extractive)
@st.cache_resource
def load_extractive_model():
    return Summarizer()

# Input area for text to be summarized
st.markdown("### 📝 Input Text")
text = st.text_area("Enter your text (up to 1000 words):", height=300)

# Run summarization when user clicks the button
if st.button("Generate Summaries"):
    if text:
        with st.spinner("⏳ Generating summaries..."):
            try:
                # Load summarization models
                abstractive_summarizer = load_abstractive_model()
                extractive_summarizer = load_extractive_model()

                # Run abstractive summarization
                abs_summary = abstractive_summarizer("summarize: " + text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

                # Run extractive summarization
                ext_summary = extractive_summarizer(text, num_sentences=5)

                # Display summaries
                
                st.subheader(" Abstractive Summary")
                st.markdown(
                    "This summary is **generated in new words** by the model (T5-Small), so it might rephrase or restructure sentences."
                )
                st.success(abs_summary)

                
                st.subheader("🔶 Extractive Summary")
                st.markdown(
                    "This summary is **copied directly from the original** text. The BERTSum model selects the top 5 most important sentences."
                )
                st.info(ext_summary)

                # Show a bar chart using Streamlit-native charting
                st.markdown("---")
                st.subheader("📊 Word Count Comparison")
                df = pd.DataFrame({
                    "Summary Type": ["Original Text", "Abstractive", "Extractive"],
                    "Word Count": [len(text.split()), len(abs_summary.split()), len(ext_summary.split())]
                })
                st.bar_chart(df.set_index("Summary Type"))

                st.markdown(
                    "This chart helps visualize how each summarization method compresses the original content:\n\n"
                    "- **Abstractive** tends to paraphrase and condense.\n"
                    "- **Extractive** keeps full original sentences, so it's usually longer."
                )

            except Exception as e:
                st.error(f"❌ An error occurred:\n\n{e}")
                # traceback.print_exc()  # Disabled for Streamlit Cloud performance
    else:
        st.warning("⚠️ Please enter some text to summarize.")
