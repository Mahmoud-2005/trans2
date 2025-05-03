# app.py
import streamlit as st
from transformers import pipeline

# Load pipelines
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_ner_pipeline():
    return pipeline("ner", grouped_entities=True)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_text_generator():
    return pipeline("text-generation", model="gpt2")

# Load models
sentiment_analyzer = load_sentiment_analyzer()
ner_pipeline = load_ner_pipeline()
summarizer = load_summarizer()
text_generator = load_text_generator()

# UI
st.title("✍️ Smart Writing Assistant")

text_input = st.text_area("📝 Enter your text:", "Streamlit is an amazing tool for building data apps!")

if st.button("🚀 Start Analysis and Suggestions") and text_input.strip():
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "✨ Smart Suggestions", "💡 Writing Tips"])

    with tab1:
        st.header("📊 Text Analysis")

        st.subheader("💬 Sentiment Analysis")
        sentiment = sentiment_analyzer(text_input)
        label = sentiment[0]['label']
        score = sentiment[0]['score']
        st.success(f"Sentiment: {label} (Confidence: {score:.2f})")

        st.subheader("🔎 Named Entity Recognition (NER)")
        entities = ner_pipeline(text_input)
        if entities:
            for entity in entities:
                st.write(f"• `{entity['word']}` - Type: **{entity['entity_group']}** - Score: {entity['score']:.2f}")
        else:
            st.info("No named entities were found in the text.")

        st.subheader("📝 Text Summarization")
        if len(text_input.split()) > 20:
            summary = summarizer(text_input, max_length=50, min_length=25, do_sample=False)
            st.success(summary[0]['summary_text'])
        else:
            st.warning("The text is too short for summarization. Please enter more than 20 words.")

    with tab2:
        st.header("✨ Smart Suggestions")

        st.subheader("🧠 Suggested Catchy Title")
        prompt = "Suggest a catchy title for the following content: " + text_input
        title = text_generator(prompt, max_length=20, num_return_sequences=1)[0]['generated_text']
        st.success(title.replace(prompt, "").strip())

        st.subheader("🔁 Paraphrase Text")
        rewrite_prompt = "Paraphrase the following paragraph to improve clarity and style:\n" + text_input
        rewritten = text_generator(rewrite_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        st.success(rewritten.replace(rewrite_prompt, "").strip())

    with tab3:
        st.header("💡 Writing Tips")

        if "negative" in label.lower():
            st.write("🔹 Consider using a more positive tone if appropriate.")
        if not entities:
            st.write("🔹 Try adding named entities (like people, places, or organizations) for more specificity.")
        if len(text_input.split()) < 50:
            st.write("🔹 The text is relatively short. Try expanding it with more details or examples.")
        st.write("🔹 Make sure your text is well-structured with proper punctuation and transitions.")
else:
    st.info("💡 Please enter your text above and click the button to begin analysis and suggestions.")
