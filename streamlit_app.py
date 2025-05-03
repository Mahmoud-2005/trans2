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

sentiment_analyzer = load_sentiment_analyzer()
ner_pipeline = load_ner_pipeline()

# Streamlit UI
st.title("Advanced Text Analysis App")

text_input = st.text_area("Enter text to analyze:", "I love programming in Python at Microsoft!")

if st.button("Analyze"):
    # Sentiment Analysis
    sentiment = sentiment_analyzer(text_input)
    label = sentiment[0]['label']
    score = sentiment[0]['score']
    st.success(f"Sentiment: {label} (Confidence: {score:.2f})")

    # Named Entity Recognition (NER)
    st.subheader("Named Entities:")
    entities = ner_pipeline(text_input)
    for entity in entities:
        st.write(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Score: {entity['score']:.2f}")
