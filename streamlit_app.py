# app.py
import streamlit as st
from transformers import pipeline

# Load sentiment analysis pipeline
@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_analyzer()

# Streamlit UI
st.title("Sentiment Analysis App")

text_input = st.text_area("Enter text to analyze sentiment:", "I love programming!")

if st.button("Analyze Sentiment"):
    sentiment = sentiment_analyzer(text_input)
    label = sentiment[0]['label']
    score = sentiment[0]['score']
    st.success(f"Sentiment: {label} (Confidence: {score:.2f})")
