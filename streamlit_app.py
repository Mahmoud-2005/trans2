# app.py
import streamlit as st
from transformers import pipeline

# Load translation pipeline (English to French)
@st.cache_resource
def load_translator():
    return pipeline("translation_en_to_fr")

translator = load_translator()

# Streamlit UI
st.title("English to French Translator")

english_text = st.text_input("Enter English text:", "The cat is on the table.")

if st.button("Translate"):
    translation = translator(english_text)
    french_text = translation[0]['translation_text']
    st.success(f"French: {french_text}")
