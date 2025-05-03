from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Streamlit UI
st.title("Advanced Text Analysis App")

text_input = st.text_area("Enter text to analyze:", "Streamlit is an amazing tool for building data apps!")

if st.button("Analyze"):
    # Language Detection
    st.subheader("Language Detection:")
    try:
        language = detect(text_input)
        st.success(f"Detected Language: {language}")
    except Exception as e:
        st.error(f"Error detecting language: {e}")

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

    # Text Summarization
    st.subheader("Text Summary:")
    if len(text_input.split()) > 20:
        summary = summarizer(text_input, max_length=50, min_length=25, do_sample=False)
        st.write(summary[0]['summary_text'])
    else:
        st.warning("Text is too short for summarization. Please enter a longer text.")

    # Text Classification
    st.subheader("Text Classification:")
    classification = text_classifier(text_input)
    for result in classification:
        st.write(f"Label: {result['label']}, Score: {result['score']:.2f}")
