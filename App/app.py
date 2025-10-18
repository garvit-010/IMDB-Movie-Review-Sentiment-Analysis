import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

model = load_model(r"Analysis/lstm_w2v_model.keras")
tokenizer = joblib.load(r"Analysis/tokenizer.pkl")

st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict sentiment.")

text = st.text_area("Review Text")

if st.button("Predict"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=200)
        pred = model.predict(pad)[0][0]
        sentiment = "😊 Positive" if pred > 0.5 else "😞 Negative"
        st.success(f"**Sentiment:** {sentiment} (Score: {pred:.4f})")
    else:
        st.warning("Please enter a review.")
