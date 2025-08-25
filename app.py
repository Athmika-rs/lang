import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🌍 Language Detection App")
st.write("Detects the language of a given sentence using TF-IDF + Logistic Regression.")

# User input
user_input = st.text_input("Enter a sentence:")

if st.button("Detect Language"):
    if user_input.strip() != "":
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        st.success(f"✅ Detected Language: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some text!")
