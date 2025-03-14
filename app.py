import streamlit as st
import pickle
import re

# Load models
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Preprocessing function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Streamlit UI
st.title("📈 Financial News Sentiment Analyzer")

st.write("Enter a financial news headline, and the model will classify its sentiment as Positive, Negative, or Neutral.")

# User input
user_input = st.text_area("Enter News Headline:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news headline.")
    else:
        # Process input
        cleaned_text = clean_text(user_input)
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = lr_model.predict(text_tfidf)[0]
        sentiment = ["Neutral", "Positive", "Negative"][prediction]

        # Display result
        st.subheader(f"Predicted Sentiment: {sentiment}")
