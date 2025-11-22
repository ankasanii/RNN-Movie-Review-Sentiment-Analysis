#Step 1: Import necessary libraries and load the model and deploy steamlit
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy
import tensorflow as tf
import streamlit as st

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for (key, value) in word_index.items()}

#load the pre-trained model with relu activation function
model = load_model('rnn_imdb_model.h5')

# Step 2: Helper function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_user_input(text):
    # Tokenize and encode the user input
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad the encoded input to the maximum length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# step 4: prediction function
def predict_sentiment(review_text):
    preprocessed_review = preprocess_user_input(review_text)
    prediction = model.predict(preprocessed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Step 3: Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment.")

# user input
user_review = st.text_area("Enter your movie review:")
if st.button("Analyze"):
    sentiment, confidence = predict_sentiment(user_review)
    st.write(f"Sentiment: {sentiment}, Prediction Score: {confidence:.2f}")
else:
    st.write("Please enter a review and click 'Analyze' to see the sentiment.")