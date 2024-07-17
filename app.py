import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 

# Load the SVM model
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load the vectorizer used during training
with open('vectorize.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess and classify messages
def classify_message(message):
    # Preprocess the message using the vectorizer
    message_vectorized = vectorizer.transform([message])
    # Predict using the SVM model
    prediction = svm_model.predict(message_vectorized)[0]
    return prediction

# Streamlit app
def main():
    st.title('Spam Filter')
    message = st.text_area('Enter your message here:')
    
    if st.button('Predict'):
        if message:
            prediction = classify_message(message)
            st.write(f'Prediction: {prediction}')
        else:
            st.warning('Please enter a message to classify.')

if __name__ == '__main__':
    main()
