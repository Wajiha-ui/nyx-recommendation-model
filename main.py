import streamlit as st
import pickle  # or torch, tensorflow, etc., based on your model
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:  # Update filename if needed
        model = pickle.load(f)
    return model

model = load_model()

# Streamlit UI
st.title("AI Size Recommendation Model")
st.write("Enter your details to get the recommended size.")

# User inputs
height = st.number_input("Enter your height (cm):", min_value=100, max_value=250, value=170)
weight = st.number_input("Enter your weight (kg):", min_value=30, max_value=200, value=70)
age = st.number_input("Enter your age:", min_value=5, max_value=100, value=25)

# Prediction button
if st.button("Recommend Size"):
    features = np.array([[height, weight, age]])
    prediction = model.predict(features)[0]  # Assuming model returns a size label

    st.success(f"Recommended Size: {prediction}")

# Run using: streamlit run main.py
