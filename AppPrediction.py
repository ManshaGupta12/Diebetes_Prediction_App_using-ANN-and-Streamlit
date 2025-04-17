import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import pickle
import joblib
from tensorflow.keras.models import load_model # type: ignore

# Load the pre-trained model
model = load_model("diabetes_ann_model.h5") 

with open("scaler.pkl", "rb") as f:
    scaler2 = pickle.load(f)

# Set up the app title and description
st.title('Diabetes Prediction App')
st.write('Enter the details below to predict whether you have diabetes or not.')

# Create the input fields for the user to enter their data
pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=50, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=100)
bmi = st.number_input('Body Mass Index (BMI)', min_value=0, max_value=50, value=30)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age (years)', min_value=18, max_value=120, value=30)

# Prepare the input data for prediction
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)

# Scale the input data
scaled_input = scaler2.transform(input_data)

# Button to make prediction
if st.button('Predict Diabetes'):
    prediction = model.predict(scaled_input)

    # If the prediction is greater than or equal to 0.5, predict positive (1), else negative (0)
    if prediction[0][0] >= 0.5:
        st.write("The model predicts: **Diabetic** 🛑")
    else:
        st.write("The model predicts: **Non-Diabetic** ✅")



