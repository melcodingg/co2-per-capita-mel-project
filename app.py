import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Indonesia CO2 Predictor", layout="centered")

st.title("🇮🇩 Indonesia CO₂ Emissions Predictor")
st.write("This app uses a Linear Regression model trained on historical data to estimate Indonesia's future CO₂ emissions per capita (SDG 13: Climate Action).")

# Load the saved model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Could not find 'model.pkl'. Please ensure the model file is uploaded to the repository.")
    st.stop()

# User input for Year
year_input = st.number_input("Enter Year to Predict:", min_value=1891, max_value=2100, value=2023, step=1)

if st.button("Predict Emissions"):
    input_df = pd.DataFrame({"Year": [year_input]})
    prediction = model.predict(input_df)[0]
    
    st.metric(label="Predicted CO₂ per capita", value=f"{prediction:.4f} metric tons")
    
    st.info("ℹ️ **Note:** Predictions are estimates based on linear historical patterns. Real future emissions are highly dependent on policies, economic factors, and technology changes.")
