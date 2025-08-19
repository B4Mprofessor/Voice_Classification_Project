# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
MODEL_FILENAME = 'svm_model.pkl'
SCALER_FILENAME = 'scaler.pkl'

# Feature names based on the dataset description
FEATURE_NAMES = [
    'mean_spectral_centroid', 'std_spectral_centroid',
    'mean_spectral_bandwidth', 'std_spectral_bandwidth',
    'mean_spectral_contrast', 'mean_spectral_flatness',
    'mean_spectral_rolloff', 'zero_crossing_rate', 'rms_energy',
    'mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch',
    'spectral_skew', 'spectral_kurtosis',
    'energy_entropy', 'log_energy',
    'mfcc_1_mean', 'mfcc_1_std', 'mfcc_2_mean', 'mfcc_2_std',
    'mfcc_3_mean', 'mfcc_3_std', 'mfcc_4_mean', 'mfcc_4_std',
    'mfcc_5_mean', 'mfcc_5_std', 'mfcc_6_mean', 'mfcc_6_std',
    'mfcc_7_mean', 'mfcc_7_std', 'mfcc_8_mean', 'mfcc_8_std',
    'mfcc_9_mean', 'mfcc_9_std', 'mfcc_10_mean', 'mfcc_10_std',
    'mfcc_11_mean', 'mfcc_11_std', 'mfcc_12_mean', 'mfcc_12_std',
    'mfcc_13_mean', 'mfcc_13_std'
]

# --- Helper Functions ---
@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    """Loads the trained model and scaler with error handling."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"**Error:** Required file not found: {e}")
        st.info("Please ensure `svm_model.pkl` and `scaler.pkl` are in the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"**Error:** An unexpected error occurred while loading files: {e}")
        st.stop()

def get_prediction(model, scaler, features_df):
    """Performs scaling, prediction, and probability calculation."""
    try:
        features_scaled = scaler.transform(features_df)
    except Exception as e:
        st.error(f"**Error during feature scaling:** {e}")
        return None, None

    try:
        prediction = model.predict(features_scaled)[0]
        probabilities = None
        if hasattr(model, "predict_proba"):
             probabilities = model.predict_proba(features_scaled)[0]
        return prediction, probabilities
    except Exception as e:
        st.error(f"**Error during prediction:** {e}")
        return None, None

# --- App Title and Info ---
st.set_page_config(page_title="Human Voice Classifier", page_icon=":microphone:")
st.title("Human Voice Gender Classification")
st.markdown("""
This application predicts the gender (Male or Female) of a voice sample based on pre-extracted audio features using a trained Support Vector Machine (SVM) model.
""")

# --- Load Model and Scaler ---
# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, MODEL_FILENAME)
scaler_path = os.path.join(current_dir, SCALER_FILENAME)

with st.spinner("Loading model and scaler..."):
    model, scaler = load_model_and_scaler(model_path, scaler_path)

# --- Input Section ---
st.header("Input Voice Features")
st.markdown("Please enter the values for the 43 audio features.")

# Create input fields for each feature
user_input = []
# Using st.columns to make the input look a bit cleaner
cols = st.columns(2)
for i, feature in enumerate(FEATURE_NAMES):
    with cols[i % 2]: # Alternate between the two columns
        # Default value of 0.0, step of 0.01 for precision
        value = st.number_input(f"{feature}", value=0.0, key=feature, format="%.6f", step=0.01, help=f"Enter the {feature} value.")
        user_input.append(value)

# --- Prediction Section ---
st.header("Prediction Result")

# Button to trigger prediction
if st.button("Classify Voice", type="primary"):
    # 1. Convert input list to DataFrame
    input_df = pd.DataFrame([user_input], columns=FEATURE_NAMES)

    # 2. Get prediction and probabilities
    prediction, probabilities = get_prediction(model, scaler, input_df)

    # 3. Display the result if successful
    if prediction is not None:
        st.subheader("Result")
        # Map label to human-readable format (1=Male, 0=Female as per PDF)
        label_map = {0: "Female", 1: "Male"}
        predicted_gender = label_map.get(prediction, f"Unknown Label ({prediction})")
        
        # Display prediction
        st.success(f"**Predicted Gender:** {predicted_gender}")

        # Display probabilities if available
        if probabilities is not None:
            prob_female = probabilities[0]
            prob_male = probabilities[1]
            st.write(f"**Confidence:**")
            st.write(f"- Female: {prob_female:.2%}")
            st.write(f"- Male: {prob_male:.2%}")
        else:
            st.info("Model does not provide probability estimates.")

# --- Sidebar Information ---
st.sidebar.title("About")
st.sidebar.info(
    "This application classifies human voice samples as Male or Female "
    "based on 43 extracted audio features using a trained machine learning model."
)
st.sidebar.markdown("**Project:** Human Voice Classification and Clustering")
st.sidebar.markdown("**Model:** Support Vector Machine (SVM)")
st.sidebar.markdown("**Features:**")
# Display features in a scrollable expander in the sidebar
with st.sidebar.expander("List of Features"):
    for feat in FEATURE_NAMES:
        st.write(f"- {feat}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** This app requires manual input of pre-extracted features. "
                    "A full version would take an audio file and extract these features automatically.")
