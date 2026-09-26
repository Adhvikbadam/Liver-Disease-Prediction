import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("svm.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("columns.pkl")

st.title("Liver Disease Prediction by ADHVIK")
st.markdown("Provide the following details to check your liver disease risk:")

age=st.slider("Age", 18, 100, 25)
gender=st.selectbox("Gender", ["Male", "Female"])
tot_bilirubin=st.number_input("Total Bilirubin", min_value=0.0, step=0.1)
direct_bilirubin=st.number_input("Direct Bilirubin", min_value=0.0, step=0.1)
alkphos=st.number_input("Alkaline Phosphotase", min_value=0.0, step=1.0)
sgpt=st.number_input("Alanine Aminotransferase", min_value=0.0, step=1.0)
sgot=st.number_input("Aspartate Aminotransferase", min_value=0.0, step=1.0)
tot_proteins=st.number_input("Total Proteins", min_value=0.0, step=0.1)
albumin=st.number_input("Albumin", min_value=0.0, step=0.1)
ag_ratio=st.number_input("Albumin-Globulin Ratio", min_value=0.0, step=0.1)

if st.button("Predict"):

 input_df = pd.DataFrame({
   
    "age": [age],
    "gender": [gender],
    "tot_bilirubin": [tot_bilirubin],
    "direct_bilirubin": [direct_bilirubin],
    "alkphos": [alkphos],
    "sgpt": [sgpt],
    "sgot": [sgot],
    "tot_proteins": [tot_proteins],
    "albumin": [albumin],
    "ag_ratio": [ag_ratio]
 })
 input_df["sgot_log"] = np.log1p(input_df["sgot"])
 input_df["sgpt_log"] = np.log1p(input_df["sgpt"])
 input_df["tot_bilirubin_log"] = np.log1p(
        input_df["tot_bilirubin"]
 )
 input_df["gender_Male"] = (
        input_df["gender"] == "Male"
    ).astype(int)


    # Remove original categorical column
 input_df.drop("gender", axis=1, inplace=True)


    

 for col in features:
    if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[features]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
 if prediction == 1:
        st.error("⚠️ High Risk of Liver Disease. Please consult a healthcare professional.")
 else:
        st.success("✅ Low Risk of Liver Disease")

