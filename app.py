import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load(r"C:\Users\saeem\Desktop\Epilepsy\random_forest_beed_model.joblib")
scaler = joblib.load(r"C:\Users\saeem\Desktop\Epilepsy\random_forest_beed_model_scaler.joblib")  # Adjust the filename if needed

st.set_page_config(page_title="🧠 NeuroHealth Epilepsy EEG Classifier", layout="wide")

st.title("🧠 NeuroHealth Epilepsy EEG Classifier")
st.markdown("Upload EEG data or manually enter values for X1 to X16 to predict the EEG stage.")

# Define column names
features = [f"X{i+1}" for i in range(16)]

# Tabs for input method
tab1, tab2 = st.tabs(["📄 Upload CSV", "⌨️ Manual Entry"])

with tab1:
    uploaded_file = st.file_uploader("Upload a CSV file with EEG features (X1 to X16) 16 EEG channels corresponding to different brain regions", type=["csv"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("📊 Preview of Uploaded Data:", data.head())
            if all(col in data.columns for col in features):
                X = scaler.transform(data[features])
                predictions = model.predict(X)
                data["Prediction"] = predictions
                st.success("✅ Predictions completed")
                st.write(data)

                # Download option
                csv = data.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            else:
                st.error(f"CSV must contain columns: {features}")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

with tab2:
    st.subheader("Enter EEG values manually")
    user_input = []
    cols = st.columns(4)
    for i, col in enumerate(cols * 4):  # 4 columns * 4 rows
        value = col.number_input(f"X{i+1}", min_value=-200.0, max_value=200.0, value=0.0, step=0.1)
        user_input.append(value)

    if st.button("🧠 Predict"):
        input_df = pd.DataFrame([user_input], columns=features)
        X_scaled = scaler.transform(input_df)
        prediction = model.predict(X_scaled)[0]

        labels = {
            0: "Healthy",
            1: "Generalized Seizure",
            2: "Focal Seizure",
            3: "Seizure with Event"
        }
        st.success(f"🧾 Predicted Class: **{labels.get(prediction, 'Unknown')}**")
