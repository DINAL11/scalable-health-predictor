import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_data, make_prediction

st.title("Health Record Risk Predictor")

#load model
#after we trained our machine learning model (e.g., using RandomForestClassifier, LogisticRegression, etc.) in Phase 2/3, we will have to save it as .pkl
#could use something else too, if easier
model = joblib.load("model.pkl")

#uploading our cleaned data here I guess 
uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    if st.button("Predict"):
        processed = preprocess_data(df)
        predictions = make_prediction(model, processed)
        df["Prediction"] = predictions
        st.success("Predictions Generated!")
        st.dataframe(df)

        st.bar_chart(df["Prediction"].value_counts())

        #optional: Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results as CSV", csv, "predictions.csv", "text/csv")
