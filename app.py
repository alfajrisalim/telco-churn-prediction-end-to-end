import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Telco Churn Dashboard", layout="wide")

# -------------------- LOAD MODEL -------------------- #
@st.cache_resource
def load_model():
    return joblib.load("churn_xgb_full_pipeline.pkl")

model = load_model()

# -------------------- LOAD DEFAULT DATA -------------------- #
@st.cache_data
def load_default_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
    return df

df = load_default_data()

# -------------------- SIDEBAR -------------------- #
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "EDA", "Batch Prediction", "Single Prediction", "Feature Importance", "SHAP Explanation"])

# -------------------- HOME -------------------- #
if menu == "Home":
    st.title("📊 Telco Customer Churn Dashboard")
    st.write("End-to-end machine learning project with XGBoost, preprocessing pipeline, SMOTE, and Streamlit deployment.")
    st.write(f"Dataset size: **{df.shape[0]} rows**, **{df.shape[1]} columns**")

# -------------------- EDA -------------------- #
elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Contract Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Contract', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Tenure Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['tenure'], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------- BATCH PREDICTION -------------------- #
elif menu == "Batch Prediction":
    st.title("📤 Batch Prediction")
    st.write("Upload dataset to get churn predictions.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        preds = model.predict(new_df)
        new_df['Churn_Prediction'] = preds

        st.write("### Preview")
        st.dataframe(new_df.head())

        csv = new_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv")

# -------------------- SINGLE PREDICTION -------------------- #
elif menu == "Single Prediction":
    st.title("🧍 Single Customer Prediction")

    st.write("Fill the form below to predict whether a customer will churn.")

    sample = {}
    for col in df.drop(columns=['Churn', 'customerID']).columns:
        if df[col].dtype == 'object':
            sample[col] = st.selectbox(col, df[col].unique())
        else:
            sample[col] = st.number_input(col, value=float(df[col].median()))

    if st.button("Predict"):
        sample_df = pd.DataFrame([sample])
        pred = model.predict(sample_df)[0]
        st.subheader(f"Prediction: {'Churn' if pred == 1 else 'Not Churn'}")

# -------------------- FEATURE IMPORTANCE -------------------- #
elif menu == "Feature Importance":
    st.title("🔥 Feature Importance (XGBoost)")

    booster = model.named_steps['model']
    importances = booster.feature_importances_
    feature_names = model.named_steps['preprocess'].get_feature_names_out()

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values(by="importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature", ax=ax)
    st.pyplot(fig)

# -------------------- SHAP EXPLANATION -------------------- #
elif menu == "SHAP Explanation":
    st.title("🧠 SHAP Model Explanation")

    st.write("Explaining predictions using SHAP values.")

    # Extract model
    booster = model.named_steps['model']
    preproc = model.named_steps['preprocess']

    X_processed = preproc.transform(df.drop(columns=['Churn', 'customerID']))

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_processed)

    st.subheader("Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, X_processed, feature_names=preproc.get_feature_names_out(), show=False)
    st.pyplot(fig)

    st.subheader("Force Plot (First Customer)")
    fig2 = shap.force_plot(explainer.expected_value, shap_values[0], matplotlib=True)
    st.pyplot(fig2)
