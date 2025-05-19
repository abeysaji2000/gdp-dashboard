import streamlit as st
import pandas as pd
import plotly.express as px

import os

# Set page title and layout
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon=":hospital:",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load the Kaggle dataset
@st.cache_data
def load_data():
    """Download and load diabetes prediction dataset from Kaggle."""
    dataset_path = "data/diabetes_prediction_dataset.csv"

    
  
    df = pd.read_csv(dataset_path)
    return df

df = load_data()

# -----------------------------------------------------------------------------
# Dashboard Header
st.title(":hospital: Diabetes Prediction Dashboard")
st.write("Explore and analyze the diabetes prediction dataset.")

# -----------------------------------------------------------------------------
# Dataset Description
st.header("📊 About the Dataset")
st.write(f"""
This dataset is sourced from Kaggle and focuses on predicting diabetes based on various health indicators.
It contains information about patients, including their age, gender, medical history, and lifestyle factors.

### **Dataset Details**
- **Source:** Kaggle (`iammustafatz/diabetes-prediction-dataset`)
- **Number of Records:** {df.shape[0]:,}
- **Number of Features:** {df.shape[1]:,}
- **Goal:** Predict the likelihood of diabetes based on health indicators.
""")

# **Column Descriptions**
st.subheader("📝 Column Descriptions")
st.write("""
- **gender** → Patient’s gender (`Female` / `Male`)
- **age** → Age of the patient (in years)
- **hypertension** → Presence of high blood pressure (`0`: No, `1`: Yes)
- **heart_disease** → History of heart disease (`0`: No, `1`: Yes)
- **smoking_history** → Smoking status (e.g., never, former, current)
- **bmi** → Body Mass Index (BMI)
- **HbA1c_level** → Hemoglobin A1c level
- **blood_glucose_level** → Fasting blood glucose level (mg/dL)
- **diabetes** → Diabetes diagnosis (`0`: No, `1`: Yes)
""")

# -----------------------------------------------------------------------------
# Sidebar Filters
st.sidebar.subheader("Dataset Sample")
st.sidebar.dataframe(df.head())

age_min, age_max = st.sidebar.slider(
    "Select Age Range", 
    int(df["age"].min()), 
    int(df["age"].max()), 
    (20, 50)
)
selected_gender = st.sidebar.selectbox("Select Gender", df["gender"].unique())

filtered_df = df[
    (df["age"] >= age_min) & 
    (df["age"] <= age_max) & 
    (df["gender"] == selected_gender)
]

# -----------------------------------------------------------------------------
# Interactive Visualizations

# 1. Age Distribution
st.subheader("📈 Age Distribution")
fig_age = px.histogram(filtered_df, x="age", nbins=30, title="Age Distribution", marginal="box")
st.plotly_chart(fig_age, use_container_width=True)

# 2. Diabetes Count
st.subheader("🩺 Diabetes Status Count")
fig_diabetes = px.bar(filtered_df, x="diabetes", color="diabetes", title="Diabetes Count")
st.plotly_chart(fig_diabetes, use_container_width=True)

# 3. BMI vs. Blood Glucose Level
st.subheader("🔬 BMI vs. Blood Glucose Level")
fig_bmi_glucose = px.scatter(filtered_df, x="bmi", y="blood_glucose_level", color="diabetes", title="BMI vs. Blood Glucose Level")
st.plotly_chart(fig_bmi_glucose, use_container_width=True)

# 4. Hypertension vs. Heart Disease
st.subheader("❤️ Hypertension vs. Heart Disease")
fig_hypertension_hd = px.bar(filtered_df, x="hypertension", y="heart_disease", color="diabetes", title="Hypertension vs. Heart Disease")
st.plotly_chart(fig_hypertension_hd, use_container_width=True)

# 5. Smoking History vs. HbA1c Level
st.subheader("🚬 Smoking History vs. HbA1c Level")
fig_smoking_hba1c = px.scatter(filtered_df, x="smoking_history", y="HbA1c_level", color="diabetes", title="Smoking History vs. HbA1c Level")
st.plotly_chart(fig_smoking_hba1c, use_container_width=True)

# 6. Blood Glucose Level Distribution
st.subheader("🩸 Blood Glucose Level Distribution")
fig_glucose = px.histogram(filtered_df, x="blood_glucose_level", nbins=20, title="Blood Glucose Level Distribution")
st.plotly_chart(fig_glucose, use_container_width=True)

# -----------------------------------------------------------------------------
# Display Data Table
st.subheader("🔍 Filtered Dataset")
st.dataframe(filtered_df)
