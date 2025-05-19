import streamlit as st
import pandas as pd
import kagglehub
import plotly.express as px
import plotly.graph_objects as go

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
    """Load diabetes prediction dataset from Kaggle."""
    file_path = "diabetes_prediction_dataset.csv"
    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "iammustafatz/diabetes-prediction-dataset",
        file_path
    )
    return df

df = load_data()

# -----------------------------------------------------------------------------
# Dashboard Header
st.title(":hospital: Diabetes Prediction Dashboard")
st.write("Explore and analyze the diabetes prediction dataset.")

# -----------------------------------------------------------------------------
# Dataset Description
st.header("📊 About the Dataset")
st.write("""
This dataset is sourced from Kaggle and focuses on predicting diabetes based on various health indicators.
It contains information about patients, including their age, gender, medical history, and lifestyle factors.

### **Dataset Details**
- **Source:** Kaggle (`iammustafatz/diabetes-prediction-dataset`)
- **Number of Records:** {:,}
- **Number of Features:** {:,}
- **Goal:** Predict the likelihood of diabetes based on health indicators.
""".format(df.shape[0], df.shape[1]))

# **Column Descriptions**
st.subheader("📝 Column Descriptions")
st.write("""
- **gender** → Patient’s gender (`0`: Female, `1`: Male)
- **age** → Age of the patient (in years)
- **hypertension** → Presence of high blood pressure (`0`: No, `1`: Yes)
- **heart_disease** → History of heart disease (`0`: No, `1`: Yes)
- **smoking_history** → Smoking status (0: Never, 1: Former, 2: Current)
- **bmi** → Body Mass Index (BMI), an indicator of body fat
- **HbA1c_level** → Hemoglobin A1c level, showing blood sugar control
- **blood_glucose_level** → Fasting blood glucose level (mg/dL)
- **diabetes** → Diabetes diagnosis (`0`: No, `1`: Yes)
""")

# Display dataset sample
st.sidebar.subheader("Dataset Sample")
st.sidebar.dataframe(df.head())

# -----------------------------------------------------------------------------
# Filters
age_range = st.sidebar.slider("Select Age Range", int(df["age"].min()), int(df["age"].max()), (20, 50))
selected_gender = st.sidebar.selectbox("Select Gender", df["gender"].unique())

filtered_df = df[(df["age"] >= age_range) & (df["age"] <= age_range) & (df["gender"] == selected_gender)]

# -----------------------------------------------------------------------------
# **Interactive Plots**

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
