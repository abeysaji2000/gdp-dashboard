import streamlit as st
import pandas as pd
import math
import kagglehub
import plotly.express as px

# Set the title and favicon
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon=":hospital:",
)

# -----------------------------------------------------------------------------
# Load the dataset
@st.cache_data
def get_diabetes_data():
    """Load diabetes prediction data from Kaggle dataset."""
    file_path = "diabetes_prediction_dataset.csv"
    df = kagglehub.load_dataset(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        "iammustafatz/diabetes-prediction-dataset",
        file_path
    )
    return df

df = get_diabetes_data()

# -----------------------------------------------------------------------------
# Dashboard Header
'''
# :hospital: Diabetes Prediction Dashboard

Explore diabetes prediction data sourced from [Kaggle](https://www.kaggle.com/).
'''

# Display dataset overview
st.subheader("Dataset Overview")
st.write(f"Total records: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")

# -----------------------------------------------------------------------------
# Filters
age_range = st.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 50))
selected_gender = st.selectbox("Select Gender", df["Gender"].unique())
filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Gender"] == selected_gender)]

# -----------------------------------------------------------------------------
# Interactive Plots

# 1. Age Distribution
st.subheader("Age Distribution")
fig_age = px.histogram(filtered_df, x="Age", nbins=30, title="Age Distribution", marginal="box")
st.plotly_chart(fig_age, use_container_width=True)

# 2. Diabetes Count
st.subheader("Diabetes Status Count")
fig_diabetes = px.bar(filtered_df, x="Diabetes", color="Diabetes", title="Diabetes Count")
st.plotly_chart(fig_diabetes, use_container_width=True)

# 3. BMI vs. Glucose Level (Scatter Plot)
st.subheader("BMI vs. Glucose Level")
fig_bmi_glucose = px.scatter(filtered_df, x="BMI", y="Glucose", color="Diabetes", title="BMI vs. Glucose Level")
st.plotly_chart(fig_bmi_glucose, use_container_width=True)

# -----------------------------------------------------------------------------
# Display Data Table
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

