import streamlit as st
import pandas as pd
import kagglehub
import plotly.express as px
import plotly.graph_objects as go

# Set the page title and icon
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon=":hospital:",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Load the dataset
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

# Display dataset overview
st.sidebar.subheader("Dataset Overview")
st.sidebar.write(f"Total records: {df.shape[0]}")
st.sidebar.write(f"Number of features: {df.shape[1]}")
st.sidebar.dataframe(df.head())

# -----------------------------------------------------------------------------
# Filters
age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (20, 50))
selected_gender = st.sidebar.selectbox("Select Gender", df["Gender"].unique())

filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Gender"] == selected_gender)]

# -----------------------------------------------------------------------------
# **Interactive Plots**

# 1. Age Distribution
st.subheader("Age Distribution")
fig_age = px.histogram(filtered_df, x="Age", nbins=30, title="Age Distribution", marginal="box")
st.plotly_chart(fig_age, use_container_width=True)

# 2. Diabetes Count (Categorical)
st.subheader("Diabetes Status Count")
fig_diabetes = px.bar(filtered_df, x="Diabetes", color="Diabetes", title="Diabetes Count")
st.plotly_chart(fig_diabetes, use_container_width=True)

# 3. BMI vs. Glucose Level (Scatter Plot)
st.subheader("BMI vs. Glucose Level")
fig_bmi_glucose = px.scatter(filtered_df, x="BMI", y="Glucose", color="Diabetes", title="BMI vs. Glucose Level")
st.plotly_chart(fig_bmi_glucose, use_container_width=True)

# 4. Heatmap of Correlations
st.subheader("Feature Correlation Heatmap")
corr_matrix = filtered_df.corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns))
fig_heatmap.update_layout(title="Correlation Heatmap", autosize=True)
st.plotly_chart(fig_heatmap, use_container_width=True)

# 5. 3D Scatter Plot of BMI, Age, and Glucose
st.subheader("3D Scatter: BMI, Age, Glucose")
fig_3d = px.scatter_3d(filtered_df, x="BMI", y="Age", z="Glucose", color="Diabetes", title="3D Scatter of BMI, Age, and Glucose")
st.plotly_chart(fig_3d, use_container_width=True)

# 6. Parallel Coordinates Plot
st.subheader("Parallel Coordinates Plot")
fig_parallel = px.parallel_coordinates(filtered_df, dimensions=["BMI", "Glucose", "Age"], color="Diabetes", title="Parallel Coordinates Visualization")
st.plotly_chart(fig_parallel, use_container_width=True)

# -----------------------------------------------------------------------------
# Display Data Table
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

# -----------------------------------------------------------------------------
# Styling Enhancements (Optional)
st.markdown("""
<style>
    .reportview-container {
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
