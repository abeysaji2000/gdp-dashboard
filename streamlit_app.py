import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page title and layout
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon=":hospital:",
    layout="wide"
)

@st.cache_data
def load_data():
    dataset_path = "data/diabetes_prediction_dataset.csv"
    df = pd.read_csv(dataset_path)
    return df

df = load_data()

st.title(":hospital: Diabetes Prediction Dashboard")
st.write("Explore and analyze the diabetes prediction dataset.")

st.header("📊 About the Dataset")
st.write(f"""
This dataset is sourced from Kaggle and focuses on predicting diabetes based on various health indicators.
It contains information about patients, including their age, gender, medical history, and lifestyle factors.

### **Dataset Details**
- **Source:** Kaggle 
- **Number of Records:** {df.shape[0]:,}
- **Number of Features:** {df.shape[1]:,}
- **Goal:** Predict the likelihood of diabetes based on health indicators.
""")

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

# Sidebar filters
st.sidebar.subheader("Filters")
st.sidebar.subheader("Dataset Sample")
st.sidebar.dataframe(df.head())


age_min, age_max = st.sidebar.slider(
    "Select Age Range", 
    int(df["age"].min()), 
    int(df["age"].max()), 
    (40, 50)
)

selected_genders = st.sidebar.multiselect(
    "Select Gender(s)",
    options=df["gender"].unique().tolist(),
    default=df["gender"].unique().tolist(),
    key="gender_multiselect"
)

bmi_min, bmi_max = st.sidebar.slider(
    "Select BMI Range",
    float(df["bmi"].min()),
    float(df["bmi"].max()),
    (float(df["bmi"].min()), float(df["bmi"].max())),
    key="bmi_slider"
)

hba1c_min, hba1c_max = st.sidebar.slider(
    "Select HbA1c Level Range",
    float(df["HbA1c_level"].min()),
    float(df["HbA1c_level"].max()),
    (float(df["HbA1c_level"].min()), float(df["HbA1c_level"].max())),
    key="hba1c_slider"
)

glucose_min, glucose_max = st.sidebar.slider(
    "Select Blood Glucose Level Range",
    float(df["blood_glucose_level"].min()),
    float(df["blood_glucose_level"].max()),
    (float(df["blood_glucose_level"].min()), float(df["blood_glucose_level"].max())),
    key="glucose_slider"
)

# New sliders for binary numeric features
hypertension_min, hypertension_max = st.sidebar.slider(
    "Hypertension (0=No, 1=Yes)",
    0, 1, (0, 1), step=1, key="hypertension_slider"
)

heart_disease_min, heart_disease_max = st.sidebar.slider(
    "Heart Disease (0=No, 1=Yes)",
    0, 1, (0, 1), step=1, key="heart_disease_slider"
)

# Smoking history filter as selectbox
smoking_filter = st.sidebar.selectbox(
    "Smoking History",
    options=["All"] + df["smoking_history"].dropna().unique().tolist(),
    key="smoking_selectbox"
)

# Apply filters
filtered_df = df[
    (df["age"] >= age_min) & (df["age"] <= age_max) &
    (df["gender"].isin(selected_genders)) &
    (df["bmi"] >= bmi_min) & (df["bmi"] <= bmi_max) &
    (df["HbA1c_level"] >= hba1c_min) & (df["HbA1c_level"] <= hba1c_max) &
    (df["blood_glucose_level"] >= glucose_min) & (df["blood_glucose_level"] <= glucose_max) &
    (df["hypertension"] >= hypertension_min) & (df["hypertension"] <= hypertension_max) &
    (df["heart_disease"] >= heart_disease_min) & (df["heart_disease"] <= heart_disease_max)
]

if smoking_filter != "All":
    filtered_df = filtered_df[filtered_df["smoking_history"] == smoking_filter]

# Continue with your existing visualizations and tables here using filtered_df
# For example:

st.subheader("📈 Age Distribution")
fig_age = px.histogram(filtered_df, x="age", nbins=30, title="Age Distribution", marginal="box")
st.plotly_chart(fig_age, use_container_width=True)

# ... rest of your dashboard code ...


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

# 3D Scatter plot
color_map = {0: "dodgerblue", 1: "crimson"}
st.subheader("🌐 3D Visualization: BMI, HbA1c, Blood Glucose")
fig_3d = px.scatter_3d(
    filtered_df,
    x="bmi",
    y="HbA1c_level",
    z="blood_glucose_level",
    color=filtered_df["diabetes"].map(color_map),
    symbol="gender",
    title="3D Scatter: BMI vs. HbA1c vs. Glucose",
    opacity=0.7
)
st.plotly_chart(fig_3d, use_container_width=True)

# Sankey Diagram (same as before)
st.subheader("🔗 Sankey Diagram: Smoking History → Hypertension → Diabetes")

smoking_categories = filtered_df["smoking_history"].dropna().unique().tolist()
hypertension_categories = ['No', 'Yes']  # 0 -> No, 1 -> Yes
diabetes_categories = ['No', 'Yes']  # 0 -> No, 1 -> Yes

all_nodes = smoking_categories + hypertension_categories + diabetes_categories
label_to_index = {label: i for i, label in enumerate(all_nodes)}

def bin_label(val, categories=['No', 'Yes']):
    return categories[val]

source = []
target = []
value = []

for smoke in smoking_categories:
    for hyper in [0, 1]:
        count = filtered_df[(filtered_df['smoking_history'] == smoke) & (filtered_df['hypertension'] == hyper)].shape[0]
        if count > 0:
            source.append(label_to_index[smoke])
            target.append(label_to_index[bin_label(hyper, hypertension_categories)])
            value.append(count)

for hyper in [0, 1]:
    for diab in [0, 1]:
        count = filtered_df[(filtered_df['hypertension'] == hyper) & (filtered_df['diabetes'] == diab)].shape[0]
        if count > 0:
            source.append(label_to_index[bin_label(hyper, hypertension_categories)])
            target.append(label_to_index[bin_label(diab, diabetes_categories)])
            value.append(count)

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color="skyblue"
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color="lightgray"
    )
)])
fig_sankey.update_layout(title_text="Sankey Diagram: Smoking History → Hypertension → Diabetes", font_size=12)
st.plotly_chart(fig_sankey, use_container_width=True)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap of Numeric Features")
numeric_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
corr_matrix = filtered_df[numeric_cols].corr()
fig_heatmap = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    origin='lower',
    title="Correlation Heatmap"
)
st.plotly_chart(fig_heatmap, use_container_width=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Select numeric features for PCA
numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

# Standardize data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[numeric_features])

# Apply PCA - 2 components
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['diabetes'] = filtered_df['diabetes'].values
pca_df['gender'] = filtered_df['gender'].values

# Explained variance
explained_var = pca.explained_variance_ratio_ * 100

st.subheader("🔍 PCA: Principal Component Analysis")

st.write(f"Explained variance by PC1: {explained_var[0]:.2f}%, PC2: {explained_var[1]:.2f}%")

# Plot PCA scatter plot
fig_pca = px.scatter(
    pca_df,
    x='PC1',
    y='PC2',
    color='diabetes',
    symbol='gender',
    title='PCA - First Two Components',
    labels={'PC1': f'PC1 ({explained_var[0]:.2f}%)', 'PC2': f'PC2 ({explained_var[1]:.2f}%)'},
    opacity=0.7
)

st.plotly_chart(fig_pca, use_container_width=True)

# Display filtered dataset
st.subheader("🔍 Filtered  Dataset")
st.dataframe(filtered_df)
