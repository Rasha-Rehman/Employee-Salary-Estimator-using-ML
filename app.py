# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# Load model
model = joblib.load("best_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Employee Salary Estimator", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: black;
        color: white;
    }
    .stSelectbox label, .stNumberInput label {
        color: pink !important;
    }
    .main-title {
        color: #00ffcc;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Employee Salary Estimator</div>', unsafe_allow_html=True)
st.markdown("Upload employee CSV file or fill in details below:")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Manual input
if not uploaded_file:
    age = st.selectbox("Age", list(range(18, 61)))
    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov'])
    education = st.selectbox("Education Level", ['Bachelors', 'HS-grad', 'Some-college', 'Masters'])
    marital = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
    occupation = st.selectbox("Occupation", [
    'Exec-managerial', 'Craft-repair', 'Sales', 'Prof-specialty',
    'Other-service', 'Adm-clerical', 'Machine-op-inspct', 'Transport-moving',
    'Handlers-cleaners', 'Tech-support', 'Protective-serv', 'Farming-fishing',
    'Priv-house-serv', 'Armed-Forces'
])

    gender = st.selectbox("Gender", ['Male', 'Female'])
    hours = st.selectbox("Hours per Week", list(range(1, 101)))

    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital,
        'occupation': occupation,
        'gender': gender,
        'hours-per-week': hours
    }])
else:
    input_df = pd.read_csv(uploaded_file)

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Prediction: {prediction[0]}")
        
        # Optional: Display evaluation chart
        st.subheader("📊 Evaluation Chart (Sample Distribution)")
        labels = input_df['gender'].value_counts().index
        counts = input_df['gender'].value_counts().values

        fig, ax = plt.subplots()
        ax.bar(labels, counts, color=['#e84393', '#0984e3'])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
