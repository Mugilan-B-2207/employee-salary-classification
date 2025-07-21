import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")  # ✅ Load the same scaler used during training

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
fnlwgt = st.sidebar.number_input("fnlwgt (Final Weight)", min_value=10000, max_value=1000000, value=50000)
workclass = st.sidebar.selectbox("Workclass", [
    'Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 
    'Self-emp-not-inc', 'State-gov', 'Others'
])
marital_status = st.sidebar.selectbox("Marital Status", [
    'Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
    'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
])
occupation = st.sidebar.selectbox("Occupation", [
    'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',
    'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
    'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving', 'Others'
])
relationship = st.sidebar.selectbox("Relationship", [
    'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'
])
race = st.sidebar.selectbox("Race", [
    'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
])
gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
native_country = st.sidebar.selectbox("Native Country", ['India', 'Mexico', 'United-States', 'Others'])

# Encoding
def encode_feature(value, category_list):
    return category_list.index(value) if value in category_list else category_list.index('Others')

# Encoding categories (same as training)
workclass_cat = ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Others']
marital_cat = ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed']
occupation_cat = ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 
                  'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 
                  'Transport-moving', 'Others']
relationship_cat = ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife']
race_cat = ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White']
gender_cat = ['Female', 'Male']
native_cat = ['India', 'Mexico', 'United-States', 'Others']

# Input data
input_data = pd.DataFrame([{
    'age': age,
    'workclass': encode_feature(workclass, workclass_cat),
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': encode_feature(marital_status, marital_cat),
    'occupation': encode_feature(occupation, occupation_cat),
    'relationship': encode_feature(relationship, relationship_cat),
    'race': encode_feature(race, race_cat),
    'gender': encode_feature(gender, gender_cat),
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': encode_feature(native_country, native_cat)
}])

st.subheader("🔎 Final Input to Model")
st.write(input_data)

# Predict with scaling
if st.button("Predict Salary Class"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.success(f"✅ Prediction: {prediction}")

# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    try:
        scaled_batch = scaler.transform(batch_data)
        batch_preds = model.predict(scaled_batch)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {str(e)}")