import streamlit as st
import pandas as pd 
import joblib 

st.set_page_config(
    page_title = "Heart Disease Prediction",
    page_icon = "❤️",
    layout = "centered"
)

st.sidebar.title("About")

st.sidebar.info("""
❤️ Heart Disease prediction
                
Developed by Diksha Dhaigude
""")

st.markdown(
    "<h1 style = 'text-align:center; color:red❤️ Heart Disease Prediction</h1>",
    unsafe_allow_html = True
)

st.write("Predict the risk of heart disease using Machine Learning")


model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl") 


st.title("Heart Disease Prediction By Diksha❤️")
st.markdown("Provide the following details to check your hear disease risk:")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("SEX",['M','F'])
    chest_pain = st.selectbox('Chest Pain type', ['ATA', 'NAP', 'TA', 'ASY'])
    resting_bp = st.number_input('Resting Blood Pressure(mm Hg)', 88, 200, 120)
    cholesterol = st.number_input('Cholesterol (mg/dL)', 100, 600, 200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mh/dl', [0,1])

with col2:
    resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    max_hr = st.slider('Max Heart Rate', 60, 220, 150)
    exercise_angina = st.selectbox('Exercise-Induced Angina', ['Y', 'N'])
    oldpeak = st.slider('Oldpeak(ST Depression)', 0.0, 6.0, 1.0)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])


if st.button('Predict Heart Disease'):
    raw_input = {
        'Age' : age,
        'RestingBP' : resting_bp,
        'Cholesterol' : cholesterol,
        'FastingBS' : fasting_bs,
        'MaxHR' : max_hr,
        'Oldpeak' : oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'Exerciseangina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope:1
    }
    
    
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of heart disease")
    else:
        st.success("✅ low risk of Heart disease")      


st.markdown("---")
st.markdown(
    "<center>Made with ❤️ by Diksha Dhaigude</center>",
    unsafe_allow_html=True
)
