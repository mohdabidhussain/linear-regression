import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('logistic_model.pkl', 'rb'))

st.title("🚢 Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare ($)", 0.0, 600.0, 50.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
sex = st.selectbox("Sex", ['male', 'female'])
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

sex_male = 1 if sex == 'male' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Age': age,
    'Fare': fare,
    'SibSp': sibsp,
    'Parch': parch,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}])

if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    result = "🟢 Survived" if prediction == 1 else "🔴 Did Not Survive"
    st.subheader(f"Prediction: {result}")