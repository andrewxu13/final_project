import numpy as np
import pandas as pd
import pickle
import streamlit as st

@st.cache_resource
def load_model():
    return pickle.load(open('andrewxu13/final_project/logistic_regression.pkl', 'rb'))

model = load_model()

def calculate_bmi(height, weight):
    """ Calculate BMI from height in cm and weight in kg. """
    if height <= 0 or weight <= 0:
        return 0
    return weight / ((height / 100) ** 2)

def encode_age(age):
    """ Encode age into specific age groups based on the logic provided. """
    if 35 <= age <= 44:
        return 5
    elif 45 <= age <= 54:
        return 6
    elif 55 <= age <= 64:
        return 7
    elif 65 <= age <= 74:
        return 8
    elif 75 <= age <= 84:
        return 9
    elif age >= 85:
        return 10
    else:
        return None  # Return None or a specific code for ages not covered
    
def encode_gender(gender):
    if gender == "male":
        return 1
    else:
        return 0
        

def main():
    st.title("Are you at risk of OSA (Obstructive Sleep Apnea)?")
    st.write("This is a simple web app to predict the likelihood of OSA given the inputs from users.")

    # Define frequency mappings and their corresponding questions
    frequency_mappings = {
        "unrested": {
            "question": "How often do you feel unrested?",
            "options": {"Never": 0, "Less than 1 per week": 1, "1 or 2 per week": 2, "Every other day": 3, "Everyday": 4, "Don't know": 8}
        },
        "snore": {
            "question": "How often do you snore?",
            "options": {"Yes": 0, "No": 1, "Don't know": 8}
        },
        "aspirin": {
            "question": "How often do you use aspirin?",
            "options": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4, "Don't know": 8}
        },
        "gender": {
            "question": "What's your gender?",
            "options": {"Male": 0, "Female": 1, "Prefer not to say": 8}
        }
    }

    user_inputs = {}
    for key, details in frequency_mappings.items():
        # Generate a unique key for each selectbox by combining the key with a suffix
        unique_key = f"selectbox_{key}"
        default_option = list(details['options'].keys())[0]  # Default to the first option in each category
        user_inputs[key] = st.selectbox(
            details['question'],
            options=list(details['options'].keys()),
            index=list(details['options'].keys()).index(default_option),
            key=unique_key  # Ensure each selectbox has a unique key
        )

    # Number inputs with reasonable defaults
    default_height = 170.0  # Default height in cm
    default_weight = 70.0   # Default weight in kg
    default_neck = 40.0     # Default neck circumference in cm
    default_age = 50        # Default age
    default_waist = 85.0    # Default waist in cm

    user_inputs['height'] = st.number_input("Height (in cm):", min_value=0.0, value=default_height, format="%.2f")
    user_inputs['weight'] = st.number_input("Weight (in kg):", min_value=0.0, value=default_weight, format="%.2f")
    user_inputs['neck_circumference'] = st.number_input("Neck Circumference (in cm):", min_value=0.0, value=default_neck, format="%.2f")
    user_inputs['age'] = st.number_input("Age:", min_value=0, value=default_age, format="%d")
    user_inputs['waist'] = st.number_input("Waist (in cm):", min_value=0.0, value=default_waist, format="%.2f")

    if st.button("Predict"):
        if not all(user_inputs.values()):
            st.error("Please fill in all required fields.")
            return
        
        bmi = calculate_bmi(user_inputs['height'], user_inputs['weight'])
        encoded_age = encode_age(user_inputs['age'])
        if encoded_age is None:
            st.error("Age out of the expected range.")
            return

        # Encode inputs
        for key, details in frequency_mappings.items():
            user_inputs[key] = details['options'][user_inputs[key]]

        user_data = {
            "age_group": encoded_age,
            "aspirin": user_inputs['aspirin'],
            "gender": user_inputs['gender'],
            "neck20": user_inputs['neck_circumference'],
            "waist": user_inputs['waist'],
            "av_bmi": bmi,
            "unrested": user_inputs['unrested'],
            "treatedSnoring": user_inputs['snore']
        }
        predict = model.predict(pd.DataFrame(user_data, index=[0]))
        predict_proba = model.predict_proba(pd.DataFrame(user_data, index=[0]))[0][1]
        
        col1, col2 = st.columns(2)

        with col1:
            if predict[0] == 0:
                st.markdown(f"### ✅ You are **not** at risk of OSA")
            else:
                st.markdown(f"### ❌ You are at risk of OSA")

        with col2:
            # Display the probability in a visually appealing way
            st.metric(label="Risk Probability", value=f"{predict_proba:.2%}")

        # Additional details or advice
        if predict[0] != 0:
            st.info("👉 We recommend consulting a healthcare professional for further evaluation.")

        # Maybe add a progress bar or a more graphical representation
        progress_bar = st.progress(0)
        progress_bar.progress(predict_proba)

if __name__ == "__main__":
    main()

