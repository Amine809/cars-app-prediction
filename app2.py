import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

def predict_price(model, scaler, observation):
    # Scale the new observation
    new_observation_scaled = scaler.transform(observation)

    # Make a prediction for the new observation
    prediction = model.predict(new_observation_scaled)

    return prediction[0]

def main():
    st.title('Car Price Prediction App')

    # Sidebar with user input
    st.sidebar.header('Input Features')

    model_year = st.sidebar.number_input('Model Year', value=2018)
    fuel_type = st.sidebar.selectbox('Fuel Type', ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel'])
    transmission = st.sidebar.selectbox('Transmission', ['Automatic', 'Manual'])
    clean_title = st.sidebar.checkbox('Clean Title')

    mil = st.sidebar.number_input('Mileage', value=53705)
    accidents = st.sidebar.checkbox('Accidents Reported')

    horsepower = st.sidebar.number_input('Horsepower', value=241.0)
    motor_l = st.sidebar.number_input('Engine Displacement (L)', value=2.0)
    num_cylinders = st.sidebar.number_input('Number of Cylinders', value=4)

    # Map categorical features to numerical format
    fuel_mapping = {'Gasoline': 1, 'Hybrid': 0, 'E85 Flex Fuel': 2, 'Diesel': -1}
    transmission_mapping = {'Automatic': 1, 'Manual': 0}

    fuel_type_numeric = fuel_mapping[fuel_type]
    transmission_numeric = transmission_mapping[transmission]

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        'model_year': [model_year],
        'fuel_type': [fuel_type_numeric],
        'transmission': [transmission_numeric],
        'clean_title': [1 if clean_title else 0],
        'mil': [mil],
        'accidents': [1 if accidents else 0],
        'horsepower': [horsepower],
        'motor(l)': [motor_l],
        'number_of_cylinders': [num_cylinders],
    })

    # Predict the price
    if st.button('Predict Price'):
        prediction = predict_price(model, scaler, user_input)
        st.success(f'Predicted Price: ${prediction:,.2f}')

if __name__ == '__main__':
    main()
