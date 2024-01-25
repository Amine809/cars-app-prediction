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
    st.title('Application de prédiction du prix des voitures par Mohamed Amine Karmous')

    # Sidebar with user input
    st.sidebar.header("Fonctionnalités d'entrée")

    model_year = st.sidebar.number_input('Année modèle', value=2018)
    fuel_type = st.sidebar.selectbox('Type de carburant', ['Gasoline', 'Hybrid', 'E85 Flex Fuel', 'Diesel'])
    transmission = st.sidebar.selectbox('Transmission', ['Automatique', 'Manuelle'])
    clean_title = st.sidebar.checkbox('Titre propre')

    mil = st.sidebar.number_input('Kilométrage', value=53705)
    accidents = st.sidebar.checkbox('Accidentée')

    horsepower = st.sidebar.number_input('Puissance chevaux(ch din)', value=241.0)
    motor_l = st.sidebar.number_input('Cylindrée du moteur (L)', value=2.0)
    num_cylinders = st.sidebar.number_input('Nombre de cylindres', value=4)

    # Map categorical features to numerical format
    fuel_mapping = {'Gasoline': 1, 'Hybrid': 0, 'E85 Flex Fuel': 2, 'Diesel': -1}
    transmission_mapping = {'Automatique': 1, 'Manuelle': 0}

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
    if st.button('prédire le prix'):
        prediction = predict_price(model, scaler, user_input)
        st.success(f'Prix prévu: ${prediction:,.2f}')

if __name__ == '__main__':
    main()
