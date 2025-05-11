import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Charger le modèle
model = joblib.load('random_forest_model1.pkl')

def get_input_data():
    st.title("Estimation de prix de voiture")
    st.write("Veuillez entrer les informations de la voiture :")

    # Définir les options prédéfinies
    brands = ['Renault', 'Citroën', 'Peugeot', 'Ford', 'Mercedes', 'BMW', 'Toyota', 'Volkswagen', 'Hyundai', 'Kia']
    models = ['Megane', 'C4', 'Partner', 'Kuga', 'C-Class', '7 Series', 'E-Class', 'Clio', '2008', 'C3', 'Hilux', 'Golf', 'Berlingo', 'Corolla', 'Accent', 'Rio', 'GLC', 'X3', 'X1', '3008', '3 Series', 'Picanto', 'Passat', 'Tucson', 'A-Class', 'Transit', 'Camry', 'Sprinter', 'Kadjar', '208', 'Ranger', 'Cerato', '5 Series', 'Rav4', 'Sportage', 'Symbol', 'C5', 'i10', 'Elantra', 'Jumpy', 'Jetta', 'Sorento', '308', 'Focus', 'Polo', 'Yaris', 'Fiesta', 'i20', 'Tiguan', 'Captur']
    car_types = ['SUV', 'Hatchback', 'Pickup', 'Sedan', 'Coupe']
    fuel_types = ['Petrol', 'Electric', 'Hybrid', 'Diesel']
    regions = ['Bizerte', 'Sousse', 'Gabès', 'Tataouine', 'Monastir', 'Kairouan', 'Tozeur', 'Sfax', 'Nabeul', 'Tunis']
    transmissions = ['Manual', 'Automatic']
    accident_histories = ['Yes', 'No']
    safety_features = ['Yes', 'No']  # For ABS, Airbags, etc.

    # User inputs
    brand = st.selectbox("Marque:", brands)
    model_car = st.selectbox("Modèle:", models)
    car_type = st.selectbox("Type de voiture:", car_types)
    fuel_type = st.selectbox("Type de carburant:", fuel_types)
    region = st.selectbox("Région:", regions)
    transmission = st.selectbox("Transmission:", transmissions)
    accident_history = st.selectbox("Historique d'accident:", accident_histories)
    abs_feature = st.selectbox("ABS:", safety_features)
    airbags = st.selectbox("Airbags:", safety_features)
    blind_spot = st.selectbox("Blind Spot Monitor:", safety_features)
    
    driven_km = st.number_input("Kilométrage (Driven_KM)", min_value=0)
    enginev = st.number_input("Cylindrée (EngineV)", min_value=0.0)
    age = st.number_input("Âge de la voiture (Age)", min_value=0)

    # Create a DataFrame with all possible features
    input_dict = {
        'Brand': brand,
        'Model': model_car,
        'Type': car_type,
        'FuelType': fuel_type,
        'Region': region,
        'Transmission': transmission,
        'Accident_History': accident_history,
        'ABS': abs_feature,
        'Airbags': airbags,
        'Blind Spot Monitor': blind_spot,
        'Driven_KM': driven_km,
        'EngineV': enginev,
        'Age': age
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Get dummy variables for all categorical features
    categorical_cols = ['Brand', 'Model', 'Type', 'FuelType', 'Region', 'Transmission', 
                       'Accident_History', 'ABS', 'Airbags', 'Blind Spot Monitor']
    
    # One-hot encode categorical variables
    encoded_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Align columns with model's training data
    # Get expected features from the model
    try:
        expected_features = model.feature_names_in_
    except AttributeError:
        # If using older scikit-learn, you'll need to manually specify features
        # This should match exactly what was used during training
        expected_features = [...]  # You need to fill this with actual feature names
    
    # Add missing columns with 0s
    for feature in expected_features:
        if feature not in encoded_df.columns:
            encoded_df[feature] = 0
    
    # Ensure columns are in correct order
    encoded_df = encoded_df[expected_features]
    
    return encoded_df

def predict_price():
    input_data = get_input_data()
    prediction = model.predict(input_data)
    st.success(f"\nLe prix estimé de la voiture est : {prediction[0]:.2f} TND")

if __name__ == "__main__":
    predict_price()