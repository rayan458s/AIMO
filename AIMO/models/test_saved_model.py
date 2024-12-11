import joblib
import pandas as pd

# Load the saved model
model_path = "random_forest_model.pk1"
try:
    rf_model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Print the model's expected features
print("Model Feature Names:", rf_model.feature_names_in_)

# Create a dummy input DataFrame with the same feature set
test_input = pd.DataFrame([{
    'postal_code': 75001,
    'size': 60.5,
    'floor': 3,
    'land_size': 0.0,
    'nb_rooms': 3,
    'nb_bedrooms': 2,
    'nb_bathrooms': 1,
    'nb_parking_places': 1,
    'nb_boxes': 0,
    'has_a_balcony': 1,
    'nb_terraces': 0,
    'has_air_conditioning': 1,
    'property_type_atelier': 0,
    'property_type_chalet': 0,
    'property_type_chambre': 0,
    'property_type_château': 0,
    'property_type_divers': 0,
    'property_type_duplex': 0,
    'property_type_ferme': 0,
    'property_type_gîte': 0,
    'property_type_hôtel': 0,
    'property_type_hôtel particulier': 0,
    'property_type_loft': 0,
    'property_type_maison': 0,
    'property_type_manoir': 0,
    'property_type_moulin': 0,
    'property_type_parking': 0,
    'property_type_propriété': 0,
    'property_type_péniche': 0,
    'property_type_terrain': 0,
    'property_type_terrain à bâtir': 0,
    'property_type_viager': 0,
    'property_type_villa': 0,
    'energy_performance_category_B': 0,
    'energy_performance_category_C': 1,
    'energy_performance_category_D': 0,
    'energy_performance_category_E': 0,
    'energy_performance_category_F': 0,
    'energy_performance_category_G': 0,
    'exposition_Est-Ouest': 0,
    'exposition_Nord': 0,
    'exposition_Nord-Est': 0,
    'exposition_Nord-Ouest': 0,
    'exposition_Nord-Sud': 0,
    'exposition_Ouest': 0,
    'exposition_Ouest-Est': 0,
    'exposition_Sud': 1,
    'exposition_Sud-Est': 0,
    'exposition_Sud-Nord': 0,
    'exposition_Sud-Ouest': 0
}])

# Align the input with the model's expected features
test_input = test_input.reindex(columns=rf_model.feature_names_in_, fill_value=0)

# Make a prediction
try:
    predicted_price = rf_model.predict(test_input)[0]
    print("Predicted Price:", predicted_price)
except Exception as e:
    print(f"Prediction Error: {e}")