from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# Define the input data schema
class FlatCharacteristics(BaseModel):
    Property_type: str
    postal_code: int
    size: float
    floor: int
    land_size: float
    energy_performance_category: str
    exposition: str
    nb_rooms: int
    nb_bedrooms: int
    nb_bathrooms: int
    nb_parking_places: int
    nb_boxes: int
    has_a_balcony: int
    nb_terraces: int
    has_air_conditioning: int

# Load the trained model
rf_model = joblib.load('/Users/rayan/PycharmProjects/rakam-systems-service-template/AIMO/random_forest_model.pk1')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_price(data: FlatCharacteristics):
    print("data{}:\n",data)
    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Align the input with the model’s expected features
    try:
        print("Input Data Before Alignment:")
        print(input_data)

        input_data = input_data.reindex(columns=rf_model.feature_names_in_, fill_value=0)

        print("Input Data After Alignment:")
        print(input_data)

        print("Model Feature Names:", rf_model.feature_names_in_)
        print("Input Data Columns:", input_data.columns)

        # Make predictions
        predicted_price = rf_model.predict(input_data)[0]
        return {"predicted_price": predicted_price}  # Ensure response is JSON-serializable
    except Exception as e:
        return {"error": str(e)}