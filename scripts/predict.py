# ============================================
# PREDICTION UTILITY
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np
import joblib
import json


class CarPricePredictor:
    def __init__(self, model_path='model/best_tuned_used_car_price_model.pkl',
                 features_path='model/model_features.pkl'):
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Features loaded: {len(self.features)}")

    def predict(self, car_data):
        """Predict price for a single car"""
        # Convert to DataFrame
        df = pd.DataFrame([car_data])

        # One-hot encode
        df_encoded = pd.get_dummies(df)

        # Align features
        for col in self.features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[self.features]

        # Predict
        prediction = self.model.predict(df_encoded)[0]
        return prediction

    def predict_batch(self, cars_data):
        """Predict prices for multiple cars"""
        predictions = []
        for car in cars_data:
            price = self.predict(car)
            predictions.append(price)
        return predictions


if __name__ == "__main__":
    # Example usage
    predictor = CarPricePredictor()

    # Single prediction
    car = {
        'year': 2018,
        'mileage': 65000,
        'engine_hp': 140,
        'owner_count': 1,
        'make': 'toyota',
        'model': 'corolla',
        'transmission': 'automatic',
        'fuel_type': 'petrol',
        'condition': 'good'
    }

    price = predictor.predict(car)
    print(f"\n💰 Predicted Price: ${price:,.2f}")