# ============================================
# TEST: STREAMLIT APP
# ============================================

import pytest
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict import CarPricePredictor


class TestApp:
    
    def setup_method(self):
        """Setup test data"""
        self.predictor = CarPricePredictor(
            model_path="../model/best_tuned_used_car_price_model.pkl",
            features_path="../model/model_features.pkl"
        )
        
        self.test_car = {
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
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        assert self.predictor.model is not None
        assert self.predictor.features is not None
        assert len(self.predictor.features) > 0
    
    def test_single_prediction(self):
        """Test single car prediction"""
        price = self.predictor.predict(self.test_car)
        
        assert isinstance(price, float)
        assert price > 0
        assert price < 100000  # Reasonable price range
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        cars = [self.test_car] * 5
        prices = self.predictor.predict_batch(cars)
        
        assert len(prices) == 5
        for price in prices:
            assert isinstance(price, float)
            assert price > 0
    
    def test_prediction_consistency(self):
        """Test prediction consistency"""
        price1 = self.predictor.predict(self.test_car)
        price2 = self.predictor.predict(self.test_car)
        
        assert price1 == price2  # Same car should give same price
    
    def test_prediction_response_time(self):
        """Test prediction response time"""
        import time
        
        start = time.time()
        self.predictor.predict(self.test_car)
        end = time.time()
        
        assert (end - start) < 1.0  # Should be under 1 second
    
    def test_handle_missing_features(self):
        """Test handling of missing features"""
        # Create car with missing features
        car_missing = {
            'year': 2018,
            'mileage': 65000,
            'engine_hp': 140,
            'make': 'toyota',
            'model': 'corolla'
        }
        
        try:
            price = self.predictor.predict(car_missing)
            assert price > 0
        except Exception as e:
            pytest.fail(f"Prediction failed with missing features: {e}")