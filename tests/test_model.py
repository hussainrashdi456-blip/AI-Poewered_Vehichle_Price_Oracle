# ============================================
# TEST: MODEL TRAINING
# ============================================

import pandas as pd
import numpy as np
import pytest
import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_model import ModelTrainer
from scripts.hyperparameter_tuning import HyperparameterTuner


class TestModel:
    
    def setup_method(self):
        """Setup test data"""
        self.trainer = ModelTrainer()
        self.tuner = HyperparameterTuner()
        
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        self.test_data = pd.DataFrame({
            'year': np.random.randint(2000, 2025, n_samples),
            'mileage': np.random.randint(0, 200000, n_samples),
            'engine_hp': np.random.randint(50, 500, n_samples),
            'owner_count': np.random.randint(1, 5, n_samples),
            'price': np.random.uniform(5000, 50000, n_samples),
            'make': np.random.choice(['toyota', 'honda', 'bmw', 'mercedes'], n_samples),
            'model': np.random.choice(['corolla', 'civic', '3series', 'cclass'], n_samples)
        })
    
    def test_prepare_features(self):
        """Test feature preparation"""
        X, y, features = self.trainer.prepare_features(self.test_data)
        
        # Check shapes
        assert X.shape[0] == len(self.test_data)
        assert len(y) == len(self.test_data)
        assert len(features) == X.shape[1]
    
    def test_train_models(self):
        """Test model training"""
        X, y, _ = self.trainer.prepare_features(self.test_data)
        
        X_train = X[:800]
        y_train = y[:800]
        
        models = self.trainer.train_models(X_train, y_train)
        
        # Check models exist
        assert 'linear' in models
        assert 'random_forest' in models
        assert 'xgboost' in models
        
        # Check models are trained
        for model in models.values():
            assert hasattr(model, 'predict')
    
    def test_evaluate_models(self):
        """Test model evaluation"""
        X, y, _ = self.trainer.prepare_features(self.test_data)
        
        X_train = X[:800]
        y_train = y[:800]
        X_test = X[800:]
        y_test = y[800:]
        
        models = self.trainer.train_models(X_train, y_train)
        self.trainer.models = models
        
        results = self.trainer.evaluate_models(X_test, y_test)
        
        # Check results have required metrics
        for name, metrics in results.items():
            assert 'mae' in metrics
            assert 'rmse' in metrics
            assert 'r2' in metrics
            assert metrics['r2'] > -1  # R² should be > -1
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning"""
        X_sample = self.test_data.drop('price', axis=1)
        y_sample = self.test_data['price']
        
        best_params, best_score = self.tuner.tune_model(X_sample, y_sample)
        
        # Check parameters found
        assert 'n_estimators' in best_params
        assert 'learning_rate' in best_params
        assert 'max_depth' in best_params
        assert best_score > 0
    
    def test_model_save_load(self):
        """Test model save and load"""
        # Create and save a simple model
        X, y, _ = self.trainer.prepare_features(self.test_data)
        X_train = X[:800]
        y_train = y[:800]
        
        models = self.trainer.train_models(X_train, y_train)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(models['xgboost'], "models/test_model.pkl")
        
        # Load model
        loaded_model = joblib.load("models/test_model.pkl")
        
        # Check predictions
        X_test = X[800:802]
        preds = loaded_model.predict(X_test)
        
        assert len(preds) == 2
        assert isinstance(preds[0], float)
        
        # Cleanup
        os.remove("models/test_model.pkl")