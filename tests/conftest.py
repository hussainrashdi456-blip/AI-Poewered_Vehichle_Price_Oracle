# ============================================
# PYTEST CONFIGURATION
# ============================================

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'year': np.random.randint(2000, 2025, n_samples),
        'mileage': np.random.randint(0, 200000, n_samples),
        'engine_hp': np.random.randint(50, 500, n_samples),
        'owner_count': np.random.randint(1, 5, n_samples),
        'price': np.random.uniform(5000, 50000, n_samples),
        'make': np.random.choice(['toyota', 'honda', 'bmw', 'mercedes'], n_samples),
        'model': np.random.choice(['corolla', 'civic', '3series', 'cclass'], n_samples),
        'transmission': np.random.choice(['automatic', 'manual'], n_samples),
        'fuel_type': np.random.choice(['petrol', 'diesel'], n_samples),
        'condition': np.random.choice(['excellent', 'good', 'fair'], n_samples)
    })


@pytest.fixture
def sample_car():
    """Create sample car for prediction test"""
    return {
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


def pytest_configure(config):
    """Configure pytest"""
    # Create required directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('model', exist_ok=True)