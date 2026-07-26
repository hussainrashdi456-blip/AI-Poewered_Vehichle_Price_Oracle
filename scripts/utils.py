# ============================================
# UTILITY FUNCTIONS
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def load_config(config_path='../config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config, config_path='../config.json'):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def create_sample_data(n_samples=100):
    """Create sample car data for testing"""
    sample_data = []
    for i in range(n_samples):
        sample_data.append({
            'year': np.random.randint(2000, 2025),
            'mileage': np.random.randint(0, 300000),
            'engine_hp': np.random.randint(50, 800),
            'owner_count': np.random.randint(1, 5),
            'make': np.random.choice(['toyota', 'honda', 'bmw', 'mercedes']),
            'model': np.random.choice(['corolla', 'civic', '3series', 'cclass']),
            'transmission': np.random.choice(['automatic', 'manual']),
            'fuel_type': np.random.choice(['petrol', 'diesel']),
            'condition': np.random.choice(['excellent', 'good', 'fair'])
        })
    return sample_data


def calculate_mae_percentage(actual, predicted):
    """Calculate MAE percentage"""
    mae = np.mean(np.abs(actual - predicted))
    mae_percentage = (mae / np.mean(actual)) * 100
    return mae_percentage


def save_results(results, filename='results.csv'):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"✅ Results saved: {filename}")


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")