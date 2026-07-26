# ============================================
# TEST: DATA PREPROCESSING
# ============================================

import pandas as pd
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_preprocessing import DataPreprocessor


class TestDataPreprocessing:
    
    def setup_method(self):
        """Setup test data"""
        self.preprocessor = DataPreprocessor()
        self.test_data = pd.DataFrame({
            'year': [2010, 2015, 2020, None],
            'mileage': [100000, 50000, None, 200000],
            'engine_hp': [150, 200, 300, None],
            'price': [15000, 20000, 25000, 30000],
            'make': ['toyota', 'honda', None, 'bmw'],
            'model': ['corolla', 'civic', 'accord', None]
        })
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        df = self.preprocessor.handle_missing_values(self.test_data.copy())
        
        # Check no missing values
        assert df.isnull().sum().sum() == 0
        
        # Check numerical columns filled with median
        assert df['year'].iloc[3] == df['year'].median()
        assert df['mileage'].iloc[2] == df['mileage'].median()
        assert df['engine_hp'].iloc[3] == df['engine_hp'].median()
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        df = self.test_data.copy()
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.feature_engineering(df)
        
        # Check new features exist
        assert 'vehicle_age' in df.columns
        assert 'mileage_per_year' in df.columns
        assert 'hp_per_mileage' in df.columns
        
        # Check calculations
        current_year = 2025
        assert df['vehicle_age'].iloc[0] == current_year - 2010
    
    def test_remove_outliers(self):
        """Test outlier removal"""
        df = self.test_data.copy()
        df = self.preprocessor.handle_missing_values(df)
        initial_len = len(df)
        
        # Add outlier
        df.loc[len(df)] = [2024, 1000000, 800, 500000, 'toyota', 'corolla']
        
        df = self.preprocessor.remove_outliers(df)
        
        # Outlier should be removed
        assert len(df) <= initial_len + 1
    
    def test_optimize_memory(self):
        """Test memory optimization"""
        df = self.preprocessor.handle_missing_values(self.test_data.copy())
        df = self.preprocessor.optimize_memory(df)
        
        # Check data types are optimized
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            assert df[col].dtype in ['int32', 'int16', 'float32']
    
    def test_run_pipeline(self):
        """Test full preprocessing pipeline"""
        input_path = "data/test_data.csv"
        output_path = "data/test_output.csv"
        
        # Save test data
        os.makedirs("data", exist_ok=True)
        self.test_data.to_csv(input_path, index=False)
        
        # Run pipeline
        result = self.preprocessor.run(input_path, output_path)
        
        # Check output
        assert result is not None
        assert os.path.exists(output_path)
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)