# ============================================
# FEATURE ENGINEERING SCRIPT
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        self.current_year = 2025
        self.derived_features = ['vehicle_age', 'mileage_per_year', 'hp_per_mileage']

    def create_features(self, df):
        """Create all derived features"""
        print("\n🔧 Creating features...")

        # vehicle_age = 2025 - year
        df['vehicle_age'] = self.current_year - df['year']
        df['vehicle_age'] = df['vehicle_age'].replace(0, 1)

        # mileage_per_year = mileage / vehicle_age
        df['mileage_per_year'] = df['mileage'] / df['vehicle_age']

        # hp_per_mileage = engine_hp / (mileage + 1)
        df['hp_per_mileage'] = df['engine_hp'] / (df['mileage'] + 1)

        print(f"   Added: {', '.join(self.derived_features)}")
        return df

    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from trained model"""
        importances = model.feature_importances_
        feature_imp = sorted(zip(feature_names, importances),
                             key=lambda x: x[1], reverse=True)
        return feature_imp

    def select_top_features(self, feature_importance, top_n=10):
        """Select top N features"""
        return feature_importance[:top_n]

    def encode_categorical(self, df, categorical_cols):
        """One-hot encode categorical columns"""
        return pd.get_dummies(df, columns=categorical_cols, drop_first=True)


if __name__ == "__main__":
    # Test feature engineering
    df = pd.DataFrame({
        'year': [2010, 2015, 2020],
        'mileage': [100000, 50000, 25000],
        'engine_hp': [150, 200, 300]
    })

    engineer = FeatureEngineer()
    df_engineered = engineer.create_features(df)
    print("\n📊 Engineered Features:")
    print(df_engineered)