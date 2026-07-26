# ============================================
# HYPERPARAMETER TUNING SCRIPT
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("HYPERPARAMETER TUNING - XGBoost")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading data...")

df = pd.read_csv("data/cleaned_vehicle_price_dataset.csv")
categorical_cols = ['make', 'model', 'transmission', 'fuel_type', 'drivetrain',
                    'body_type', 'exterior_color', 'interior_color',
                    'accident_history', 'seller_type', 'condition', 'trim']

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Sample for faster tuning
X_sample = X.sample(n=50000, random_state=42)
y_sample = y.loc[X_sample.index]

print(f"✅ Sample size: {X_sample.shape[0]:,} records")

# ============================================
# 2. TUNING
# ============================================
print("\n🔧 Tuning hyperparameters...")

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.3]
}

random_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_sample, y_sample)

print(f"\n🏆 Best Parameters:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

# ============================================
# 3. TRAIN FINAL MODEL
# ============================================
print("\n🤖 Training tuned model...")

best_model = XGBRegressor(**random_search.best_params_, random_state=42)
best_model.fit(X, y)

# ============================================
# 4. SAVE MODEL
# ============================================
print("\n💾 Saving tuned model...")

joblib.dump(best_model, 'model/best_tuned_used_car_price_model.pkl')
print("   ✅ best_tuned_used_car_price_model.pkl")

print("\n" + "="*60)
print("✅ TUNING COMPLETE!")
print("="*60)