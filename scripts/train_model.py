# ============================================
# MODEL TRAINING SCRIPT
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AI USED CAR PRICE PREDICTOR - MODEL TRAINING")
print("="*60)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading dataset...")

# Try multiple paths
data_paths = [
    "../data/cleaned_vehicle_price_dataset.csv",
    "data/cleaned_vehicle_price_dataset.csv",
    "../data/vehicle_price_dataset.csv",
    "data/vehicle_price_dataset.csv",
]

df = None
for path in data_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ Loaded: {path}")
        break

if df is None:
    print("❌ Dataset not found! Please check file path.")
    exit()

print(f"   Shape: {df.shape}")
print(f"   Columns: {len(df.columns)}")

# ============================================
# 2. PREPARE FEATURES
# ============================================
print("\n🔧 Preparing features...")

# Identify categorical columns
categorical_cols = ['make', 'model', 'transmission', 'fuel_type', 'drivetrain',
                    'body_type', 'exterior_color', 'interior_color',
                    'accident_history', 'seller_type', 'condition', 'trim']

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features and target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

print(f"✅ Features: {X.shape[1]}")
print(f"✅ Target: {y.shape[0]} records")

# ============================================
# 3. TRAIN-TEST SPLIT
# ============================================
print("\n📊 Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training: {X_train.shape[0]:,} records")
print(f"   Testing: {X_test.shape[0]:,} records")

# ============================================
# 4. TRAIN MODELS
# ============================================
print("\n🤖 Training models...")

# Model 1: Linear Regression
print("\n   📊 Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Model 2: Random Forest
print("   🌲 Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Model 3: XGBoost
print("   ⚡ XGBoost...")
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

# ============================================
# 5. EVALUATION
# ============================================
print("\n📊 Model Evaluation:")
print("-"*50)

def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:20} | MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

results = {}
results['Linear Regression'] = evaluate_model(y_test, lr_pred, 'Linear Regression')
results['Random Forest'] = evaluate_model(y_test, rf_pred, 'Random Forest')
results['XGBoost'] = evaluate_model(y_test, xgb_pred, 'XGBoost')

# ============================================
# 6. SELECT BEST MODEL
# ============================================
print("\n🏆 Selecting best model...")

best_name = max(results, key=lambda x: results[x]['r2'])
print(f"   Best Model: {best_name}")
print(f"   R² Score: {results[best_name]['r2']:.4f}")

if best_name == 'XGBoost':
    best_model = xgb
elif best_name == 'Random Forest':
    best_model = rf
else:
    best_model = lr

# ============================================
# 7. SAVE MODEL
# ============================================
print("\n💾 Saving model...")

# Save best model
joblib.dump(best_model, 'model/best_used_car_price_model.pkl')
print("   ✅ best_used_car_price_model.pkl")

# Save features
joblib.dump(X.columns.tolist(), 'model/model_features.pkl')
print("   ✅ model_features.pkl")

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv('outputs/model_results.csv')
print("   ✅ model_results.csv")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)