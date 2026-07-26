# ============================================
# DATA PREPROCESSING SCRIPT
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np
import os

print("="*50)
print("DATA PREPROCESSING")
print("="*50)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n📂 Loading dataset...")

# data_paths = [
#     "vehicle_price_prediction.csv"
# ]

# df = None
# for path in data_paths:
#     if os.path.exists(path):
#         df = pd.read_csv("vehicle_price_prediction.csv")
#         print(f"✅ Loaded: {path}")
#         break

# if df is None:
#     print("❌ Dataset not found!")
#     exit()
df = pd.read_csv("data/vehicle_price_prediction.csv")

print(f"   Initial Shape: {df.shape}")
print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ============================================
# 2. STANDARDIZE COLUMNS
# ============================================
print("\n🔧 Standardizing columns...")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print(f"   Columns: {df.columns.tolist()}")

# ============================================
# 3. HANDLE MISSING VALUES
# ============================================
print("\n📊 Handling missing values...")

print(f"   Missing before: {df.isnull().sum().sum()}")

# Numerical → Median
numerical_cols = ['year', 'mileage', 'engine_hp', 'owner_count', 'price']
for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Categorical → Mode
categorical_cols = ['make', 'model', 'transmission', 'fuel_type', 'drivetrain',
                    'body_type', 'exterior_color', 'interior_color',
                    'accident_history', 'seller_type', 'condition', 'trim']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'none')

print(f"   Missing after: {df.isnull().sum().sum()}")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n🔧 Feature engineering...")

current_year = 2025

df['vehicle_age'] = current_year - df['year']
df['vehicle_age'] = df['vehicle_age'].replace(0, 1)

df['mileage_per_year'] = df['mileage'] / df['vehicle_age']
df['hp_per_mileage'] = df['engine_hp'] / (df['mileage'] + 1)

print(f"   New features: vehicle_age, mileage_per_year, hp_per_mileage")

# ============================================
# 5. REMOVE OUTLIERS
# ============================================
print("\n📊 Removing outliers...")

initial_count = len(df)

for col in ['price', 'mileage', 'engine_hp']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

outliers_removed = initial_count - len(df)
print(f"   Outliers removed: {outliers_removed:,} ({outliers_removed/initial_count*100:.1f}%)")

# ============================================
# 6. OPTIMIZE MEMORY
# ============================================
print("\n💾 Optimizing memory...")

for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

memory_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"   Memory: {memory_after:.1f} MB")

# ============================================
# 7. SAVE CLEANED DATA
# ============================================
print("\n💾 Saving cleaned data...")

df.to_csv('data/cleaned_vehicle_price_dataset.csv', index=False)
print(f"   ✅ cleaned_vehicle_price_dataset.csv")
print(f"   Shape: {df.shape}")
print(f"   Records: {len(df):,}")

# ============================================
# 8. SUMMARY
# ============================================
print("\n" + "="*50)
print("✅ PREPROCESSING COMPLETE!")
print("="*50)

print(f"\n📊 Final Dataset:")
print(f"   Records: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print(f"   Memory: {memory_after:.1f} MB")
print(f"   Missing: {df.isnull().sum().sum()}")
print(f"   Duplicates: {df.duplicated().sum()}")