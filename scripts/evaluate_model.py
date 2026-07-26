# ============================================
# MODEL EVALUATION SCRIPT
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


def evaluate_model(y_true, y_pred, name="Model"):
    """Evaluate model performance"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 {name}:")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   R²: {r2:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def evaluate_on_test_data(model_path, test_data_path):
    """Evaluate model on test data"""
    # Load model
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {model_path}")

    # Load test data
    df = pd.read_csv(test_data_path)
    print(f"✅ Test data: {df.shape}")

    # Prepare test data
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and target
    X_test = df_encoded.drop('price', axis=1)
    y_test = df_encoded['price']

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate_model(y_test, y_pred, "XGBoost (Tuned)")

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv('outputs/evaluation_results.csv', index=False)
    print("\n✅ Results saved: outputs/evaluation_results.csv")

    return results


if __name__ == "__main__":
    evaluate_on_test_data(
        model_path="model/best_tuned_used_car_price_model.pkl",
        test_data_path="data/cleaned_vehicle_price_dataset.csv"
    )