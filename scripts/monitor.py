# ============================================
# MODEL MONITORING SCRIPT
# AI Used Car Price Predictor
# ============================================

import pandas as pd
import json
from datetime import datetime
import os

print("=" * 60)
print("MODEL MONITORING")
print("=" * 60)


class ModelMonitor:
    def __init__(self, log_file='logs/predictions_log.csv'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_prediction(self, input_data, prediction, actual=None, confidence=None):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': json.dumps(input_data),
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence
        }

        df = pd.DataFrame([log_entry])

        try:
            existing = pd.read_csv(self.log_file)
            df = pd.concat([existing, df])
        except FileNotFoundError:
            pass

        df.to_csv(self.log_file, index=False)
        print(f"✅ Prediction logged: ${prediction:,.2f}")

    def get_performance_metrics(self):
        """Get performance metrics from logs"""
        try:
            df = pd.read_csv(self.log_file)

            metrics = {
                'total_predictions': len(df),
                'avg_prediction': df['prediction'].mean(),
                'min_prediction': df['prediction'].min(),
                'max_prediction': df['prediction'].max(),
                'last_24h': len(df[df['timestamp'] > datetime.now().isoformat()])
            }

            if 'actual' in df.columns:
                df['error'] = abs(df['prediction'] - df['actual'])
                metrics['avg_error'] = df['error'].mean()
                metrics['max_error'] = df['error'].max()

            return metrics
        except FileNotFoundError:
            return {'error': 'No logs found'}

    def get_prediction_history(self, limit=100):
        """Get prediction history"""
        try:
            df = pd.read_csv(self.log_file)
            return df.tail(limit)
        except FileNotFoundError:
            return pd.DataFrame()

    def clear_logs(self):
        """Clear all logs"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            print("✅ Logs cleared")


if __name__ == "__main__":
    monitor = ModelMonitor()

    # Example: Log a prediction
    monitor.log_prediction(
        input_data={'make': 'toyota', 'model': 'corolla', 'year': 2018},
        prediction=14250.00,
        actual=15000.00,
        confidence=0.94
    )

    # Get metrics
    print("\n📊 Performance Metrics:")
    metrics = monitor.get_performance_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")