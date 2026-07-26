# ============================================
# GENERATE PERFORMANCE FIGURES
# AI Used Car Price Predictor
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Create figures folder
os.makedirs('../figures', exist_ok=True)

print("=" * 60)
print("GENERATING PERFORMANCE FIGURES")
print("=" * 60)

# Model performance data
models = ['Linear\nRegression', 'Random\nForest', 'XGBoost']
r2_scores = [0.9365, 0.9831, 0.9858]
mae_scores = [2340.51, 1217.58, 1105.23]
rmse_scores = [3441.93, 1775.19, 1629.89]

# ============================================
# Figure 1: Model Performance Comparison
# ============================================
print("\n📊 Generating model performance comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R²
bars1 = axes[0].bar(models, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_ylim(0.9, 1.0)
for bar, val in zip(bars1, r2_scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.01,
                 f'{val:.4f}', ha='center', color='white', fontweight='bold')

# MAE
bars2 = axes[1].bar(models, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[1].set_ylabel('MAE ($)')
axes[1].set_title('MAE Comparison')
for bar, val in zip(bars2, mae_scores):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'${val:,.0f}', ha='center', fontweight='bold')

# RMSE
bars3 = axes[2].bar(models, rmse_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[2].set_ylabel('RMSE ($)')
axes[2].set_title('RMSE Comparison')
for bar, val in zip(bars3, rmse_scores):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'${val:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/model_performance.png', dpi=300)
plt.close()
print("✅ figures/model_performance.png")

# ============================================
# Figure 2: Feature Importance
# ============================================
print("\n📊 Generating feature importance...")

features = ['vehicle_age', 'hp_per_mileage', 'make_porsche', 'make_land rover',
            'engine_hp', 'year', 'brand_popularity', 'make_honda', 'make_nissan']
importance = [23.54, 19.38, 10.86, 10.18, 3.42, 3.21, 2.90, 2.36, 2.25]

plt.figure(figsize=(10, 6))
bars = plt.barh(features, importance, color='steelblue')
plt.xlabel('Importance (%)')
plt.title('Top Feature Importance')
for bar, val in zip(bars, importance):
    plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val}%', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300)
plt.close()
print("✅ figures/feature_importance.png")

# ============================================
# Figure 3: Final Dashboard
# ============================================
print("\n📊 Generating final dashboard...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

metrics = ['R² Score', 'MAE', 'RMSE', 'Error %', 'Prediction\nTime']
values = ['0.9858', '$1,146', '$1,610', '4.8%', '0.3 sec']
colors = ['#2ecc71', '#3498db', '#3498db', '#f39c12', '#2ecc71']

ax.set_title('FINAL MODEL PERFORMANCE DASHBOARD', fontsize=16, fontweight='bold', pad=20)

for i, (metric, value, color) in enumerate(zip(metrics, values, colors)):
    x = 0.1 + i * 0.18
    circle = plt.Circle((x, 0.5), 0.12, color=color, alpha=0.8)
    ax.add_patch(circle)
    ax.text(x, 0.5, value, ha='center', va='center', color='white', fontweight='bold', fontsize=14)
    ax.text(x, 0.25, metric, ha='center', va='center', fontsize=10, fontweight='bold')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('figures/final_dashboard.png', dpi=300)
plt.close()
print("✅ figures/final_dashboard.png")

# ============================================
# Figure 4: Cross-Validation Results
# ============================================
print("\n📊 Generating cross-validation results...")

cv_scores = [0.9857, 0.9859, 0.9856, 0.9858, 0.9858]
folds = [1, 2, 3, 4, 5]
mean_cv = np.mean(cv_scores)

plt.figure(figsize=(8, 5))
plt.plot(folds, cv_scores, 'o-', color='#2ecc71', linewidth=2, markersize=10)
plt.axhline(y=mean_cv, color='red', linestyle='--', label=f'Mean: {mean_cv:.4f}')
plt.fill_between(folds, mean_cv - 0.0002, mean_cv + 0.0002, alpha=0.2, color='green')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('5-Fold Cross-Validation Results')
plt.ylim(0.985, 0.9865)
plt.legend()
plt.tight_layout()
plt.savefig('figures/cross_validation.png', dpi=300)
plt.close()
print("✅ figures/cross_validation.png")

# ============================================
# Figure 5: Tuning Improvement
# ============================================
print("\n📊 Generating tuning improvement...")

metrics = ['MAE ($)', 'RMSE ($)', 'R² Score']
base = [1292.53, 1793.47, 0.9772]
tuned = [1146.33, 1610.36, 0.9816]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = range(len(metrics))
width = 0.35

ax1.bar([i - width / 2 for i in x], base, width, label='Base XGBoost', color='#3498db')
ax1.bar([i + width / 2 for i in x], tuned, width, label='Tuned XGBoost', color='#e74c3c')
ax1.set_ylabel('Value')
ax1.set_title('Base vs Tuned Model Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

improvements = [base[0] - tuned[0], base[1] - tuned[1], tuned[2] - base[2]]
ax2.bar(metrics, improvements, color='green')
ax2.set_ylabel('Improvement')
ax2.set_title('Performance Improvement from Tuning')
for bar, val in zip(ax2.patches, improvements):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{val:.0f}' if val > 1 else f'+{val:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/tuning_improvement.png', dpi=300)
plt.close()
print("✅ figures/tuning_improvement.png")

print("\n" + "=" * 60)
print("✅ ALL FIGURES GENERATED!")
print("=" * 60)