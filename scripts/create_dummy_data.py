# ============================================
# CREATE DUMMY DATASET
# For testing and development
# ============================================

import pandas as pd
import numpy as np
import os

print("="*50)
print("CREATING DUMMY DATASET")
print("="*50)

# Define options for dropdown
options = {
    'make': ['toyota', 'honda', 'suzuki', 'bmw', 'mercedes', 'audi', 'ford', 
             'hyundai', 'kia', 'nissan', 'chevrolet', 'mazda', 'subaru', 
             'volkswagen', 'jeep', 'land rover', 'porsche', 'tesla'],
    
    'model': ['corolla', 'camry', 'yaris', 'civic', 'city', 'accord', 'alto', 
              'wagonr', 'swift', '3series', '5series', 'cclass', 'eclass', 
              'a4', 'a6', 'focus', 'mustang', 'elantra', 'sportage', 'sorento',
              'sunny', 'altima', 'camaro', '3', 'cx5', 'impreza', 'outback',
              'jetta', 'golf', 'wrangler', 'range rover', '911', 'model3'],
    
    'year': list(range(2000, 2026)),
    'transmission': ['automatic', 'manual', 'cvt'],
    'fuel_type': ['petrol', 'diesel', 'electric', 'hybrid'],
    'drivetrain': ['fwd', 'rwd', 'awd', '4wd'],
    'body_type': ['sedan', 'suv', 'hatchback', 'coupe', 'truck'],
    'exterior_color': ['black', 'white', 'red', 'blue', 'silver', 'gray', 'green'],
    'interior_color': ['black', 'beige', 'gray', 'brown'],
    'accident_history': ['none', 'minor', 'major'],
    'seller_type': ['dealer', 'private'],
    'condition': ['excellent', 'good', 'fair', 'poor'],
    'trim': ['base', 'sport', 'luxury', 'ex', 'lx', 'touring'],
    'price': np.random.uniform(5000, 50000, 1)  # Placeholder, not used for dummy
}

# Create combinations (max 1000 rows for dummy)
np.random.seed(42)
n_rows = 500

data = []
for i in range(n_rows):
    data.append({
        'make': np.random.choice(options['make']),
        'model': np.random.choice(options['model']),
        'year': np.random.choice(options['year']),
        'mileage': np.random.randint(0, 300000),
        'engine_hp': np.random.randint(50, 800),
        'transmission': np.random.choice(options['transmission']),
        'fuel_type': np.random.choice(options['fuel_type']),
        'drivetrain': np.random.choice(options['drivetrain']),
        'body_type': np.random.choice(options['body_type']),
        'exterior_color': np.random.choice(options['exterior_color']),
        'interior_color': np.random.choice(options['interior_color']),
        'owner_count': np.random.randint(1, 5),
        'accident_history': np.random.choice(options['accident_history']),
        'seller_type': np.random.choice(options['seller_type']),
        'condition': np.random.choice(options['condition']),
        'trim': np.random.choice(options['trim'])
    })

df = pd.DataFrame(data)

# Save
df.to_csv('dummy_data.csv', index=False)

print(f"✅ Dummy dataset created!")
print(f"   Records: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   File: dummy_data.csv")

# Show sample
print("\n📊 Sample:")
print(df.head(3).to_string())