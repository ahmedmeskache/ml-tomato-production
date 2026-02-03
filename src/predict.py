import os
import joblib
import pandas as pd

# -------------------------
# 1. Load trained model
# -------------------------
model_file = r"C:\Users\ASUS\OneDrive\Desktop\tomato-quality production\data\tomato_quality_model.joblib"
if not os.path.exists(model_file):
    raise Exception(f"Model file not found at {model_file}")

model = joblib.load(model_file)
print("Model loaded successfully.")

# -------------------------
# 2. Prepare new batch data
# -------------------------
new_batch = pd.DataFrame([{
    'temp_morning': 23.0,
    'temp_evening': 24.5,
    'humidity': 68,
    'ph': 6.7,
    'water_flow': 30,
    'fertilizer': 110,
    'production_speed': 120,
    'avg_temp': (23.0 + 24.5)/2
}])

# -------------------------
# 3. Predict quality
# -------------------------
features = ['temp_morning', 'temp_evening', 'humidity', 'ph', 'water_flow', 'fertilizer', 'production_speed', 'avg_temp']
predicted_quality = model.predict(new_batch[features])
print("Predicted quality for new batch:", round(predicted_quality[0], 2))
