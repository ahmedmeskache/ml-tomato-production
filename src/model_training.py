import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
import numpy as np

import joblib
from data_preprocessing import load_data
from feature_engineering import add_features

# -------------------------
# 1. Load data
# -------------------------
data_path = r"C:\Users\ASUS\OneDrive\Desktop\tomato-quality production\data\sample_data.csv"
data = load_data(data_path)
if data is None:
    raise Exception("Data loading failed. Exiting.")

# -------------------------
# 2. Feature engineering
# -------------------------
data = add_features(data)

# -------------------------
# 3. Prepare features and target
# -------------------------
features = ['temp_morning', 'temp_evening', 'humidity', 'ph', 'water_flow', 'fertilizer', 'production_speed', 'avg_temp']
X = data[features]
y = data['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 4. Train Random Forest
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully.")

# -------------------------
# 5. Evaluate
# -------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# -------------------------
# 6. Save model
# -------------------------
model_file = os.path.join(os.path.dirname(data_path), "tomato_quality_model.joblib")
joblib.dump(model, model_file)
print(f"Trained model saved at: {model_file}")
