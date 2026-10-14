"""Train and evaluate the Aerisense demo AQI regression model."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "air_quality_demo.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
FEATURES = ["temperature_c", "humidity_pct", "pm25", "pm10", "no2", "co", "so2", "o3"]

df = pd.read_csv(DATA)
required = FEATURES + ["aqi"]
missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df[required].isna().any().any():
    raise ValueError("Missing values found. Clean/impute them before training.")
if len(df) < 50:
    raise ValueError("Dataset is too small for this demo training workflow.")

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df["aqi"], test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "MAE": mean_absolute_error(y_test, pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
    "R2": r2_score(y_test, pred),
}
joblib.dump({"model": model, "features": FEATURES, "metrics": metrics},
            MODEL_DIR / "aqi_random_forest.joblib")
print("Model saved to models/aqi_random_forest.joblib")
print(f"Rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
