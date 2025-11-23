import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

# Load dataset
df = pd.read_csv("data.csv")

# Extract zipcode
df["zipcode"] = df["statezip"].str.extract(r"(\d+)$").astype(float)

# Drop unnecessary text columns
df = df.drop(columns=["date", "street", "city", "statezip", "country"])

# Select features
features = [
    "bedrooms","bathrooms","sqft_living","sqft_lot","floors",
    "waterfront","view","condition","sqft_above","sqft_basement",
    "yr_built","yr_renovated","zipcode"
]

X = df[features].fillna(df[features].median())
y = np.log1p(df["price"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Smaller, lighter RandomForest model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=80,    # reduced trees
        max_depth=15,       # limit depth
        min_samples_leaf=5, # reduces size a lot
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Evaluate
y_pred = np.expm1(model.predict(X_test))
y_test_real = np.expm1(y_test)

rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
r2 = float(r2_score(y_test_real, y_pred))

# Save folder
os.makedirs("improved_model", exist_ok=True)

joblib.dump(model, "improved_model/model.pkl")

# Save feature importance
fi = model.named_steps["rf"].feature_importances_
pd.DataFrame({"feature": features, "importance": fi}).to_csv(
    "improved_model/feature_importances.csv", index=False
)

# Save metrics
with open("improved_model/metrics.json", "w") as f:
    json.dump({"rmse": rmse, "r2": r2}, f)

print("Training complete!")
print("New model size is MUCH smaller.")
print("RMSE:", rmse)
print("R2:", r2)
