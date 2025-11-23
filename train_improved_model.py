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

# ------------------------------------------------
# LOAD YOUR DATA
# ------------------------------------------------
df = pd.read_csv("data.csv")

# Extract zipcode (most important feature)
if "statezip" in df.columns:
    df["zipcode"] = df["statezip"].str.extract(r"(\d+)$").astype(float)

# Drop unnecessary text columns
drop_cols = ["date", "street", "city", "statezip", "country"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Select features
numeric_features = [
    "bedrooms","bathrooms","sqft_living","sqft_lot","floors",
    "waterfront","view","condition","sqft_above","sqft_basement",
    "yr_built","yr_renovated","zipcode"
]

numeric_features = [f for f in numeric_features if f in df.columns]

# Define X and y with log-transform
X = df[numeric_features].fillna(df[numeric_features].median())
y = np.log1p(df["price"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing + Model
preprocessor = ColumnTransformer([
    ("scaler", StandardScaler(), numeric_features)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=22,
        random_state=42,
        n_jobs=-1
    ))
])

# Train model
model.fit(X_train, y_train)

# Evaluate (convert back with expm1)
y_pred = np.expm1(model.predict(X_test))
y_test_exp = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred))
r2 = r2_score(y_test_exp, y_pred)

print("\nMODEL TRAINED SUCCESSFULLY!")
print("RMSE:", rmse)
print("R2  :", r2)

# Create output directory
os.makedirs("improved_model", exist_ok=True)

# Save model
joblib.dump(model, "improved_model/model.pkl")

# Save feature importances
rf_importance = model.named_steps["rf"].feature_importances_
fi_df = pd.DataFrame({"feature": numeric_features, "importance": rf_importance})
fi_df.to_csv("improved_model/feature_importances.csv", index=False)

# Save metrics
with open("improved_model/metrics.json", "w") as f:
    json.dump({"rmse": float(rmse), "r2": float(r2)}, f)

print("\nSaved:")
print("- improved_model/model.pkl")
print("- improved_model/feature_importances.csv")
print("- improved_model/metrics.json")
