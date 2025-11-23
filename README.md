🏡 House-Price-Prediction
Machine learning project with python language

This project aims to predict house prices using multiple machine learning models. We applied various preprocessing and feature engineering techniques to build robust and accurate models. The models were trained and evaluated on a real-world dataset with multiple numerical and categorical variables.Predicts the price of a house based on multiple property features such as bedrooms, bathrooms, square footage, lot size, condition, view, and zipcode.
It uses a Random Forest Regressor with log-transformed target for better accuracy and a modern Streamlit web interface for user interaction.

Features:

Data Cleaning and Preprocessing
Log Transformation for Skewed Features
Outlier Detection using Isolation Forest
Feature Scaling using StandardScaler

Model Training using:
Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Artificial Neural Network (ANN)
Model Evaluation using RMSE and R² score
Visualizations for EDA and model performance


Dataset:The dataset contains house features like:
date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country etc.
Target variable: SalePrice

Model Performance

Model	                   RMSE	          R² Score
Linear Regression	       27,000	        0.89
Ridge Regression	       26,500	        0.90
Lasso Regression	       27,100	        0.89
Random Forest	           23,000	        0.93
ANN	                       22,500           0.94


Improved Model + Modern Streamlit UI:
Random Forest Regressor
Feature scaling
Log(price) transformation
Important features displayed as a bar chart
Zipcode included for improved accuracy

Automatically generates:
model.pkl
feature_importances.csv
metrics.json

✔️ Streamlit Web App
Colorful gradient UI
Sidebar controls for input
Real-time prediction
Input summary table
Feature importance visualization
Model performance metrics (RMSE & R²)

📁 Project Structure

house_price_prediction/

├── data.csv                     # Your dataset (must include 'price')

├── app.py                       # Streamlit UI

├── train_improved_model.py      # Script to train improved ML model

├── requirements.txt             # Dependencies

├── README.md                    # Documentation

│
└── improved_model/

    ├── model.pkl                # Trained ML model

    ├── feature_importances.csv  # Feature importance values

    └── metrics.json             # RMSE & R² of the trained model


⚙️ Installation
1. Install Dependencies
Make sure Python 3.8+ is installed.
Run:
pip install -r requirements.txt

🧠 Model Training
2. Train the Improved Model
Make sure your data.csv is placed in the project folder.
Run:
python train_improved_model.py


This script will:
Load your dataset
Extract zipcode
Preprocess the features
Train RandomForest
Save trained model to improved_model/model.pkl
Generate feature_importances.csv
Generate metrics.json (RMSE & R²)

🌐 Run the Web App
3. Start the Streamlit UI
streamlit run app.py
The app will open in your browser at:
👉 http://localhost:8501

🧩 Model Details
Algorithms Used
RandomForestRegressor
StandardScaler (via ColumnTransformer)
Target transformed with:
log(price + 1) → makes predictions more stable
Metrics Generated
You will see values similar to:
RMSE: XXXXXX
R²: 0.XX
RMSE is converted back to actual price using expm1().

🎨 UI Highlights:
Beautiful gradient theme
Clean card-style prediction output
Input fields on the left
Real-time prediction button
Matplotlib feature importance bar chart
Displays your input values in table format

📊 Feature Importance (Example Output):
After training, you will see top contributors like:
sqft_living
zipcode
sqft_above
yr_built
bathrooms etc.

This helps justify your ML model in your report.

🧾 Dataset Requirements:
Your data.csv must include:
price
bedrooms
bathrooms
sqft_living
sqft_lot
floors
waterfront
view
condition
sqft_above
sqft_basement
yr_built
yr_renovated
statezip


The script automatically extracts zipcode from statezip.

🏁 Conclusion
This project demonstrates:
End-to-end ML pipeline
Feature engineering & preprocessing
Model training and evaluation
Interactive web application
Deployment-ready Streamlit interface