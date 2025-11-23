
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Custom background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        min-height: 100vh;
    }
    .card {
        background: rgba(255,255,255,0.85);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .big-text {
        font-size: 28px;
        font-weight: 700;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 House Price Prediction App (Improved Model)")

# Load model
model = joblib.load("improved_model/model.pkl")

# Sidebar Inputs
st.sidebar.header("Enter House Details")

def get_inputs():
    d = {}
    d["bedrooms"] = st.sidebar.number_input("Bedrooms", 0, 10, 3)
    d["bathrooms"] = st.sidebar.number_input("Bathrooms", 0.0, 10.0, 2.0)
    d["sqft_living"] = st.sidebar.number_input("Living Area (sqft)", 200, 20000, 1500)
    d["sqft_lot"] = st.sidebar.number_input("Lot Size (sqft)", 200, 500000, 5000)
    d["floors"] = st.sidebar.number_input("Floors", 1.0, 4.0, 1.0)
    d["waterfront"] = st.sidebar.selectbox("Waterfront Home?", [0,1])
    d["view"] = st.sidebar.slider("View Rating (0–4)", 0, 4, 0)
    d["condition"] = st.sidebar.slider("Condition (1–5)", 1, 5, 3)
    d["sqft_above"] = st.sidebar.number_input("Sqft Above", 0, 20000, 1400)
    d["sqft_basement"] = st.sidebar.number_input("Sqft Basement", 0, 20000, 100)
    d["yr_built"] = st.sidebar.number_input("Year Built", 1800, 2025, 1990)
    d["yr_renovated"] = st.sidebar.number_input("Year Renovated", 0, 2025, 0)
    d["zipcode"] = st.sidebar.number_input("Zipcode", 98000, 99999, 98052)

    return pd.DataFrame([d])

df_input = get_inputs()

# Layout
col1, col2 = st.columns([2,1])

# ---- LEFT PANEL ----
with col1:
    st.subheader("Prediction Results")

    if st.button("Predict Price"):
        pred_log = model.predict(df_input)[0]
        pred = np.expm1(pred_log)

        st.markdown(f"""
            <div class="card">
                <p class="big-text">Predicted Price: ${pred:,.0f}</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model Feature Importance")

    fi = pd.read_csv("improved_model/feature_importances.csv")

    fig, ax = plt.subplots(figsize=(6,4))
    fi.sort_values("importance").plot.barh(x="feature", y="importance", ax=ax)
    ax.set_title("Top Features")
    st.pyplot(fig)

# ---- RIGHT PANEL ----
with col2:
    st.subheader("Your Input Summary")
    st.write(df_input.T)

    st.subheader("Model Performance")
    with open("improved_model/metrics.json") as f:
        m = json.load(f)
    st.write(m)

st.markdown("---")
st.caption("Improved Model | Random Forest + Scaling + Log Transform + Zipcode Feature")
