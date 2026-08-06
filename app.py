import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Load dataset (only to get dropdown values)
df = pd.read_csv("House_Prediction.csv")

# -----------------------------
# Title
# -----------------------------
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction System")
st.write("Predict the estimated price of a house using Machine Learning.")

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 3)

bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5
)

sqft_living = st.sidebar.number_input(
    "Living Area (sqft)",
    value=2000
)

sqft_lot = st.sidebar.number_input(
    "Lot Area (sqft)",
    value=5000
)

floors = st.sidebar.selectbox(
    "Floors",
    [1,1.5,2,2.5,3,3.5]
)

waterfront = st.sidebar.selectbox(
    "Waterfront",
    [0,1]
)

view = st.sidebar.selectbox(
    "View",
    [0,1,2,3,4]
)

condition = st.sidebar.selectbox(
    "Condition",
    [1,2,3,4,5]
)

sqft_above = st.sidebar.number_input(
    "Sqft Above",
    value=1800
)

sqft_basement = st.sidebar.number_input(
    "Sqft Basement",
    value=200
)

yr_built = st.sidebar.number_input(
    "Year Built",
    1900,
    2025,
    2000
)

yr_renovated = st.sidebar.number_input(
    "Year Renovated",
    0,
    2025,
    0
)

# Date values
year = 2014
month = 6
day = 15

# Dropdowns
city = st.sidebar.selectbox(
    "City",
    sorted(df["city"].unique())
)

statezip = st.sidebar.selectbox(
    "State ZIP",
    sorted(df["statezip"].unique())
)

# -----------------------------
# Create Input DataFrame
# -----------------------------

input_dict = {}

for feature in feature_names:
    input_dict[feature] = 0

# Numerical Features

input_dict["bedrooms"] = bedrooms
input_dict["bathrooms"] = bathrooms
input_dict["sqft_living"] = sqft_living
input_dict["sqft_lot"] = sqft_lot
input_dict["floors"] = floors
input_dict["waterfront"] = waterfront
input_dict["view"] = view
input_dict["condition"] = condition
input_dict["sqft_above"] = sqft_above
input_dict["sqft_basement"] = sqft_basement
input_dict["yr_built"] = yr_built
input_dict["yr_renovated"] = yr_renovated
input_dict["year"] = year
input_dict["month"] = month
input_dict["day"] = day

# One-Hot Encoding

city_col = f"city_{city}"
zip_col = f"statezip_{statezip}"

if city_col in input_dict:
    input_dict[city_col] = 1

if zip_col in input_dict:
    input_dict[zip_col] = 1

input_df = pd.DataFrame([input_dict])

# Scale
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict House Price"):

    prediction = model.predict(input_scaled)[0]

    st.success(f"🏠 Estimated House Price : ${prediction:,.2f}")

    st.balloons()

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.write("Developed using Streamlit and Scikit-Learn")
