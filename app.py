# app.py

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------
# Title
# ---------------------------
st.title("🏠 Housing Price Prediction (Ridge Regression)")
st.write("Predict house prices with demographic + geographic data")

# ---------------------------
# Load Dataset (CSV)
# ---------------------------
df = pd.read_csv("housing.csv")

# ---------------------------
# Handle Missing Values
# ---------------------------
df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------------------
# One-Hot Encoding
# ---------------------------
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# ---------------------------
# Features & Target
# ---------------------------
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------------------
# Model
# ---------------------------
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Enter House Details")

longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
housingMedianAge = st.sidebar.slider("House Age", 1.0, 50.0, 20.0)
totalRooms = st.sidebar.slider("Total Rooms", 100.0, 10000.0, 2000.0)
totalBedrooms = st.sidebar.slider("Total Bedrooms", 50.0, 5000.0, 500.0)
population = st.sidebar.slider("Population", 100.0, 5000.0, 1000.0)
households = st.sidebar.slider("Households", 100.0, 3000.0, 500.0)
medianIncome = st.sidebar.slider("Median Income", 0.0, 15.0, 3.0)

ocean = st.sidebar.selectbox(
    "Ocean Proximity",
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
)

# ---------------------------
# Prepare Input Data
# ---------------------------
input_dict = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housingMedianAge,
    "total_rooms": totalRooms,
    "total_bedrooms": totalBedrooms,
    "population": population,
    "households": households,
    "median_income": medianIncome
}

# Add encoded ocean columns
for col in X.columns:
    if "ocean_proximity" in col:
        input_dict[col] = 0

# Set selected category
selected_col = f"ocean_proximity_{ocean}"
if selected_col in input_dict:
    input_dict[selected_col] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

# ---------------------------
# Output
# ---------------------------
st.subheader("🏡 Predicted House Price")
st.write(f"💰 Estimated Value: ${prediction[0]:,.2f}")

# ---------------------------
# Show Data
# ---------------------------
if st.checkbox("Show Dataset"):
    st.write(df.head())
