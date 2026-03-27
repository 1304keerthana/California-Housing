# app.py

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------
# Title
# ---------------------------
st.title("🏠 California Housing Price Prediction")
st.write("Predict house prices using Ridge Regression")

# ---------------------------
# Load Data
# ---------------------------
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# ---------------------------
# Prepare Data
# ---------------------------
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Model
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# ---------------------------
# User Input
# ---------------------------
st.sidebar.header("Enter House Details")

MedInc = st.sidebar.slider("Median Income", 0.0, 15.0, 3.0)
HouseAge = st.sidebar.slider("House Age", 1.0, 50.0, 20.0)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.sidebar.slider("Population", 100.0, 5000.0, 1000.0)
AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
Longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -120.0)

# Convert input into array
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                        Population, AveOccup, Latitude, Longitude]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# ---------------------------
# Output
# ---------------------------
st.subheader("🏡 Predicted House Price")
st.write(f"💰 Estimated Value: ${prediction[0]*100000:.2f}")

# ---------------------------
# Show Dataset
# ---------------------------
if st.checkbox("Show Dataset"):
    st.write(df.head())
