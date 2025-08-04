import streamlit as st
import pickle
import json
import numpy as np

model = pickle.load(open('dataset/house_price_pridiction/RandomForestRegressor.pkl', 'rb'))
columns = json.load(open('dataset/house_price_pridiction/columns.json', 'r'))['data_columns']

st.title("🏡 Bengaluru House Price Predictor")

# User Inputs
sqft = st.number_input("Total Square Feet", 500, 10000)
bath = st.number_input("Bathrooms", 1, 10)
bedroom = st.number_input("bedroom", 1, 10)
location = st.selectbox("Location", sorted([loc for loc in columns if loc not in ['total_sqft', 'bath', 'bhk']]))
balcony = st.number_input("Balcony", 0, 10)

if st.button("Predict Price"):
    x = np.zeros(len(columns))
    x[columns.index('total_sqft')] = sqft
    x[columns.index('bath')] = bath
    x[columns.index('bedroom')] = bedroom
    x[columns.index('balcony')] = balcony
    if location in columns:
        x[columns.index(location)] = 1

    price = model.predict(np.array([x]))[0]
    st.success(f"Estimated Price: ₹{round(price, 2)} Lakhs")
