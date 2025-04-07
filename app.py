import streamlit as st
import pandas as pd
import numpy as np
import pickle 


with(open('rf_model.pkl', 'rb')) as f: 
    model = pickle.load(f)

with (open('feature_columns.pkl', 'rb')) as f: 
    feature_cols = pickle.load(f)

with open('city_encoder.pkl', 'rb') as f:
    city_encoder = pickle.load(f)

with open('city_coords.pkl', 'rb') as f:
    city_coords = pickle.load(f)

st.title("Airbnb Price Estimator")

city = st.selectbox("Select City", sorted(city_coords.keys()))

lat_default = city_coords[city]['latitude']
lon_default = city_coords[city]['longitude']

latitude = st.number_input("Latitude", value=lat_default, format="%.6f")
longitude = st.number_input("Longitude", value=lon_default, format="%.6f")

bedrooms = st.slider("Bedrooms", 0, 10, 1)
bathrooms = st.slider("Bathrooms", 0, 5, 1)
tenure = st.slider("Host Tenure (Years)", 0, 20, 1)
num_amenities = st.slider("Number of Amenities", 0, 50, 10)
accommodates = st.slider("Accommodates", 1, 16, 4)
longitude = st.number_input("Longitude", value=-87.6298)  
latitude = st.number_input("Latitude", value=41.8781)

input_dict = {
    'bedrooms': bedrooms,
    'num_bathrooms': bathrooms,
    'tenure': tenure,
    'num_amenities': num_amenities,
    'accommodates': accommodates,
    'latitude': latitude,
    'longitude': longitude
}
X_input = pd.DataFrame([input_dict])

for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]


if st.button("Estimate Price"):
    predicted_price = model.predict(X_input)[0]
    st.success(f"Estimated Airbnb Price: ${predicted_price:.2f}")
    st.caption(f"Based on location: **{city}** at ({latitude:.4f}, {longitude:.4f})")