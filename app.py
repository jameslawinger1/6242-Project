import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime


with(open('rf_model.pkl', 'rb')) as f: 
    model = pickle.load(f)

with (open('feature_columns.pkl', 'rb')) as f: 
    feature_cols = pickle.load(f)

with open('city_encoder.pkl', 'rb') as f:
    city_encoder = pickle.load(f)

with open('city_coords.pkl', 'rb') as f:
    city_coords = pickle.load(f)

st.title("Airbnb Price Estimator")

#City selection and coordinates
city = st.selectbox("Select City", sorted(city_coords.keys()))
lat_default = city_coords[city]['latitude']
lon_default = city_coords[city]['longitude']
#get coordinates from city
latitude = st.number_input("Latitude", value=lat_default, format="%.6f")
longitude = st.number_input("Longitude", value=lon_default, format="%.6f")
#map it
st.map(pd.DataFrame([{
    'lat': latitude,
    'lon': longitude
}]))

bedrooms = st.slider("Bedrooms", 0, 10, 1)
bathrooms = st.slider("Bathrooms", 0, 5, 1)
tenure = st.slider("Host Tenure (Years)", 0, 20, 1)
accommodates = st.slider("Accommodates", 1, 16, 4)

stay_date = st.date_input("Stay Date", datetime.date.today())
days_since = (datetime.date.today() - stay_date).days

with st.expander("Advanced Listing Settings"):
    host_listings = st.slider("Host's Total Listings", 0, 100, value=6)
    reviews_per_month = st.slider("Reviews Per Month", 0.0, 10.0, value=1.0, step=0.1)
    minimum_nights = st.slider("Avg Minimum Nights", 1, 60, value=3)

input_dict = {
    'bedrooms': bedrooms,
    'num_bathrooms': bathrooms,
    'tenure': tenure,
    'accommodates': accommodates,
    'latitude': latitude,
    'longitude': longitude,
    'date': days_since,
    'host_total_listings_count': host_listings,
    'reviews_per_month': reviews_per_month,
    'minimum_nights_avg_ntm': minimum_nights
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