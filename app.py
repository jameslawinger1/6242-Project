import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import pydeck as pdk


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

lat = city_coords[city]['latitude']
lon = city_coords[city]['longitude']

# Create pydeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lat": lat, "lon": lon}],
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=5000,
)

# Create zoomed-out map view
view_state = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=6,           # ← adjust zoom here (4–12 is typical)
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[layer],
))

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