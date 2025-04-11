import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import pydeck as pdk
import matplotlib.pyplot as plt

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

st.subheader("Select Variables to Use")
use_bedrooms = st.checkbox("Bedrooms", value=True)
use_bathrooms = st.checkbox("Bathrooms", value=True)
use_tenure = st.checkbox("Host Tenure", value=True)
use_accommodates = st.checkbox("Accommodates", value=True)
use_date = st.checkbox("Date of Stay", value=True)
use_listings = st.checkbox("Host's Total Listings", value=True)
use_reviews = st.checkbox("Reviews Per Month", value=True)
use_min_nights = st.checkbox("Avg Minimum Nights", value=True)

input_dict = {}
if use_bedrooms:
    input_dict['bedrooms'] = bedrooms
if use_bathrooms:
    input_dict['num_bathrooms'] = bathrooms
if use_tenure:
    input_dict['tenure'] = tenure
if use_accommodates:
    input_dict['accommodates'] = accommodates
if use_date:
    input_dict['date'] = days_since
if use_listings:
    input_dict['host_total_listings_count'] = host_listings
if use_reviews:
    input_dict['reviews_per_month'] = reviews_per_month
if use_min_nights:
    input_dict['minimum_nights_avg_ntm'] = minimum_nights
input_dict['latitude'] = lat
input_dict['longitude'] = lon

X_input = pd.DataFrame([input_dict])

for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]


if st.button("Estimate Price"):
    predicted_price = model.predict(X_input)[0]
    st.success(f"Estimated Airbnb Price: ${predicted_price:.2f}")

if st.button("Show Feature Importance"):
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='red', align='center')
        ax.set_xlabel("Relative Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

if st.button("Compare Prices Across Cities"):
    city_predictions = []
    
    for city_name in city_coords:
        lat_city = city_coords[city_name]['latitude']
        lon_city = city_coords[city_name]['longitude']
        
        city_input = input_dict.copy()
        city_input['latitude'] = lat_city
        city_input['longitude'] = lon_city

        city_df = pd.DataFrame([city_input])
        for col in feature_cols:
            if col not in city_df.columns:
                city_df[col] = 0
        city_df = city_df[feature_cols]
        
        price = model.predict(city_df)[0]
        city_predictions.append((city_name, price))

    city_predictions.sort(key=lambda x: x[1], reverse=True) 

    st.subheader("Predicted Prices with Seelected Features Across Cities")
    pred_df = pd.DataFrame(city_predictions, columns=["City", "Predicted Price"])
    st.bar_chart(pred_df.set_index("City"))
