import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import pydeck as pdk
import matplotlib.pyplot as plt
import altair as alt

with(open('rf_model.pkl', 'rb')) as f: 
    model = pickle.load(f)

with (open('feature_columns.pkl', 'rb')) as f: 
    feature_cols = pickle.load(f)

with open('city_encoder.pkl', 'rb') as f:
    city_encoder = pickle.load(f)

with open('city_coords.pkl', 'rb') as f:
    city_coords = pickle.load(f)

st.title("Airbnb Price Estimator")

st.markdown("### Location")
city = st.selectbox("Select City", sorted(city_coords.keys()))
lat = city_coords[city]['latitude']
lon = city_coords[city]['longitude']

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=6,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": lat, "lon": lon}],
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=5000,
        )
    ]
), use_container_width=True)

with st.form("estimator_form"):
    st.markdown("### Basic Listing Info")
    col1, col2 = st.columns(2)
    with col1:
        accommodates = st.slider("Accommodates", 1, 16, 4)
        bedrooms = st.slider("Bedrooms", 0, 10, 1)
    with col2:
        bathrooms = st.slider("Bathrooms", 0, 5, 1)
        stay_date = st.date_input("Stay Date", datetime.date.today())
        days_since = (datetime.date.today() - stay_date).days

    st.markdown("### Host Information")
    col3, col4 = st.columns(2)
    with col3:
        tenure = st.slider("Host Tenure (Years)", 0, 20, 1)
    with col4:
        host_listings = st.slider("Host's Total Listings", 0, 100, 6)

    st.markdown("### Advanced Settings")
    with st.expander("Show Advanced Settings"):
        reviews_per_month = st.slider("Reviews Per Month", 0.0, 10.0, 1.0, step=0.1)
        minimum_nights = st.slider("Avg Minimum Nights", 1, 60, 3)

    st.markdown("---")
    st.subheader("Select Variables to Include in Model")
    col5, col6 = st.columns(2)
    with col5:
        use_accommodates = st.checkbox("Accommodates", value=True)
        use_bedrooms = st.checkbox("Bedrooms", value=True)
        use_bathrooms = st.checkbox("Bathrooms", value=True)
        use_date = st.checkbox("Date of Stay", value=False)
    with col6:
        use_tenure = st.checkbox("Host Tenure", value=True)
        use_listings = st.checkbox("Host's Total Listings", value=True)
        use_reviews = st.checkbox("Reviews Per Month", value=False)
        use_min_nights = st.checkbox("Avg Minimum Nights", value=False)

    submitted = st.form_submit_button("Estimate Price")



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

if submitted:
    predicted_price = model.predict(X_input)[0]
    st.success(f"Estimated Airbnb Price: ${predicted_price:.2f}")

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
    st.subheader("Predicted Prices Across Cities")
    pred_df = pd.DataFrame(city_predictions, columns=["City", "Predicted Price"])
    pred_df["Predicted Price"] = pred_df["Predicted Price"].round(2)

    selection = alt.selection_single(on="mouseover", empty='none', fields=['City'])
    chart = alt.Chart(pred_df).mark_bar().encode(
        x=alt.X("City:N", sort="-y"),
        y=alt.Y("Predicted Price:Q"),
        tooltip=[
            alt.Tooltip("City:N"),
            alt.Tooltip("Predicted Price:Q", format=".2f")
        ],
        color=alt.condition(selection, alt.value('#f53f2c'), alt.value('#aaa'))
    ).properties(
        width=600,
        height=400,
        title="Predicted Prices with Selected Features Across Cities"
    ).add_selection(selection).interactive()

    st.altair_chart(chart, use_container_width=True)

if st.button("Show Feature Importance"):
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        top_10_df = importance_df.head(10).copy()
        selection_feat = alt.selection_single(on="mouseover", empty='none', fields=['Feature'])
        
        chart = (
            alt.Chart(top_10_df)
            .mark_bar()
            .encode(
                x=alt.X("Importance:Q", title="Relative Importance"),
                y=alt.Y(
                    "Feature:N",
                    sort=top_10_df["Feature"].tolist()[::-1],
                    axis=alt.Axis(labelOverlap=False, labelLimit=300)  
                ),
                tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".4f")],
                color=alt.condition(selection_feat, alt.value('#f53f2c'), alt.value('#aaa'))
            )
            .properties(
                width=600,
                height=30 * len(top_10_df),
                title="Top 10 Feature Importance"
            )
            .add_selection(selection_feat)
            .interactive()
        )
        
        st.altair_chart(chart, use_container_width=True)
