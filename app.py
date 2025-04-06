import streamlit as st
import pandas as pd
import numpy as np
import pickle 


with(open('rf_model.pkl', 'rb')) as f: 
    model = pickle.load(f)

with (open('feature_cols.pkl', 'rb')) as f: 
    feature_cols = pickle.load(f)

st.title("Airbnb Price Estimator")

city = st.selectbox("City", ["austin", "hawaii", "new-orleans", "chicago"])
bedrooms = st.slider("Bedrooms", 0, 10, 1)
bathrooms = st.slider("Bathrooms", 0, 5, 1)
tenure = st.slider("Host Tenure (Years)", 0, 20, 1)
num_amenities = st.slider("Number of Amenities", 0, 50, 10)


input_dict = {
    'bedrooms': bedrooms,
    'num_bathrooms': bathrooms,
    'tenure': tenure,
    'num_amenities': num_amenities,
    'city_encoded': 0  
}
X_input = pd.DataFrame([input_dict])

for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]


if st.button("Estimate Price"):
    predicted_price = model.predict(X_input)[0]
    st.success(f"Estimated Airbnb Price: ${predicted_price:.2f}")