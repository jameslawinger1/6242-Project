import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import matplotlib.pyplot as plt

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

st.subheader("Feature Selection")
st.write("Select the features you want to include in the prediction.")

include_bedrooms = st.checkbox("Include Bedrooms", value=True)
include_bathrooms = st.checkbox("Include Bathrooms", value=True)
include_tenure = st.checkbox("Include Host Tenure", value=True)
include_amenities = st.checkbox("Include Number of Amenities", value=True)
include_city = st.checkbox("Include City", value=True)

input_dict = {}

if include_bedrooms:
    input_dict['bedrooms'] = bedrooms
if include_bathrooms:
    input_dict['num_bathrooms'] = bathrooms
if include_tenure:
    input_dict['tenure'] = tenure
if include_amenities:
    input_dict['num_amenities'] = num_amenities
if include_city:
    input_dict['city_encoded'] = 0  
else:
    input_dict['city_encoded'] = 0 

X_input = pd.DataFrame([input_dict])

for col in feature_cols:
    if col not in X_input.columns:
        X_input[col] = 0
X_input = X_input[feature_cols]

input_dict = {
    'bedrooms': bedrooms,
    'num_bathrooms': bathrooms,
    'tenure': tenure,
    'num_amenities': num_amenities,
    'city_encoded': 0  
}
X_input = pd.DataFrame([input_dict])

if st.button("Estimate Price"):
    predicted_price = model.predict(X_input)[0]
    st.success(f"Estimated Airbnb Price: ${predicted_price:.2f}")

if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', align='center')
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance from RF Model")
    
    st.pyplot(fig)
