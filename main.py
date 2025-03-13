import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and data
@st.cache_data
def load_models():
    return joblib.load("laptop_recommended.joblib")

models = load_models()
cleaned_data = models["cleaned_data"]
unscaled_data = models["unscaled_data"]
scaler = models["scaler"]
best_model = models["best_model"]

# Get unique original values for dropdowns
available_cores = sorted(unscaled_data["num_cores"].unique())
available_ram = sorted(unscaled_data["ram_memory"].unique())
available_prices = sorted(unscaled_data["Price"].unique())

# Streamlit UI
st.sidebar.title("Laptop Recommendation System")
st.sidebar.subheader("Select Preferences")

selected_cores = st.sidebar.selectbox("Select Number of Cores", available_cores)
selected_ram = st.sidebar.selectbox("Select RAM Memory (GB)", available_ram)
selected_price = st.sidebar.selectbox("Select Price", available_prices)

# Prepare user input
input_unscaled = np.array([[selected_cores, selected_ram, selected_price]])
expected_columns = ['Price', 'Rating', 'num_cores', 'num_threads', 'ram_memory', 
                    'display_size', 'resolution_width', 'resolution_height']

input_unscaled_df = pd.DataFrame(input_unscaled, columns=['num_cores', 'ram_memory', 'Price'])

# Fill missing columns with 0 (or default values) to match model expectations
for col in expected_columns:
    if col not in input_unscaled_df:
        input_unscaled_df[col] = 0  

input_scaled = scaler.transform(input_unscaled_df[expected_columns])

# Recommendation Button
if st.sidebar.button("Recommend Laptops"):
    st.title("Recommended Laptops")

    # Use best_model directly
    recommendations = best_model.kneighbors(input_scaled, n_neighbors=5)  # Assuming KNN is best
    indices = recommendations[1][0]  # Extract indices of recommended laptops
    recommended_laptops = cleaned_data.iloc[indices]

    # Display results
    st.table(recommended_laptops[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
