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
pca = models.get("pca", None)  # Load PCA if available

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

# Apply Standard Scaling
input_scaled = scaler.transform(input_unscaled_df[expected_columns])

# **Apply PCA transformation (Must be same as in .ipynb)**
if pca is not None:
    input_scaled = pca.transform(input_scaled)  # Reduce to 5 features

# Debugging: Print input shape
st.write(f"Input shape after transformation: {input_scaled.shape}")
st.write(f"Expected input shape for model: {best_model.n_features_in_}")

# Recommendation Button
if st.sidebar.button("Recommend Laptops"):
    st.title("Recommended Laptops")

    try:
        # **Check that input_scaled has 5 features before prediction**
        if input_scaled.shape[1] != best_model.n_features_in_:
            raise ValueError(f"Feature mismatch: Model expects {best_model.n_features_in_} features, but received {input_scaled.shape[1]}.")

        # **Get recommendations using best_model**
        distances, indices = best_model.kneighbors(input_scaled, n_neighbors=5)
        recommended_laptops = cleaned_data.iloc[indices[0]]

        # **Display results**
        st.subheader("Top 5 Recommended Laptops")
        st.table(recommended_laptops[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])

    except ValueError as e:
        st.error(f"Model input mismatch: {str(e)}")
