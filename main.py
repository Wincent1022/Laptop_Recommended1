import streamlit as st
import joblib
import numpy as np
import pandas as pd

def closest_value(user_value, available_values):
    """Find the closest match in available values"""
    return min(available_values, key=lambda x: abs(x - user_value))

# Load models and data
@st.cache_data
def load_models():
    return joblib.load("laptop_recommended.joblib")

models = load_models()
unscaled_data = models["unscaled_data"]  # Use unscaled data for dropdowns
cleaned_data = models["cleaned_data"]  # Scaled data for recommendation models
scaler = models["scaler"]

# Get unique original values for dropdowns
available_cores = sorted(unscaled_data["num_cores"].unique())
available_ram = sorted(unscaled_data["ram_memory"].unique())
available_prices = sorted(unscaled_data["Price"].unique())

# Streamlit UI
st.sidebar.title("Laptop Recommendation System")
st.sidebar.subheader("Select Preferences")

# Sidebar user inputs with dropdown selections
selected_cores = st.sidebar.selectbox("Select Number of Cores", available_cores)
selected_ram = st.sidebar.selectbox("Select RAM Memory (GB)", available_ram)
selected_price = st.sidebar.selectbox("Select Price", available_prices)

# Convert input to scaled values for model input
input_unscaled = np.array([[selected_cores, selected_ram, selected_price]])
input_scaled = scaler.transform(input_unscaled)  # Ensure proper scaling before prediction

# Recommendation Button
if st.sidebar.button("Recommend Laptops"):
    # KNN Recommendation
    knn_pca = models["knn_pca"]
    pca = models["pca"]
    input_transformed = pca.transform(input_scaled)
    distances, indices = knn_pca.kneighbors(input_transformed, n_neighbors=5)
    knn_recommendations = cleaned_data.iloc[indices[0]]

    # Cosine Similarity Recommendation
    cosine_sim = models["cosine_sim"]
    similarities = cosine_sim.dot(input_scaled.T).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    cosine_recommendations = cleaned_data.iloc[top_indices]

    # Display Recommendations
    st.title("Recommended Laptops")
    st.subheader("Top 5 Laptops Based on KNN")
    st.table(knn_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
    
    st.subheader("Top 5 Laptops Based on Cosine Similarity")
    st.table(cosine_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
