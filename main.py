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
    return joblib.load("laptop_recommender.joblib")

models = load_models()
cleaned_data = models["cleaned_data"]

# Unique values for dropdown selections
available_cores = sorted(cleaned_data["num_cores"].unique())
available_ram = sorted(cleaned_data["ram_memory"].unique())
available_prices = sorted(cleaned_data["Price"].unique())

# Streamlit UI
st.sidebar.title("Laptop Recommendation System")
st.sidebar.subheader("Select Preferences")

# Sidebar user inputs with dropdown selections
selected_cores = st.sidebar.selectbox("Select Number of Cores", available_cores)
selected_ram = st.sidebar.selectbox("Select RAM Memory (GB)", available_ram)
selected_price = st.sidebar.selectbox("Select Price", available_prices)

# Convert input to match available values
selected_cores = closest_value(selected_cores, available_cores)
selected_ram = closest_value(selected_ram, available_ram)
selected_price = closest_value(selected_price, available_prices)

input_array = np.array([selected_cores, selected_ram, selected_price]).reshape(1, -1)

# Recommendation Button
if st.sidebar.button("Recommend Laptops"):
    # KNN Recommendation
    knn_pca = models["knn_pca"]
    pca = models["pca"]
    input_transformed = pca.transform(input_array)
    distances, indices = knn_pca.kneighbors(input_transformed, n_neighbors=5)
    knn_recommendations = cleaned_data.iloc[indices[0]]

    # Cosine Similarity Recommendation
    cosine_sim = models["cosine_sim"]
    similarities = cosine_sim.dot(input_array.T).flatten()
    top_indices = np.argsort(similarities)[-5:][::-1]
    cosine_recommendations = cleaned_data.iloc[top_indices]

    # Display Recommendations
    st.title("Recommended Laptops")
    st.subheader("Top 5 Laptops Based on KNN")
    st.table(knn_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
    
    st.subheader("Top 5 Laptops Based on Cosine Similarity")
    st.table(cosine_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
