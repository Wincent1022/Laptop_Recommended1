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
scaler = models["scaler"]

# Select only numerical columns used in scaling
numerical_columns = ['num_cores', 'ram_memory', 'Price']
scaled_data = cleaned_data[numerical_columns].to_numpy()

# Ensure scaler was fitted properly before applying inverse transformation
if hasattr(scaler, "scale_") and scaled_data.shape[1] == len(scaler.scale_):
    original_data = pd.DataFrame(scaler.inverse_transform(scaled_data),
                                 columns=numerical_columns)
else:
    original_data = cleaned_data[numerical_columns].copy()  # Use unscaled data if inverse transform fails

# Get unique original values for dropdowns
available_cores = sorted(original_data["num_cores"].unique())
available_ram = sorted(original_data["ram_memory"].unique())
available_prices = sorted(original_data["Price"].unique())

# Streamlit UI
st.sidebar.title("Laptop Recommendation System")
st.sidebar.subheader("Select Preferences")

# Sidebar user inputs with dropdown selections
selected_cores = st.sidebar.selectbox("Select Number of Cores", available_cores)
selected_ram = st.sidebar.selectbox("Select RAM Memory (GB)", available_ram)
selected_price = st.sidebar.selectbox("Select Price", available_prices)

# Convert input to scaled values for model input
input_unscaled = np.array([[selected_cores, selected_ram, selected_price]])
if hasattr(scaler, "scale_") and input_unscaled.shape[1] == len(scaler.scale_):
    input_scaled = scaler.transform(input_unscaled)
else:
    input_scaled = input_unscaled  # Use unscaled input if scaler is unavailable

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
