import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and data
@st.cache_data
def load_models():
    return joblib.load("laptop_recommender.joblib")

models = load_models()
cleaned_data = models["cleaned_data"]
numerical_features = cleaned_data.select_dtypes(include=['number']).columns.tolist()

# Streamlit UI
st.sidebar.title("Laptop Recommendation System")
st.sidebar.subheader("Select Preferences")

# Sidebar user inputs
user_inputs = {}
for feature in numerical_features:
    user_inputs[feature] = st.sidebar.slider(
        f"Select {feature}", float(cleaned_data[feature].min()), float(cleaned_data[feature].max()), float(cleaned_data[feature].mean())
    )

# Convert input to array
input_array = np.array([user_inputs[feature] for feature in numerical_features]).reshape(1, -1)

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
