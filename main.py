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
cleaned_data = models["cleaned_data"]
unscaled_data = models["unscaled_data"]  # Use unscaled data for dropdowns
scaler = models["scaler"]
pca = models["pca"]
knn_pca = models["knn_pca"]
cosine_sim = models["cosine_sim"]

# Select only numerical columns used in scaling
numerical_columns = ['num_cores', 'ram_memory', 'Price']

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

# Prepare user input
input_unscaled = np.array([[selected_cores, selected_ram, selected_price]])

# Ensure input_unscaled has the correct number of features for transformation
input_unscaled_df = pd.DataFrame(input_unscaled, columns=['num_cores', 'ram_memory', 'Price'])

# Match the expected order of features used in StandardScaler()
expected_columns = ['Price', 'Rating', 'num_cores', 'num_threads', 'ram_memory', 
                    'display_size', 'resolution_width', 'resolution_height']

# Fill missing columns with 0 (or mean values) to match scaler's expected shape
for col in expected_columns:
    if col not in input_unscaled_df:
        input_unscaled_df[col] = 0  # Replace with meaningful default values if necessary

# Convert to numpy array and apply scaling
input_scaled = scaler.transform(input_unscaled_df[expected_columns])

# Ensure input_scaled has the correct shape for PCA
if input_scaled.shape[1] != pca.n_features_:
    st.error(f"Feature mismatch! PCA expects {pca.n_features_} features, but received {input_scaled.shape[1]}.")
else:
    # Recommendation Button
    if st.sidebar.button("Recommend Laptops"):
        # Apply PCA transformation
        input_transformed = pca.transform(input_scaled)
        
        # KNN Recommendation
        distances, indices = knn_pca.kneighbors(input_transformed, n_neighbors=5)
        knn_recommendations = cleaned_data.iloc[indices[0]]

        # Cosine Similarity Recommendation
        similarities = cosine_sim.dot(input_scaled.T).flatten()
        top_indices = np.argsort(similarities)[-5:][::-1]
        cosine_recommendations = cleaned_data.iloc[top_indices]

        # Display Recommendations
        st.title("Recommended Laptops")
        st.subheader("Top 5 Laptops Based on KNN")
        st.table(knn_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
        
        st.subheader("Top 5 Laptops Based on Cosine Similarity")
        st.table(cosine_recommendations[['brand', 'processor_tier', 'Price', 'ram_memory', 'display_size', 'gpu_brand']])
