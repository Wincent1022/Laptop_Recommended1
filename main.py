import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def load_models():
    """Load trained models and data from joblib file."""
    return joblib.load("laptop_recommender.joblib")

def recommend_knn(models, input_features, n_recommendations=5):
    """Recommend laptops using KNN based on PCA-reduced features."""
    knn_pca = models["knn_pca"]
    pca = models["pca"]
    input_transformed = pca.transform([input_features])
    distances, indices = knn_pca.kneighbors(input_transformed, n_neighbors=n_recommendations)
    return models["cleaned_data"].iloc[indices[0]]

def recommend_cosine(models, input_features, n_recommendations=5):
    """Recommend laptops using Cosine Similarity."""
    cosine_sim = models["cosine_sim"]
    similarities = cosine_similarity([input_features], models["cleaned_data"].select_dtypes(include=["number"]))[0]
    top_indices = np.argsort(similarities)[-n_recommendations:][::-1]
    return models["cleaned_data"].iloc[top_indices]

def main():
    """Main function for deployment."""
    print("Loading models...")
    models = load_models()
    print("Models loaded successfully.")

    print("Silhouette Score (K-Means):", models["silhouette_score"])
    
    # Example Input (User selects values manually for testing)
    example_input = np.random.rand(models["cleaned_data"].select_dtypes(include=['number']).shape[1])
    
    print("\nTop 5 Recommendations using KNN:")
    knn_recommendations = recommend_knn(models, example_input)
    print(knn_recommendations)
    
    print("\nTop 5 Recommendations using Cosine Similarity:")
    cosine_recommendations = recommend_cosine(models, example_input)
    print(cosine_recommendations)

if __name__ == "__main__":
    main()
