import streamlit as st
import joblib
import pandas as pd

# Load the saved model and encoders
model_data = joblib.load("laptop_recommendation.joblib")

# Extract components
clf = model_data["classification_model"]
reg = model_data["regression_model"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]

# Remove unwanted features
features = [f for f in features if f not in ["primary_storage_capacity", "secondary_storage_capacity", "year_of_warranty"]]

# Define possible values for display size and resolution
possible_display_sizes = ["13.3", "14.0", "15.6", "16.0", "17.3"]
possible_resolution_widths = ["1366", "1920", "2560", "3840"]
possible_resolution_heights = ["768", "1080", "1440", "2160"]

# Streamlit UI
st.title("Laptop Recommendation System")
st.write("Enter your laptop preferences to get recommendations!")

# User input form
user_input = {}
for feature in features:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    elif feature == "display_size":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_display_sizes)
    elif feature == "resolution_width":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_resolution_widths)
    elif feature == "resolution_height":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_resolution_heights)
    else:
        user_input[feature] = st.selectbox(f"Select {feature}", list(range(1, 17)))  # Dropdown for numerical values

# Predict button
if st.button("Get Recommendation"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical values
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    
    # Make predictions
    classification_prediction = clf.predict(input_df)[0]
    regression_prediction = reg.predict(input_df)[0]
    
    # Display results
    st.subheader("Recommendation Results:")
    st.write(f"Predicted Laptop Rating: {classification_prediction}")
    st.write(f"Estimated Laptop Price: ${regression_prediction:.2f}")
