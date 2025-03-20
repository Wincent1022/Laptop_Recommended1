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

# Define possible values for display size, resolution, cores, and RAM
possible_display_sizes = ["13.3", "14.0", "15.6", "16.0", "17.3"]
possible_resolution_widths = ["1366", "1920", "2560", "3840"]
possible_resolution_heights = ["768", "1080", "1440", "2160"]
possible_num_cores = list(range(2, 17, 2))  # Even numbers from 2 to 16
possible_num_threads = list(range(2, 17, 2))  # Even numbers from 2 to 16
possible_ram_memory = list(range(4, 33, 4))  # Increments of 4 up to 32

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
    elif feature == "num_cores":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_num_cores)
    elif feature == "num_threads":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_num_threads)
    elif feature == "ram_memory":
        user_input[feature] = st.selectbox(f"Select {feature}", possible_ram_memory)
    else:
        user_input[feature] = st.selectbox(f"Select {feature}", list(range(1, 17)))  # Default dropdown for other numerical values

# Predict button
if st.button("Get Recommendation"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Ensure input_df has correct column names and order
    input_df = input_df.reindex(columns=features, fill_value=0)
    
    # Encode categorical values if they exist in input_df
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])
    
    # Convert all values to float to match model expectations
    input_df = input_df.astype(float)
    
    # Make predictions
    try:
        classification_prediction = clf.predict(input_df)[0]
        regression_prediction = reg.predict(input_df)[0]
        
        # Display results
        st.subheader("Recommendation Results:")
        st.write(f"Predicted Laptop Rating: {classification_prediction}")
        st.write(f"Estimated Laptop Price: ${regression_prediction:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
