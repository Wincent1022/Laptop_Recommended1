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

# Streamlit UI
st.title("Laptop Recommendation System")
st.write("Enter your laptop preferences to get recommendations!")

# User input form
user_input = {}
for feature in features:
    if feature in label_encoders:
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    else:
        user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, step=1)

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
