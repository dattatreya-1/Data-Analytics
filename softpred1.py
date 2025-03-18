import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the trained ANN model
model = tf.keras.models.load_model("model2.h5")

st.title("Software Reliability Prediction")

st.write("Enter the software metrics below to check if the software is reliable.")

# Define input fields
maxNUM_UNIQUE_OPERANDS = st.number_input("Max Unique Operands", min_value=0.0, max_value=1000.0, step=0.1)
maxNUM_UNIQUE_OPERATORS = st.number_input("Max Unique Operators", min_value=0.0, max_value=1000.0, step=0.1)
COUPLING_BETWEEN_OBJECTS = st.number_input("Coupling Between Objects", min_value=0.0, max_value=1000.0, step=0.1)
maxHALSTEAD_DIFFICULTY = st.number_input("Max Halstead Difficulty", min_value=0.0, max_value=1000.0, step=0.1)
maxNUM_OPERANDS = st.number_input("Max Num Operands", min_value=0.0, max_value=1000.0, step=0.1)

try:
    df = pd.read_csv("promise_dataset.csv")  # Load the dataset to get mean values
    mean_values = df.mean()  # Compute mean values for all features
except:
    mean_values = {feature: 0 for feature in range(95)}  # If dataset is not available, use zeros

# **📌 Step 3: Construct the Full Feature Array (User Inputs + Default Values)**
full_feature_list = []  # Stores all 95 features
for feature in range(95):  # Loop through all expected 95 features
    feature_name = f"Feature_{feature+1}"  # Placeholder name

    if feature_name in selected_features:
        full_feature_list.append(user_inputs[feature_name])  # Add user input
    else:
        full_feature_list.append(mean_values.get(feature_name, 0))  # Use mean or zero


# Convert to NumPy array
user_input_array = np.array([full_feature_list]).astype(np.float32)

# Compute fuzzy feature (same logic as in training)
def fuzzy_membership(value, feature, low_threshold, medium_threshold):
    if value <= low_threshold[feature]:
        return 0.2  # Low
    elif value <= medium_threshold[feature]:
        return 0.5  # Medium
    else:
        return 0.8  # High

# Load dataset to calculate percentiles (replace with actual dataset)
df = pd.read_csv("promise_dataset.csv")  # Load the dataset
low_threshold = df[['maxNUM_UNIQUE_OPERANDS', 'maxNUM_UNIQUE_OPERATORS',
                    'COUPLING_BETWEEN_OBJECTS', 'maxHALSTEAD_DIFFICULTY', 'maxNUM_OPERANDS']].quantile(0.33)
medium_threshold = df[['maxNUM_UNIQUE_OPERANDS', 'maxNUM_UNIQUE_OPERATORS',
                       'COUPLING_BETWEEN_OBJECTS', 'maxHALSTEAD_DIFFICULTY', 'maxNUM_OPERANDS']].quantile(0.66)

# Compute fuzzy likelihood for user input
refined_fuzzy_defect_likelihood = np.mean([
    fuzzy_membership(maxNUM_UNIQUE_OPERANDS, 'maxNUM_UNIQUE_OPERANDS', low_threshold, medium_threshold),
    fuzzy_membership(maxNUM_UNIQUE_OPERATORS, 'maxNUM_UNIQUE_OPERATORS', low_threshold, medium_threshold),
    fuzzy_membership(COUPLING_BETWEEN_OBJECTS, 'COUPLING_BETWEEN_OBJECTS', low_threshold, medium_threshold),
    fuzzy_membership(maxHALSTEAD_DIFFICULTY, 'maxHALSTEAD_DIFFICULTY', low_threshold, medium_threshold),
    fuzzy_membership(maxNUM_OPERANDS, 'maxNUM_OPERANDS', low_threshold, medium_threshold)
])



# Predict on button click
if st.button("Predict Software Reliability"):
    prediction = model.predict(user_input)
    result = "Reliable" if prediction[0][0] > 0.5 else "Not Reliable"
    st.write(f"### Prediction: {result}")

    # Show confidence score
    confidence = float(prediction[0][0]) * 100
    st.write(f"Confidence: {confidence:.2f}%")
