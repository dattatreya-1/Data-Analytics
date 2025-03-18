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

# Convert input to numpy array
user_input = np.array([[maxNUM_UNIQUE_OPERANDS, maxNUM_UNIQUE_OPERATORS, COUPLING_BETWEEN_OBJECTS,
                        maxHALSTEAD_DIFFICULTY, maxNUM_OPERANDS, refined_fuzzy_defect_likelihood]])

# Predict on button click
if st.button("Predict Software Reliability"):
    prediction = model.predict(user_input)
    result = "Reliable" if prediction[0][0] > 0.5 else "Not Reliable"
    st.write(f"### Prediction: {result}")

    # Show confidence score
    confidence = float(prediction[0][0]) * 100
    st.write(f"Confidence: {confidence:.2f}%")
