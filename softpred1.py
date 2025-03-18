import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the trained ANN model
model = tf.keras.models.load_model("model2.h5")

st.title("🔍 Software Reliability Prediction")

st.write("Enter the key software metrics below. Other required features will be automatically filled.")

# **📌 Step 1: Define Selected Features for User Input (MANUAL ENTRY)**
selected_features = ["maxNUM_UNIQUE_OPERANDS", "maxNUM_UNIQUE_OPERATORS",
                     "COUPLING_BETWEEN_OBJECTS", "maxHALSTEAD_DIFFICULTY", "maxNUM_OPERANDS"]

# Create an empty dictionary to store user inputs
user_inputs = {}

# **📌 Step 2: Accept user input only for selected features**
for feature in selected_features:
    user_inputs[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=1000.0, step=0.1)

# **📌 Step 3: Load the default values for all 95 features**
try:
    df = pd.read_csv("promise_dataset.csv")  # Load dataset to get feature names & mean values
    all_feature_names = list(df.columns)  # Get the exact names used in training
    mean_values = df.mean()  # Compute mean for all columns
except:
    all_feature_names = [f"Feature_{i+1}" for i in range(95)]  # If dataset is missing, use Feature_1, Feature_2, etc.
    mean_values = {feature: 0 for feature in all_feature_names}  # Default missing values to zero

# **📌 Step 4: Construct the Full Feature Array (User Inputs + Default Values)**
full_feature_list = []  # Stores all 95 feature values

for feature_name in all_feature_names:  # Loop through ALL 95 features used in model training
    if feature_name in selected_features:
        full_feature_list.append(user_inputs[feature_name])  # Use user-entered values
    else:
        full_feature_list.append(mean_values.get(feature_name, 0))  # Fill with mean value (or zero if missing)

# Convert to NumPy array
user_input_array = np.array([full_feature_list]).astype(np.float32)

# **📌 Step 5: Predict on button click**
if st.button("🚀 Predict Software Reliability"):
    prediction = model.predict(user_input_array)
    result = "✅ Reliable" if prediction[0][0] > 0.5 else "❌ Not Reliable"

    st.subheader(f"🔮 Prediction: {result}")
    st.write(f"Confidence Score: **{float(prediction[0][0]) * 100:.2f}%**")
