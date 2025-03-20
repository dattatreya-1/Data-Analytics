import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "brain_tumor_model.h5"  # Update with correct path if needed
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (update with actual classes from your dataset)
class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']  # Replace with your dataset classes

def preprocess_image(image):
    """Preprocess image to match model input."""
    image = image.resize((224, 224))  # Resize to match CNN input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload a brain scan image and the model will predict the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Upload Brain Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display results
    st.subheader("🔍 Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    st.success("✅ Prediction Complete!")
