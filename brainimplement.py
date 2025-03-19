import streamlit as st
import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np

st.title("Brain Tumor Classification")

# ✅ File uploader for dataset ZIP
uploaded_file = st.file_uploader("Upload ZIP file of dataset", type=["zip"])

if uploaded_file is not None:
    zip_path = "dataset.zip"  # Save uploaded file

    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_path = "dataset"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    st.success("✅ Dataset extracted successfully!")
    st.write("Extracted folders:", os.listdir(extract_path))

# ✅ Check dataset before loading
if not os.path.exists(extract_path):
    st.error("⚠️ Dataset path NOT found! Please check ZIP extraction.")

# ✅ Load Dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Reduce batch size to prevent memory issues

try:
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        extract_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        extract_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    st.write("✅ Dataset loaded successfully!")
    class_names = train_ds.class_names
    num_classes = len(class_names)
    st.write(f"📌 Detected Classes: {class_names}")

    # ✅ Display a sample batch
    for image_batch, label_batch in train_ds.take(1):
        st.write(f"🖼 Batch Shape: {image_batch.shape}")
        st.write(f"🏷 Labels: {label_batch.numpy()}")  # Ensure labels are valid integers

except Exception as e:
    st.error(f"⚠️ Error loading training dataset: {e}")

# ✅ Load Pre-Trained Model Instead of Training in Streamlit
@st.cache_resource
def load_trained_model():
    model_path = "brain_tumor_model.h5"  # Load the saved model
    if not os.path.exists(model_path):
        st.error("⚠️ Trained model not found! Please train in Google Colab and upload.")
    return keras.models.load_model(model_path)

# ✅ Use the pre-trained model for prediction
model = load_trained_model()

# ✅ Streamlit UI for Image Upload & Prediction
st.subheader("Upload an Image for Prediction")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # ✅ Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # ✅ Show result
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
