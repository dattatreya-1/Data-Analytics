# -*- coding: utf-8 -*-
"""brainimplement.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GKJc_LSU1BXfZBv7soqCqqlbrckw1IH-
"""

import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import streamlit as st
import zipfile
import os

st.title("Upload Brain Tumor Dataset")

# File uploader
uploaded_file = st.file_uploader("Upload ZIP file of dataset", type=["zip"])

if uploaded_file is not None:
    zip_path = os.path.join("temp_dataset.zip")  # Save to temporary directory

    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_path = "dataset"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    st.success("✅ Dataset extracted successfully!")
    st.write("Extracted folders:", os.listdir(extract_path))


# Check if extraction has already been done
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully!")

# ✅ Step 2: Verify Extracted Folders
print("Extracted Folders:", os.listdir(extract_path))

# Check if images exist inside class folders
for folder_name in os.listdir(extract_path):
    folder_path = os.path.join(extract_path, folder_name)
    if os.path.isdir(folder_path):  # Check if it's a directory
        image_count = sum(1 for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg')))
        print(f"📂 Folder '{folder_name}' contains {image_count} images.")

# ✅ Step 3: Load Dataset for Training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    extract_path,
    validation_split=0.2,  # 20% for validation
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

# ✅ Step 4: Get Class Names
class_names = train_ds.class_names
num_classes = len(class_names)
print("📌 Detected Classes:", class_names)

st.write(f"Train Dataset: {train_ds}")
st.write(f"Test Dataset: {test_ds}")

# ✅ Step 5: Define CNN Model
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize pixel values

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer for classification
])

# ✅ Step 6: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Change to categorical_crossentropy if labels are one-hot encoded
    metrics=['accuracy']
)

# Show Model Summary
model.summary()

# ✅ Step 7: Train the Model
epochs = 10
history = model.fit(train_ds, epochs=epochs, validation_data=test_ds, verbose=1)

# ✅ Step 8: Evaluate the Model
loss, accuracy = model.evaluate(test_ds, verbose=1)
print("🎯 Test Loss:", loss)
print("🎯 Test Accuracy:", accuracy)

# ✅ Step 9: Save the Model
model.save("/content/brain_tumor_model.h5")
print("✅ Model saved successfully!")
