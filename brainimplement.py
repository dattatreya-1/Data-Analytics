import zipfile
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# ✅ Step 1: Extract ZIP File
zip_path = "/content/ccn_brain_images.zip"  # Replace with your uploaded file name
extract_path = "/content/dataset"  # Define extraction location

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

