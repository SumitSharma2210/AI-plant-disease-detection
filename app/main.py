import os
import json
import gdown
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64

# Google Drive File ID of your model
FILE_ID = "1ruvxlkBJt1KpVq-wz6cGrOy-EqNpPWaQ"
MODEL_PATH = "model/disease_prediction_model.h5"

# Function to download the model using gdown
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... Please wait.")
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download the model: {e}")

# Load the model
@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_indices_path = "app/class_indices.json"
if not os.path.exists(class_indices_path):
    st.error("class_indices.json file is missing!")
else:
    with open(class_indices_path, "r") as file:
        class_indices = json.load(file)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)  # Load the image
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype("float32") / 255.0  # Scale the image values
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

# Convert the background image to Base64
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Path to the downloaded background image
background_image_path = "app/picture1.jpg"  # Adjust path if necessary
background_base64 = get_base64_encoded_image(background_image_path)

# Custom CSS for background image and dark overlay
st.markdown(
    f"""
   <style>
    html, body {{
        height: 100%;
        margin: 0;
        padding: 0;
    }}
    body {{
        background-image: url("data:image/jpeg;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed; /* Ensures the background stays fixed while scrolling */
    }}
    .stApp {{
        background-color: rgba(0, 0, 0, 0.5); /* Dark overlay for better contrast */
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("AI-based Plant Disease Detection")

# Upload Image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")

    with col2:
        st.info("Loading model...")
        model = load_model()  # Load the model

        if st.button("Classify"):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")
