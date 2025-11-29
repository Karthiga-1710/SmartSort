import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("waste_classifier.h5")

# Class names (adjust as per your dataset)
class_names = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash", "Food Organic"]

# Page config
st.set_page_config(page_title="♻️ Smart Waste Classifier", page_icon="🗑️", layout="centered")

# Title
st.title("♻️ Smart Waste Classification App")
st.markdown("Upload your waste image and let AI tell you what type it is 🚀")

# Sidebar
st.sidebar.title("⚙️ Options")
st.sidebar.info("This app classifies waste images into categories.")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a Waste Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    predicted_class = class_names[class_index]

    # Show result
    st.success(f"✅ Predicted Class: **{predicted_class}**")

# Background style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f9f9;
        font-family: 'Segoe UI';
    }
    </style>
    """,
    unsafe_allow_html=True
)
