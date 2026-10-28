import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page settings
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Title
st.title("🫁 Pneumonia Detection App")

# 🔍 DEBUG INFO (you can remove later)
st.write("📂 Current Directory:", os.getcwd())
st.write("📁 Files in Directory:", os.listdir())

# Load model safely
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "pneumonia_model.h5")
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model not found or failed to load: {e}")
        st.stop()

model = load_model()

st.success("✅ Model loaded successfully")

# File uploader
uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

IMG_SIZE = 224

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_array)[0][0]

    # Confidence
    pneumonia_conf = prediction * 100
    normal_conf = (1 - prediction) * 100

    st.subheader("🧠 Prediction Result")

    if prediction > 0.5:
        st.error("🦠 Pneumonia Detected")
        st.write(f"Confidence: {pneumonia_conf:.2f}%")
        st.progress(int(pneumonia_conf))
    else:
        st.success("✅ Normal")
        st.write(f"Confidence: {normal_conf:.2f}%")
        st.progress(int(normal_conf))

    st.info("⚠️ This is an AI prediction and not a medical diagnosis.")