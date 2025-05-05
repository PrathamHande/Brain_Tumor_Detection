# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from utils.gradcam import get_img_array, make_gradcam_heatmap, overlay_heatmap


# Load your model
model = tf.keras.models.load_model('model/Tumor-Model.h5')
target_size = (128, 128)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Grad-CAM layer
LAST_CONV_LAYER_NAME = 'conv5_block20_concat'  # Change based on your model architecture

st.set_page_config(page_title="Brain Tumor Classifier + Grad-CAM", layout="centered")

st.title("🧠 Brain Tumor Detector")
st.write("Upload a brain MRI scan to predict tumor type and visualize the model's attention with Grad-CAM.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize(target_size)
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100
    class_name = class_labels[class_idx]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"### 🎯 Predicted: `{class_name}` ({confidence:.2f}% confidence)")

    st.subheader("Grad-CAM Heatmap")

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block20_concat")


    heatmap = cv2.resize(heatmap, (image.size))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap_color, 0.4, 0)

    st.image(superimposed_img, caption="Grad-CAM Overlay", use_column_width=True)
