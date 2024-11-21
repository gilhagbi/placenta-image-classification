import streamlit as st
from PIL import Image
import inference  # Import your function from inference.py
from fastai.vision.all import *
import os

# Main app setup
st.set_page_config(page_title="Medical Images Classification", layout="wide")

# Title of the app
st.title("Medical Images Classification")

# Display a default image
image_path = os.path.join(os.getcwd(), "robot_image.png")
st.image(image_path, caption="AI Analyzing Medical Images", width=400)

# File uploader container
st.subheader("Upload Medical Images for Classification")
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Process uploaded images
if uploaded_images:
    for uploaded_image in uploaded_images:
        try:
            # Read image from the uploaded file (in memory)
            image_to_analyze = Image.open(uploaded_image)

            # Show loading spinner during the classification process
            with st.spinner('Processing your image...'):
                try:
                    # Display the uploaded image without 'use_container_width'
                    st.image(image_to_analyze, caption=f"Uploaded Image: {uploaded_image.name}", width=600)
                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")

        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")
