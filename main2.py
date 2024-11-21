import streamlit as st
import io
from PIL import Image
import inference  # Import your function from inference.py
from fastai.vision.all import *
import os

# Main app setup
st.set_page_config(page_title="Medical Images Classification", layout="wide")

# Title of the app
st.title("Medical Images Classification")
# Get the absolute path to the image
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
            image = Image.open(uploaded_image)

            # Specify save path for cropped tiles (can be in memory as well)
            #save_path = "Image_to_predict"  # Use string paths

            # Load the trained model
            learn_inf = load_learner("placenta_classification_export.pkl", pickle_module=pickle)

            # Show loading spinner during the classification process
            with st.spinner('Processing your image...'):
                try:
                    # Perform classification and aggregation
                    detailed_predictions, final_prediction, avg_probs = inference.classify_and_aggregate(
                        learn_inf, image
                    )

                    # Display the result
                    st.write(f"### Final Prediction: **{final_prediction}**")
                    st.subheader(f"Original Image: {uploaded_image.name}")
                    st.image(image, caption=f"Aggregate Prediction: {final_prediction}", use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")

        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")