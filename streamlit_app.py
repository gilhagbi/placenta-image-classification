import streamlit as st
from inference import classify_and_aggregate
from fastai.vision.all import *
import os
from pathlib import Path, PosixPath

# Main app setup
st.set_page_config(page_title="Medical Images Classification", layout="wide")

# Title of the app
st.title("Medical Images Classification")

# Display a default image (can be used as a placeholder or example)
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

            # Specify the path for the model (could be modified to dynamically find the model file)
            model_path = Path('placenta_classification_export.pkl')

            # Load the trained model from the pickle file
            learn_inf = load_learner(model_path, pickle_module=pickle)

            # Show a loading spinner during the classification process
            with st.spinner('Processing your image...'):
                try:
                    # Perform classification and aggregation on the image
                    detailed_predictions, final_prediction, avg_probs = classify_and_aggregate(
                        learn_inf, image_to_analyze
                    )

                    # Display the final prediction
                    st.write(f"### Final Prediction: **{final_prediction}**")
                    st.subheader(f"Original Image: {uploaded_image.name}")
                    st.image(image_to_analyze, caption=f"Aggregate Prediction: {final_prediction}", width=600)

                except Exception as e:
                    # Error handling during the classification process
                    st.error(f"An error occurred during classification: {e}")

        except Exception as e:
            # Error handling when processing the uploaded image
            st.error(f"Failed to process uploaded image: {e}")
