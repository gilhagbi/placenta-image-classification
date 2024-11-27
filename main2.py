import streamlit as st
from inference import classify_and_aggregate
from fastai.vision.all import *
import os
from pathlib import Path, PosixPath

# Monkey-patch Path to use PosixPath
Path = PosixPath
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
            # Specify save path for cropped tiles
            save_path = "Image_to_predict"  # Use string paths
            st.image(image_to_analyze, caption=f"Uploaded Image: {uploaded_image.name}", width=600)
            # Load the model from the pickle file
            #model_path = Path('model_to_streamlit.pkl')
            #
            # with open(model_path, 'rb') as f:
            #     learn_inf = pickle.load(f)
            #
            # print("Model loaded successfully and is platform-independent.")

            # Load the model using load_learner
            # model_path = 'model_to_streamlit.pkl'
            # try:
            #     learn_inf = load_learner(model_path)
            #     st.success("Model loaded successfully.")
            # except Exception as e:
            #     st.error(f"Failed to load model: {e}")

            # Load the trained model
            learn_inf = load_learner(Path("placenta_classification_export.pkl"), pickle_module=pickle)
            #learn_inf = load_learner(model_path, pickle_module=pickle)

            # Show loading spinner during the classification process
            with st.spinner('Processing your image...'):
                try:
                    # Perform classification and aggregation
                    detailed_predictions, final_prediction, avg_probs = classify_and_aggregate(
                        learn_inf, image_to_analyze
                    )

                    # Display the result
                    st.write(f"### Final Prediction: **{final_prediction}**")
                    st.subheader(f"Original Image: {uploaded_image.name}")
                    st.image(image_to_analyze, caption=f"Aggregate Prediction: {final_prediction}", use_container_width=True)
                    #st.image(image_to_analyze, caption=f"Uploaded Image: {uploaded_image.name}", width=600)
                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")

        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")
