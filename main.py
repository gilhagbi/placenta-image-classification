import streamlit as st
from pathlib import Path
import tempfile
import inference  # Import your function from inference.py
from fastai.vision.all import *
# Main app setup
st.set_page_config(page_title="Medical Images Classification", layout="wide")

#Background Image and Styling (Optional)
css = """
<style>
    .stApp {
        background-image: url('background_image_url');
        background-size: cover;
        background-position: center;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Title of the app
st.title("Medical Images Classification")
st.image("robot_image.png", caption="AI Analyzing Medical Images", use_container_width=False, width=300)

# File uploader container
st.subheader("Upload Medical Images for Classification")
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Process uploaded images
if uploaded_images:
    for uploaded_image in uploaded_images:

        # Save the uploaded image to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(uploaded_image.read())
            tmp_image_path = Path(tmp_file.name)

        # Specify save path for cropped tiles
        save_path = Path("Image_to_predict")  # Change this to your desired directory
        learn_inf = load_learner("placenta_classification_export.pkl", pickle_module=pickle)
        # Show loading spinner during the classification process
        with st.spinner('Processing your image...'):
            try:
                # Perform classification and aggregation
                detailed_predictions, final_prediction, avg_probs = inference.classify_and_aggregate(learn_inf,tmp_image_path, save_path)

                # Display the result
                st.write(f"### Final Prediction: **{final_prediction}**")
                st.subheader(f"Original Image: {uploaded_image.name}")
                st.image(str(tmp_image_path), caption=f"Aggregate Prediction: {final_prediction}", use_container_width=True)

                # Optionally, display detailed predictions for each cropped tile
                # st.subheader("Cropped Image Predictions")
                # for idx, (tile_path, (pred, _)) in enumerate(detailed_predictions):
                #     tile = Image.open(tile_path)
                #     st.image(tile, caption=f"Tile {idx + 1}: Prediction: {pred}", use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred during classification: {e}")

        # Clean up temporary file after processing
        tmp_image_path.unlink()
# else:
    # st.write("Please upload images to analyze.")
