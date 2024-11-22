import streamlit as st
import tempfile
import inference  # Import your function from inference.py
from fastai.vision.all import *
# Main app setup
st.set_page_config(page_title="Medical Images Classification", layout="wide")



# Title of the app
st.title("Medical Images Classification")
# Get the absolute path to the image
image_path = os.path.join(os.getcwd(), "robot_image.png")

st.image("robot_image.png", caption="AI Analyzing Medical Images", width=400)
# File uploader container
st.subheader("Upload Medical Images for Classification")
uploaded_images = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# Process uploaded images

if uploaded_images:
    for uploaded_image in uploaded_images:
        # Save the uploaded image to a temporary location
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            tmp_file.write(uploaded_image.read())
            tmp_file.close()
            tmp_image_path = str(tmp_file.name)
            #tmp_image_path=os.path.dirname(tmp_file)
            #tmp_image_path = str(os.path.basename(tmp_file))


            # Specify save path for cropped tiles
            save_path = "Image_to_predict"  # Use string paths
            #learn_inf = inference.load_cross_platform_model(str("placenta_classification_export.pkl"))
            learn_inf = load_learner(str("placenta_classification_export.pkl"), pickle_module=pickle)
            # Show loading spinner during the classification process
            with st.spinner('Processing your image...'):
                try:
                    # Perform classification and aggregation
                    detailed_predictions, final_prediction, avg_probs = inference.classify_and_aggregate(
                        learn_inf, tmp_image_path, save_path
                    )

                    # Display the result
                    st.write(f"### Final Prediction: **{final_prediction}**")
                    st.subheader(f"Original Image: {uploaded_image.name}")
                    st.image(tmp_image_path, caption=f"Aggregate Prediction: {final_prediction}", use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")

        except Exception as e:
            st.error(f"Failed to process uploaded image: {e}")
        finally:
            # Clean up temporary file after processing
            try:
                os.remove(tmp_image_path)
            except Exception as e:
                st.warning(f"Failed to delete temporary file: {e}")

# else:
    # st.write("Please upload images to analyze.")
