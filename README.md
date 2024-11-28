# Medical Image Classification Pipeline

This repository provides a complete pipeline for medical image classification, including data preprocessing, training a ResNet18-based model, and inference with a user-friendly Streamlit interface. The pipeline is designed to process input images, crop them into overlapping tiles, train a robust model, and classify medical images interactively.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Training](#model-training)
   - [Run Locally for Inference](#run-locally-for-inference)
   - [Using the Streamlit App](#using-the-streamlit-app)
6. [Configuration](#configuration)
7. [Requirements](#requirements)
8. [Output](#output)

---

## Overview

The project is built to automate the classification of medical images into categories. It includes:

- **Preprocessing**: Cropping large images into smaller overlapping tiles.
- **Training**: Fine-tuning a ResNet18 model on the processed data.
- **Inference**: Predicting image categories based on aggregated tile predictions.
- **Interactive App**: A Streamlit-based interface for easy image classification.

---

## Features

### Preprocessing
- **Automated Data Handling**: Downloads and extracts datasets automatically.
- **Tiling**: Crops large images into smaller tiles for more granular analysis.

### Training
- **Transfer Learning**: Utilizes a pre-trained ResNet18 for robust performance.
- **Data Augmentation**: Includes augmentations like rotations, flips, and normalization for better generalization.
- **Exported Model**: Saves the trained model for inference.

### Inference
- **Tile-Wise Predictions**: Classifies individual tiles using the trained model.
- **Aggregated Results**: Provides final predictions by combining tile predictions.
- **Interactive Web App**: User-friendly interface for real-time image classification.

---

## Setup

1. Clone the repository:
    ```bash
    git clone <https://github.com/gilhagbi/placenta-image-classification.git>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure paths and settings in `config.py` .

---

## Project Structure

```plaintext
├── ImagesDatapipline.py                # Preprocessing script
├── train_model.py                      # Training script
├── inference.py                        # Inference script
├── streamlit.py                        # Streamlit app for interactive classification
├── config.py                           # Configuration file
├── requirements.txt                    # Python dependencies
├── placenta_classification_export.pkl  # Exported model for inference
└── README.md                           # Project documentation
```



# Usage

### 1. Data Preprocessing

Run the `ImagesDatapipline.py` script to download, extract, and preprocess the dataset by cropping images into overlapping tiles.

```bash
python ImagesDatapipline.py
```
### 2. Model Training

Train the ResNet18 model by running the `train_model.py` script. The script will preprocess the data (if not already done) and train the model, saving the trained model as `placenta_classification_export.pkl`.

```bash
python train_model.py
```


### 3. Run Locally for Inference

To use the inference functions programmatically:

```python
from inference import classify_and_aggregate
from PIL import Image
from fastai.vision.all import load_learner

# Load the pre-trained model
learn_inf = load_learner('model_to_streamlit.pkl')

# Open an image
image = Image.open("path_to_image.jpg")

# Perform classification
detailed_predictions, final_prediction, avg_probs = classify_and_aggregate(learn_inf, image)

print(f"Final Prediction: {final_prediction}")
```

### 4. Using the Streamlit App

Launch the Streamlit app for interactive image classification:

```bash
streamlit run streamlit_app.py
```
## Configuration

The `config.py` file manages pipeline settings. Key variables include:

- **FILE_ID**: Google Drive file ID of the ZIP file containing the dataset.
- **ZIP_FILE_PATH**: Path to save the downloaded ZIP file.
- **EXTRACT_FOLDER**: Path to extract the dataset.
- **SAVE_PATH**: Path to save processed image tiles.

---

## Requirements

- Python 3.10 or later

Libraries:
- `fastai`
- `torch`
- `torchvision`
- `Pillow`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`
- `gdown`
- `pandas`

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Output

- **Processed Images**: Cropped image tiles saved in the specified directory.
- **Trained Model**: A serialized model (`placenta_classification_export.pkl`).
- **Predictions**: Tile-wise and aggregated predictions during inference.
- **Interactive Results**: Real-time results displayed in the Streamlit app.
