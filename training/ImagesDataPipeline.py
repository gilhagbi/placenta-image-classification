
# Import required libraries
import os
import shutil
from pathlib import Path
from fastai.vision.all import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import gdown
import config



def download_and_extract(file_id, zip_path, extract_folder):
    if not os.path.exists(extract_folder):
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', zip_path, quiet=False)
        shutil.unpack_archive(zip_path, extract_folder)
        print("Files extracted:", os.listdir(extract_folder))
    else:
        print("Files already extracted.")




def crop_with_overlap(image_path, save_path, tile_size=(512, 512), stride=(256, 256)):
    img = Image.open(image_path)
    width, height = img.size
    label = image_path.parent.name  # Extract label from parent folder
    class_save_path = save_path / label
    class_save_path.mkdir(parents=True, exist_ok=True)

    for i in range(0, width - tile_size[0] + 1, stride[0]):
        for j in range(0, height - tile_size[1] + 1, stride[1]):
            left, upper = i, j
            right, lower = left + tile_size[0], upper + tile_size[1]
            tile = img.crop((left, upper, right, lower))
            tile.save(class_save_path / f"{image_path.stem}_tile_{i}_{j}.jpg")

def images_data_process (config):
    """
    Downloads, extracts, and processes images by cropping them into overlapping tiles.

    Parameters:
    - config: A configuration object with the following attributes:
        - FILE_ID: ID of the file to download.
        - ZIP_FILE_PATH: Path to save the downloaded ZIP file.
        - EXTRACT_FOLDER: Path to extract the ZIP contents.
        - SAVE_PATH: Path to save the cropped image tiles.
    """
    # Step 1: Download and Extract Images
    download_and_extract(config.FILE_ID, config.ZIP_FILE_PATH, config.EXTRACT_FOLDER)

    # Step 2: Crop Images into Overlapping Tiles
    IMAGE_PATH = f"{Path(config.EXTRACT_FOLDER)}/Images"
    image_files = get_image_files(IMAGE_PATH)

    # Check if the crop folder already exists and contains files
    if config.SAVE_PATH.exists() and any(config.SAVE_PATH.iterdir()):
        print(f"Skipping: {config.SAVE_PATH} already exists and contains files.")
    else:
        for image_file in image_files:
            crop_with_overlap(image_file, config.SAVE_PATH)

    print(f"Total cropped images: {len(get_image_files(config.SAVE_PATH))}")