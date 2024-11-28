from fastai.vision.all import *
import torch
from PIL import Image
import pickle
import os


def crop_with_overlap(image, tile_size=(512, 512), stride=(256, 256)):
    tiles = []  # List to store cropped images in memory

    # Process image to crop it into tiles
    for i in range(0, image.width - tile_size[0] + 1, stride[0]):
        for j in range(0, image.height - tile_size[1] + 1, stride[1]):
            tile = image.crop((i, j, i + tile_size[0], j + tile_size[1]))
            tiles.append(tile)  # Append the cropped image (PIL.Image object) to the list

    return tiles  # Return the list of tiles


def classify_and_aggregate(learn_inf, image, tile_size=(512, 512), stride=(256, 256)):
    # Crop the image into tiles (in memory)
    tiles = crop_with_overlap(image, tile_size, stride)

    # Predict on each tile (in memory)
    predictions = [(f"tile_{idx}", learn_inf.predict(PILImage.create(tile))) for idx, tile in enumerate(tiles)]

    # Convert probabilities to float32 before stacking and computing mean
    avg_probs = torch.stack([pred[1][1].float() for pred in predictions]).mean(dim=0)

    # Determine final prediction based on average probability
    final_pred = "Week 30-32" if avg_probs.mean() < 0.5 else "Week 38-40"

    return predictions, final_pred, avg_probs
