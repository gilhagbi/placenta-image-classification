from fastai.vision.all import *
import torch
from PIL import Image
import config
import cloudpickle

# Load the trained model
#learn_inf = load_learner(str("training/placenta_classification_export.pkl"))
import pickle



# Crop function with overlap
# def crop_with_overlap(image_path, save_path, tile_size=(512, 512), stride=(256, 256)):
#     img = Image.open(image_path)
#     tiles = []
#     save_path = str(save_path)  # Convert save_path to a Path object
#     #save_path.mkdir(parents=True, exist_ok=True)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     image_path = str(image_path)  # Convert image_path to a Path object
#     for i in range(0, img.width - tile_size[0] + 1, stride[0]):
#         for j in range(0, img.height - tile_size[1] + 1, stride[1]):
#             tile = img.crop((i, j, i + tile_size[0], j + tile_size[1]))
#             #tile_path = save_path / f"{image_path.stem}_tile_{i}_{j}.jpg"  # Now image_path is a Path object
#             tile_path = os.path.join(save_path, f"{image_path}_tile_{i}_{j}.jpg")
#             tile.save(tile_path)
#             tiles.append(tile_path)
#
#     return tiles
#
#
# # Classify tiles and aggregate predictions
# def classify_and_aggregate(learn_inf, image_path, save_path, tile_size=(512, 512), stride=(256, 256)):
#     tiles = crop_with_overlap(image_path, save_path, tile_size, stride)
#     predictions = [(os.path.basename(tile), learn_inf.predict(PILImage.create(tile))) for tile in tiles] # Unpacking two values
#
#     # Convert probabilities to float32 before stacking and computing mean
#     avg_probs = torch.stack([pred[1][1].float() for pred in predictions]).mean(dim=0)
#     final_pred = "Week 30-32" if avg_probs.mean() < 0.5 else "Week 38-40"
#
#     return predictions, final_pred, avg_probs


def show_image_with_predictions(image_path, detailed_predictions, final_prediction, avg_probs, tiles):
    img = Image.open(image_path)

    # Plot the original image with its final prediction score
    fig, ax = plt.subplots(1, len(tiles) + 1, figsize=(20, 5))  # Create subplots for original + tiles
    ax[0].imshow(img)
    ax[0].set_title(f"Original: {final_prediction}\nAvg Probs: {avg_probs}")
    ax[0].axis('off')

    # Plot each cropped tile with its prediction score
    for idx, (tile_path, (pred, probs, _)) in enumerate(zip(tiles, detailed_predictions)):
        tile_img = Image.open(tile_path)
        ax[idx + 1].imshow(tile_img)
        ax[idx + 1].set_title(f"Tile {idx}: {pred}\nProbs: {probs}")
        ax[idx + 1].axis('off')

    plt.show()



from PIL import Image
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
