# Placenta Classification Model

import gdown
from fastai.vision.all import *
from torchvision.models import resnet18
import torch
import torch.nn as nn
import config
from ImagesDataPipeline import images_data_process


def train_model():
    """
    Set up and train the ResNet18 model for placenta classification.
    The model is fine-tuned on the data and then exported.
    """
    # Prepare the DataBlock with augmentations
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,  # Load image files
        splitter=RandomSplitter(valid_pct=0.2),  # Split data into train/validation
        get_y=parent_label,  # Get labels from parent folder names
        item_tfms=[Resize(256)],  # Resize all images to 256x256
        batch_tfms=[
            Rotate(max_deg=20),  # Random rotation
            FlipItem(),  # Random horizontal flip
            RandomResizedCrop(256, min_scale=0.8),  # Random cropped patches
            Normalize.from_stats(*imagenet_stats),  # Normalize using ImageNet stats
        ],
    ).dataloaders(str(config.SAVE_PATH), bs=32)

    # Initialize the ResNet18 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, dls.c)  # Adjust final layer for classification

    # Create a Learner and fine-tune the model
    learner = Learner(dls, model, metrics=accuracy)
    learner.fine_tune(1)

    # Export the trained model
    learner.export(Path("placenta_classification_export.pkl"), pickle_module=pickle)


def main():
    """
    Main execution flow for processing data and training the model.
    """
    images_data_process(config)
    train_model()
    print("Model saved as 'placenta_classification_export.pkl'.")


if __name__ == "__main__":
    main()
