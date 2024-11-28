from inference import classify_and_aggregate  # Assuming this is in your inference.py

# Example usage
# 1. Path to the image you want to classify
image_path = 'training/Images/Images/Week38-40/0_1.png'

# 2. Path to save the cropped tiles (this folder should exist or will be created automatically)
save_path = "test_file/"

# 3. Perform the classification and aggregation of the image and its tiles
detailed_predictions, final_prediction, avg_probs = classify_and_aggregate(image_path, save_path)

# Optional: Print detailed predictions for each tile
print("Detailed Predictions for Each Tile:")
for tile_name, (pred, probs, _) in detailed_predictions:
    print(f"Tile: {tile_name} | Prediction: {pred} | Probabilities: {probs}")

# Final Prediction and Average Probabilities
print(f"\nFinal Prediction: {final_prediction}")
print(f"Average Probabilities: {avg_probs}")
