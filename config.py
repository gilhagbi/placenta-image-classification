from pathlib import Path

# Define global paths
FILE_ID = '1YHaxS8f6fQ5IMT3JJVeBH40eZSS-UC9h'
ZIP_FILE_PATH = 'Images.zip'
EXTRACT_FOLDER = 'Images'
SAVE_PATH = Path("cropped/images")

# Ensure the save path exists
SAVE_PATH.mkdir(parents=True, exist_ok=True)