from fastai.learner import load_learner
import pickle
import os

# Load your Fastai model (if not already loaded)
model_path = 'placenta_classification_export.pkl'  # Replace with the path to your trained model
learn = load_learner(model_path)

# Save the model as a pickle file
pickle_path = 'model_to_streamlit.pkl'

# Ensure it's platform-independent
with open(pickle_path, 'wb') as f:
    pickle.dump(learn, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Model saved as pickle at {os.path.abspath(pickle_path)}")
