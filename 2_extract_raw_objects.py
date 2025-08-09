import os
import time
from glob import glob
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np 
from utils import preprocess_image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import config
import utils

import warnings
warnings.filterwarnings("ignore")

# --- Worker Initialization ---
# This ensures each parallel worker has its own model instance
def init_worker():
    global model
    model = InceptionV3(weights='imagenet')

def process_image(img_path):
    """Processes a single image: slices it and extracts raw objects."""
    try:
        img_batch = preprocess_image(img_path) 
        preds = model.predict(img_batch, verbose=0)
        # Decode and collect objects
        decoded_preds = decode_predictions(preds, top=10) # Get top 10 predictions
        all_raw_objects = [pred[1] for sublist in decoded_preds for pred in sublist]
         
        return img_path, all_raw_objects
    except Exception as e: 
        print(f"Could not process {img_path}: {e}")
        return img_path, []

if __name__ == '__main__':
    
    image_paths = glob(os.path.join(config.MIT67_PATH, '*/*.jpg'))
    print(f"Found {len(image_paths)} images to process.")

    start_time = time.time()
    init_worker() 

    # Use joblib for parallel processing
    results = Parallel(n_jobs=config.NUM_JOBS, verbose=10)(
        delayed(process_image)(p) for p in image_paths
    )
    
    # Structure the results into a dictionary
    raw_objects_data = {path: objects for path, objects in results if objects}
    
    # Save the extracted data
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    utils.save_data(raw_objects_data, config.RAW_OBJECTS_FILE)

    end_time = time.time()
    print(f"\nFinished extracting raw objects for {len(raw_objects_data)} images.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")