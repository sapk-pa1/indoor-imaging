import os
import time
from glob import glob
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np 

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
        # 1. Slice image in memory
        image_slices = utils.slice_image_in_memory(img_path, config.NUM_SLICES)
        
        # 2. Extract objects from each slice
        all_raw_objects = []
        for slice_img in image_slices:
            slice_img = slice_img.resize(config.IMAGE_TARGET_SIZE)
            x = image.img_to_array(slice_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x, verbose=0)
            decoded_preds = decode_predictions(preds, top=config.TOP_N_OBJECTS)[0]
            object_names = [pred[1] for pred in decoded_preds]
            all_raw_objects.extend(object_names)
            
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