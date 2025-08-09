import pickle
from PIL import Image
import config 
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input  
from tensorflow.keras.preprocessing import image


def slice_image_in_memory(image_path, num_slices):
    """Slices a PIL image into sub-images in memory."""
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    rows = cols = int(np.sqrt(num_slices))
    slice_width, slice_height = width // cols, height // rows
    
    slices = []
    for i in range(rows):
        for j in range(cols):
            box = (j * slice_width, i * slice_height, (j + 1) * slice_width, (i + 1) * slice_height)
            slices.append(img.crop(box))
    return slices

def save_data(data, filepath):
    """Saves data to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_data(filepath):
    """Loads data from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def preprocess_image(img_path:str) -> np.ndarray : 
    """Preprocess the images"""
    # 1. Slice image in memory
    image_slices = slice_image_in_memory(img_path, config.NUM_SLICES)
    batch = []
    for slice_img in image_slices:
        slice_img = slice_img.resize(config.IMAGE_TARGET_SIZE)  # Resize
        img_arr = image.img_to_array(slice_img)  # Convert to array
        img_preprocess = preprocess_input(img_arr)  
        batch.append(img_preprocess)

    # Stack all images into a single batch
    batch = np.stack(batch, axis=0)  # Shape: (num_slices, height, width, channels)
    return batch 
            
        