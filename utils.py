import pickle
from PIL import Image
import numpy as np

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