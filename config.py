# --- Project Configuration ---
import os

# Base directory for all datasets
DATASET_BASE_PATH = 'datasets'
DELTA_TYPE = "normal"

MIT67_PATH = os.path.join(DATASET_BASE_PATH, 'MIT-67', 'images')
SCENE15_PATH = os.path.join(DATASET_BASE_PATH, '15-Scene')
NYUV1_PATH = os.path.join(DATASET_BASE_PATH, 'NYU-V1')

# Directory to save all generated outputs and trained models
OUTPUT_DIR = 'output'

# --- FILE NAMES for saved data ---
# These files will be created in the OUTPUT_DIR
RAW_OBJECTS_FILE = os.path.join(OUTPUT_DIR, 'raw_objects_data.pkl')
DICTIONARIES_FILE = os.path.join(OUTPUT_DIR, 'dictionaries.pkl')
TRAINED_SVM_MODEL_FILE = os.path.join(OUTPUT_DIR, 'svm_classifier.pkl')
FEATURE_SCALER_FILE = os.path.join(OUTPUT_DIR, 'feature_scaler.pkl')


# --- PROCESSING PARAMETERS ---
# The paper experiments with 9, 16, and 25 slices. 16 is a robust choice.
NUM_SLICES = 16

# Top N objects to extract from each image slice using InceptionV3
TOP_N_OBJECTS = 10

# Number of CPU cores for parallel processing. -1 uses all available cores.
# Set to 1 if you encounter memory issues.
NUM_JOBS = 1


# --- MODEL PARAMETERS ---
# Target image size for the InceptionV3 model
IMAGE_TARGET_SIZE = (299, 299)

# Kernel for the Support Vector Machine (SVM). 'rbf' is a strong default.
SVM_KERNEL = 'rbf'

# --- DATA SPLIT PARAMETERS ---
# Proportion of the data to be used for testing
TEST_SPLIT_SIZE = 0.2
# Seed for reproducibility of the train/test split
RANDOM_STATE_SEED = 42