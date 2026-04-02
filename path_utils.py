import os

# Get the absolute path of the project root (where this file is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define specific paths for data, models, and outputs
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')

def get_path(base_dir, filename):
    return os.path.join(base_dir, filename)
