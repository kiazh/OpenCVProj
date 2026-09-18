"""Central configuration. All paths resolve relative to the package root
(FacialRecgTensorFlow/) so tests and notebook wrappers share one source of truth."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANC_PATH = os.path.join(BASE_DIR, "data", "anchor")
POS_PATH = os.path.join(BASE_DIR, "data", "positive")
NEG_PATH = os.path.join(BASE_DIR, "data", "negative")

APP_DIR = os.path.join(BASE_DIR, "application_data")
VERIFICATION_IMAGES_DIR = os.path.join(APP_DIR, "verification_images")
INPUT_IMAGE_PATH = os.path.join(APP_DIR, "input_image", "input_image.jpg")

MODEL_KERAS_PATH = os.path.join(BASE_DIR, "my_model.keras")
MODEL_H5_PATH = os.path.join(BASE_DIR, "siamesemodelv2.h5")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "training_checkpoints")

# Hyperparameters (README sections 5.4 / 7)
IMAGE_SIZE = (100, 100)
EMBEDDING_DIM = 4096
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 50
TRAIN_SPLIT = 0.7
SHUFFLE_BUFFER = 10000
PREFETCH_DEPTH = 8
MAX_SAMPLES_PER_SPLIT = 3000
DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5
VERIFICATION_GALLERY_SIZE = 200
