import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

# Define paths
data_dir = "../dataset-original"

BATCH_SIZE = 128
IMG_SIZE = (256, 256)

# Image data generators for preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_set = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

valid_set = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save data generators for later use
joblib.dump(train_set, "train_set.pkl")
joblib.dump(valid_set, "valid_set.pkl")