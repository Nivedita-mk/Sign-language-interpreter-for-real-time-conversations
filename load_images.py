import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

# Function to load and label images
def pickle_images_labels():
    images_labels = []
    gesture_folders = glob("gestures/*")  # All gesture subfolders

    for folder_path in gesture_folders:
        if not os.path.isdir(folder_path):
            continue
        label = os.path.basename(folder_path)  # Folder name becomes the label (e.g., 'A', '1', 'space', 'full-stop')
        images = glob(os.path.join(folder_path, "*.jpg"))

        for image_path in images:
            img = cv2.imread(image_path, 0)  # Read image in grayscale
            if img is not None:
                images_labels.append((img, label))

    return images_labels

# Load and shuffle
images_labels = pickle_images_labels()
images_labels = shuffle(images_labels, random_state=42)
images, labels = zip(*images_labels)

print("Total samples:", len(images))

# Train/Test/Validation split
total_len = len(images)
train_split = int(0.75 * total_len)
val_split = int(0.9 * total_len)

# Save train
with open("train_images", "wb") as f:
    pickle.dump(images[:train_split], f)
with open("train_labels", "wb") as f:
    pickle.dump(labels[:train_split], f)

# Save validation
with open("val_images", "wb") as f:
    pickle.dump(images[train_split:val_split], f)
with open("val_labels", "wb") as f:
    pickle.dump(labels[train_split:val_split], f)

# Save test
with open("test_images", "wb") as f:
    pickle.dump(images[val_split:], f)
with open("test_labels", "wb") as f:
    pickle.dump(labels[val_split:], f)

print("✅ Pickle files created successfully.")
