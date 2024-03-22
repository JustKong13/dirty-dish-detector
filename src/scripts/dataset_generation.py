import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

CLEAN_SINK_DIR = './data/clean_sink'
DIRTY_SINK_DIR = './data/dirty_sink'


def load_data(folder_path): 
    images = []
    labels = []

    for filename in os.listdir(folder_path): 
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((64, 64))
        img = img.convert('L')
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img_array)

        if folder_path == CLEAN_SINK_DIR: 
            labels.append(1)
        else: 
            labels.append(0)

    return np.array(images), np.array(labels)



clean_images, clean_labels = load_data(CLEAN_SINK_DIR)
dirty_images, dirty_labels = load_data(DIRTY_SINK_DIR)

X = np.concatenate(clean_images, dirty_images)
y = np.concatenate(clean_labels, dirty_images)

xTr, xTe, yTr, yTe = train_test_split(X, y, test_size = 0.2, random_state = 1738)

