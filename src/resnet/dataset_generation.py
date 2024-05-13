import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

CLEAN_SINK_DIR = './data/clean_sink'
DIRTY_SINK_DIR = './data/dirty_sink'


def load_data(folder_path): 
    images = []
    labels = []

    for filename in os.listdir(folder_path): 
        img = Image.open(os.path.join(folder_path, filename))
        img = img.resize((256, 256))
        img = img.convert('RGB')
        img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)

        if folder_path == CLEAN_SINK_DIR: 
            labels.append(1)
        else: 
            labels.append(0)

    return np.array(images), np.array(labels)



clean_images, clean_labels = load_data(CLEAN_SINK_DIR)
dirty_images, dirty_labels = load_data(DIRTY_SINK_DIR)

X = np.concatenate((clean_images, dirty_images))
y = np.concatenate((clean_labels, dirty_labels))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1738)


class SinkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {
            'image': torch.tensor(self.X[idx], dtype=torch.float), # 3 x 256 x 256
            'label': torch.tensor(self.y[idx], dtype=torch.int)  # Change dtype if needed
        }
        return sample['image'].view([3,256,256]), sample['label']
    

train_dataset = SinkDataset(X_train, y_train)
test_dataset = SinkDataset(X_test, y_test)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)