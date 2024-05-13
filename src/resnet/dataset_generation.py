import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from augmentation import Contrast, HorizontalFlip, Rotate, Shift
import torch
from torch.utils.data import Dataset, DataLoader

CLEAN_SINK_DIR = '../../data/clean_sink'
DIRTY_SINK_DIR = '../../data/dirty_sink'

shift_transform = Shift(max_shift=10)
contrast_transform = Contrast(min_contrast=0.3, max_contrast=1.0)
rotate_transform = Rotate(max_angle=10)
horizontal_flip_transform = HorizontalFlip(p=0.5)

train_transforms = [shift_transform, contrast_transform, rotate_transform, horizontal_flip_transform]

def load_data(folder_path): 
    images = []
    labels = []

    for filename in os.listdir(folder_path): 
        og_img = Image.open(os.path.join(folder_path, filename))

        img = og_img.resize((256, 256))
        img = img.convert('RGB')
        img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32) 
        img_tensor = img_tensor.permute(2, 0, 1) 

        images.append(img_tensor.numpy())

        if folder_path == CLEAN_SINK_DIR: 
            labels.append(1)
        else: 
            labels.append(0)

        print(img_tensor.shape)

        for transform in train_transforms: 
            transformed_img = transform(img_tensor)
            images.append(transformed_img.numpy())
            if folder_path == CLEAN_SINK_DIR: 
                labels.append(1)
            else: 
                labels.append(0)

    return np.stack(images), np.array(labels)



clean_images, clean_labels = load_data(CLEAN_SINK_DIR)
dirty_images, dirty_labels = load_data(DIRTY_SINK_DIR)

X = np.concatenate((clean_images, dirty_images))
y = np.concatenate((clean_labels, dirty_labels))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1738)


class SinkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        d = {'image': torch.tensor(self.X[idx], dtype=torch.float),
            'label': torch.tensor(self.y[idx], dtype=torch.float) }


        return d['image'].view([3,256,256]), d['label']

train_dataset = SinkDataset(X_train, y_train)
test_dataset = SinkDataset(X_test, y_test)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)