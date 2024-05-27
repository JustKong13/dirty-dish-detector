import cv2
import torch
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from PIL import Image
from dataset_generation import *
from model import ResnetArchitecture

# Model
# pickle approach
# filename = 'unbias_tuned_model_resnet.pkl'
# with open(filename, 'rb') as file:
#   resNetModel = pickle.load(file)

# resNetModel.eval()

# torch approach
filename = './models/v10_unbias_tuned_model_resnet.pt'
resNetModel = ResnetArchitecture()
resNetModel.load_state_dict(torch.load(filename))
resNetModel.eval()

model = resNetModel

def load_images(dataset, label):
    imgs = [(dataset + f, label) for f in listdir(dataset) if isfile(join(dataset, f))]
    return imgs


imgs = load_images("../../data/clean_sink/", 1)
clean_count = len(imgs)
print("total clean sink images: ", clean_count)
imgs += load_images("../../data/dirty_sink/", 0)
print("total dirty sink images: ", len(imgs) - clean_count)

print("total image count: ", len(imgs))

predicted_dirty = predicted_clean = 0
tp = fp = fn = tn = 0


def classify_image(img_path):
        global predicted_dirty, predicted_clean, tp, fp, fn, tn

        # imread returns a Numpy array with channel in dim=2
        # image = cv2.imread(img_path[0]).transpose((2, 0, 1))
        image = Image.open(img_path[0])

        # img processing
        img = image.resize((256, 256))
        img = img.convert('RGB')
        img = np.array(img) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        print("img tensor shape: ", img_tensor.shape)

        dirty = False

        # Classify image as dirty or clean
        with torch.no_grad():
            logits = model(img_tensor)
            print("model output: ", logits)
            # predicted = 1 if torch.sum(logits) > -0.15 else 0
            _, predicted = torch.max(logits, dim=1)
            dirty = (predicted == 0) # predicted is 1 if clean, 0 if dirty
            if dirty:
                predicted_dirty += 1
            else:
                predicted_clean += 1

        ##################################

        if dirty and img_path[1] == 0:
            print(img_path[0])
            tp += 1
        elif dirty and img_path[1] == 1:
            print(img_path[0])
            fp += 1
        elif not dirty and img_path[1] == 0:
            print(img_path[0])
            fn += 1
        else:
            print(img_path[0])
            tn += 1
        return dirty


for i in range(len(imgs)):
    classify_image(imgs[i])

print("Predicted Dirty: ", predicted_dirty)
print("Predicted Clean: ", predicted_clean)
print("True Positives: ", tp)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Negatives: ", tn)
