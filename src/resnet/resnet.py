import cv2
import torch
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from PIL import Image


# Model
class ResNetModel:
    def __init__(self):
        filename = 'tuned_model_resnet.pkl'
        with open(filename, 'rb') as file:
            resNetModel = pickle.load(file)

        resNetModel.eval()
        self.model = resNetModel
        self.images = []
        self.tp = self.fp = self.fn = self.tn = 0

    def load_image(self, img_path, label):
        self.images = [(img_path, label)]

    def load_images(self, dataset, label):
        img = [
            (dataset + f, label) for f in listdir(dataset) if isfile(join(dataset, f))
        ]
        self.images.extend(img)

    def classify_image(self, img_path):
        # imread returns a Numpy array with channel in dim=2
        # image = cv2.imread(img_path[0]).transpose((2, 0, 1))
        image = Image.open(img_path[0])

        # img processing
        img = image.resize((256, 256))
        img = img.convert('RGB')
        img = np.array(img) / 255.0
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0)

        dirty = False

        # Classify image as dirty or clean
        with torch.no_grad():
            output = self.model(img)
            _, predicted = torch.max(output, 1)
            dirty = not predicted # predicted is 1 if clean, 0 if dirty

        ##################################

        if dirty and img_path[1] == 1:
            self.tp += 1
        elif dirty and img_path[1] == 0:
            print(img_path[0])
            self.fp += 1
        elif not dirty and img_path[1] == 1:
            print(img_path[0])
            self.fn += 1
        else:
            self.tn += 1
        return dirty

    def classify_images(self):
        output = []
        for i in range(len(self.images)):
            output.append(
                self.classify_image(self.images[i])
            )
        return output

    def print_results(self):
        print("True Positives: ", self.tp)
        print("False Positives: ", self.fp)
        print("False Negatives: ", self.fn)
        print("True Negatives: ", self.tn)


def __main__():
    resNetModel = ResNetModel()
    resNetModel.load_images("data/clean_sink/", 0)
    resNetModel.load_images("data/dirty_sink/", 1)
    resNetModel.classify_images()
    resNetModel.print_results()
