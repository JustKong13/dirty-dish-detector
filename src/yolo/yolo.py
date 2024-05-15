import cv2
import torch
from os import listdir
from os.path import isfile, join


# Model
class YoloModel:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        self.images = []
        self.results = None
        self.tp = self.fp = self.fn = self.tn = 0

    def load_image(self, img_path, label):
        self.images = [(img_path, label)]

    def load_images(self, dataset, label):
        img = [
            (dataset + f, label) for f in listdir(dataset) if isfile(join(dataset, f))
        ]
        self.images.extend(img)

    def check_intersecting(self, box1, box2):
        intersection = 0
        dx = min(box1["xmax"], box2["xmax"]) - max(box1["xmin"], box2["xmin"])
        dy = min(box1["ymax"], box2["ymax"]) - max(box1["ymin"], box2["ymin"])
        if (dx >= 0) and (dy >= 0):
            intersection = dx * dy
        area = (box2["ymax"] - box2["ymin"]) * (box2["xmax"] - box2["xmin"])
        # print(intersection, area)
        return intersection > 0.8 * area

    def classify_image(self, img_path, df):
        interested = ["bowl", "cup", "spoon", "knife", "fork"]
        image = cv2.imread(img_path[0])
        h, w, c = image.shape

        dirty = df["name"].isin(interested).any()

        if "sink" in df["name"].values:
            sink = df[df["name"] == "sink"].iloc[0]
            items = df[df["name"].isin(interested)]
            dirty = False
            for i in range(len(items)):
                if self.check_intersecting(sink, items.iloc[i]):
                    dirty = True
                    break

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
        self.results = self.model([img[0] for img in self.images])
        output = []
        for i in range(len(self.images)):
            output.append(
                self.classify_image(self.images[i], self.results.pandas().xyxy[i])
            )
        return output

    def print_results(self):
        print("True Positives: ", self.tp)
        print("False Positives: ", self.fp)
        print("False Negatives: ", self.fn)
        print("True Negatives: ", self.tn)


def __main__():
    yoloModel = YoloModel()
    yoloModel.load_images("data/clean_sink/", 0)
    yoloModel.load_images("data/dirty_sink/", 1)
    yoloModel.classify_images()
    yoloModel.print_results()
