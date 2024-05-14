import cv2
import torch
from os import listdir
from os.path import isfile, join

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def load_images(dataset, label):
    imgs = [(dataset + f, label) for f in listdir(dataset) if isfile(join(dataset, f))]
    return imgs


imgs = load_images("data/clean_sink/", 0)
imgs += load_images("data/dirty_sink/", 1)

# imgs = imgs[:10]
# Inference
results = model(
    [img[0] for img in imgs],
)

# Results
# results.print()
results.save()  # or .show()


interested = ["bowl", "cup", "spoon", "knife", "fork"]


tp = fp = fn = tn = 0


def check_intersecting(box1, box2):
    intersection = 0
    dx = min(box1["xmax"], box2["xmax"]) - max(box1["xmin"], box2["xmin"])
    dy = min(box1["ymax"], box2["ymax"]) - max(box1["ymin"], box2["ymin"])
    if (dx >= 0) and (dy >= 0):
        intersection = dx * dy
    area = (box2["ymax"] - box2["ymin"]) * (box2["xmax"] - box2["xmin"])
    # print(intersection, area)
    return intersection > 0.8 * area


def classify_image(img_path, df):
    global tp, fp, fn, tn
    image = cv2.imread(img_path[0])
    h, w, c = image.shape

    dirty = df["name"].isin(interested).any()

    if "sink" in df["name"].values:
        sink = df[df["name"] == "sink"].iloc[0]
        items = df[df["name"].isin(interested)]
        dirty = False
        for i in range(len(items)):
            if check_intersecting(sink, items.iloc[i]):
                dirty = True
                break

    if dirty and img_path[1] == 1:
        tp += 1
    elif dirty and img_path[1] == 0:
        print(img_path[0])
        fp += 1
    elif not dirty and img_path[1] == 1:
        print(img_path[0])
        fn += 1
    else:
        tn += 1


for i in range(len(imgs)):
    classify_image(imgs[i], results.pandas().xyxy[i])

print("True Positives: ", tp)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Negatives: ", tn)
