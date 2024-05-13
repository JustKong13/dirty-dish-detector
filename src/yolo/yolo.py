import torch
from os import listdir
from os.path import isfile, join

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Images
imgs = [f for f in listdir("data/clean_sink") if isfile(join("data/clean_sink", f))]

# # Inference
# results = model(imgs)

# # Results
# results.print()
# results.save()  # or .show()
# print(results.pandas().xyxy[0])
