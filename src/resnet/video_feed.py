### MAY NEED TO `brew reinstall openssl@1.2`
### Then run 'brew install --cask opensc
### Probably should use a venv

import time
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import pickle 
import numpy as np
import torch

############# ResNet Model File Path #############
filename = 'tuned_model_resnet.pkl'

def __draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 10
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


video = cv2.VideoCapture(0)
video2 = cv2.VideoCapture(1)
windowName = "Sink Detector"

start = time.time()

################## Load ResNet Model ##################
with open(filename, 'rb') as file:
    resNetModel = pickle.load(file)

resNetModel.eval()
################## Helper Functions ##################

def load_image(img):
    img = img.resize((256, 256))
    img = img.convert('RGB')
    img = np.array(img) / 255.0
    img = img.transpose((2, 0, 1))
    return img

def classify_images(img):
    with torch.no_grad():
        output = resNetModel(img)
        _, predicted = torch.max(output, 1)
        return predicted
    
#######################################################

num_caught = 0
past_dirty = False
dirty = False

while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame, model="yolov4-tiny")
    output_image = draw_bbox(frame, bbox, label, confidence=conf)

    if time.time() - start > 1:
        cv2.imwrite("frame.jpg", frame)
        start = time.time()
        processed_img = load_image(frame) 
        dirty = not classify_images(processed_img) # model classifies clean as 1 and 0 as dirty
        if not past_dirty and dirty:
            ret2, frame2 = video2.read()
            cv2.imwrite("culprit/captured-" + str(num_caught) + ".jpg", frame2)
            num_caught += 1
        past_dirty = dirty

    if dirty:
        __draw_label(output_image, "Dirty", (20, 70), (0, 0, 200))
    else:
        __draw_label(output_image, "Clean", (20, 70), (0, 200, 0))

    cv2.imshow(windowName, output_image)

    if cv2.waitKey(1) & 0xFF == ord("c"):
        print("capture")
        cv2.imwrite("frame.jpg", frame)

    # Closes the window if you press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
