### MAY NEED TO `brew reinstall openssl@1.2`
### Then run 'brew install --cask opensc
### Probably should use a venv

import time
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

from yolo import YoloModel


video = cv2.VideoCapture(0)
video2 = cv2.VideoCapture(1)
windowName = "Sink Detector"

start = time.time()
yoloModel = YoloModel()
num_caught = 0
while True:
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame, model="yolov4-tiny")
    output_image = draw_bbox(frame, bbox, label, confidence=conf)

    cv2.imshow(windowName, output_image)
    if time.time() - start > 1:
        cv2.imwrite("frame.jpg", frame)
        start = time.time()
        yoloModel.load_image("frame.jpg", -1)
        dirty = yoloModel.classify_images()[0]
        if dirty:
            print("dirty")
            ret2, frame2 = video2.read()
            cv2.imwrite("culprit/captured-" + str(num_caught) + ".jpg", frame2)
            num_caught += 1

    if cv2.waitKey(1) & 0xFF == ord("c"):
        print("capture")
        cv2.imwrite("frame.jpg", frame)

    # Closes the window if you press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
