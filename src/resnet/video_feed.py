### MAY NEED TO `brew reinstall openssl@1.2`
### Then run 'brew install --cask opensc
### Probably should use a venv

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

video = cv2.VideoCapture(1)
windowName = "Sink Detector"

while True: 
    ret, frame = video.read()
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov4-tiny')
    output_image = draw_bbox(frame, bbox, label, confidence=conf) # why isnt confidence percentage showing up ugh

    cv2.imshow(windowName, output_image)


    # Closes the window if you press q
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break


cv2.destroyAllWindows()