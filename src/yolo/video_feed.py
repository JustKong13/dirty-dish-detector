### MAY NEED TO `brew reinstall openssl@1.2`
### Then run 'brew install --cask opensc
### Probably should use a venv

import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

video = cv2.VideoCapture(0)
windowName = "Sink Detector"

while True:
    ret, frame = video.read()

    cv2.imshow(windowName, frame)

    if cv2.waitKey(1) & 0xFF == ord("c"):
        print("capture")
        cv2.imwrite("frame.jpg", frame)

    # Closes the window if you press q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
