# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        width = int(img.shape[1] * 1)
        height = int(img.shape[0] * 1)
        img = imutils.resize(img, width=min(width, height))
        (rects, weights) = hog.detectMultiScale(img, winStride=(16, 16),
                padding=(4,4), scale=1.05)
               
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.45)

        for (x, y, w, h) in pick:
                cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 4)
        cv2.imshow("test", img)
    else:
        break
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()