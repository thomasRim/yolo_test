import cv2
import numpy as np
import os
import sys
import yolo
import utils
from datetime import datetime

# Load Yolo
# Net
sysPath = os.path.dirname(os.path.abspath(__file__))

weights = os.path.join(sysPath, "lib/yolov3-custom.weights")
config = os.path.join(sysPath, "lib/yolov3-custom.cfg")
names = os.path.join(sysPath, "lib/custom.names")

sources = ["sources/spring.jpeg"]

imagePath = os.path.join(sysPath, sources[0])
img = cv2.imread(imagePath)

height, width, _ = img.shape
pre_scale = 0.5
post_scale = 0.25

# Crop image to get smaller region to detect from.
x, y, w, h = (
    0,
    0,
    width,
    height,
)  # (int(width * 0.48), int(height * 0.45), int(width * 0.2), int(height * 0.45))

img = img[y : y + h, x : x + w]

img = cv2.resize(img, (int(pre_scale * width), int(pre_scale * height)))

# send to yolo
yo = yolo.Yolo(weights, config, names, x, y)
yo.confidence = 0.2
yo.detectFrom(img)

# Visualize detected on source image
font = cv2.FONT_HERSHEY_PLAIN

for obj in yo.objects:
    label = str(obj.x) + ", " + str(obj.y)
    textColor = (0, 0, 0)
    boxColor = (150, 180, 20)
    cv2.rectangle(
        img, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), boxColor, 1
    )
    cv2.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)


img = cv2.resize(img, (int(post_scale * width), int(post_scale * height)))

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
