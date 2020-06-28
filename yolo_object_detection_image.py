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

weights = os.path.join(sysPath, 'lib/yolov3-custom.weights')
config = os.path.join(sysPath, 'lib/yolov3-custom.cfg')
names = os.path.join(sysPath, 'lib/custom.names')

# ['sources/washer_1.jpeg', 'sources/washer_2.jpeg', 'sources/washer_3.jpeg', 'sources/washer_4.jpeg', 'sources/washer_5.jpeg', 'sources/washer_6.jpeg']
sources = ['sources/rs.png']

imagePath = os.path.join(sysPath, sources[0])
img = cv2.imread(imagePath)

height, width, _ = img.shape

# Crop image to get smaller region to detect from.
x = int(width * 0.48)
w = int(width * 0.2)
y = int(height * 0.45)
h = int(height * 0.45)

crop_img = img[y: y + h, x: x + w]

# send cropped to yolo
yo = yolo.Yolo(weights, config, names, x, y)
yo.detectFrom(crop_img)

# Visualize detected on source image
font = cv2.FONT_HERSHEY_PLAIN

for obj in yo.objects:
    label = str(obj.x) + ', ' + str(obj.y)
    textColor = (0, 0, 0)
    boxColor = (150, 180, 20)
    cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.width,
                                        obj.y + obj.height), boxColor, 1)
    cv2.putText(img, label, (obj.x, obj.y - 5), font, 1, textColor, 2)

cv2.imshow("Image", img)


# for (i, name) in enumerate(sources):

#     # Loading image
#     imagePath = os.path.join(sysPath, name)


#     if utils.fileExist(imagePath):
#         img = cv2.imread(imagePath)
#         yo.detectFrom(img)

#         # imageName = os.path.join(sysPath, 'result/image_' + str(int(datetime.timestamp(datetime.now()))) + '.jpg')
#         # cv2.imwrite(imageName, img)

#         cv2.imshow("Image" + str(i), img)


cv2.waitKey(0)
cv2.destroyAllWindows()
