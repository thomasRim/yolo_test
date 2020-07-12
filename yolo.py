import cv2 as cv
import numpy as np
import os
import sys
import utils


class FoundObject(object):
    def __init__(self, name="", x=0, y=0, width=0, height=0, confidence=0):
        self.name = name
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence


class Yolo(object):
    def __init__(self, weightPath, configPath, names, xAdd=0, yAdd=0):

        self.objects = []
        self.confidence = 0.5
        self.blobResize = 416
        self.xAdd = xAdd
        self.yAdd = yAdd

        # Load Yolo
        self.weightPath = weightPath
        self.configPath = configPath
        if not utils.fileExist(self.weightPath):
            sys.exit(1)
        if not utils.fileExist(self.configPath):
            sys.exit(1)

        self.classes = []
        namesPath = names
        if not utils.fileExist(namesPath):
            sys.exit(1)
        with open(namesPath, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.net = cv.dnn.readNet(self.weightPath, self.configPath)
        # layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
        ]

    def detectFrom(self, img):
        self.objects = []
        height, width, _ = img.shape

        # Detecting objects
        blob = cv.dnn.blobFromImage(
            img, 1 / 255.0, (self.blobResize, self.blobResize), (0, 0, 0), True, crop=False
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                obj = FoundObject(
                    str(self.classes[class_ids[i]]),
                    x + self.xAdd,
                    y + self.yAdd,
                    w,
                    h,
                    confidences[i],
                )
                self.objects.append(obj)
