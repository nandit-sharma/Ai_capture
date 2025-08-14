import cv2
import numpy as np

# Note: Before running, download the model files:
# deploy.prototxt.txt from https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt
# MobileNetSSD_deploy.caffemodel from https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view
# Place them in the project directory.

class ObjectDetector:
    def __init__(self, prototxt="deploy.prototxt.txt", model="MobileNetSSD_deploy.caffemodel"):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def detect(self, frame, confidence_threshold=0.2):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        results = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if self.classes[idx] == "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(self.classes[idx], confidence * 100)
                results.append(((startX, startY, endX - startX, endY - startY), label))
        return results