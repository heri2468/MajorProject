from calendar import c
from genericpath import exists
from glob import glob
from pydoc import classname
from time import time
from turtle import home
import cv2
from cv2 import imshow
from matplotlib.image import imsave
import numpy as np
import os
from datetime import datetime
from RaiseTrigger import raise_alarm

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
whT = 320


confThreshold = 0.2
nmsThreshold = 0.5

classesFile = "yolo.names"

classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = "yolov4-tiny_custom.cfg"
modelWeights = "yolov4-tiny_custom_last.weights"


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CapturedImageDirectory = r'C:\Users\z004fznd\Desktop\Major\outputs'

#my_dir = Path(r"outputs")

framecaptured = 0
timetaken = 0


def findObjects(outputs, img):

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # print(indices[0])
    for i in indices:
        global framecaptured
        global timetaken
        timetaken += 10
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y),
                      (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, str(datetime.now()), (20, 40),
                    font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if timetaken == 50:
            if(classNames[classId] == 'fire'):
                os.chdir(CapturedImageDirectory)
                format = 'Frame'+str(framecaptured)+'.jpg'
                cv2.imwrite(format, img)
                framecaptured += 1
            timetaken = 0
        # if framecaptured > 10:
        #     raise_alarm()


while True:

    _, img = cap.read()
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    # print(layersNames)
    outputNames = [(layersNames[i - 1])
                   for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    cv2.imshow("original", img)
    if cv2.waitKey(1) == ord("q"):
        break
