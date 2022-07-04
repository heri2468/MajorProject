import cv2
import argparse

from cv2 import CAP_DSHOW
from matplotlib.pyplot import flag
from get_background import get_background
from calendar import c
from genericpath import exists
from glob import glob
from pydoc import classname
from re import I
from time import time
from turtle import home
from matplotlib.image import imsave
import numpy as np
import os
from datetime import datetime
from RaiseTrigger import raise_alarm
from Alerts import emailme

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


cap = cv2.VideoCapture('http://192.168.124.233:4747/video')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_real7.avi', fourcc, 10.0, (800, 600))
# get the video frame height and width
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * \
    np.random.uniform(size=50)
# we will store the frames in array
frames = []
for idx in frame_indices:
    # set the frame id to read that particular frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    frames.append(frame)
    # calculate the median
median_frame = np.median(frames, axis=0).astype(np.uint8)

# convert the background model to grayscale format
background = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
frame_count = 0
consecutive_frame = 4

flag = True


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
        if classNames[classIds[i]] == 'fire':
            global flag
            if(flag == True):
                raise_alarm()
                emailme.email_alert('http://192.168.124.233:4747/video')
                flag = False


while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imwrite("fire.jpg", frame)
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    # print(layersNames)
    outputNames = [(layersNames[i - 1])
                   for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)

    if ret == True:
        frame_count += 1
        orig_frame = frame.copy()
        # IMPORTANT STEP: convert the frame to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_count % consecutive_frame == 0 or frame_count == 1:
            frame_diff_list = []
        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(gray, background)
        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=2)
        # append the final result into the `frame_diff_list`
        frame_diff_list.append(dilate_frame)
        # if we have reached `consecutive_frame` number of frames
        if len(frame_diff_list) == consecutive_frame:
            # add all the frames in the `frame_diff_list`
            sum_frames = sum(frame_diff_list)
            # find the contours around the white segmented areas
            contours, hierarchy = cv2.findContours(
                sum_frames, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours, not strictly necessary
            for i, cnt in enumerate(contours):
                cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
            for contour in contours:
                # continue through the loop if contour area is less than 500...
                # ... helps in removing noise detection
                if cv2.contourArea(contour) < 500:
                    continue
                findObjects(outputs, orig_frame)

            cv2.imshow('Detected Objects', orig_frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    else:
        break
cap.release()
cv2.destroyAllWindows()
