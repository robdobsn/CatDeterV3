# import the necessary packages
import argparse
import datetime
import time
import os
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas
import json
import queue

def procVideoFrame(videoFrame):
    userBreak = False

    # initialize the first frame in the video stream
    referenceFrame = None

    # loop over the frames of the video
    imgIdx = 0
    while True:
        # grab the current frame and process
        (grabbed, frame) = camera.read()
        motionDetected = False

        # finished?
        if not grabbed:
            break

        # handle resizing (if required), greying and blurring
        #frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the reference frame hasn't been set ...
        if referenceFrame is None:
            referenceFrame = gray
            continue

        # compute the absolute difference between the current frame and reference
        frameDelta = cv2.absdiff(referenceFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate (if required) the thresholded image to fill in holes, then find contours
        # thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        withContours = frame.copy()
        boundingRect = None
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < args["min_area"]:
                continue

            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # check the shape of the contour - looking for side views so should be a landscape rectangle
            if h > w * 1.1:
                continue

            # Increase the bounding rect to ensure we cover the whole cat
            imgHeight, imgWidth = frame.shape[:2]
            x = x - imgWidth // 16
            y = y - imgHeight // 16
            w = w + imgWidth // 8
            h = h + imgWidth // 8

            # check the resized rectangle is all within the frame
            if x < 0 or x + w > imgWidth:
                continue
            if y < 0 or y + h > imgHeight:
                continue
            boundingRect = (x, y, w, h)

            # Draw onto the frame
            cv2.rectangle(withContours, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Flag motion detected
            motionDetected = True

        # Only proceed if we've found a suitable bounding rect
        if boundingRect is None:
            continue

        # show the frame and record if the user presses a key
        cv2.imshow("Contoured", withContours)
        cv2.imshow("Threshhold", thresh)
        cv2.imshow("Delta", frameDelta)

        # Crop the image using the bounding rect
        (x, y, w, h) = boundingRect
        cropImg = frame[y:y+h,x:x+w]

        # Export the frame as a jpeg
        fileNameNoExt, fileExt = os.path.splitext(fileNameOnly)
        imgFileNameOnly = fileNameNoExt + "_" + "{:0>3}".format(imgIdx) + ".jpg"
        imgIdx += 1
        outFileName = os.path.join(destImageFolder,imgFileNameOnly)
        cv2.imwrite(outFileName, cropImg)

        # Add to output records
        cdf = pandas.DataFrame({"filename":[imgFileNameOnly],"cat":[cat]})
        imageFileData = imageFileData.append(cdf, ignore_index=True)


        # Handle user keypresses
        keyPressed = cv2.waitKey(100) & 0xFF

        # if the `q` key is pressed, break the whole program, space skips the rest of this video
        if keyPressed == 27:
            userBreak = True
            break
        elif keyPressed == ord(" "):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return userBreak

class VideoSource():
    def __init__(self, videoSourceType, videoSourceStr, motionDetectType, motionDetectNumFrames):
        self.videoSourceType = videoSourceType
        self.videoSourceStr = videoSourceStr
        self.motionDetectType = motionDetectType
        self.motionDetectNumFrames = motionDetectNumFrames

    def __enter__(self):
        # Reference queue for detecting motion
        self.referenceQueue = queue.Queue(self.motionDetectNumFrames)
        if self.videoSourceType == 'folder':
            self.fileList = os.listdir(self.videoSourceStr)
        print("Entering VideoSource")
        return self

    def getNextFrame(self):
        if self.videoSourceType == 'folder':
            for fileIdx in range(len(self.fileList)):
                fileName = os.path.join(self.videoSourceStr, self.fileList[fileIdx])
                camera = cv2.VideoCapture(fileName)
                frameIdx = 0
                # Clear reference queue
                referenceQueue = queue.Queue(self.motionDetectNumFrames)
                while True:
                    (grabbed, frame) = camera.read()
                    if not grabbed:
                        break
                    yield (frame, self.fileList[fileIdx], fileIdx, frameIdx)
                    frameIdx += 1

    def findBoundingRect(self,thresholdedImg):
        boundingRect = None
        dilated = cv2.dilate(thresholdedImg, None, iterations=2)
        (_, cnts, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 1000:
                continue

            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # check the shape of the contour - looking for side views so should be a landscape rectangle
            if h > w * 1.1:
                continue

            # Increase the bounding rect to ensure we cover the whole cat
            imgHeight, imgWidth = thresholdedImg.shape[:2]
            x = x - imgWidth // 16
            y = y - imgHeight // 16
            w = w + imgWidth // 8
            h = h + imgWidth // 8

            # check the resized rectangle is all within the frame
            if x < 0 or x + w > imgWidth:
                continue
            if y < 0 or y + h > imgHeight:
                continue
            boundingRect = (x, y, w, h)
            break

        return boundingRect

    def motionDetectFrame(self, videoFrame):
        # Resize, grat and blur
        frame = imutils.resize(videoFrame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Handle reference
        if self.motionDetectType == 'one-frame':
            self.referenceQueue.put(blurred)
            if not self.referenceQueue.full():
                return None
            # Calc abs difference between frame and reference
            frameDelta = cv2.absdiff(self.referenceQueue.get_nowait(), blurred)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # get contour of motion
            bounds = self.findBoundingRect(thresh)
            if bounds is None:
                return None
            (x,y,w,h) = bounds
            # Draw onto the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return frame
        return None


    def __exit__(self, *exc):
        print("Cleaning up VideoSource")

# Init
print("Extract Video Frames")

# Get video source and destination for stills and database
config = {}
try:
    with open("config.json") as jsonConfig:
        config = json.load(jsonConfig)
except EnvironmentError:
    print("Failed to load config")
    exit(0)

# Folders
videoSourceType = config["videoSourceType"]
videoSourceStr = config["videoSourceStr"]
destImageFolder = config["destFolder"]

# Create the empty record for the image file dataset
imageFileData = pandas.DataFrame({"filename":[],"cat":[]})

# Go through each video frame
with VideoSource(videoSourceType, videoSourceStr, 'one-frame', 5) as videoSource:
    for (videoFrame, fileName, fileIdx, frameIdx) in videoSource.getNextFrame():
        motionDetectFrame = videoSource.motionDetectFrame(videoFrame)
        if motionDetectFrame is not None:
            # print ("Frame", fileIdx, frameIdx)

        # userBreak = procVideoFrame(videoFrame)
        # print("imageFileData rows", len(imageFileData))
        # if userBreak:
        #     break
    # Display the resulting frame
            cv2.imshow('frame', motionDetectFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# imageFileData.to_csv("imageFileData.csv")

cv2.destroyAllWindows()