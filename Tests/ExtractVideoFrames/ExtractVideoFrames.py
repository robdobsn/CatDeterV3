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
import socket

class VideoSource():
    def __init__(self, videoSourceType, videoSourceStr, motionDetectType, motionDetectNumFrames):
        self.videoSourceType = videoSourceType
        self.videoSourceStr = videoSourceStr
        self.motionDetectType = motionDetectType
        self.motionDetectNumFrames = motionDetectNumFrames

    def __enter__(self):
        # Reference for detecting motion
        self.referenceFrame = None
        self.boundingList = []
        self.boundingListPos = 0
        if self.videoSourceType == 'folder':
            fileList = os.listdir(self.videoSourceStr)
            self.fileList = sorted(fileList, reverse=True, key=lambda fName: os.path.getctime(os.path.join(self.videoSourceStr, fName)))
        print("Entering VideoSource")
        return self

    def __exit__(self, *exc):
        print("Cleaning up VideoSource")

    def getNextFrame(self):
        if self.videoSourceType == 'folder':
            for fileIdx in range(len(self.fileList)):
                fileName = os.path.join(self.videoSourceStr, self.fileList[fileIdx])
                fileTime = os.path.getmtime(fileName)
                camera = cv2.VideoCapture(fileName)
                frameIdx = 0
                # Clear reference queue
                self.boundingList = []
                self.boundingListPos = 0
                self.referenceFrame = None
                while True:
                    (grabbed, frame) = camera.read()
                    if not grabbed:
                        break
                    yield (frame, self.fileList[fileIdx], fileIdx, frameIdx, fileTime)
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
        boundsCoords = None if boundingRect is None else (boundingRect[0], boundingRect[1], boundingRect[0]+boundingRect[2], boundingRect[1]+boundingRect[3])
        return (boundingRect, boundsCoords)

    def motionDetectFrame(self, videoFrame):
        # Resize, grat and blur
        frame = imutils.resize(videoFrame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Handle reference
        if self.motionDetectType == 'one-frame':
            if self.referenceFrame is None:
                self.referenceFrame = blurred
                return (frame, None)
            # Calc abs difference between frame and reference
            frameDelta = cv2.absdiff(self.referenceFrame, blurred)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # get contour of motion
            (boundingRect, boundsCoords) = self.findBoundingRect(thresh)
            if boundingRect is None:
                return (frame, None)
            (x,y,w,h) = boundingRect
            # Draw onto the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.referenceFrame = blurred
            return (frame, boundingRect)
        elif self.motionDetectType == 'multi-frame-bb':
            if self.referenceFrame is None:
                self.referenceFrame = blurred
                return (frame, None)
            # Calc abs difference between frame and reference
            frameDelta = cv2.absdiff(self.referenceFrame, blurred)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow('thresh', thresh)
            # get contour of motion
            (boundingRect, boundsCoords) = self.findBoundingRect(thresh)
            if boundingRect is None:
                return (frame, None)
            # Add to bounds list
            if len(self.boundingList) < self.motionDetectNumFrames:
                self.boundingList.append(boundsCoords)
            else:
                self.boundingList[self.boundingListPos] = boundsCoords
            self.boundingListPos += 1
            if (self.boundingListPos >= self.motionDetectNumFrames):
                self.boundingListPos = 0
            # print(len(self.boundingList), self.boundingListPos)
            # Calculate the max bounding box from the list
            bl = np.min(self.boundingList, axis=0)
            tr = np.max(self.boundingList, axis=0)
            maxBounds = (bl[0],bl[1],tr[2]-bl[0],tr[3]-bl[1])
            (x,y,w,h) = maxBounds
            self.referenceFrame = blurred
            return (frame, maxBounds)
        return (frame, None)

    def isFrameColoured(self, frame):
        # Take a scatter of pixels across the image and see how far they are from gray
        # If not far then indicate that the image may not be coloured enough to be useful
        (w, h, _) = frame.shape
        pixelList = [frame[y * h // 10, x * w // 10] for x in range(1, 8, 2) for y in range(1, 8, 2)]
        distFromGray = self.distFromGray(pixelList)
        return distFromGray > 10

    def distFromGray(self, colourList):
        l1 = np.array([0, 0, 0])
        l2 = np.array([255, 255, 255])
        maxDist = 0
        for colr in colourList:
            if colr[0] == 0 or colr[1] == 0 or colr[2] == 0 or colr[0] == 255 or colr[1] == 255 or colr[2] == 255:
                continue
            d = np.linalg.norm(np.cross(l2 - l1, l1 - colr)) / np.linalg.norm(l2 - l1)
            if maxDist < d:
                maxDist = d
        return maxDist

    def saveAsJpeg(self, destFolder, fileName, fileIdx, frameIdx, videoFrame, boundingRect):
        # Crop the image using the bounding rect
        (x, y, w, h) = boundingRect
        cropImg = videoFrame[y:y+h,x:x+w]
        # Export the frame as a jpeg
        fileNameNoExt, fileExt = os.path.splitext(fileName)
        imgFileNameOnly = fileNameNoExt + "_" + "{:0>3}".format(frameIdx) + ".jpg"
        outFileName = os.path.join(destFolder,imgFileNameOnly)
        cv2.imwrite(outFileName, cropImg)

# Init
print("Extract Video Frames")

# Get hostname
hostname = socket.gethostname()
configFileName = "config_" + hostname + ".json"
if not os.path.exists(configFileName):
    configFileName = "config.json"
print("Getting config from", configFileName)

# Get video source and destination for stills and database
config = {}
try:
    with open(configFileName) as jsonConfig:
        config = json.load(jsonConfig)
except EnvironmentError:
    print("Failed to load config")
    exit(0)

# Folders
videoSourceType = config["videoSourceType"]
videoSourceStr = config["videoSourceStr"]
destImageFolder = config["destFolder"]
frameDetectType = config["frameDetectType"]
frameDetectLen = config["frameDetectLen"]
showDebugImages = True

# Create the empty record for the image file dataset
imageFileData = pandas.DataFrame({"filename":[],"cat":[]})

# Go through each video frame
frameCount = 0
startTime = datetime.datetime.now()
with VideoSource(videoSourceType, videoSourceStr, frameDetectType, frameDetectLen) as videoSource:
    for (videoFrame, fileName, fileIdx, frameIdx, frameTime) in videoSource.getNextFrame():
        frameCount += 1
        # Check if frame has insufficient colour
        if not videoSource.isFrameColoured(videoFrame):
            continue
        # Detect motion
        (motionDetectFrame, maxBounds) = videoSource.motionDetectFrame(videoFrame)
        if maxBounds is None:
            continue
        # Save out as an image if required
        # if destImageFolder != "":
        #     imgFileName = videoSource.saveAsJpeg(destImageFolder, fileName, fileIdx, frameIdx, motionDetectFrame, maxBounds)

        # Add to output records
        # cdf = pandas.DataFrame({"filename":[imgFileNameOnly],"cat":[cat]})
        # imageFileData = imageFileData.append(cdf, ignore_index=True)



        # userBreak = procVideoFrame(videoFrame)
        # print("imageFileData rows", len(imageFileData))
        # if userBreak:
        #     break
    # Display the resulting frame
            # Show
        if showDebugImages:
            debugFrame = motionDetectFrame.copy()
            (x,y,w,h) = maxBounds
            cv2.rectangle(debugFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('frame', debugFrame)
            if cv2.waitKey(10000) & 0xFF == ord('q'):
                break

enddatetime = datetime.datetime.now()
print("Start", startTime)
print("End", enddatetime)
print("Count", frameCount)
if frameCount > 0:
    print("Elapsed", enddatetime-startTime, "Average", (enddatetime-startTime)/frameCount)

# imageFileData.to_csv("imageFileData.csv")

cv2.destroyAllWindows()