import datetime
import json
import os
import socket
import time

import cv2
import imutils
import numpy as np
import tensorflow as tf

from Utils import DebugTimer
from ClassifyBadCats import BadCatClassifier
from InceptionV3ImageNetClassifier import Inception3ImageNetClassifier


class VideoSource():
    def __init__(self, videoSourceType, videoSourceStr):
        self.videoSourceType = videoSourceType
        self.videoSourceStr = videoSourceStr
        self.debugTimer = DebugTimer.DebugTimer(["IsColoured", "MotionDetect", "GetFrame"])
        self.boundsValid = 0
        self.boundsInvalid_NoBounds = 1
        self.boundsInvalid_TooSmall = 2
        self.boundsInvalid_NotLandscape = 3
        self.boundsInvalid_TooBig = 4
        self.boundsInvalid_NoReference = 5
        self.boundsReason = ["Ok", "None", "Small", "Shape", "Big", "NoRef"]

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
                    self.debugTimer.start(2)
                    (grabbed, frame) = camera.read()
                    self.debugTimer.end(2)
                    if not grabbed:
                        break
                    yield (frame, self.fileList[fileIdx], fileIdx, frameIdx, fileTime)
                    frameIdx += 1
        elif self.videoSourceType == 'stream':
            camera = cv2.VideoCapture(self.videoSourceStr)
            frameIdx = 0
            while True:
                self.debugTimer.start(2)
                (grabbed, frame) = camera.read()
                self.debugTimer.end(2)
                if not grabbed:
                    break
                yield (frame, "", 0, frameIdx, time.time())
                frameIdx += 1

    def findBoundingRect(self,thresholdedImg,maxBounds,paddingPercent,detectPcX1Y1X2Y2,maxPcX1Y1X2Y2):
        boundingOutline = None
        imgHeight, imgWidth = thresholdedImg.shape[:2]
        (validLeft, validTop, validRight, validBottom) = ((imgWidth * detectPcX1Y1X2Y2[0]) // 100,
                              (imgHeight * detectPcX1Y1X2Y2[1]) // 100,
                              (imgWidth * detectPcX1Y1X2Y2[2]) // 100,
                              (imgHeight * detectPcX1Y1X2Y2[3]) // 100)
        dilated = cv2.dilate(thresholdedImg, None, iterations=2)
        # Find contours
        (_, cnts, _) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # Only process contours which go close to the image centre
            (x, y, w, h) = cv2.boundingRect(c)
            newRectOutline = (x, y, x + w, y + h)
            (x1, y1, x2, y2) = newRectOutline
            if x2 < validLeft  or x1 > validRight or y2 < validTop or y1 > validBottom:
                continue
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 300:
                continue
            if boundingOutline is not None:
                bl = np.min((boundingOutline, newRectOutline),axis=0)
                tr = np.max((boundingOutline, newRectOutline),axis=0)
                newBoundingOutline = (bl[0],bl[1],tr[2],tr[3])
            else:
                newBoundingOutline = newRectOutline
            # print("C", newRectOutline, newBoundingOutline)
            boundingOutline = newBoundingOutline
        if boundingOutline is None:
            return (self.boundsInvalid_NoBounds, boundingOutline, cnts)

        # check the width of the box is greater than 20% of image
        (x1,y1,x2,y2) = boundingOutline
        boxWidth = x2-x1
        boxHeight = y2-y1
        if boxWidth < imgWidth // 5:
            return (self.boundsInvalid_TooSmall, boundingOutline, cnts)

        # check the shape of the box - looking for side views so should be square to 2x1 landscape rectangle
        if boxHeight > boxWidth or boxWidth > 2 * boxHeight:
            return (self.boundsInvalid_NotLandscape, boundingOutline, cnts)

        # check if image is too big
        (maxLeft, maxTop, maxRight, maxBottom) = ((imgWidth * maxPcX1Y1X2Y2[0]) // 100,
                              (imgHeight * maxPcX1Y1X2Y2[1]) // 100,
                              (imgWidth * maxPcX1Y1X2Y2[2]) // 100,
                              (imgHeight * maxPcX1Y1X2Y2[3]) // 100)
        if x1 < maxLeft or x2 > maxRight or y1 < maxTop or y2 > maxBottom:
            return (self.boundsInvalid_TooBig, boundingOutline, cnts)

        # Increase the bounding rect to ensure we cover the whole object
        x1 = x1 - (boxWidth * paddingPercent) // 100
        y1 = y1 - (boxHeight * paddingPercent) // 100
        x2 = x2 + (boxWidth * paddingPercent) // 100
        y2 = y2 + (boxHeight * paddingPercent) // 100
        return (self.boundsValid, (x1,y1,x2,y2), cnts)

    def motionDetectFrame(self, videoFrame):
        self.debugTimer.start(1)
        # Resize, gray and blur
        frame = imutils.resize(videoFrame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Handle reference
        if self.referenceFrame is None:
            self.referenceFrame = blurred
            self.debugTimer.end(1)
            return (self.boundsInvalid_NoReference, frame, None, None, None)
        # Calc abs difference between frame and reference
        frameDelta = cv2.absdiff(self.referenceFrame, blurred)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        self.referenceFrame = blurred
        # get contour of motion
        (validBounds, boundsCoords, contours) = self.findBoundingRect(thresh, True, 10, (10,10,90,90), (5,10,95,90))
        if validBounds != self.boundsValid:
            self.debugTimer.end(1)
            return (validBounds, frame, boundsCoords, None, contours)
        (x1,y1,x2,y2) = boundsCoords
        # Draw onto the frame
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.debugTimer.end(1)
        return (validBounds, frame, boundsCoords, None, contours)

    def isFrameColoured(self, frame):
        self.debugTimer.start(0)
        # Take a scatter of pixels across the image and see how far they are from gray
        # If not far then indicate that the image may not be coloured enough to be useful
        (w, h, _) = frame.shape
        pixelList = [frame[y * h // 10, x * w // 10] for x in range(1, 8, 2) for y in range(1, 8, 2)]
        distFromGray = self.distFromGray(pixelList)
        self.debugTimer.end(0)
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

    def getCroppedImage(self, videoFrame, resizedImage, boundingRect, reqdSizeHW):
        # Scaling factors
        h1,w1 = videoFrame.shape[:2]
        h2,w2 = resizedImage.shape[:2]
        hScale,wScale = (h1/h2,w1/w2)
        # Scale the bounding rect to the original image
        (x1, y1, x2, y2) = boundingRect
        x1,x2 = wScale * x1, wScale * x2
        y1,y2 = hScale * y1, hScale * y2
        # Reqd shape
        (hReqd,wReqd) = reqdSizeHW
        fReqd = hReqd/wReqd
        wBounds = x2-x1
        hBounds = y2-y1
        fBounds = hBounds/wBounds
        if fReqd > fBounds:
            hBoundsNew = wBounds * fReqd
            y1 -= (hBoundsNew-hBounds) / 2
            y2 += (hBoundsNew-hBounds) / 2
        else:
            wBoundsNew = hBounds / fReqd
            x1 -= (wBoundsNew-wBounds) / 2
            x2 += (wBoundsNew-wBounds) / 2
        if y1 < 0: y1 = 0
        if x1 < 0: x1 = 0
        if y2 >= h1: y2 = h1-1
        if x2 >= w1: x2 = w1-1
        # Crop the original image
        cropImg = videoFrame[int(y1):int(y2),int(x1):int(x2)]
        # Resize
        hCrop, wCrop = cropImg.shape[:2]
        if hCrop <= 0 or wCrop <= 0:
            return None
        resImg = cv2.resize(cropImg, dsize=reqdSizeHW, interpolation=cv2.INTER_CUBIC)
        return resImg

    def saveAsJpeg(self, destFolder, fileName, fileIdx, frameIdx, image):
        # Export the frame as a jpeg
        fileNameNoExt, fileExt = os.path.splitext(fileName)
        imgFileNameOnly = fileNameNoExt + "_" + "{:0>3}".format(frameIdx) + ".jpg"
        outFileName = os.path.join(destFolder,imgFileNameOnly)
        cv2.imwrite(outFileName, image)

def getConfig():
    # Config
    config = {}
    # Get base info
    try:
        with open("config.json") as jsonConfig:
            config = json.load(jsonConfig)
    except EnvironmentError:
        print("Failed to load base config")
        exit(0)
    # Get hostname
    hostname = socket.gethostname()
    configFileName = "config_" + hostname + ".json"
    mcConfig = {}
    if os.path.exists(configFileName):
        print("Getting config from", configFileName)
        try:
            with open(configFileName) as jsonConfig:
                mcConfig = json.load(jsonConfig)
        except EnvironmentError:
            print("Failed to load machine config", configFileName)
            exit(0)
    # Merge
    for k,v in mcConfig.items():
        if isinstance(v,str) and "{" in v:
            v = v.format(config["username"], config["password"])
        config[k] = v
    return config

def squirtTheBadAnimal(config):
    if config["squirtProtocol"] == "UDP":
        ip = config["squirterIP"]
        port = config["squirterPort"]
        cmd = bytes(config["squirtOnCmd"], "utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(cmd, (ip, port))
        print("UDP Sent {}:{} {}".format(ip, port, cmd))

def main(_):
    # Init
    print("Extract Video Frames")

    # Folders
    config = getConfig()
    videoSourceType = config["videoSourceType"]
    videoSourceStr = config["videoSourceStr"]
    destImageFolder = config["destFolder"]
    imageNetFolder = config["imageNetFolder"]
    showDebugImages = (config["showDebugImages"] != 0)
    imageRecogniser = config["imageRecogniser"]
    badCatClassifierFolder = config["badCatClassifierFolder"]

    # Create the image recogniser
    if imageRecogniser == "BadCat":
        imageRecogniser = BadCatClassifier.ImageRecogniser(badCatClassifierFolder, 50)
    else:
        imageRecogniser = Inception3ImageNetClassifier.ImageRecogniser(imageNetFolder)

    print("Starting image recogniser ... ", end="")
    modelLoadOk = imageRecogniser.start()
    if not modelLoadOk:
        print("Model not loaded correctly")
        exit(0)

    # Count consecutive bad cat detection events
    numConsecutiveBadCats = 0

    # Start tensorflow session
    startTime = datetime.datetime.now()
    with tf.Session() as sess:
        frameCount = 0
        # Go through each video frame
        with VideoSource(videoSourceType, videoSourceStr) as videoSource:
            for (videoFrame, fileName, fileIdx, frameIdx, frameTime) in videoSource.getNextFrame():
                frameCount += 1
                # Check if frame has insufficient colour
                # if not videoSource.isFrameColoured(videoFrame):
                #     continue
                # Detect motion
                (validBounds, motionDetectFrame, maxBoundsCoords, boundsList, contours) = videoSource.motionDetectFrame(videoFrame)

                # Detect cat
                good_bad_string = ""
                if validBounds == videoSource.boundsValid:
                    croppedImage = videoSource.getCroppedImage(videoFrame, motionDetectFrame, maxBoundsCoords, (299,299))
                    if croppedImage is None:
                        continue
                    (good_bad_string, score) = imageRecogniser.recogniseImage(sess, croppedImage, 1)

                    # Save out as an image if required
                    if destImageFolder != "":
                        fName = good_bad_string + "_" + fileName
                        imgFileName = videoSource.saveAsJpeg(destImageFolder, fName, fileIdx, frameIdx, croppedImage)

                    # Count consecutive "bad" cats
                    if "bad" in good_bad_string:
                        numConsecutiveBadCats += 1
                        # Squirt if enough consecutive images seen
                        if "consecutiveBadDetections" in config:
                            consecutiveBadDetections = config["consecutiveBadDetections"]
                            if numConsecutiveBadCats >= consecutiveBadDetections and "squirtProtocol" in config:
                                squirtTheBadAnimal(config)
                    else:
                        numConsecutiveBadCats = 0



                # Display the resulting frame if required
                if showDebugImages:
                    # print(videoSource.boundsReason[validBounds])
                    debugFrame = motionDetectFrame.copy()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(debugFrame, videoSource.boundsReason[validBounds], (220, 40), font, 1, (0, 255, 0), 2)
                    cv2.putText(debugFrame, good_bad_string, (350, 40), font, 1, (0, 255, 0) if "good" in good_bad_string else (0,0,255) , 2)
                    showBoundsList = False
                    showContours = True
                    if showContours and contours is not None:
                        for c in contours:
                            # compute the bounding box for the contour
                            (x, y, w, h) = cv2.boundingRect(c)
                            cv2.rectangle(debugFrame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    if validBounds == videoSource.boundsValid:
                        (x1,y1,x2,y2) = maxBoundsCoords
                        cv2.rectangle(debugFrame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        if showBoundsList:
                            for b in boundsList:
                                (x1, y1, x2, y2) = b
                                cv2.rectangle(debugFrame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.imshow('frame', debugFrame)
                    userKey = cv2.waitKey(50 if validBounds == videoSource.boundsValid else 1) & 0xFF
                    if userKey == ord('q'):
                        break
                    elif userKey == ord('t'):
                        if "squirtProtocol" in config:
                            squirtTheBadAnimal(config)


    # Debug info
    enddatetime = datetime.datetime.now()
    print("Start", startTime, "End", enddatetime, "Count", frameCount)
    if frameCount > 0:
        print("Elapsed", enddatetime-startTime, "Average", (enddatetime-startTime)/frameCount)
    videoSource.debugTimer.printTimings()
    imageRecogniser.debugTimer.printTimings()
    # Clean up
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run(main=main)
