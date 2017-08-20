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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=7500, help="minimum area size")
args = vars(ap.parse_args())

# Folders
sourceVideoFolder = "video"
destImageFolder = "images"

def distFromGrey(colourList):
    l1 = np.array([0,0,0])
    l2 = np.array([255,255,255])
    maxColr = [0,0,0]
    maxDist = 0
    for colr in colourList:
        d = np.linalg.norm(np.cross(l2 - l1, l1 - colr)) / np.linalg.norm(l2 - l1)
        if maxDist < d:
            maxDist = d
            maxColr = colr
    #     if d > 7:
    #         print(colr,d)
    # # for colr in colourList:
    #     hsv = cv2.cvtColor(np.uint8([[colr]]), cv2.COLOR_BGR2HSV)
    #     print(hsv)
    return maxColr

def procVideoFile(fileNameOnly,fullPathName,cat):
    global imageFileData
    camera = cv2.VideoCapture(fullPathName)
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

# Start with the set of video files and cat info
videoData = pandas.read_csv("videoData.csv")
videoData.set_index('filename',inplace=True)
print("videoData...")
print(videoData)

# Create the empty record for the image file dataset
imageFileData = pandas.DataFrame({"filename":[],"cat":[]})

# Go through each video
for index,row in videoData.iterrows():
    file = index
    print(file)
    fName = os.path.join(sourceVideoFolder, file)
    userBreak = procVideoFile(file,fName,row['cat'])
    print("imageFileData rows", len(imageFileData))
    if userBreak:
        break

imageFileData.to_csv("imageFileData.csv")
