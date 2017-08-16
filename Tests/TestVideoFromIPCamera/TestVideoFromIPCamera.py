import cv2
import numpy as np
import imutils
import json

try:
    with open("config.json") as jsonConfig:
        config = json.load(jsonConfig)
except EnvironmentError:
    print("Failed to load config")
    exit(0)

cap = cv2.VideoCapture(config["videoURL"])

referenceFrame = None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    resized = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
#    thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    # if the reference frame hasn't been set ...
    if referenceFrame is None:
        referenceFrame = blurred
        continue

    # compute the absolute difference between the current frame and reference
    frameDelta = cv2.absdiff(referenceFrame, blurred)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Update reference
    referenceFrame = blurred
    
    # Display the resulting frame
    cv2.imshow('frame', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
