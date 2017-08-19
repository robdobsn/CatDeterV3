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
import tensorflow as tf
import re

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,model_dir,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
          label_lookup_path: string UID to integer node ID.
          uid_lookup_path: string UID to human-readable string.

        Returns:
          dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

class ImageRecogniser():
    def __init__(self, model_dir):
        # Creates graph from saved GraphDef.
        self.model_dir = model_dir
        self.create_graph()
        self.debugTimer = DebugTimer(["ConvertImage", "RecogniseImage","Lookup"])

    def create_graph(self):
        """Creates a graph from saved GraphDef file and returns a saver."""
        # Creates graph from saved graph_def.pb.
        with tf.gfile.FastGFile(os.path.join(
                self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def recogniseImage(self, sess, image, num_top_predictions):

        self.debugTimer.start(0)
        # Convert image
        img2 = cv2.resize(image, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
        # Numpy array
        np_image_data = np.asarray(img2)
        np_image_data = cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        # maybe insert float convertion here - see edit remark!
        np_final = np.expand_dims(np_image_data, axis=0)
        self.debugTimer.end(0)

        self.debugTimer.start(1)
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,{'Mul:0': np_final})
        predictions = np.squeeze(predictions)
        self.debugTimer.end(1)

        self.debugTimer.start(2)
        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup(self.model_dir)

        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        # for node_id in top_k:
        #   human_string = node_lookup.id_to_string(node_id)
        #   score = predictions[node_id]
        #   print('%s (score = %.5f)' % (human_string, score))
        node_id = top_k[0]
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        self.debugTimer.end(2)
        return (human_string, score)

class VideoSource():
    def __init__(self, videoSourceType, videoSourceStr, motionDetectType, motionDetectNumFrames):
        self.videoSourceType = videoSourceType
        self.videoSourceStr = videoSourceStr
        self.motionDetectType = motionDetectType
        self.motionDetectNumFrames = motionDetectNumFrames
        self.debugTimer = DebugTimer(["IsColoured", "MotionDetect","GetFrame"])
        self.boundsValid = 0
        self.boundsInvalid_NoBounds = 1
        self.boundsInvalid_TooSmall = 2
        self.boundsInvalid_NotLandscape = 3
        self.boundsInvalid_TooBig = 4
        self.boundsInvalid_NoReference = 5
        self.boundsReason = ["Ok", "NoBounds", "TooSmall", "NotLandscape", "TooBig","NoReference"]

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
        (_, cnts, _) = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            # Only process contours which go close to the image centre
            (x, y, w, h) = cv2.boundingRect(c)
            newRectOutline = (x, y, x + w, y + h)
            (x1, y1, x2, y2) = newRectOutline
            # print("T1", x1 < ((imgWidth * centringBorder) // 100) or x2 > (imgWidth * (100 - centringBorder)) // 100)
            # print("T2", y1 < ((imgHeight * centringBorder) // 100) or y2 > (imgHeight * (100 - centringBorder)) // 100)
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

        # return ((x1, y1, x2, y2), cnts)

        # Increase the bounding rect to ensure we cover the whole object
        x1 = x1 - (boxWidth * paddingPercent) // 100
        y1 = y1 - (boxHeight * paddingPercent) // 100
        x2 = x2 + (boxWidth * paddingPercent) // 100
        y2 = y2 + (boxHeight * paddingPercent) // 100

            # boundingRect = (x, y, w, h)
#        boundsCoords = None if boundingRect is None else (boundingRect[0], boundingRect[1], boundingRect[0]+boundingRect[2], boundingRect[1]+boundingRect[3])
        return (self.boundsValid, (x1,y1,x2,y2), cnts)

    def motionDetectFrame(self, videoFrame):
        self.debugTimer.start(1)
        # Resize, gray and blur
        frame = imutils.resize(videoFrame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Handle reference
        if self.motionDetectType == 'one-frame':
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
        elif self.motionDetectType == 'multi-frame-bb':
            if self.referenceFrame is None:
                self.referenceFrame = blurred
                self.debugTimer.end(1)
                return (frame, None, None, None)
            # Calc abs difference between frame and reference
            frameDelta = cv2.absdiff(self.referenceFrame, blurred)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow('thresh', thresh)
            # get contour of motion
            (boundingRect, boundsCoords, contours) = self.findBoundingRect(thresh)
            if boundingRect is None:
                self.debugTimer.end(1)
                return (frame, None, None, contours)
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
            self.debugTimer.end(1)
            return (frame, maxBounds, self.boundingList, contours)
        self.debugTimer.end(1)
        return (self.boundsInvalid_NoReference, frame, None, None, None)

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

    def getCroppedImage(self, videoFrame, boundingRect):
        # Crop the image using the bounding rect
        (x1, y1, x2, y2) = boundingRect
        cropImg = videoFrame[y1:y2,x1:x2]
        return cropImg

    def saveAsJpeg(self, destFolder, fileName, fileIdx, frameIdx, image):
        # Export the frame as a jpeg
        fileNameNoExt, fileExt = os.path.splitext(fileName)
        imgFileNameOnly = fileNameNoExt + "_" + "{:0>3}".format(frameIdx) + ".jpg"
        outFileName = os.path.join(destFolder,imgFileNameOnly)
        cv2.imwrite(outFileName, image)

class DebugTimer():
    def __init__(self, countLabels):
        self.counts = [0] * len(countLabels)
        self.starts = [0] * len(countLabels)
        self.times = [0] * len(countLabels)
        self.countLabels = countLabels

    def start(self, i):
        self.starts[i] = time.time()

    def end(self, i):
        self.times[i] += time.time() - self.starts[i]
        self.counts[i] += 1

    def mean(self, i):
        if self.counts[i] > 0:
            return self.times[i] / self.counts[i]
        return 0

    def getTimings(self):
        timeVals = []
        for i in range(len(self.countLabels)):
            timeVal = { "n": self.countLabels[i], "t": self.times[i], "i": self.counts[i], "m": self.mean(i)}
            timeVals.append(timeVal)
        return timeVals

    def printTimings(self):
        timings = self.getTimings()
        for tim in timings:
            print("{:20}\t{:.3f}\t{:.3f}\t{}".format(tim['n'],tim['m'],tim['t'],tim['i']))

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

def main(_):
    # Init
    print("Extract Video Frames")

    # Folders
    config = getConfig()
    videoSourceType = config["videoSourceType"]
    videoSourceStr = config["videoSourceStr"]
    destImageFolder = config["destFolder"]
    outDataFile = config["outDataFile"]
    frameDetectType = config["frameDetectType"]
    frameDetectLen = config["frameDetectLen"]
    imageNetFolder = config["imageNetFolder"]
    showDebugImages = False

    # Create the empty record for the image file dataset
    imageFileData = pandas.DataFrame({"filename":[],"cat":[]})

    # Create the image recogniser
    imageRecogniser = ImageRecogniser(imageNetFolder)

    # Start tensorflow session
    startTime = datetime.datetime.now()
    with tf.Session() as sess:
        frameCount = 0
        # Go through each video frame
        with VideoSource(videoSourceType, videoSourceStr, frameDetectType, frameDetectLen) as videoSource:
            for (videoFrame, fileName, fileIdx, frameIdx, frameTime) in videoSource.getNextFrame():
                frameCount += 1
                # Check if frame has insufficient colour
                # if not videoSource.isFrameColoured(videoFrame):
                #     continue
                # Detect motion
                (validBounds, motionDetectFrame, maxBoundsCoords, boundsList, contours) = videoSource.motionDetectFrame(videoFrame)
                # if not validBounds:
                #     continue

                # Detect cat
                if validBounds == videoSource.boundsValid:
                    croppedImage = videoSource.getCroppedImage(motionDetectFrame, maxBoundsCoords)
                    (human_string, score) = imageRecogniser.recogniseImage(sess, croppedImage, 1)
                    # if score > 0.2 and "iamese" in human_string:
                    #     print('%s (score = %.5f)' % (human_string, score))

                    # Save out as an image if required
                    if destImageFolder != "":
                        fName = "bad_" + fileName if (score > 0.2 and "iamese" in human_string) else "good_" + fileName
                        imgFileName = videoSource.saveAsJpeg(destImageFolder, fName, fileIdx, frameIdx, croppedImage)

                # Add to output records
                # if outDataFile != "":
                #     cdf = pandas.DataFrame({"filename":[fileName],"cat":[cat]})
                #     imageFileData = imageFileData.append(cdf, ignore_index=True)



                # userBreak = procVideoFrame(videoFrame)
                # print("imageFileData rows", len(imageFileData))
                # if userBreak:
                #     break
            # Display the resulting frame
                    # Show
                if showDebugImages:
                    # print(videoSource.boundsReason[validBounds])
                    debugFrame = motionDetectFrame.copy()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(debugFrame, videoSource.boundsReason[validBounds], (120, 40), font, 1, (0, 255, 0), 2)
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
                    if cv2.waitKey(50 if validBounds == videoSource.boundsValid else 50) & 0xFF == ord('q'):
                        break

    enddatetime = datetime.datetime.now()
    print("Start", startTime, "End", enddatetime, "Count", frameCount)
    if frameCount > 0:
        print("Elapsed", enddatetime-startTime, "Average", (enddatetime-startTime)/frameCount)
    videoSource.debugTimer.printTimings()
    imageRecogniser.debugTimer.printTimings()

    # imageFileData.to_csv("imageFileData.csv")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run(main=main)
