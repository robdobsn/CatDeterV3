# import the necessary packages
import argparse
import datetime
import time
import os
import imutils
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.stats import itemfreq
import pandas

def histogram(image, mask):
    # extract a 3D color histogram from the masked region of the
    # image, using the supplied number of bins per channel; then
    # normalize the histogram
    hist = cv2.calcHist([image], [0, 1, 2], mask, (8, 12, 3),
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist).flatten()

    # return the histogram
    return hist


def plotHist(image):
    # grab the image channels, initialize the tuple of colors,
    # the figure and the flattened feature vector
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    features = []

    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and
        # concatenate the resulting histograms for each
        # channel
        hist = cv2.calcHist([chan], [0], None, [256], [1, 256])
        features.extend(hist)

        # plot the histogram
        plt.plot(hist, color = color)
        plt.xlim([0, 256])
        print("Chan", color)

    # here we are simply showing the dimensionality of the
    # flattened color histogram 256 bins for each channel
    # x 3 channels = 768 total values -- in practice, we would
    # normally not use 256 bins for each channel, a choice
    # between 32-96 bins are normally used, but this tends
    # to be application dependent
    print ("flattened feature vector size: %d" % (np.array(features).flatten().shape))
    plt.show()


def plot3DHist(image):
    chans = cv2.split(image)
    # let's move on to 2D histograms -- I am reducing the
    # number of bins in the histogram from 256 to 32 so we
    # can better visualize the results
    fig = plt.figure()

    # plot a 2D color histogram for green and blue
    ax = fig.add_subplot(131)
    hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None,
                        [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for Green and Blue")
    plt.colorbar(p)

    # plot a 2D color histogram for green and red
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None,
                        [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for Green and Red")
    plt.colorbar(p)

    # plot a 2D color histogram for blue and red
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None,
                        [32, 32], [0, 256, 0, 256])
    p = ax.imshow(hist, interpolation="nearest")
    ax.set_title("2D Color Histogram for Blue and Red")
    plt.colorbar(p)

    # finally, let's examine the dimensionality of one of
    # the 2D histograms
    print("2D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))
    # our 2D histogram could only take into account 2 out
    # of the 3 channels in the image so now let's build a
    # 3D color histogram (utilizing all channels) with 8 bins
    # in each direction -- we can't plot the 3D histogram, but
    # the theory is exactly like that of a 2D histogram, so
    # we'll just show the shape of the histogram
    hist = cv2.calcHist([image], [0, 1, 2],
                        None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    print("3D histogram shape: %s, with %d values" % (hist.shape, hist.flatten().shape[0]))
    plt.show()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=3000, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
# 	camera = cv2.VideoCapture(0)
# 	time.sleep(0.25)
#
# # otherwise, we are reading from a video file
# else:
# 	camera = cv2.VideoCapture(args["video"])

# fName = 'video/8_2017-07-19_06-00-49.mp4'
# fName = 'video/8_2017-05-17_09-39-37.mp4'
# fName = 'video/8_2017-05-19_07-50-20.mp4'
# fName = 'video/8_2017-05-19_08-43-29.mp4'
#fName = 'video/8_2017-05-19_08-43-53.mp4'

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
    global videoData
    global categoryData
    camera = cv2.VideoCapture(fullPathName)
    broken = False

    # initialize the first frame in the video stream
    firstFrame = None

    # loop over the frames of the video
    while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        (grabbed, frame) = camera.read()
        text = "Unoccupied"
        motionDetected = False

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break

        # resize the frame, convert it to grayscale, and blur it
        #frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
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

            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            boundingRect = cv2.boundingRect(c)
            cv2.rectangle(withContours, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = "Occupied"
            motionDetected = True

        if boundingRect is None:
            continue

        # draw the text and timestamp on the frame
        cv2.putText(withContours, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(withContours, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", withContours)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)

        # Check bounding rect
        res = cv2.bitwise_and(frame, frame, mask=thresh)
#        cv2.imshow("AND", res)
        (x, y, w, h) = boundingRect
        cropImg = res[y:y+h,x:x+w]

        # Get a stripe through the bounding rect near the top
        imgStripe = cropImg[cropImg.shape[0]//4:cropImg.shape[0]//2,:]
        #print(imgStripe.shape)
        cv2.imshow("Crop", imgStripe)

        # KMeans - convert to float and reshape to linear vectors for each colour channel
        arr = np.float32(imgStripe)
        pixels = arr.reshape((-1, 3))
        n_colors = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        palette = np.uint8(centroids)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(imgStripe.shape)
        cv2.imshow("Quant", quantized)
        #dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
        # hsv = cv2.cvtColor(np.uint8([palette]), cv2.COLOR_BGR2HSV)
        # for hsvVal in hsv.reshape((-1, 3)):
        #     if (hsvVal[0] > 10 and hsvVal[0] < 30) and hsvVal[1] > 50 and hsvVal[2] > 30:
        #         print("Reddish colour found", hsvVal)
        nonGreyColr = distFromGrey(palette)
        cdf = pandas.DataFrame({"filename":[fileNameOnly],"cat":[cat],"colr":[nonGreyColr]})
        categoryData = categoryData.append(cdf, ignore_index=True)
        #print(categoryData)

        key = cv2.waitKey(100) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == 27:
            broken = True
            break
        elif key == ord(" "):
            # hist2 = histogram(frame, mask=thresh)
            # # hist = cv2.calcHist([res], [0], None, [256], [0, 256])
            # # plt.hist(res.ravel(), 256, [0, 256])
            # plt.hist(hist2)
            # plt.title('Histogram for motion')
            # plt.show()
            res = cv2.bitwise_and(frame, frame, mask=thresh)
            cv2.imshow("AND", res)
            plot3DHist(frame)
        elif ord(chr(key).lower()) >= ord('a') and ord(chr(key).lower()) <= ord("z"):
            df2 = pandas.DataFrame({'filename': [fileNameOnly], 'cat': [chr(key).lower()]})
            #videoData.set_index('filename', inplace=True)
            df2.set_index('filename',inplace=True)
            videoData = pandas.concat([videoData[~videoData.index.isin(df2.index)], df2])
            #videoData.update(df2)
            print(videoData)
        elif key != 255:
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
    return broken


# f = open("CatVideoFileInfo.txt")
# lines = f.readlines()
# for lin in lines:
#     fields = lin.split()
#     if len(fields) < 1:
#         continue
#     fName = fields[0]
#     broken = procVideoFile("video/" + fName + ".mp4")
#     if broken:
#         break

# directory = os.fsencode("video")

try:
    videoData = pandas.read_csv("videoData.csv")
except:
    videoData = pandas.DataFrame({"filename":[],"cat":[]})
videoData.set_index('filename',inplace=True)
print("videoData...")
print(videoData)

categoryData = pandas.DataFrame({"filename":[],"cat":[],"colr":[]})

INDEXING_DATA = False
directory = "video"
if INDEXING_DATA:
    for file in os.listdir(directory):
        # filename = os.fsdecode(file)
        print(file)
        if not file.endswith(".mp4") or file[0] != '8':
            continue
        fName = os.path.join(directory, file)
        broken = procVideoFile(file,fName,'')
        print(videoData)
        if broken:
            break
    videoData.to_csv("videoData.csv")
else:
    for index,row in videoData.iterrows():
        file = index
        print(file)
        fName = os.path.join(directory, file)
        broken = procVideoFile(file,fName,row['cat'])
        print("categoryData rows", len(categoryData))
        if broken:
            break

categoryData.to_csv("categoryData.csv")
