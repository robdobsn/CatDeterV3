import cv2
import os

sourceFolder = 'C:/Users/rob/Dropbox (TheDobsons)/Main/RobDev/Projects/AutomationIoT/DoorCameraAndLocks/Cat Deterrent/CatDeterV3/trainImages'

fileList = os.listdir(sourceFolder)
fileList = sorted(fileList, reverse=False,
                       key=lambda fName: os.path.getctime(os.path.join(sourceFolder, fName)))

imgIdx = 0
while True:
    if imgIdx >= len(fileList):
        imgIdx = len(fileList)-1
    fName = fileList[imgIdx]
    fileName = os.path.join(sourceFolder, fName)
    fileNameNoExt, fileExt = os.path.splitext(fileName)
    fNameOnly = os.path.split(fileName)[1]
    if fileExt != ".jpg":
        imgIdx += 1
        continue
    frame = cv2.imread(fileName)
    font = cv2.FONT_HERSHEY_SIMPLEX
    curPref = fNameOnly.split("_")[0]
    cv2.putText(frame, curPref, (5, 30), font, 1, (0, 255, 0) if curPref == "good" else ((0,0,255) if curPref != "del" else (255,0,0)), 2)
    cv2.putText(frame, str(imgIdx), (155, 30), font, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    keyCode = cv2.waitKey(10000)
    if keyCode == ord('q'):
        break
    prefix = ""
    if keyCode & 0xFF == ord('g'):
        prefix = "good"
    elif keyCode & 0xFF == ord('b'):
        prefix = "bad"
    elif keyCode & 0xFF == ord('d'):
        prefix = "del"
    if prefix != "":
        if curPref == "good" or curPref == "bad" or curPref == "del":
            newFName = prefix + fNameOnly[len(curPref):]
            print(newFName)
            os.rename(fileName, os.path.join(sourceFolder,newFName))
            fileList[imgIdx] = newFName
    if keyCode & 0xFF == ord(',') or keyCode & 0xFF == ord('<') or keyCode == 0:
        imgIdx -= 1
        if imgIdx < 0:
            imgIdx = 0
    else:
        imgIdx += 1
