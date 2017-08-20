import numpy as np

def distFromGrey(colourList):
    maxd = 0
    l1 = np.array([0,0,0])
    l2 = np.array([255,255,255])
    for colr in colourList:
        brightness = (colr[2] / 255.0) * 0.3 + (colr[1] / 255.0) * 0.59 + (colr[0] / 255.0) * 0.11
        d = np.linalg.norm(np.cross(l2 - l1, l1 - colr)) / np.linalg.norm(l2 - l1)
        if brightness > 0.03:
            d = d / brightness;
        print(colr,d,brightness)
        if maxd < d:
            maxd = d
    return maxd

print(distFromGrey([[0,0,0]]))
print(distFromGrey([[10,10,10]]))
print(distFromGrey([[250,250,250]]))
print(distFromGrey([[0,10,10]]))
print(distFromGrey([[0,250,10]]))
print(distFromGrey([[250,250,10]]))
