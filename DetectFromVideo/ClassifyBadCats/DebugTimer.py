import time

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
