import os
import cv2
import cv2 as cv
import numpy as np
from collections import Counter
from pathlib import Path
import time

import PATH


class BGS:
    def __init__(self, periodrange, thresh=3, kernel=(4, 4)):
        self.PeriodRange = periodrange
        self.threshold = thresh
        self.kernel = kernel
        self.Timer = Counter()
        self.TransitionCounter = Counter()
        self.rows = []
        self.cols = []
        self.length = int()
        self.shape = ()
        self.path = os.path.join(PATH.imgpath, 'cache')
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def filter(self, fgMask, framecount):
        self.shape = fgMask.shape
        np.save(os.path.join(self.path, 'fgMask{}'.format(framecount)), fgMask)

        # current nonzero index
        cols, rows = np.nonzero(fgMask)
        gamma = set(zip(rows, cols))

        # Timer keys
        index = set(self.Timer.keys())

        # update Timer
        for p in (index | gamma) - (gamma & index):
            self.Timer.update([p])
        for p in gamma & index:
            if self.PeriodRange[0] < self.Timer[p] < self.PeriodRange[1]:
                self.Timer[p] = 1
                self.TransitionCounter.update([p])
            else:
                self.Timer.update([p])

        self.length = framecount
        print('%d frame have been processed' % framecount)

    def get_rec_point(self, verbose=False):
        RecPoint = list({p: c for p, c in self.TransitionCounter.items() if c > self.threshold}.keys())
        print('%d recurrent points are detected' % len(RecPoint))
        self.rows, self.cols = list(zip(*RecPoint))

        # show rec points
        present = np.zeros(self.shape)
        present[self.cols, self.rows] = 255
        while verbose:
            present = cv.resize(present, (1280, 720))
            cv.imshow('recurrent points', present)
            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

    def post_process(self, framecount):
        # get filtered fgMask
        fgMask = np.load(os.path.join(self.path, 'fgMask{}.npy'.format(framecount)))
        # fgMask_ = cv.cvtColor(fgMask_, cv.COLOR_GRAY2BGR)

        fgMask[self.cols, self.rows] = 0  # (0, 0, 0)

        # # load ROI config file and reshape the coordinate according to the points of ROI
        # model = joblib.load(r'D:\workspace\python\objectTracking\segmentation\config.pkl')
        # coordinate = np.array(model['ROI'])
        # coordinate = coordinate.reshape((-1, 2))
        #
        # # cropping according to the ROI
        # mask = np.zeros((fgMask.shape[0], fgMask.shape[1]))
        # cv2.fillConvexPoly(mask, coordinate, 1)
        # mask = mask.astype(np.bool)
        # out = np.zeros_like(fgMask)
        # out[mask] = fgMask[mask]
        # fgMask = out

        # fill up holes and fuse fragmented elements
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        # show the mask
        fgMask_show = cv2.resize(fgMask.copy(), (960, 540))
        np.save(os.path.join(self.path, 'fgMask{}'.format(framecount)), fgMask)
        cv2.imshow('Mask', fgMask_show)
