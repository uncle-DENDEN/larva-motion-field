import copy
import numpy as np
import cv2
import math


class Interface(object):
    def __init__(self, img):
        self.img = img
        self.cache = img
        self.drawing = False  # true if mouse is pressed
        self.ix, self.iy = -1, -1
        self.circles = []
        self.points = []

    # Create a function based on a CV2 Event (Left button click)
    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            # we take note of where that mouse was located
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            radius = int(math.sqrt(((self.ix - x) ** 2) + ((self.iy - y) ** 2)))
            centre = (self.ix, self.iy)
            self.cache = copy.deepcopy(self.img)
            cv2.circle(self.img, centre, radius, (0, 0, 255), thickness=2)
            self.drawing = False
            self.circles.append([centre, radius])
            # cv2.imshow('src', self.img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.img = copy.deepcopy(self.cache)
            del self.circles[-1]

    def draw_points(self, event, x, y, flags, params):

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x, y), 7, (0, 0, 255), thickness=-1)
            self.points.append([x, y])
            # cv2.imshow('src', self.img)


class ROI(object):
    def __init__(self, img, maxp):
        if len(img.shape) == 2:
            self.img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.mask = np.zeros((img.shape[0], img.shape[1]))
        else:
            self.img = img
            self.mask = np.zeros((img.shape[0], img.shape[1], 1))
        self.img2 = deepcopy(img)
        self.point = ()
        self.lsPointsChoose = []
        self.pointsCount = 0
        self.pointsMax = maxp
        self.mask = np.zeros((img.shape[0], img.shape[1]))

    def on_mouse(self, event, x, y, flag, para):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pointsCount += 1
            print('pointsCount:', self.pointsCount)
            self.point = (x, y)
            print('your points:', self.point)
            # draw the selecting point
            cv2.circle(self.img, self.point, 5, (0, 255, 0), 2)
            # save the point into list
            self.lsPointsChoose.append([x, y])
            # when number of points reaches max, extract image
            if self.pointsCount == self.pointsMax:
                # draw ROI
                ROI_byMouse()
                self.lsPointsChoose = []
        # right click to clear track
        if event == cv2.EVENT_RBUTTONDOWN:
            print("right-mouse")
            self.img = img2
            self.pointsCount = 0
            self.lsPointsChoose = []

    def ROI_byMouse(self):
        # mask = np.ones((src.shape[0], src.shape[1]))
        pts = np.array([self.lsPointsChoose], np.int32)
        # reshape the points
        pts = pts.reshape((-1, 1, 2))
        # draw polygon
        cv2.polylines(self.img, [pts], True, (0, 255, 255))
        # fill the polygon
        self.mask = cv2.fillPoly(self.mask, [pts], 1)
        print('mask extracted')

    def __call__(self, *args, **kwargs):
        cv2.namedWindow('select ROI', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('select ROI', on_mouse)
        while True:
            cv2.imshow('src', self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                print("[INFO] ROI coordinates has been saved ")
                cv2.destroyAllWindows()
                return self.mask


