import copy
import numpy as np
import cv2
import math


class Interface:
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

