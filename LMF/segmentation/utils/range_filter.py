import cv2
import numpy as np


class Rangefilter:

    def __init__(self, range_filter, frame, preview=True):
        self.filter = range_filter
        self.image = frame
        self.preview = preview

    def callback(self, value):
        pass

    def setup_trackbars(self):
        cv2.namedWindow("Trackbars", 0)

        for i in ["MIN", "MAX"]:
            v = 0 if i == "MIN" else 255

            for j in self.filter:
                cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, self.callback)

    def get_trackbar_values(self):
        values = []

        for i in ["MIN", "MAX"]:
            for j in self.filter:
                v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
                values.append(v)

        return values

    def main(self):

        range_filter = self.filter.upper()

        if range_filter == 'RGB':
            frame_to_thresh = self.image.copy()
        else:
            frame_to_thresh = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        self.setup_trackbars()

        while True:
            v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = self.get_trackbar_values()

            thresh = cv2.inRange(frame_to_thresh, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

            if self.preview:
                preview = cv2.bitwise_and(self.image, self.image, mask=thresh)
                previewS = cv2.resize(preview, (960, 540))
                cv2.imshow("Preview", previewS)
            else:
                cv2.imshow("Original", self.image)
                cv2.imshow("Thresh", thresh)

            if cv2.waitKey(1) & 0xFF is ord('s'):
                break

        return v1_min, v2_min, v3_min, v1_max, v2_max, v3_max
