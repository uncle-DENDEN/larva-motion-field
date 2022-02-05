from __future__ import print_function
import argparse
import cv2 as cv

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
args = parser.parse_args()

# arguments
backSub = cv.createBackgroundSubtractorMOG2(history=400, varThreshold=39, detectShadows=False)
capture = cv.VideoCapture(args.input + "/%04d.jpg", cv.CAP_IMAGES)

# bgs = BGS(periodrange=(63, 69), thresh=0)
# _, frame = capture.read()
# bgs.filter(frame, 1, False)
# print(bgs.Timer, bgs.TransitionCounter)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel1)
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel1)

    frame = cv.resize(frame, (1280, 720))
    cv.imshow('Frame', frame)

    fgMask = cv.resize(fgMask, (1280, 720))
    cv.imshow('mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
