from utils.range_filter import Rangefilter
from morphotracker import MorphoTracker
from detector import detector
from BGS import BGS
import numpy as np
import argparse
import shutil
import joblib
import time
import copy
import sys
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# configuration of BGS
period_range = (457, 461)  # estimated period of recurrent background movement
periodic_thresh = 1  # threshold for detecting a pixel as a recurrent pixel
kernel_size = (6, 6)  # kernel used for morphological opening and closing
bgs = BGS(periodrange=period_range, thresh=periodic_thresh, kernel=kernel_size)

# configuration of the target detector
beta = 1.0  # pruning accuracy tolerance, 0.0 < beta <= 1.0
size_range = (0, 1000)  # size filter

# configuration  of the morphotracker
length = 64  # length of larva in pixel, must be integer
maxDisappeared = 270  # maxDisappeared frames of a subjects. If exceeds, then the object is deregistered
warmup = 10  # start tracking after the warmup period
mt = MorphoTracker(maxDisappeared, length)
OBJ = []


# initialize the video
print("[INFO] starting the video")
# vs = cv2.VideoCapture(args["video"])
# vs_init = cv2.VideoCapture(args["video"] + "/%04d.jpg", cv2.CAP_IMAGES)
# vs = cv2.VideoCapture(args["video"] + "/%04d.jpg", cv2.CAP_IMAGES)

vs_init = cv2.VideoCapture(r'D:\workspace\python\objectTracking\sample\multi' + "/%04d.jpg", cv2.CAP_IMAGES)
vs = cv2.VideoCapture(r'D:\workspace\python\objectTracking\sample\multi' + "/%04d.jpg", cv2.CAP_IMAGES)

# allow the video file to warm up
time.sleep(1.0)

# adj threshold
bg = cv2.imread(r'D:\workspace\python\objectTracking\sample\multi' + '/bg.jpg')
f = cv2.imread(r'D:\workspace\python\objectTracking\sample\multi' + '/0001.jpg')
subtracted = cv2.absdiff(f, bg)
Rf = Rangefilter('HSV', subtracted)
v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = Rf.main()
cv2.destroyAllWindows()

# start segmentation
print('start detecting recurrent point')
time.sleep(3.0)
framecount = 1
while True:
    _, frame = vs_init.read()
    if frame is None:
        print('filtering finished')
        break

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # skip the bg
    if framecount == 1:
        framecount += 1
        continue

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    subtracted = cv2.absdiff(frame, bg)
    fgMask = cv2.inRange(subtracted, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("frame", frame)
    fgMask_ = cv2.resize(fgMask, (1280, 720))
    cv2.imshow('mask', fgMask_)

    bgs.filter(fgMask, framecount)
    framecount += 1

    # terminate if unsatisfactory
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:
        sys.exit()
cv2.destroyAllWindows()

# post-processing
bgs.get_rec_point()
for i in range(2, framecount):
    bgs.post_process(i)

    # terminate if unsatisfactory
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:
        sys.exit()
print('processed foreground masks have been saved')
cv2.destroyAllWindows()
#
# for i in range(2, 1000):
#     fgMask = np.load(r'D:\workspace\python\objectTracking\sample\cache\fgMask%d.npy' % i)
#     draw = cv2.resize(fgMask, (1280, 720))
#     cv2.rectangle(draw, (10, 50), (130, 76), (255, 255, 255), -1)
#     cv2.putText(draw, str(i), (70, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     cv2.imshow('mask', draw)
#     time.sleep(0.2)
#
#     key = cv2.waitKey(30) & 0xFF
#     if key == ord("q") or key == 27:
#         break
# cv2.destroyAllWindows()

# loop over the frames from the video
print('tracking starts')
time.sleep(2.0)
framecount = 1
while True:
    # read the next frame from the video
    _, frame = vs.read()
    if frame is None:
        break

    # skip frame1
    if framecount == 1:
        framecount += 1
        continue

    # get target contours from target detector
    framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # convert to RGB to draw contour
    contours, skel_edp = detector(framecount, size_range)
    if (len(contours) != 0) and (len(skel_edp) != 0) and (framecount > warmup):
        objects = mt.update(contours, skel_edp, framergb)
        dish_map = mt.dishmap
        obj = copy.deepcopy(objects)
        OBJ.append(obj)
    else:
        objects, dish_map = {}, {}

    # show contours
    draw = cv2.drawContours(framergb, contours, -1, (0, 255, 255), 3)  # yellow is contour

    # show centroid, head, tail and object ID
    for (objectID, pos) in objects.items():
        text = "ID {}".format(objectID)
        centroid, head, tail = np.int32(pos[0]), np.int32(pos[1]), np.int32(pos[2])
        cv2.putText(draw, text, (centroid[0] - 20, centroid[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(draw, (centroid[0], centroid[1]), 3, (255, 0, 0), -1)  # blue is centroid
        cv2.circle(draw, (head[0], head[1]), 3, (0, 255, 0), -1)  # green is head
        cv2.circle(draw, (tail[0], tail[1]), 3, (0, 0, 255), -1)  # red is tail

    # show dishes
    for (objectID, dishes) in dish_map.items():
        text = "ID {}".format(objectID)
        centre, radius = dishes[0], dishes[1]
        cv2.putText(draw, text, (centre[0] + radius//2, centre[1] + radius//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(draw, tuple(centre), radius, (255, 255, 0), 1)  # cyan is dishes

    # # show preprocessed inputs
    # for i, cent in enumerate(input_centroids):
    #     text = "{}".format(i+1)
    #     cent = tuple(np.int32(cent.squeeze()))
    #     cv2.circle(draw, cent, 3, (255, 0, 0), -1)  # blue is centre
    #     cv2.putText(draw, text, (cent[0] - 10, cent[1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #
    # for i, edp in enumerate(head_tail):
    #     text = "{}".format(i+1)
    #     for p in edp:
    #         cv2.circle(draw, tuple(p), 3, (0, 0, 255), -1)
    #     cv2.putText(draw, text, (edp[0, 0] - 10, edp[0, 1] - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    draw = cv2.resize(draw, (1280, 720))
    cv2.rectangle(draw, (10, 50), (130, 76), (255, 255, 255), -1)
    cv2.putText(draw, str(framecount), (70, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow('src', draw)

    key = cv2.waitKey(30) & 0xFF
    # early stop
    if key == ord('s'):
        break

    # stop the programme
    if key == ord("q") or key == 27:
        sys.exit()

    # count the frame
    framecount += 1

# write the result, remove cache
print('tracking finished')
time.sleep(1.0)

dish_map = mt.dishmap
joblib.dump(OBJ, 'objects.pkl')
joblib.dump(dish_map, 'map.pkl')
print('data saved')

shutil.rmtree(r'D:\workspace\python\objectTracking\sample\cache')

# cleanup
cv2.destroyAllWindows()
vs.release()
