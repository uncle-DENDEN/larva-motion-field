from offline_kinetic_analysis.motion_base import Motion_base
from offline_kinetic_analysis.sensory_map_final import *
from offline_kinetic_analysis.utils import *
from online_tracking.utils.range_filter import Rangefilter
from online_tracking.utils.ROI import ROI
from online_tracking.morphotracker import MorphoTracker
from online_tracking.detector import extract_skeleton
from online_tracking.BGS import BGS
from Hparams import HParams

from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import shutil
import joblib
import time
import copy
import sys
import cv2
import os

config_path = 'config.yaml'
c = HParams.from_yaml(config_path)
# load the video before this step
video_path = os.path.join(c.root_path, c.video_name)

"""
PART 1 ONLINE TRACKING
transform the dict tracking data to analysable data.
input data is a dict {[coordinate, centroid, head, tail, plate, target][...][...][...][...][...]}
the number of [] is the index of the plate.
the dict will be transformed to numpy arrays
"""
bgs = BGS(
    periodrange=c.period_range,
    thresh=c.periodic_thresh,
    kernel=c.kernel_size,
    root_path=c.root_path
)
mt = MorphoTracker(c.maxDisappeared, c.length)
OBJ = []

# initialize the video
print("[INFO] starting the video")
# vs = cv2.VideoCapture(args["video"])
vs_init = cv2.VideoCapture(os.path.join(video_path, "%04d.jpg"), cv2.CAP_IMAGES)
vs = cv2.VideoCapture(os.path.join(video_path, "%04d.jpg"), cv2.CAP_IMAGES)
bg = cv2.imread(os.path.join(video_path, 'bg.jpg'))
f = cv2.imread(os.path.join(video_path, '0001.jpg'))
time.sleep(1.0)

# mask
sel = ROI(f, 6)
mask = sel()

# adj threshold
subtracted = cv2.absdiff(f, bg)
Rf = Rangefilter('HSV', subtracted)
v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = Rf()
cv2.destroyAllWindows()

# start online_tracking
print('start detecting recurrent point')
time.sleep(3.0)
framecount = 1
while True:
    _, frame = vs_init.read()
    if frame is None:
        print('filtering finished')
        break

    # skip the bg
    if framecount == 1:
        framecount += 1
        continue

    # bg subtraction and thresholding
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    subtracted = cv2.absdiff(frame, bg)
    fgMask = cv2.inRange(subtracted, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

    # crop by mask
    assert mask.shape == frame.shape
    fgMask[~mask] = 0

    # show mask
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("frame", frame)
    fgMask_ = cv2.resize(fgMask, (1280, 720))
    cv2.imshow('mask', fgMask_)

    # filter mask
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
    contours, skel_edp = extract_skeleton(root_path, framecount, c.size_range, c.beta)
    if (len(contours) != 0) and (len(skel_edp) != 0) and (framecount > c.warmup):
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
        cv2.putText(draw, text, (centre[0] + radius // 2, centre[1] + radius // 2),
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
joblib.dump(OBJ, os.path.join(c.out_path, 'objects.pkl'))
joblib.dump(dish_map, os.path.join(c.out_path, 'map.pkl'))
print('data saved')
shutil.rmtree(os.path.join(c.root_path, 'cache'))

# cleanup
cv2.destroyAllWindows()
vs.release()

"""
PART 2 TRACKING ANALYSIS

    Script aim to analyze the basic trajectory parameters of moving animal
    and visualization. Initialize the Motion-base class and its subclass using 
    raw data or specify the centroid, head and tail array. Then normalize all 
    positions using a reference position, sparsify data, and visualize
    

PARAMETER INFORMATION:
- raw input: [{i∈N: [*centroid, *head, *tail]} for _ in range(T)]， N is the number of plates, T is the video length
- centroid, head and tail array: (N, T, 2), N is the number of plates, T is the video length
- fps is the frame rate of the recorded video.
"""

raw = OBJ
centroid_dict, h, t = data_trans(raw)
framenum = len(centroid_dict[0])
platenum = len(centroid_dict)
fps = c.framerate
t_radius = c.target_radii
matplotlib.use(c.matplotlib_backend)

# get kinetic params
M = Motion_base.from_raw(raw, fps)
centroid = M.centroid
rate = M.speed
vector = M.velocity
turn_angle_diff = M.delta_trajectory_angle
head_angle = M.head_angle

# CROP & MOVE
st_traj, st_plate, st_target = crop_and_move_v2(centroid_dict, t_cod_dict, framenum, platenum)

# binning trajectory coordinates and vectors
centroid_binned = binning(centroid, 2, 1)
vector_binned = binning(vector, 2, 1)
st_traj_binned = binning(st_traj, 2, 1)
C = MA.angle_identifier(t_cod, t_radius, centroid_binned, vector_binned)

# vector coordinate, direction & quiver
N = min(st_traj_binned.shape[1], vector_binned.shape[1])
X = (st_traj_binned[:, :N, 0]).flatten()
Y = (st_traj_binned[:, :N, 1]).flatten()
U = (vector_binned[:, :N, 0]).flatten()
V = (vector_binned[:, :N, 1]).flatten()
C = C.flatten()
vector_visualization(X, Y, U, V, C, st_target, t_radius, st_plate, scale=7)

# SPARSEN the data
resolution = 2 * st_plate[0]
sp_traj = sparsen(st_traj, resolution)
sp_rate = sparsen(st_traj, resolution, rate)
sp_turn_angle_diff = sparsen(st_traj, resolution, turn_angle_diff)
sp_head_cast_angle = sparsen(st_traj, resolution, head_angle)
# reduce sum at axis1
sp_traj_cum = sp_traj.sum(axis=0)
sp_rate_mean = sp_rate.mean(axis=0)
sp_turn_angle_diff_mean = sp_turn_angle_diff.mean(axis=0)
sp_head_cast_angle_mean = sp_head_cast_angle.mean(axis=0)

# clear outliers (a hack)
sp_traj_cum = reject_outliers(sp_traj_cum, 5)
sp_rate_mean = reject_outliers(sp_rate_mean, 50)
sp_turn_angle_diff_mean = reject_outliers(sp_turn_angle_diff_mean, 5)
sp_head_cast_angle_mean = reject_outliers(sp_head_cast_angle_mean, 50)

# filter gaussian (a hack)
sp_traj_cum = gaussian_filter(sp_traj_cum, 2)
sp_rate_mean = gaussian_filter(sp_rate_mean, 3)
sp_turn_angle_diff_mean = gaussian_filter(sp_turn_angle_diff_mean, 2)
sp_head_cast_angle_mean = gaussian_filter(sp_head_cast_angle_mean, 7)

heatmap_visualization_v2(
    [sp_traj_cum, sp_rate_mean, sp_turn_angle_diff_mean, sp_head_cast_angle_mean],
    ['trajectory', 'instantaneous velocity', 'turn_angle_difference', 'head_angle'],
    st_target, t_radius, st_plate)
