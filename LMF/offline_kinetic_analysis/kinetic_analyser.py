from motion_analyser import Motion_Analyser
from sensory_map_final import *
from utils import *
import PATH

from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib
import argparse
import cv2
import os

# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help="name of the video folder")
# args = vars(ap.parse_args())

video_root = PATH.imgpath
io_path = PATH.filepath
matplotlib.use('qt5agg')

"""
PART 1 DATA TRANSFORM
transform the dict tracking data to analysable data.
input data is a dict {[coordinate, centroid, head, tail, plate, target][...][...][...][...][...]}
the number of [] is the index of the plate.
the dict will be transformed to numpy arrays
"""

obj_path = os.path.join(io_path, 'objects.pkl')

raw = read_data(obj_path)
centroid_dict, h, t = data_trans(raw)
centroid, h_, t_, _ = data_trans_v2(raw)
framenum = len(centroid_dict[0])
platenum = len(centroid_dict)
raw_resolution = np.array([2048, 3072])
fps = int(input("frame per second = "))


"""
PART 2 INFORMATION GATHER
Point out the centre of the target and set the radius
And other information gathering
"""

# import bg and dishmap
bg_path = os.path.join(video_root, 'multi', 'bg.jpg')
map_path = os.path.join(io_path, 'map.pkl')
bg = cv2.imread(bg_path)
Map = read_data(map_path)

# interface of target localization
target_coordinate = []
t_radius = 45  # t_radius is the target radius, may changes with different videos


def draw_circle(event, x, y, flags, param):
    global target_coordinate_array, t_radius

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(bg, (x, y), t_radius, (255, 0, 0), 2)
        target_coordinate.append([x, y])


# drawing
cv2.namedWindow('circle', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('circle', draw_circle)
print('PLEASE LABEL THE TARGET COUNTERCLOCKWISE!')
while 1:
    cv2.imshow('circle', bg)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print('target coordinate have been saved')
        break
cv2.destroyAllWindows()


# clustering to identify the targets
target_coordinate_array = np.array(target_coordinate)
modified_target = target_coordinate_array.reshape((platenum, -1, 2))

Map_arr = np.array([list(Map.values())[i][0] for i in range(len(Map))])
target_coordinate_array = np.vstack((target_coordinate_array, Map_arr))
kmeans = KMeans(n_clusters=6, random_state=10)
kmeans.fit(target_coordinate_array)
labels = kmeans.labels_

bound_ = modified_target.shape[0] * modified_target.shape[1]
labels1 = labels[:bound_]
labels2 = []
for i in range(0, bound_, 3):
    labels2.append(labels1[i])

t_cod_dict = {}
targetlink = labels[bound_:]
for i in range(0, platenum):
    t_cod_dict[i] = modified_target[np.argwhere(labels2 == targetlink[i])]
t_cod = np.array(list(t_cod_dict.values())).squeeze()


"""
PART 3 TRACKING ANALYSIS
    
    Script aim to analyze the basic trjectory factors of moving animal
    and visualization

PARAMETER INFORMATION:
- traject_array means the input coordinate information, it's a 2 col numpy array.
- Number of rows is depend on the frame number.
- the coordinate information can be head, tail or centroid, depend on the efficiency.
- fps is the frame rate of the recorded video.
"""

# get kinetic params
MA = Motion_Analyser(raw, fps)
rate, vector, turn_angle_diff = MA.trajectory_analyser()
head_cast_angle = MA.headcast_detector()

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

vector_visualization(X, Y, U, V, C, st_target, t_radius, st_plate, scale=3)

# SPARSEN the data
resolution = 2 * st_plate[0]
sp_traj = sparsen(st_traj, resolution)
sp_rate = sparsen(st_traj, resolution, rate)
sp_turn_angle_diff = sparsen(st_traj, resolution, turn_angle_diff)
sp_head_cast_angle = sparsen(st_traj, resolution, head_cast_angle)

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

heatmap_visualization_v2([sp_traj_cum, sp_rate_mean, sp_turn_angle_diff_mean, sp_head_cast_angle_mean],
                         ['trajectory', 'instantaneous velocity', 'turn_angle_difference', 'head_cast_angle'],
                         st_target, t_radius, st_plate)

