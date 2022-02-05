from sklearn.cluster import KMeans
import numpy as np
import cv2
import os
from epoch_finder import Epoch_Finder
from utils import read_data
import dump_config
import matplotlib

video_root = dump_config.imgpath
io_path = dump_config.filepath
matplotlib.use('qt5agg')

obj_path = os.path.join(io_path, 'objects.pkl')
bg_path = os.path.join(video_root, 'multi', 'bg.jpg')
map_path = os.path.join(io_path, 'map.pkl')

raw = read_data(obj_path)
bg = cv2.imread(bg_path)
Map = read_data(map_path)

platenum = len(Map)
framenum = len(raw)
fps = 7
t_radius = 45  # t_radius is the target radius, may changes with different videos
margin = 35
theta = 1.047

# interface of target localization
target_coordinate = []


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

ef = Epoch_Finder.from_raw(raw, 7)
ef(t_cod, t_radius + margin, theta)
print('喳')
