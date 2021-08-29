from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from utils import data_trans_v2
import numpy as np


class Motion_Analyser:

    def __init__(self, raw, fps):
        self.centroid, self.head, self.tail, _ = data_trans_v2(raw)
        self.platenum = self.centroid.shape[0]
        self.framenum = self.centroid.shape[1]
        self.fps = fps

    @staticmethod
    def k_order_difference(arr, k):
        delta_arr = None
        for k in range(k):
            arrn = np.roll(arr, -1, axis=-1)
            arr = (arrn - arr)[..., :-1]
            delta_arr = arr.copy()

        return delta_arr

    # get velocity and vetcor
    def trajectory_analyser(self):

        x = self.centroid[:, :, 0]
        y = self.centroid[:, :, 1]
        x_p = self.k_order_difference(x, 1)
        y_p = self.k_order_difference(y, 1)

        velo_array = (x_p ** 2 + y_p ** 2) ** 0.5 / (1 / self.fps)
        x_p_ = np.expand_dims(x_p, -1)
        y_p_ = np.expand_dims(y_p, -1)
        vector_array = np.concatenate((x_p_, y_p_), axis=-1)

        x_pn = np.roll(x_p, -1, axis=1)
        y_pn = np.roll(y_p, -1, axis=1)

        cos = (x_p * x_pn + y_p * y_pn) / ((x_p ** 2 + y_p ** 2) ** 0.5 * (x_pn ** 2 + y_pn ** 2) ** 0.5)
        turn_angle = np.arccos(cos)[..., :-1]
        turn_angle = np.nan_to_num(turn_angle, nan=0.)
        delta_turn_angle = self.k_order_difference(turn_angle, 1)

        return velo_array, vector_array, np.abs(delta_turn_angle)

    # identify if toward target
    def angle_identifier(self, target_coordinate, target_radius, centroid=None, vector=None):
        targetnum = target_coordinate.shape[1]
        r = target_radius

        # trajectory velocity vector
        if vector is None:
            _, d_xy, _ = self.trajectory_analyser()
        else:
            d_xy = vector

        # centroid position
        if centroid is None:
            centroid = self.centroid
        else:
            centroid = centroid

        N = min(d_xy.shape[1], centroid.shape[1])
        d_xx = d_xy[:, :N, [0]]  # d_xx (6, fn, 1)
        d_yy = d_xy[:, :N, [1]]
        x = centroid[:, :N, [0]]  # x (6, fn, 1)
        y = centroid[:, :N, [1]]
        quiver_id = np.zeros([self.platenum, N, targetnum])

        xt = np.expand_dims(target_coordinate[:, :, 0], -2)  # target coordinate (6, 3, 2) -> (6, 1, 3)
        yt = np.expand_dims(target_coordinate[:, :, 1], -2)

        # compute angle between 2 tangents
        distl = ((xt - x) ** 2 + (yt - y) ** 2) ** 0.5  # centroid to target centre
        angle = np.arcsin(r / distl)
        tan2tan = np.nan_to_num(2 * angle, nan=np.inf)

        # rotation
        xu = (xt - x) / distl  # unit vector from centroid to target centre
        yu = (yt - y) / distl
        distq = (distl ** 2 - r ** 2) ** 0.5  # centroid to tangent point
        xq1 = xu * np.cos(angle) - yu * np.sin(angle)  # rotating
        yq1 = xu * np.sin(angle) + yu * np.cos(angle)
        xq2 = xu * np.cos(-angle) - yu * np.sin(-angle)
        yq2 = xu * np.sin(-angle) + yu * np.cos(-angle)

        # find the tangent points
        xq1 = xq1 * distq + x
        yq1 = yq1 * distq + y
        xq2 = xq2 * distq + x
        yq2 = yq2 * distq + y

        # find the tangent vector
        dxq1x = (xq1 - x)
        dyq1y = (yq1 - y)
        dxq2x = (xq2 - x)
        dyq2y = (yq2 - y)

        # find the angle between one of the tangent line and the trajectory vector
        cos1 = (d_xx * dxq1x + d_yy * dyq1y) / ((d_xx ** 2 + d_yy ** 2) ** 0.5 * (dxq1x ** 2 + dyq1y ** 2) ** 0.5)
        cos2 = (d_xx * dxq2x + d_yy * dyq2y) / ((d_xx ** 2 + d_yy ** 2) ** 0.5 * (dxq2x ** 2 + dyq2y ** 2) ** 0.5)
        traj2tan1 = np.arccos(cos1)
        traj2tan2 = np.arccos(cos2)

        # logic filtering by angles
        quiver_id[(traj2tan1 < tan2tan) & (traj2tan2 < tan2tan)] = 1

        # one hot filter of minimum distance
        Dct = []
        for d in range(centroid.shape[0]):
            dct = cdist(centroid[d], target_coordinate[d])
            Dct.append(np.expand_dims(dct, axis=0))
        dct = np.concatenate(Dct, axis=0)[:, :N, :]
        dctmin = np.expand_dims(dct.argmin(axis=-1), -1)
        onehotmin = np.zeros_like(dct)
        np.put_along_axis(onehotmin, dctmin, 1, axis=-1)

        # logic filtering
        quiver_id += onehotmin * quiver_id
        quiver_id /= quiver_id.max(axis=-1, keepdims=True)  # normalize by the maximum
        quiver_id = np.nan_to_num(quiver_id, nan=0.)

        # convert to categorical
        row = np.where(quiver_id == 1)[0]
        col = np.where(quiver_id == 1)[1]
        val = np.where(quiver_id == 1)[2] + 1
        label = csr_matrix((val, (row, col)), shape=(self.platenum, N), dtype=np.float32).toarray()
        label[label == 0.] = np.nan

        return label

    # calculate head cast angle according to head tail and centroid
    def headcast_detector(self):

        x = self.centroid[:, :, 0]
        y = self.centroid[:, :, 1]
        xh = self.head[:, :, 0]
        yh = self.head[:, :, 1]
        xt = self.tail[:, :, 0]
        yt = self.tail[:, :, 1]

        t2c = [x - xt, y - yt]
        h2c = [xh - x, yh - y]

        cos = (t2c[0] * h2c[0] + t2c[1] * h2c[1]) / ((t2c[0] ** 2 + t2c[1] ** 2) ** 0.5 *
                                                     (h2c[0] ** 2 + h2c[1] ** 2) ** 0.5)
        head_cast_angle = np.arccos(cos)
        head_cast_angle = np.nan_to_num(head_cast_angle, nan=0.)

        return head_cast_angle
