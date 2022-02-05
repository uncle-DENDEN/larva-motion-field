from utils import data_trans_v2
import numpy as np
from functools import cached_property


def k_order_difference(arr, k):
    delta_arr = None
    for k in range(k):
        arrn = np.roll(arr, -1, axis=-1)
        arr = (arrn - arr)[..., :-1]
        delta_arr = arr.copy()

    return delta_arr


class Motion_base:
    def __init__(self, centroid, head, tail, fps, ):
        # basic motion data
        self.centroid = centroid
        self.head = head
        self.tail = tail
        # properties
        self.fps = fps
        self.platenum = self.centroid.shape[0]
        self.framenum = self.centroid.shape[1]

    @classmethod
    def from_raw(cls, raw, fps):
        centroid, head, tail, _ = data_trans_v2(raw)
        kwargs = {
            'centroid': centroid,
            'head': head,
            'tail': tail,
            'fps': fps
        }
        return Motion_base(**kwargs)

    @cached_property
    def speed(self):
        x = self.centroid[:, :, 0]
        y = self.centroid[:, :, 1]
        x_p = k_order_difference(x, 1)
        y_p = k_order_difference(y, 1)
        self._speed = (x_p ** 2 + y_p ** 2) ** 0.5 / (1 / fps)
        return self._speed

    @cached_property
    def velocity(self):
        x = self.centroid[:, :, 0]
        y = self.centroid[:, :, 1]
        x_p = k_order_difference(x, 1)
        y_p = k_order_difference(y, 1)
        x_p_ = np.expand_dims(x_p, -1)
        y_p_ = np.expand_dims(y_p, -1)
        self._velocity = np.concatenate((x_p_, y_p_), axis=-1)
        return self._velocity

    @cached_property
    def delta_trajectory_angle(self):
        x = self.centroid[:, :, 0]
        y = self.centroid[:, :, 1]
        x_p = k_order_difference(x, 1)
        y_p = k_order_difference(y, 1)

        x_pn = np.roll(x_p, -1, axis=1)
        y_pn = np.roll(y_p, -1, axis=1)

        cos = (x_p * x_pn + y_p * y_pn) / ((x_p ** 2 + y_p ** 2) ** 0.5 * (x_pn ** 2 + y_pn ** 2) ** 0.5)
        turn_angle = np.arccos(cos)[..., :-1]
        turn_angle = np.nan_to_num(turn_angle, nan=0.)
        self._ang_diff = k_order_difference(turn_angle, 1)
        return self._ang_diff

    @cached_property
    def head_angle(self):
        xh = self.head[:, :, 0]
        yh = self.head[:, :, 1]
        xt = self.tail[:, :, 0]
        yt = self.tail[:, :, 1]

        t2c = [x - xt, y - yt]
        h2c = [xh - x, yh - y]

        cos = (t2c[0] * h2c[0] + t2c[1] * h2c[1]) / ((t2c[0] ** 2 + t2c[1] ** 2) ** 0.5 *
                                                     (h2c[0] ** 2 + h2c[1] ** 2) ** 0.5)
        head_cast_angle = np.arccos(cos)
        self._head_angle = np.nan_to_num(head_cast_angle, nan=0.)
        return self._head_angle

    @cached_property
    def body_angle(self):
        return

    @cached_property
    def reorientation_angle(self):
        return

    @cached_property
    def turn(self):
        return

    def bearing_angle(self, target, r):
        return

    def per_region_time(self, target):
        pass
