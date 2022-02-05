from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from motion_base import Motion_base
import numpy as np


# deprecated class
class Angle_Identifier(Motion_base):
    def __init__(self, kwargs):
        super(Angle_Identifier, self).__init__(**kwargs)

    @classmethod
    def from_raw(cls, raw, fps):
        kwargs = super(Angle_Identifier, cls).from_raw(raw, fps)
        return Angle_Identifier(kwargs)

    def __call__(self, target_coordinate, target_radius):
        # identify if toward target
        targetnum = target_coordinate.shape[1]
        r = target_radius

        d_xy = self.velocity
        centroid = self.centroid
        Nd1 = self.framenum - 1

        d_xx = d_xy[:, :Nd1, [0]]  # d_xx (6, fn, 1)
        d_yy = d_xy[:, :Nd1, [1]]
        x = centroid[:, :Nd1, [0]]  # x (6, fn, 1)
        y = centroid[:, :Nd1, [1]]
        quiver_id = np.zeros([self.platenum, Nd1, targetnum])

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
        dct = np.concatenate(Dct, axis=0)[:, :Nd1, :]
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
        label = csr_matrix((val, (row, col)), shape=(self.platenum, Nd1), dtype=np.float32).toarray()
        label[label == 0.] = np.nan

        return label
