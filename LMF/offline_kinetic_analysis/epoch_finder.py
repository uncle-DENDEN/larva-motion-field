import numpy as np
from collections import defaultdict
from numpy.core.umath_tests import inner1d
from scipy.spatial.distance import cdist
from motion_base import Motion_base


def target_theta(p, v, target):
    p = np.expand_dims(p, -2)
    target = np.expand_dims(target, 1)
    ptar = (p - target)[:, :-1, :, :]
    v = np.expand_dims(v, -2)

    ptar_L2 = np.linalg.norm(ptar, axis=-1, keepdims=True)
    v_L2 = np.linalg.norm(v, axis=-1, keepdims=True)

    # unit vector
    ptar_unit = ptar / ptar_L2
    v_unit = v / v_L2

    # row-wise dot product
    cos = inner1d(ptar_unit, v_unit)

    return np.arccos(np.clip(cos, -1.0, 1.0))


def outer_diff(arr1, arr2):
    """
    arr1-arr2 in an outer product fashion, where arr1 represent column vector and arr2 represent row vector
    """
    l1 = len(arr1)
    l2 = len(arr2)
    b1 = np.ones(l2)
    b2 = np.ones(l1) * -1
    return np.outer(arr1, b1) + np.outer(b2, arr2)


def binning(array, binsize):
    bins = np.arange(0, len(array), binsize)
    x = np.arange(0, len(array))
    digitized = np.digitize(x, bins)
    return np.array([array[digitized == j, 0].mean() for j in range(1, len(bins))]).reshape((-1, 1))


def align():
    pass


class Epoch_Finder(Motion_base):
    def __init__(self, kwargs):
        super(Epoch_Finder, self).__init__(**kwargs)
        self.start_end = []
        self.epoch = []

    @classmethod
    def from_raw(cls, raw, fps):
        kwargs = super(Epoch_Finder, cls).from_raw(raw, fps)
        return Epoch_Finder(kwargs)

    def __call__(self, target, r, theta):

        targetnum = target.shape[1]
        Nd1 = self.framenum - 1

        target_angle = target_theta(self.centroid, self.velocity, target)
        Dct = []
        for d in range(self.platenum):
            dct = cdist(self.centroid[d], target[d])
            Dct.append(np.expand_dims(dct, axis=0))
        dct = np.concatenate(Dct, axis=0)[:, :Nd1, :]
        target_angle = np.moveaxis(target_angle, 2, 1)
        dct = np.moveaxis(dct, 2, 1)

        # logic filter
        target_angle[target_angle <= theta] = 2
        target_angle[target_angle > theta] = 1
        dct[dct <= r] = 1
        dct[dct > r] = 0
        sig = target_angle * dct

        # end search
        sig_minus_t = np.roll(sig, -1)
        end = np.array(list(zip(np.where((sig == 2) & (sig_minus_t == 1)))))

        # nn search of start
        target_angle_plus_t = np.roll(target_angle, 1)
        soft_start = np.array(list(zip(np.where((target_angle == 1) & (target_angle_plus_t == 2)))))

        for i in self.platenum:
            for j in targetnum:
                mask = (end[:, 0] == i) & (end[:, 1] == j)
                end_ij = end[mask, -1]
                ss_ij = soft_start[mask, -1]
                dist_ij = outer_diff(end_ij, ss_ij)
                dist_ij[dist_ij <= 0] = np.inf
                matched_start_ind = dist_ij.argmin(axis=1)
                ss_ij_sorted = ss_ij[matched_start_ind]
                ind_pairs = list(zip(end_ij, ss_ij_sorted))

                # find epoch
                epochs = []
                for pair in ind_pairs:
                    start, end = pair[0], pair[1]
                    epoch = self.centroid[i, start:end, :]
                    epochs.append(epoch)

                self.start_end[i][j] = ind_pairs
                self.epoch[i][j] = epochs

        return self.start_end, self.epoch

    def epoch_length_analysis(self, t_thresh):
        """
        :param t_thresh: > length threshold, the epoch trajectory is considered analyzable
        :return: filtered(or normalized) epoch, length of the epoch
        """

        def fil(L, thresh):
            for l in L:
                if len(l) < thresh:
                    L.remove(l)

        len_thresh = t_thresh * self.fps
        epoch_fil = [[map(lambda L: fil(L, len_thresh), subt) for subt in subp] for subp in self.epoch]
        length = [[map(lambda L: map(len, L), subt) for subt in subp] for subp in self.epoch]

        return epoch_fil, length

    def headcast_enrichment_analysis(self):
        pass
