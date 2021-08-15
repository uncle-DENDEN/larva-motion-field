from skimage.morphology import skeletonize
from crack_detector import DSE
import numpy as np
import os
import cv2
import imutils
import joblib


def detector(framecount, size_range, beta=1.0):
    # read the mask
    cache_path = r'D:\workspace\python\objectTracking\sample\cache'
    fgMask = np.load(os.path.join(cache_path, 'fgMask%d.npy' % framecount))

    # skeletonize by thinning to infinity
    fgMask_skt_bin = fgMask.copy()
    fgMask_skt_bin[fgMask_skt_bin == 255] = 1
    fgMask_skt_bin = skeletonize(fgMask_skt_bin)

    # skeleton pruning by maximizing the simplicity
    dse = DSE(fgMask_skt_bin, beta)
    try:
        fgMask_skt_pruned_bin = dse.prune()
        fgMask_skt_pruned = fgMask_skt_pruned_bin.astype(np.uint8) * 255
        # wind = ImgExp('skeleton', fgMask_skt_pruned)
        # wind.main()

        # find contours and skeleton in the mask
        cnts = cv2.findContours(fgMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        skts = cv2.findContours(fgMask_skt_pruned.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        skts = imutils.grab_contours(skts)
        congru = len(cnts) - len(skts)
        if not congru == 0:
            print(framecount, (len(cnts), len(skts)))

        # if no more branched skeleton left, extract skeleton
        contours = []
        skel_edp = []
        E = dse.skeleton_endpoints(fgMask_skt_pruned)
        if len(E) <= 2 * len(skts):
            # size filter
            lower, upper = size_range
            skts_ = skts.copy()
            for i in range(len(cnts)):
                if lower < cnts[i].shape[0] < upper:
                    contours.append(cnts[i])
                    skts_.remove(skts[i])

            fgMask_skt_pruned_filtered = fgMask_skt_pruned_bin.copy()
            for skt in skts_:
                skt = np.squeeze(skt)
                # print(skt)
                try:
                    cols, rows = list(zip(*skt))
                except TypeError:
                    cols, rows = skt[0], skt[1]
                fgMask_skt_pruned_filtered[rows, cols] = 0

            skel_edp = dse.skeleton_endpoints(fgMask_skt_pruned_filtered)

        else:
            # cnts = []
            # skts = []
            print("incomplete pruning, frame %d skipped" % framecount)

    except ValueError as e:
        print(e, ',frame %d skipped' % framecount)
        contours = []
        skel_edp = []

    return contours, skel_edp
