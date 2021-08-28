import cv2
import numpy as np
from decimal import Decimal
from collections import OrderedDict
from scipy.spatial import distance as dist
from scipy.cluster.hierarchy import fclusterdata
from utils.interface import Interface


class MorphoTracker:
    def __init__(self, Max, length):
        self.maxDisappeared = Max
        self.length = length
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.dishes = []  # [[(centroid), r]]
        self.dishmap = OrderedDict()

    @staticmethod
    def get_centroid(contours):
        """get centroid of a contour"""

        centroids = []
        for cnt in contours:
            cnt = np.squeeze(cnt)
            area = Decimal(0.0)
            x, y = Decimal(0.0), Decimal(0.0)
            for i in range(len(cnt)):
                lng = Decimal(cnt[i][0].item())
                lat = Decimal(cnt[i][1].item())
                nextlng = Decimal(cnt[i - 1][0].item())
                nextlat = Decimal(cnt[i - 1][1].item())

                tmp_area = (nextlng * lat - nextlat * lng) / Decimal(2.0)
                area += tmp_area
                x += tmp_area * (lng + nextlng) / Decimal(3.0)
                y += tmp_area * (lat + nextlat) / Decimal(3.0)
            x = np.ceil(x / area)
            y = np.ceil(y / area)
            centroids.append((x, y))
        return np.array(centroids)

    def preprocess(self, contours, skel_edp):
        """
        take the contours and skeleton endpoint and
        return the aligned, clustered centroid and head-tail pair

        :param contours: list of contours obtained from detector
        :param skel_edp: list of skeleton endpoints from detector
        :return: aligned, clustered array of contour centroid and list of head-tail pair(list of arrays)
        """

        input_centroids = self.get_centroid(contours)
        skel_edp = np.array(skel_edp)

        # assign skel_edp to centroids according to the distance.
        edp2centroid = dist.cdist(input_centroids, skel_edp)
        corresp = edp2centroid.argsort(axis=1)[:, :2]
        head_tail = [skel_edp[corresp[i]] for i in range(corresp.shape[0])]

        # concat fragment pieces through hierarchical clustering
        if len(input_centroids) > 1:
            fcluster = fclusterdata(input_centroids, self.length, 'distance')
            cluster = np.unique(fcluster)
            input_centroids_concat = []
            head_tail_concat = []
            for cls in cluster:
                mean_cent = [input_centroids[fcluster == cls].mean(axis=0)]
                input_centroids_concat += mean_cent
                head_tail_set = np.vstack([head_tail[idx] for idx in (fcluster == cls).nonzero()[0]])
                head_tail_concat.append(head_tail_set)
            return np.array(input_centroids_concat), head_tail_concat
        else:
            return input_centroids, head_tail

    def minimum_rect_ROI(self, anchor, frame):
        """
        find the binding rectangles of contours
        and crop the image according to the binding rect

        :param frame: current frame to be cropped
        :param anchor: array: one of the centroid
        :return: mask of the bounding rect, coordinate calibration
        """

        anchor = np.int0(anchor)
        calib = anchor - self.length
        frame_cropped = frame[anchor[1] - self.length:anchor[1] + self.length,
                        anchor[0] - self.length:anchor[0] + self.length]
        return frame_cropped, calib

    def add_new_object(self, input_centroids, head_tail, idx, frame):
        """
        add new object from the aligned, concatenated input
        and specify the head and tail manually

        :param frame: current frame
        :param input_centroids: aligned, concatenated centroids
        :param head_tail: aligned, concatenated head tail pair
        :param idx: index of the input to be added into objects dict
        :return: None
        """

        # labelling head and tail manually
        out, calib = self.minimum_rect_ROI(input_centroids[idx].squeeze(), frame)
        itf = Interface(out)
        cv2.namedWindow('No.%d' % idx)
        cv2.setMouseCallback('No.%d' % idx, itf.draw_points)
        while 1:
            cv2.imshow('No.%d' % idx, itf.img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                tail, head = None, None
                print('not an object')
                break
            if k == ord("s"):
                tail = np.array(itf.points[1] + calib).reshape((1, 2))
                head = np.array(itf.points[0] + calib).reshape((1, 2))
                print('head tail anchor points have been saved')
                break
        cv2.destroyAllWindows()

        if (head is not None) & (tail is not None):
            tail_dist = dist.cdist(tail, head_tail[idx])
            head_dist = dist.cdist(head, head_tail[idx])
            head_tail[idx] = head_tail[idx][[head_dist.argmin(), tail_dist.argmin()]]

            self.objects[self.nextObjectID] = np.concatenate((input_centroids[idx].reshape(1, 2), head_tail[idx]))
            self.disappeared[self.nextObjectID] = 0
            self.nextObjectID += 1

    def update(self, contours, skel_edp, frame):
        # obtain input centroid and head-tail pair
        input_centroids, head_tail = self.preprocess(contours, skel_edp)

        # if currently there is no tracking object, take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.add_new_object(input_centroids, head_tail, i, frame)

            # get ids and centroid after initial register
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))[:, 0, :]

            # labelling the dish manually
            itf = Interface(frame)
            cv2.namedWindow('labelling the dishes area', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('labelling the dishes area', itf.draw_circle)
            while 1:
                cv2.imshow('labelling the dishes area', itf.img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("s"):
                    self.dishes = itf.circles
                    print('dish areas have been saved')
                    break
            cv2.destroyAllWindows()

            dishes_center = np.array([self.dishes[i][0] for i in range(len(self.dishes))])
            dishes_r = np.array([self.dishes[i][1] for i in range(len(self.dishes))])

            # assign dishes restraint to the object
            obj2dishes = dist.cdist(object_centroids, dishes_center)
            dish_ind = obj2dishes.argmin(axis=1)
            aligned_dish_cent = dishes_center[dish_ind]
            aligned_dish_r = dishes_r[dish_ind]
            self.dishmap = {object_ids[i]: [aligned_dish_cent[i], aligned_dish_r[i]] for i in range(len(object_ids))}

        # otherwise, there are currently tracking objects match the input centroids to existing object centroids
        else:
            # initialize the set of object IDs and corresponding centroids
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))[:, 0, :]
            dishes_centers = [list(self.dishmap.values())[i][0] for i in range(len(self.dishmap))]
            dishes_rs = [list(self.dishmap.values())[i][1] for i in range(len(self.dishmap))]

            # compute the Euclidean distance between each pair of object centroids and input centroids
            distance = dist.cdist(object_centroids, input_centroids)

            # sort the distance from small to large within the restraint of dish radii
            # rows = distance.min(axis=1).argsort()
            # cols = distance.argmin(axis=1)[rows]
            rows = []
            cols = []
            distance_sorted = distance.argsort(axis=1)
            for i in range(distance_sorted.shape[0]):
                input_sorted = input_centroids[distance_sorted[i]]
                dishes_center = dishes_centers[i].reshape((1, 2))
                cand2dish = dist.cdist(dishes_center, input_sorted)
                candidate = distance_sorted[i][(cand2dish < dishes_rs[i]).nonzero()[0]]
                if len(candidate) == 0:
                    col = distance_sorted[i][-1]
                else:
                    col = candidate[0]
                cols.append(col)
                rows.append(distance[i, col])
            rows = np.array(rows).argsort()
            cols = np.array(cols)[rows]
            combination = zip(rows, cols)

            # initialize rows and cols already examined
            examined_rows = set()
            examined_cols = set()

            # loop over the combination of distance
            for (row, col) in combination:
                # skip the examined rows and cols
                if row in examined_rows or col in examined_cols:
                    continue

                # set new centroid as object ID for the current row
                object_id = object_ids[row]

                valid = True
                if self.disappeared[object_id] > self.maxDisappeared:
                    out, _ = self.minimum_rect_ROI(input_centroids[col], frame)
                    while True:
                        cv2.imshow('newly matched', out)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("s"):
                            break
                        if k == 27:
                            valid = False
                            break
                cv2.destroyWindow('newly matched')

                if valid:
                    self.objects[object_id][0] = input_centroids[col]
                    self.disappeared[object_id] = 0

                    # get head and tail
                    head_tail_pair = head_tail[col]
                    edp2tail = dist.cdist(self.objects[object_id][2].reshape((1, 2)), head_tail_pair)
                    tail = head_tail_pair[edp2tail.argmin()]  # current tail is the closest to prev tail
                    self.objects[object_id][2] = tail
                    if len(head_tail_pair) == 2:
                        head = np.delete(head_tail_pair, edp2tail.argmin(), 0)[0]  # the one remained is the head
                    else:
                        head_tail_pair = np.delete(head_tail_pair, edp2tail.argmin(), 0)
                        edp2head = dist.cdist(self.objects[object_id][1].reshape((1, 2)), head_tail_pair)
                        head = head_tail_pair[edp2head.argmin()]
                    self.objects[object_id][1] = head

                    # update examined rows and cols
                    examined_rows.add(row)
                    examined_cols.add(col)

            # compute both the rows and cols that not examined
            unexamined_rows = set(range(0, distance.shape[0])).difference(examined_rows)
            unexamined_cols = set(range(0, distance.shape[1])).difference(examined_cols)

            # if number of current object centroids is equal or greater than the number of input centroids,
            # check whether some objects disappeared
            if distance.shape[0] >= distance.shape[1]:
                # loop over the unused row indexes
                for row in unexamined_rows:
                    # mark the object as disappeared
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

            # if number of current object centroids is smaller than the number of input centroids, register the newly
            # presented input centroid and assign a dish to it
            else:
                dishes_center = np.array([self.dishes[i][0] for i in range(len(self.dishes))])
                dishes_r = np.array([self.dishes[i][1] for i in range(len(self.dishes))])

                # register new object
                for col in unexamined_cols:
                    id1 = self.nextObjectID
                    self.add_new_object(input_centroids, head_tail, col, frame)
                    id2 = self.nextObjectID

                    # assign a dish to it if it is not an artifact
                    if id2 == id1 + 1:
                        obj2dishes = dist.cdist(input_centroids[col].reshape((1, 2)), dishes_center)
                        dish_ind = obj2dishes.argmin(axis=1)
                        self.dishmap.update({id1: [dishes_center[dish_ind][0], dishes_r[dish_ind][0]]})

        # return the set of trackable objects
        return self.objects
