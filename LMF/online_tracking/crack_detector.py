import numpy as np
import math
from skimage import morphology
import cv2
from online_tracking.utils.image_explorer import ImgExp


class DSE:
    def __init__(self, skel, beta):
        self.skel = skel
        # three branch intersection template
        selems = list()
        selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
        selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
        selems = [np.rot90(selems[i], k=j) for i in range(4) for j in range(4)]

        # four branches intersection template
        # there will be no branches > 4, for a 1 pixel thick skeleton
        selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))
        self.selems = selems
        self.beta = beta

    @staticmethod
    def skeleton_endpoints(skel):
        # Make our input nice, possibly necessary.
        skel = np.uint8(skel.copy())

        # Apply the convolution.
        kernel = np.uint8([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]])
        src_depth = -1
        filtered = cv2.filter2D(skel, src_depth, kernel)

        # coordinate of endpoints
        cols, rows = np.where(filtered == 11)
        return list(zip(rows, cols))

    def show_endpoints(self):
        s = self.skel.copy().astype(np.uint8) * 255
        E = self.skeleton_endpoints(self.skel)
        for e in E:
            s = cv2.circle(s, e, radius=2, thickness=-1, color=128)
        wind = ImgExp('Ep', s)
        wind.main()

    def get_branches(self, skel, verbose=False):
        """
        This algorithm gets the branches of the skeleton image from end points to
        intersection point. The determined branches are used for DSE pruning method.

        Branches from one intersection point to another intersection point is not considered
        because it is definitely important to the reconstruction of the binary image.

        i.e. Only end point to intersection point is considered.

        Arguments:
        skel - Skeletonized image

        Return
        all_branches - A list of lists of coordinates (x,y) of the branches
        """

        w = skel.copy()

        # intersection template
        selems = self.selems

        # endpoints
        E = self.skeleton_endpoints(skel)
        if verbose:
            print('Endpoints', E)
        if len(E) == 0:
            return [], []

        def _eight_connected(r, c, s):
            # finding neighbors
            (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]),
                                                 np.array([r - 1, r, r + 1]))
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')
            neighbor = s[row_neigh, col_neigh]
            return neighbor

        def _intersection(neighbor):
            insec = False
            if np.sum(neighbor) > 3:
                for selem in selems:
                    if (neighbor == selem).all():
                        insec = True
                        break
            return insec

        def _is_branch(cache, s):
            groundtruth = True
            for b_p in cache:
                b_p_neigh = _eight_connected(b_p[1], b_p[0], s)
                insec = _intersection(b_p_neigh)
                if insec:
                    groundtruth = False
            return groundtruth

        # heuristic search for the branches
        intersection = []
        all_branches = []
        for e in E:
            search = [e]
            branch = [e]
            searched = []
            for p in search:
                # deal with the no-branch case
                if (p in E) and (p != e):
                    branch = []
                    # searched.append(p)
                    break
                # if p in searched:
                #     branch = []
                #     print('the branch starts from ', e, 'is deprecated because', p, 'is in searched already')
                #     break

                cache = []
                r_g, c_g = p[1], p[0]
                pix_neighbor = _eight_connected(r_g, c_g, w)
                is_intersection = _intersection(pix_neighbor)
                searched.append(p)

                if not is_intersection:
                    (r_t, c_t) = np.nonzero(pix_neighbor)
                    r_t = r_t - 1 + r_g
                    c_t = c_t - 1 + c_g
                    for point in zip(c_t, r_t):
                        if point not in searched:
                            search.append(point)
                        if point not in branch:
                            cache.append(point)

                    if _is_branch(cache, w):
                        branch = branch + cache
                        # branch = [point for sublist in branch for point in sublist]
                        # print('after flatten', branch)

                else:
                    intersection.append(p)
                    break

            if len(branch) > 0:
                if verbose:
                    print('this branch is about to append', branch, 'start from', e)
                all_branches.append(branch)

        # int_r, int_c = zip(*intersection)
        # w[int_c, int_r] = 0
        # # wind = ImgExp('gg', w)
        # # wind.main()
        # cnts = cv2.findContours(w, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # branches = imutils.grab_contours(cnts)

        # intersection2intersection or end2end branch is excluded
        # all_branches = []
        # for branch in branches:
        #     is_valid = [_ in E for _ in branch]
        #     is_valid = [i for i, x in enumerate(is_valid) if x]
        #     if len(is_valid) == 1:
        #         all_branches.append(branch)

        return intersection, all_branches

    def show_intersection(self):
        t = self.skel.copy().astype(np.uint8) * 255
        intersection, _ = self.get_branches(self.skel)
        for ins in intersection:
            t = cv2.circle(t, ins, radius=2, thickness=-1, color=128)
        wind = ImgExp('Ep', t)
        wind.main()

    def show_branch(self):
        _, all_branches = self.get_branches(self.skel.copy())
        s = self.skel.copy().astype(np.uint8) * 255
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2RGB)
        for branch in all_branches:
            branch = np.array(branch).reshape((-1, 1, 2))
            s = cv2.polylines(s, branch, True, (0, 0, 255), 1)
        wind = ImgExp('branches', s)
        wind.main()

    def get_avg_curve_len(self, skel):
        """
        This algorithm gets the averaged curve length

        Arguments:
        img - Skeletonized image

        Return
        avg_len - average length of all curves in the skeleton
        """

        all_branch_len = []
        s = skel.copy() * 1
        i = 0

        while True:

            all_branch = self.get_branches(s)

            # when no more branch detected, break from the loop
            if len(all_branch) == 0:
                break

            # calculate branch length in all_branch
            for branch in all_branch:
                all_branch_len.append(len(branch))
                r = [i[1] for i in branch]
                c = [i[0] for i in branch]

                # pruning the branches in all_branch off from the mask
                s[r, c] = 0

            #  calculate the length of remnant in the mask
            if i != 0:
                all_branch_len.append(np.sum(s))

            i += 1

        try:
            avg_len = np.mean(all_branch_len)
        except:
            return 0

        return avg_len

    @staticmethod
    def reconstruct(skel_img, dist_tr):
        """
        Attempt to reconstruct the binary image from the skeleton

        Arguments:
        img - Skeleton image using thinning algorithm
        dist_tr - Distance transform matrix

        Return:
        bn_img - Binary image
        """
        row, col = np.nonzero(skel_img)
        bn_img = skel_img.copy() * 1
        for (r, c) in zip(row, col):
            radius = math.ceil(dist_tr[r, c] - 1)
            if radius >= 1:
                stel = morphology.disk(radius)
                bn_img[r - radius:r + radius + 1, c - radius:c + radius + 1] += stel

        return bn_img >= 1

    def prune(self, verbose=False):
        """
        Discrete Skeletonization Evolution algorithm which finds the trade
        off between skeleton simplicity and reconstruction error.

        Arguments:
        img - skeletonized image
        dist - distance transform matrix

        Returns:
        pruned_img - Pruned binary image using DSE
        """
        skel = self.skel.copy().astype(np.uint8)

        # padding
        top = 1  # shape[0] = rows
        bottom = top
        left = 1  # shape[1] = cols
        right = left
        borderType = cv2.BORDER_CONSTANT
        skel = cv2.copyMakeBorder(skel, top, bottom, left, right, borderType, None, 0)

        beta = self.beta
        norm_dist = lambda s: np.log(np.sum(s) + 1)
        # norm_area = lambda s, d: (np.sum(d) - np.sum(self.reconstruct(s, dist))) / (np.sum(d))

        # initialize S_all and scores
        # S_all = [self.skel]
        # scores = [norm_area(self.skel, self.reconstruct(self.skel, dist)) + norm_dist(self.skel)]
        scores = []

        Ins, all_branches = self.get_branches(skel, verbose)
        # print('ins:', Ins)
        # print('branch:', all_branches)
        while len(Ins) > 0:
            weights = []  # Initialize the weights
            for branch in all_branches:
                """
                Iteratively removes each branch and then assigns the weight for each branch
                """
                S = skel.copy()  # S_(i)
                r = [i[1] for i in branch]
                c = [i[0] for i in branch]

                S[r, c] = 0

                ER = beta * len(self.skeleton_endpoints(S))
                LR = norm_dist(S)
                weights.append(LR - ER)

            max_idx = np.argmax(weights)
            E = all_branches[max_idx]  # The minimum branch to be removed
            r = [i[1] for i in E]  # Get the rows of minimum weight branch
            c = [i[0] for i in E]  # Get the columns of minimum weight branch
            skel[r, c] = 0  # Remove the minimum branch from the medial axis, S_(i+1)
            Ins, _ = self.get_branches(skel)
            del all_branches[max_idx]

            # show pruned skel and pruned branch in every iteration
            if verbose:
                skel_s = skel.copy().astype(np.uint8) * 255
                skel_s = cv2.cvtColor(skel_s, cv2.COLOR_GRAY2RGB)
                branch = list(zip(c, r))
                branch = np.array(branch).reshape((-1, 1, 2))
                skel_s = cv2.polylines(skel_s, branch, True, (0, 0, 255), 1)
                for ins in Ins:
                    skel_s = cv2.circle(skel_s, ins, radius=1, thickness=-1, color=(0, 255, 0))
                wind = ImgExp('branches', skel_s)
                wind.main()

        # for S in S_all:
        #     # AR = norm_area(img, self.reconstruct(S, dist))
        #     LR = norm_dist(self.get_avg_curve_len(S))
        #     # scores.append(self.beta * AR + LR)
        #     scores.append(LR)
        #     S_best = S_all[np.argmin(scores)]

        skel = skel[1: -1, 1: -1]
        return skel
