import joblib
import numpy as np
import scipy.sparse as sp


def read_data(path):
    d = joblib.load(path)
    return d


def reject_outliers(data, m=2.):
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    return data


# deprecated
def data_trans(raw_data):
    framenum = len(raw_data)
    plate_number = len(raw_data[1])
    centroid_trajectory_dict = {}
    head_trajectory_dict = {}
    tail_trajectory_dict = {}
    centroid_trajectory = np.zeros((1, 2))
    head_trajectory = np.zeros((1, 2))
    tail_trajectory = np.zeros((1, 2))
    for i in range(0, plate_number):
        for j in range(0, framenum):
            centroid_trajectory = np.concatenate((centroid_trajectory, [raw_data[j][i][0]]), axis=0)
            head_trajectory = np.concatenate((head_trajectory, [raw_data[j][i][1]]), axis=0)
            tail_trajectory = np.concatenate((tail_trajectory, [raw_data[j][i][2]]), axis=0)
        centroid_trajectory_dict[i] = centroid_trajectory
        head_trajectory_dict[i] = head_trajectory
        tail_trajectory_dict[i] = tail_trajectory
        centroid_trajectory = np.zeros((1, 2))
        head_trajectory = np.zeros((1, 2))
        tail_trajectory = np.zeros((1, 2))

    return centroid_trajectory_dict, head_trajectory_dict, tail_trajectory_dict


def data_trans_v2(raw_data):
    array_like = [list(raw_data[i].values()) for i in range(len(raw_data))]
    key = np.array(list(raw_data[0].keys()))
    obj = np.array(array_like)
    obj = np.moveaxis(obj, 0, 1)
    centroid = obj[:, :, 0, :]
    head = obj[:, :, 1, :]
    tail = obj[:, :, 2, :]

    return centroid, head, tail, key


def sparsen(idx, rsl, arr=None):
    x = idx[:, :, 0]
    y = idx[:, :, 1]

    trans = []
    for n in range(x.shape[0]):
        if arr is None:
            ones = np.ones_like(x[0])
            st_arr = sp.csr_matrix((ones, (x[n], y[n])), shape=(rsl, rsl)).toarray()
        else:
            if len(arr[n]) != len(x[n]):
                N = min(len(arr[n]), len(x[n]))
                x_ = x[n, :N]
                y_ = y[n, :N]
                arr_ = arr[n, :N]
            else:
                x_, y_, arr_ = x[n], y[n], arr[n]
            st_arr = sp.csr_matrix((arr_, (x_, y_)), shape=(rsl, rsl)).toarray()

        trans.append(np.expand_dims(st_arr, axis=0))

    trans = np.concatenate(trans, axis=0)
    return trans


# move according to target position
def crop_and_move_v2(traject_raw,
                     target_coordinate,
                     framenum,
                     actual_plate_number):
    # plate_info = [x, y, radius], [], [], []
    # seting up new coordinate
    ideal_plate_coordinate = [560, 560]
    target_num = int(input("target in 1 dish = "))
    ideal_target_coordinate = np.zeros([target_num, 2])
    traject_moved = np.zeros([actual_plate_number, framenum, 2])
    for i in range(0, actual_plate_number):
        plate_centre_x = (target_coordinate[i][0, 0, 0, 0] + target_coordinate[i][0, 0, 1, 0] +
                          target_coordinate[i][0, 0, 2, 0]) / 3
        plate_centre_y = (target_coordinate[i][0, 0, 0, 1] + target_coordinate[i][0, 0, 1, 1] +
                          target_coordinate[i][0, 0, 2, 1]) / 3
        xv = ideal_plate_coordinate[0] - plate_centre_x
        yv = ideal_plate_coordinate[1] - plate_centre_y  # the template vector of the movement
        for j in range(0, target_num):
            ideal_target_coordinate[j, 0] = target_coordinate[i][0, 0, j, 0] + xv
            ideal_target_coordinate[j, 1] = target_coordinate[i][0, 0, j, 1] + yv
        for j in range(1, framenum):
            traject_moved[i, j, 0] = traject_raw[i][j][0] + xv
            traject_moved[i, j, 1] = traject_raw[i][j][1] + yv

    return traject_moved, ideal_plate_coordinate, ideal_target_coordinate


def sumup(previous_sumed_file_path, new_cum_data):
    pf = joblib.load(previous_sumed_file_path)
    pf = new_cum_data + pf
    joblib.dump(pf, 'currentdata.pkl')
    return

