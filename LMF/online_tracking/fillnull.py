import pandas as pd
import numpy as np
import math


def avfill(se, ce, thre, ):
    """
    Fill nulls in the tracking output according to the distance to the targets

    Args:
        se: Dataframe, data to be filled
        ce: Ndarray, coordinate of the center
        thre: float, sum of the target radii and larva length
    Return:
        Filled dataframe
    """

    # select null frames in the data
    se = se.drop(0)
    se['indic'] = 0
    for m in range(0, len(se)):
        if se['X_coord'].iloc[m] != se['X_coord'].iloc[m]:
            se['indic'].iloc[m] = 1

    # select the nearest valid frame around the null period
    Xhead = []
    Yhead = []
    Xtail = []
    Ytail = []
    for n in range(0, len(se) - 1):
        if (se['indic'].iloc[n] == 0) & (se['indic'].iloc[n + 1] == 1):
            Xhead.append((n, se['X_coord'].iloc[n]))
            Yhead.append((n, se['Y_coord'].iloc[n]))
        if (se['indic'].iloc[n] == 0) & (se['indic'].iloc[n - 1] == 1):
            Xtail.append((n, se['X_coord'].iloc[n]))
            Ytail.append((n, se['Y_coord'].iloc[n]))
    x_in = np.array(Xhead)
    x_out = np.array(Xtail)
    y_in = np.array(Yhead)
    y_out = np.array(Ytail)

    # pack head and tail coordinate
    if len(x_in) != len(x_out):
        x = np.zeros((max(len(x_in), len(x_out)), 2))
        y = np.zeros((max(len(y_in), len(y_out)), 2))
    else:
        if x_in[0, 0] >= x_out[0, 0]:
            x = np.zeros((len(x_in) + 1, 2))
            y = np.zeros((len(y_in) + 1, 2))
        else:
            x = np.zeros((len(x_in), 2))
            y = np.zeros((len(y_in), 2))

    if x_in[0, 0] >= x_out[0, 0]:
        x[0] = (x_out[0, 1], x_out[0, 1])
    for i in range(len(x_in)):
        if (x_in[i, 0] < x_out[i, 0]) & (i <= len(x_out) - 1):
            x[i] = (x_in[i, 1], x_out[i, 1])
        if (x_in[i, 0] >= x_out[i, 0]) & (i + 1 <= len(x_out) - 1):
            x[i + 1] = (x_in[i, 1], x_out[i + 1, 1])
        if i > len(x_out) - 1:
            x[i] = (x_in[i, 1], x_in[i, 1])
        if (x_in[i, 0] >= x_out[i, 0]) & (i + 1 > len(x_out) - 1):
            x[i + 1] = (x_in[i, 1], x_in[i, 1])

    if y_in[0, 0] > y_out[0, 0]:
        y[0] = (y_out[0, 1], y_out[0, 1])
    for i in range(len(y_in)):
        if (y_in[i, 0] < y_out[i, 0]) & (i <= len(y_out) - 1):
            y[i] = (y_in[i, 1], y_out[i, 1])
        if (y_in[i, 0] >= y_out[i, 0]) & (i + 1 <= len(y_out) - 1):
            y[i + 1] = (y_in[i, 1], y_out[i + 1, 1])
        if i > len(y_out) - 1:
            y[i] = (y_in[i, 1], y_in[i, 1])
        if (y_in[i, 0] >= y_out[i, 0]) & (i + 1 > len(y_out) - 1):
            y[i + 1] = (y_in[i, 1], y_in[i, 1])

    # if (len(Xhead) == len(Xtail)) & (len(Yhead) == len(Ytail)):
    #    X = list(zip(Xhead, Xtail))
    #    Y = list(zip(Yhead, Ytail))
    # end with null frame would lead to loss of the last tail frame. replace this frame with the head frame.
    # elif (len(Xhead) > len(Xtail)) & (len(Yhead) > len(Ytail)):
    #    Xtail.append(Xhead[-1])
    #    Ytail.append(Yhead[-1])
    #    X = list(zip(Xhead, Xtail))
    #    Y = list(zip(Yhead, Ytail))
    # elif (len(Xhead) < len(Xtail)) & (len(Yhead) < len(Ytail)):
    #    Xhead.append(Xtail[0])
    #    Yhead.append(Ytail[0])
    #    X = list(zip(Xhead, Xtail))
    #    Y = list(zip(Yhead, Ytail))

    # if coordinate of tail frame and head frame differes within 150 pixal(1 unit block), the filling
    # value is their mean value, if not within 150, the filling value is the head frame coordinate
    # Xfill = []; Yfill = []
    # for tupx in X:
    # if abs(tupx[0] - tupx[1]) < 150:
    #    Xfill.append(np.mean(tupx))
    # else:
    #   Xfill.append(tupx[0])
    # for tupy in Y:
    # if abs(tupy[0] - tupy[1]) < 150:
    #   Yfill.append(np.mean(tupy))
    # else:
    #   Yfill.append(tupy[0])
    # print(len(Xfill))

    # generate filling coordinate
    dist = np.zeros((len(x), 6, 3))
    xfill = np.zeros(len(x))
    yfill = np.zeros(len(y))
    cand = []
    for i in range(len(x)):
        for j in range(6):
            dist_in = math.sqrt((x[i, 0] - ce[j, 0]) ** 2 + (y[i, 0] - ce[j, 1]) ** 2)
            dist_out = math.sqrt((x[i, 1] - ce[j, 0]) ** 2 + (y[i, 1] - ce[j, 1]) ** 2)
            dist[i, j] = (j, dist_in, dist_out)
    for i in range(len(x)):
        for j in range(6):
            if (dist[i, j, 1] < thre) & (dist[i, j, 2] < thre):
                cand.append(dist[i, j, 0])
        if len(cand) > 1:
            raise Exception('threshold is too big, please adjust the threshold')
        if len(cand) == 1:
            xfill[i] = ce[int(cand[0]), 0]
            yfill[i] = ce[int(cand[0]), 1]
        if len(cand) == 0:
            xfill[i] = x[i, 0]
            yfill[i] = y[i, 0]
        cand = []
    # xfill = np.split(x, 2, axis = 1)[0].flatten()
    # yfill = np.split(y, 2, axis = 1)[0].flatten()

    # fill in the filling coordinate
    j = 0
    for i in range(0, len(se) - 1):
        if (se['indic'].iloc[i] == 0) & (se['indic'].iloc[i + 1] == 1):
            se['X_coord'].iloc[i + 1] = xfill[j]
            se['Y_coord'].iloc[i + 1] = yfill[j]
        if (se['indic'].iloc[i] == 1) & (se['indic'].iloc[i + 1] == 0):
            se['X_coord'].iloc[i] = xfill[j]
            se['Y_coord'].iloc[i] = yfill[j]
            j = j + 1
    se = se.fillna(method='ffill')
    se = se.fillna(method='bfill')
    sep = se[['X_coord', 'Y_coord']]

    return sep
