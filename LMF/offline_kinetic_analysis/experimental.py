import matplotlib.pyplot as plt
from matplotlib import cm
from mat4py import savemat
from scipy.interpolate import griddata
import numpy as np
import joblib


def reject_outliers(data, m=2.):
    data[abs(data - np.mean(data)) > m * np.std(data)] = 0
    return data


# hyperparams
resolution = 1000
interpolation = 'linear'
outlier_thresh = 10

# load value
trajectory = joblib.load('centroid_trajectory_data.pkl')
angle = joblib.load('slope_diff.pkl')
rate = joblib.load('velocity_data.pkl')

# preprocessing
trajectory = reject_outliers(trajectory, outlier_thresh)
angle = np.nan_to_num(angle, nan=0, posinf=0, neginf=0)
angle = reject_outliers(angle, outlier_thresh)
rate = np.nan_to_num(rate, nan=0, posinf=0, neginf=0)
rate = reject_outliers(rate, outlier_thresh)

# non-zero
non0_traj = trajectory.nonzero()
non0_ang = angle.nonzero()
non0_rate = rate.nonzero()

# coordinate of non-zero
xy_traj = np.vstack(non0_traj).T
xy_ang = np.vstack(non0_ang).T
xy_rate = np.vstack(non0_rate).T

# value of non-zero coordinate
val_traj = trajectory[non0_traj]
val_ang = angle[non0_ang]
val_rate = rate[non0_rate]

# state space
x = np.linspace(0., trajectory.shape[0], resolution)
y = np.linspace(0., trajectory.shape[1], resolution)
grid_x, grid_y = np.meshgrid(x, y)

# grid fill
grid_traj = griddata(xy_traj, val_traj, (grid_x, grid_y), method=interpolation)
grid_ang = griddata(xy_ang, val_ang, (grid_x, grid_y), method=interpolation)
grid_rate = griddata(xy_rate, val_rate, (grid_x, grid_y), method=interpolation)

# data_itp = {'trajectory': grid_traj.T.tolist(),
#             'angle': grid_ang.T.tolist(),
#             'rate': grid_rate.T.tolist()}
# data_raw = {'trajectory': trajectory.tolist(),
#             'angle': angle.tolist(),
#             'rate': rate.tolist()}
# savemat('interpolated.mat', data_itp)
# savemat('raw.mat', data_raw)

# plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
cm1 = ax1.imshow(grid_traj.T, extent=(0, 1, 0, 1), origin='lower', interpolation='spline16', cmap='cividis')
fig.colorbar(cm1, ax=ax1)
cm2 = ax2.imshow(grid_ang.T, extent=(0, 1, 0, 1), origin='lower', interpolation='spline16', cmap='cividis')
fig.colorbar(cm2, ax=ax2)
cm3 = ax3.imshow(grid_rate.T, extent=(0, 1, 0, 1), origin='lower', interpolation='spline16', cmap='cividis')
fig.colorbar(cm3, ax=ax3)

plt.show()
print('l')
