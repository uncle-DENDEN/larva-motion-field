import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors


# heatmap with text
def heatmap_visualization(resolution_info, transformed_centroid_array):
    fig, ax = plt.subplots()
    # transformed_velocity_array,
    # transformed_vector_array, transformed_immobdis_array, transformed_immobv_array,
    # transformed_slope_array, transformed_twtarget_array, transformed_turnangle_array
    ax.imshow(transformed_centroid_array)
    ax.set_xticks([0, 1120])
    # 设置x轴刻度间隔
    ax.set_yticks([0, 1120])
    # 设置y轴刻度间隔
    ax.set_xticklabels("xaxis(pixel)")
    # 设置x轴标签'''
    ax.set_yticklabels("yaxis(pixel)")
    # 设置y轴标签'''
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # 设置标签 旋转45度 ha有三个选择：right,center,left（对齐方式）

    # warning: make sure that the resolution is a sqaure
    for i in range(resolution_info[0]):
        for j in range(resolution_info[1]):
            ax.text(j, i, transformed_centroid_array[i, j],
                    ha="center", va="center", color="w")

    '''
    画图
    j,i:表示坐标值上的值
    harvest[i, j]表示内容
    ha有三个选择：right,center,left（对齐方式）
    va有四个选择：'top', 'bottom', 'center', 'baseline'（对齐方式）
    color:设置颜色
    '''

    ax.set_title(" The trajectory of the centroid")
    # 设置题目
    fig.tight_layout()  # 自动调整子图参数,使之填充整个图像区域。
    x = 3
    y = 3
    r = 3
    a = np.arange(x - r, x + r, 0.000001)
    # 点的纵坐标为b
    b = np.sqrt(np.power(r, 2) - np.power((a - x), 2))
    plt.plot(a, b + y, color='r', linestyle='-')
    plt.plot(a, -b + y, color='r', linestyle='-')
    plt.scatter(x, y, c='b', marker='o')
    # plt.grid(True)   #网格

    plt.show()


def heatmap_visualization_v2(arrs, names):
    def trim_axs(axs_, N):
        """
        Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
        """
        axs_ = axs_.flat
        for ax_ in axs[N:]:
            ax_.remove()
        return axs_[:N]

    n_fig = len(arrs)
    n_col = n_fig//2
    n_row = n_fig - n_col

    figure, axs = plt.subplots(n_row, n_col, sharex='all', sharey='all', constrained_layout=True)
    axs = trim_axs(axs, n_fig)
    for ax, arr, name in zip(axs, arrs, names):
        ax.set_title(name)
        Map = ax.imshow(arr, cmap='nipy_spectral', norm=colors.Normalize())
        figure.colorbar(Map, ax=ax, cax=None, shrink=0.5)
    # vmin = a.min(), vmax = a.max()
    plt.show()  # 图像展示


def vector_visualization(vector_x, vector_y, traject_v_x, traject_v_y, label):
    # data transform, from vector to vector matrix, the vector array is cummulative array
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('Accent')
    x = traject_v_x
    y = traject_v_y
    u = vector_x
    v = vector_y
    C = label
    q = ax.quiver(x, y, u, v, C, angles='xy', scale_units='xy', scale=1, cmap=cmap)
    fig.colorbar(q, shrink=0.5)
    # remember to set the resolution shown in the figure
    # full size 1080p is too big and can cause crash
    plt.show()


#  draw a circle (the plate edge)
def pint(r, x, y):
    plt.figure()  # 创建一个空白窗体
    plt.title('Circle')
    plt.xlabel('x')
    plt.ylabel('y')
    # 点的横坐标为a
    a = np.arange(x - r, x + r, 0.000001)
    # 点的纵坐标为b
    b = np.sqrt(np.power(r, 2) - np.power((a - x), 2)) + y
    plt.plot(a, b, color='r', linestyle='-')
    plt.plot(a, -b, color='r', linestyle='-')
    plt.scatter(0, 0, c='b', marker='o')
    plt.grid(True)
    plt.show()
