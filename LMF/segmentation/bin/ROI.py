# import the necessary packages
import cv2
import numpy as np
import joblib
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-p", "--point", help="indicate the maximal point of ROI")
args = vars(ap.parse_args())

# initialize points, points count and points list
lsPointsChoose = []
tpPointsChoose = []

pointsCount = 0
count = 0
pointsMax = int(args["point"])
pts_list = []


def on_mouse(event, x, y, flag, para):
    global img, point1, point2, count, pointsMax
    global lsPointsChoose, tpPointsChoose
    global pointsCount

    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        pointsCount = pointsCount + 1
        print('pointsCount:', pointsCount)
        point1 = (x, y)
        print(x, y)
        # draw the selecting point
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)

        # save the point into list
        lsPointsChoose.append([x, y])
        tpPointsChoose.append((x, y))

        # link the points by straight line
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 5)
        # when number of points reaches max, extract image
        if pointsCount == pointsMax:
            # draw ROI
            ROI_byMouse()
            lsPointsChoose = []
        # show result
        cv2.imshow('src', img2)
    # right click to clear track
    if event == cv2.EVENT_RBUTTONDOWN:
        print("right-mouse")
        pointsCount = 0
        tpPointsChoose = []
        lsPointsChoose = []
        print(len(tpPointsChoose))
        for i in range(len(tpPointsChoose) - 1):
            print('i', i)
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 5)
        cv2.imshow('src', img2)


def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([lsPointsChoose], np.int32)
    # reshape the points
    pts = pts.reshape((-1, 1, 2))  # -1代表剩下的维度自动计算
    # draw the polygon
    mask = cv2.polylines(mask, [pts], True, (0, 255, 255))
    # fill the polygon
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    # show the image
    # cv2.imshow('mask', mask2)
    ROI = cv2.bitwise_and(mask2, img)
    pts_list.append(pts)


def main():
    global img, img2, ROI
    vs = cv2.VideoCapture(args["video"] + "/%04d.jpg", cv2.CAP_IMAGES)
    frame = vs.read()
    img = frame[1]
    ROI = img.copy()
    cv2.namedWindow('src', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('src', on_mouse)
    cv2.imshow('src', img)
    cv2.waitKey(0)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord("s"):
            saved_data = {"ROI": pts_list}
            joblib.dump(value=saved_data, filename="config.pkl")
            print("[INFO] ROI coordinates has been saved ")
            break
    cv2.destroyAllWindows()
    # cv2.waitKey(1)


if __name__ == '__main__':
    main()
