import cv2
import numpy as np
from skimage.morphology import skeletonize
from crack_detector import DSE
import imutils
from image_explorer import ImgExp


frame = cv2.imread(r'/sample/overlap/0169.jpg')
fgMask = np.load(r'D:\workspace\python\objectTracking\sample\e_frame\problem_frame169.npy')

# get original skeleton
fgMask_skt_bin = fgMask.copy()
print(np.unique(fgMask_skt_bin))
fgMask_skt_bin[fgMask_skt_bin == 255] = 1
fgMask_skt_bin = skeletonize(fgMask_skt_bin)
endpoint2 = DSE.skeleton_endpoints(fgMask_skt_bin)
fgMask_skt = fgMask_skt_bin.astype(np.uint8) * 255

# get pruned skeleton
beta = 1
dse = DSE(fgMask_skt_bin, beta)
dse.show_branch()
fgMask_skt_pruned_bin = dse.prune(True)
fgMask_skt_pruned = fgMask_skt_pruned_bin.astype(np.uint8) * 255
endpoint1 = dse.skeleton_endpoints(fgMask_skt_pruned_bin)

# extract contour and skeleton
cnts = cv2.findContours(fgMask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
skts1 = cv2.findContours(fgMask_skt_pruned.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
skts1 = imutils.grab_contours(skts1)
skts2 = cv2.findContours(fgMask_skt.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
skts2 = imutils.grab_contours(skts2)
print('pruned skel:', len(endpoint1), len(skts1))
print('raw skel:', len(endpoint2), len(skts2))

# overlay contour and pruned skeleton
frame2 = frame.copy()
# framergb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # convert to RGB to draw contour
cv2.drawContours(frame, skts1, -1, (0, 255, 0), 3)
draw1 = cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)

# overlay contour and raw skeleton
# framergb2 = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # convert to RGB to draw contour
cv2.drawContours(frame2, skts2, -1, (0, 255, 0), 3)
draw2 = cv2.drawContours(frame2, cnts, -1, (0, 255, 0), 3)


# fgMask_skt_show = cv2.resize(fgMask_skt.copy(), (1536, 1024))
# cv2.imshow('raw skel', fgMask_skt_show)
wind1 = ImgExp('raw skel', fgMask_skt)
wind1.main()

# fgMask_skt_pruned_show = cv2.resize(fgMask_skt_pruned.copy(), (1536, 1024))
# cv2.imshow('pruned skel', fgMask_skt_pruned_show)
wind2 = ImgExp('pruned skel', fgMask_skt_pruned)
wind2.main()

# draw1 = cv2.resize(draw1, (1536, 1024))
# cv2.imshow('contour + pruned contour', draw1)
wind3 = ImgExp('contour + pruned contour', draw1)
wind3.main()

# draw2 = cv2.resize(draw2, (1536, 1024))
# cv2.imshow('contour + raw contour', draw2)
wind4 = ImgExp('contour + raw contour', draw2)
wind4.main()

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break










