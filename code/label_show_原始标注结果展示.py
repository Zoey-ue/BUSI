# -*- coding: utf-8 -*-


import cv2
import numpy as np

alpha = 0.4
color_mask = np.array(list((125, 0, 180)))
img_path = r'D:\Project2\DTC-master\data\BUSI\test\img\00007.bmp'
img_gt_path = r'D:\Project2\DTC-master\data\BUSI\test\gt\00007_anno.bmp'

img = cv2.imread(img_path)
img_gt = cv2.imread(img_gt_path, cv2.IMREAD_GRAYSCALE)
img_gt = img_gt/255
mask = img_gt.astype(bool)
img[mask] = img[mask] * (1 - alpha) + color_mask * alpha
cv2.imwrite('1.jpg', img)
cv2.imshow('1.jpg', img)
cv2.waitKey(0)



