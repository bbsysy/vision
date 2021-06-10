# -*- coding: utf-8 -*-

import numpy as np
import cv2,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#-------------------------------


os.chdir('_image')

img = cv2.imread('00001_image.png')

result = img.copy()
#cv2.imshow('Origian',img)

img_plt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img_plt)

print('This image is:', type(img), 'with dimensions:', img.shape)

img = cv2.GaussianBlur(img, (3, 3), 0)

#canny 
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
   
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img,lines


#################################-----roi
#roi_img = img[280:480,:640]

#======================== bird eye view

roi_img = img_plt[280:480,:640]
plt.imshow(img_plt)
plt.figure()
plt.imshow(roi_img)
plt.figure()
height, width = roi_img.shape[:2]
print(height)
print(width)

pts2 = np.float32([[245, 0], [40,height],[430, 0],[width,height]])
pts1 = np.float32([[0, 0], [0, height] , [width, 0],[width, height]])    #ori_coordi


M = cv2.getPerspectiveTransform(pts2,pts1)

warped_img = cv2.warpPerspective(roi_img, M, (width+10 ,height))
plt.imshow(warped_img),plt.title('warped_img')
plt.figure()




cv2.waitKey(0)
cv2.destroyAllWindows()
