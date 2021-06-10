# -*- coding: utf-8 -*-

import numpy as np
import cv2,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#-------------------------------



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

#################################----hsv
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# HSV; RED color range values; 
lower1 = np.array([160,50,25])
upper1 = np.array([180,255,255])


# RGB; yellow color range values; 
lower2 = np.array([17,120,120])
upper2 = np.array([38,255,255])

red_mask = cv2.inRange(img_hsv, lower1, upper1)
yellow_mask = cv2.inRange(img_hsv,lower2,upper2)

full_mask = red_mask + yellow_mask;

img_result = cv2.bitwise_and(result, result, mask = full_mask)

canny_img = canny(img_result,100,200)

#cv2.imshow('hsv_canny',canny_img)
plt.subplot(1,2,2)
plt.imshow(canny_img,cmap='gray'),plt.title('canny')
plt.figure()

#################################-----roi
H=240
W=640
roi_area = canny_img[280:H+240,:W]

plt.imshow(roi_area,cmap='gray'),plt.title('ROI')
plt.figure()

################################-----hough
line_img,line_arr = hough_lines(roi_area, 1, 1 * np.pi/180, 30, 50, 300) 
plt.imshow(line_img),plt.title('hough')
plt.figure()


cv2.waitKey(0)
cv2.destroyAllWindows()
