# -*- coding: utf-8 -*-

import numpy as np
import cv2,os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math,itertools
#-------------------------------


os.chdir('_image')

img = cv2.imread('00025_image.png')

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
img_hsv = cv2.cvtColor(img_plt, cv2.COLOR_RGB2HSV)
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


plt.subplot(1,2,2)
plt.imshow(canny_img,cmap='gray'),plt.title('canny')
plt.figure()
#======================== roi

roi_img = canny_img[280:480,:640]
roi_img2 = img_plt[280:480,:640]

plt.imshow(roi_img,cmap='gray'),plt.title('roi')
plt.figure()
height, width = roi_img.shape[:2]
print(height)
print(width)
#======================== bird eye view
pts2 = np.float32([[0, 200], [600,200],[0, 0],[width,0]])
pts1 = np.float32([[235,200], [395, 200] , [0, 0],[width, 0]])   


M = cv2.getPerspectiveTransform(pts2,pts1)

warped_img = cv2.warpPerspective(roi_img, M, (width ,height))
warped_img2 = cv2.warpPerspective(roi_img2, M, (width ,height))

plt.imshow(warped_img,cmap='gray'),plt.title('warped_img')
plt.figure()
plt.imshow(warped_img2),plt.title('warped_img2')
plt.figure()

################################-----hough
line_img,line_arr = hough_lines(warped_img, 1, 1 * np.pi/180, 15, 20, 70) 

plt.imshow(line_img),plt.title('hough')
plt.figure()
#=============================== angle
slope_degree = (np.arctan2(line_arr[:,0,1] - line_arr[:,0,3], line_arr[:,0,0] - line_arr[:,0,2]) * 180) / np.pi
print(slope_degree)

line_arr = line_arr[np.abs(slope_degree)<160]
slope_degree = slope_degree[np.abs(slope_degree)<160]


line_arr = line_arr[np.abs(slope_degree)>90]
slope_degree = slope_degree[np.abs(slope_degree)>90]


L_lines, R_lines = line_arr[(slope_degree>0),0,:], line_arr[(slope_degree<0),0,:]
L_lines, R_lines = L_lines[:,None], R_lines[:,None]

cv2.waitKey(0)
cv2.destroyAllWindows()
