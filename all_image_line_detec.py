# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:56:59 2021

@author: pmrda
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2,os
import math
import glob

os.chdir('_image')

#reading in an image


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):  
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:          #color
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
   
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img,lines

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def left_right(line_arr):
    slope_degree = (np.arctan2(line_arr[:,0,1] - line_arr[:,0,3], line_arr[:,0,0] - line_arr[:,0,2]) * 180) / np.pi
    
    line_arr = line_arr[np.abs(slope_degree)<160]
    slope_degree = slope_degree[np.abs(slope_degree)<160]

    line_arr = line_arr[np.abs(slope_degree)>90]
    slope_degree = slope_degree[np.abs(slope_degree)>90]
    
    L_lines, R_lines = line_arr[(slope_degree>0),0,:], line_arr[(slope_degree<0),0,:]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    return L_lines, R_lines

def Representative_line(img,line):
    line = np.squeeze(line)
    line = line.reshape(line.shape[0]*2,2)
    output = cv2.fitLine(line,cv2.DIST_L2,0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x) , img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)
    result = [[[x1,y1,x2,y2]]]
    return result

images = glob.glob('*.png')

for name in images:
    image_bgr = cv2.imread(name)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    kernel_size = 3     
    low_threshold = 100 
    high_threshold = 200
    vertices = np.array([[[0,height], [width,height],[width, height/2+50],[0, height/2+50]]], dtype=np.int32)
    min_line_len = 100
    max_line_gap = 150
    blur_img = gaussian_blur(image,kernel_size)
    canny_img = canny(blur_img,100,200)
    
    roi_img = region_of_interest(canny_img, vertices)
    #plt.imshow(roi_img,cmap='gray'),plt.title('ROI')
    #plt.figure()
    
    line_img,line_arr = hough_lines(roi_img, 1, 1 * np.pi/180, 120, min_line_len, max_line_gap) 
    #plt.imshow(line_img),plt.title('hough')
    #plt.figure()
    
    result = weighted_img(line_img, image) 
    plt.imshow(result),plt.title('result')
    plt.figure()
