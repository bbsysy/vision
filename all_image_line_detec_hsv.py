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

    
images = glob.glob('*.png')

for name in images:
    img = cv2.imread(name)
    result = img.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    height, width = img.shape[:2]
    kernel_size = 3     
    low_threshold = 100 
    high_threshold = 200
    vertices = np.array([[[0,height], [width,height],[width, height/2+50],[0, height/2+50]]], dtype=np.int32)
    min_line_len = 100
    max_line_gap = 150
    
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
    #plt.subplot(1,2,2)
    #plt.imshow(canny_img,cmap='gray'),plt.title('canny')
    #plt.figure()
    
    #################################-----roi
    
    
    height, width = img.shape[:2]
    vertices = np.array([[[0,height], [width,height],[width, height/2+50],[0, height/2+50]]], dtype=np.int32)
    
    roi_img = region_of_interest(canny_img, vertices)
    #plt.imshow(roi_img,cmap='gray'),plt.title('ROI')
    #plt.figure()
    
    ################################-----hough
    line_img,line_arr = hough_lines(roi_img, 1, 1 * np.pi/180, 30, 50, 300) 
    #plt.imshow(line_img),plt.title('hough')
    #plt.figure()
    
    #======================= add
    
    result = weighted_img(line_img, img_rgb) 
    plt.imshow(result),plt.title('result')
    plt.figure()
    
        