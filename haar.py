# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 17:52:08 2021

@author: pmrda
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('00001_image.png')
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_rgb2= img_rgb.copy()
grayimg = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

full_body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
full_body = full_body_cascade.detectMultiScale(grayimg, 1.03, 10, minSize=(30, 30))

lower_body_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
body = lower_body_cascade.detectMultiScale(grayimg, 1.01, 10, minSize=(50, 50))

for (x,y,w,h) in full_body :         
    cv2.rectangle(img_rgb,(x,y-50),(x+w,y+h),(0,0,255),5)

for (x,y,w,h) in body :         
    cv2.rectangle(img_rgb2,(x,y),(x+w,y+h),(0,0,255),5)
    
plt.imshow(img_rgb)
plt.figure()
plt.imshow(img_rgb2)

