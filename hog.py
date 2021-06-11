# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:22:42 2021

@author: pmrda
"""


import numpy as np
import cv2,os
from matplotlib import pyplot as plt
import random
import glob

os.chdir('_image')

images = glob.glob('*.png')

for name in images:
    img_bgr = cv2.imread(name)
    img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    grayimg = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    detected, _ = hog.detectMultiScale(img_rgb)
    
    for (x, y, w, h) in detected:
            c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(img_rgb, (x, y, w, h), c, 2)
            
    plt.imshow(img_rgb)
    plt.figure()
    
    