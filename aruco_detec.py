# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:12:41 2021

@author: pmrda
"""
import argparse
#import imutils
import cv2,os
import sys
from matplotlib import pyplot as plt
import glob
import cv2.aruco as aruco
import numpy as np

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

os.chdir('_image')

images = glob.glob('*.png')

for name in images:
    #print("[INFO] loading image...")
    image =cv2.imread(name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image = imutils.resize(image, width=600)
    # verify that the supplied ArUCo tag exists and is supported by
    # OpenCV
    
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_4X4_50'])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,parameters=arucoParams)
    print(corners)
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
    	# flatten the ArUco IDs list
    	ids = ids.flatten()
    	# loop over the detected ArUCo corners
    	for (markerCorner, markerID) in zip(corners, ids):
    		# extract the marker corners (which are always returned in
    		# top-left, top-right, bottom-right, and bottom-left order)
    		corners = markerCorner.reshape((4, 2))
    		(topLeft, topRight, bottomRight, bottomLeft) = corners
    		# convert each of the (x, y)-coordinate pairs to integers
    		topRight = (int(topRight[0]), int(topRight[1]))
    		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    		topLeft = (int(topLeft[0]), int(topLeft[1]))
            # draw the bounding box of the ArUCo detection
    		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
    		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
    		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
    		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
    		# compute and draw the center (x, y)-coordinates of the ArUco
    		# marker
    		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
    		# draw the ArUco marker ID on the image
    		cv2.putText(image, str(markerID),
    			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
    			0.5, (0, 255, 0), 2)
    		print("[INFO] ArUco marker ID: {}".format(markerID))
    		# show the output image
    		plt.imshow(image),plt.figure()
            
   
            
