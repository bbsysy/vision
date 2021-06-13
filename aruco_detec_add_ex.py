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

matrix_coefficients = np.array([[574.94769264 ,  0.       ,  304.45999329],
                [  0.     ,    580.99846129 ,230.35481223]
               ,[  0.     ,      0.    ,       1.        ]])

distortion_coefficients = np.array([[ 0.02423695 , 0.08291847 ,-0.00184008 ,-0.00187223, -0.3784548 ]])



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
            print('wyrano')
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
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
            #show the output image
            plt.imshow(image)
            plt.figure()
            while True:
            # operations on the frame come here
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Change grayscale
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Use 5x5 dictionary to find markers
                parameters = aruco.DetectorParameters_create()  # Marker detection parameters
                # lists of ids and the corners beloning to each id
                corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                        parameters=parameters,
                                                                        cameraMatrix=matrix_coefficients,
                                                                        distCoeff=distortion_coefficients)
                if np.all(ids is not None):  # If there are markers found by detector
                    for i in range(0, len(ids)):  # Iterate in markers
                       # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                                   distortion_coefficients)
                        (rvec - tvec).any()  # get rid of that nasty numpy value array error
                        aruco.drawDetectedMarkers(image, corners)  # Draw A square around the markers
                        aruco.drawAxis(image, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
                # Display the resulting frame
                plt.imshow(image)
                plt.figure()
                result = 1
                # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
                key = cv2.waitKey(3) & 0xFF
                if result == 1: # Quit
                    break
            
       
        