# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 13:23:18 2021

@author: pmrda
"""


import numpy as np
import cv2,glob
import cv2.aruco as aruco
from matplotlib import pyplot as plt
'''
# 종료 기준(termination criteria)를 정한다.
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30 ,0.001)

# Object Point(3D)를 준비한다. (0,0,0),(1,0,0),(2,0,0)... 처럼
objp = np.zeros((6*8,3),np.float32)
# np,mgrid[0:7,0:6]으로 (2,7,6) 배열 생성
# Transpose 해줘서 (6,7,2)로, reshpae(-1,2)로 flat 시켜서 (42,2)로 변환
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# 이미지로 부터의 Object point와 Image points를 저장하기 위한 배열
objpoints = [] # 실제 세계의 3D 점들 
imgpoints = [] # 2D 이미지의 점들

# 전체 path를 받기 위해 os말고 glob 사용
images = glob.glob('C:/Users/pmrda/Desktop/calibration/*.png')

for name in images:
    img = cv2.imread(name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스판의 코너들 찾기
    ret, corners = cv2.findChessboardCorners(gray,(8,6),None)

    # 찾았으면, Object points, Image points 추가하기 (이후에 수정한다)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # 코너를 그리고 봐보자
        img = cv2.drawChessboardCorners(img,(8,6),corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(2000)
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#camera matrix & distortion coeff 확인
print("camera matrix") 
print(mtx) 
print("distortion coeff") 
print(dist) 
'''

#camera matrix & distortion 
mtx = np.array([[574.94769264 ,  0.       ,  304.45999329],
                [  0.     ,    580.99846129 ,230.35481223]
               ,[  0.     ,      0.    ,       1.        ]])

dist = np.array([[ 0.02423695 , 0.08291847 ,-0.00184008 ,-0.00187223, -0.3784548 ]])

def track(matrix_coefficients, distortion_coefficients):
    while True:
        frame = cv2.imread('00001_image.png')
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Change grayscale
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
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
        # Display the resulting frame
        plt.imshow(frame)
        plt.figure()
        result = 1
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if result == 1: # Quit
            break
    
    # When everything done, release the capture
    cv2.destroyAllWindows()
    
track(mtx,dist)