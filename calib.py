# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:53:38 2021

@author: pmrda
"""


import numpy as np
import glob, cv2

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

# 왜곡 보정된 사진 저장

img = cv2.imread('C:/Users/pmrda/Desktop/calibration/00001_rgb.png')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)