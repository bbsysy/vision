# vision
## Robot Vision for Smart Factory

### lane.py는 원래 rgb를 canny 후 허프로 (허프 threshold를 높여서 검출)
### line_detection.py 는 hsv로 색깔을 검출 후 -> 캐니 -> 허프 
### all_image_line_detec : 모든 이미지 한번에 검출 // rgb --- 코너돌때랑 그 후 라인 인식 불가
### all_image_line_detec_hsv : 모든 이미지 hsv로 검출 // ---코너 돌고도 잘하는데 여러개 나올 떄도 있다.
