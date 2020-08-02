# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:54:47 2020

@author: jaydi
"""

import numpy as np 
import cv2 


# 비디오캠 켜기
cap = cv2.VideoCapture('차차.mp4') 
   
while True: 
      
    # 비디오에서 이미지 읽어오기
    _, imageFrame = cap.read() 
  
    # 이미지 BGR에서 HSV로 변경하기
    hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 
  
    # 빨간색 마스크
    #red_lower = np.array([136, 87, 111], np.uint8) 
    #red_upper = np.array([180, 255, 255], np.uint8) 
    #red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
    # lower range
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # upper range
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    # 둘이 합쳐서 최종 마스크(detect red color)
    red_mask = mask1+mask2
      
    # 커널만들기
    kernal = np.ones((5, 5), "uint8") 
      
    # 빨간색 확실히 구분위해 팽창 등 노이즈 제거 For red color 
    red_mask = cv2.dilate(red_mask, kernal) 
    res_red = cv2.bitwise_and(imageFrame, imageFrame,  
                              mask = red_mask) 
   
    # 빨간색 영역 추적위해 컨투어 생성
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y),  
                                       (x + w, y + h),  
                                       (0, 0, 255), 2) 
              
            cv2.putText(imageFrame, "Red Detection", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 0, 255))     
  
              
    # 프로그램 시작!! ----------------------------
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yellow_frame = np.full((h,w,3), (0,255,255), dtype=np.uint8)
    red_mask_inv = cv2.bitwise_not(red_mask)
    final_image = cv2.copyTo(imageFrame, red_mask_inv, yellow_frame)
    
    yellow_area = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)
    yellow_area[:,:,1] += 50
    yellow_area = cv2.cvtColor(yellow_area, cv2.COLOR_HSV2BGR)
    final_image2 = cv2.copyTo(imageFrame, red_mask_inv, yellow_area)
    
    stacked = np.hstack((imageFrame, final_image, final_image2))
    cv2.imshow('result', stacked)
    
    if cv2.waitKey(15) == 27: 
        cap.release() 
        cv2.destroyAllWindows() 
        break