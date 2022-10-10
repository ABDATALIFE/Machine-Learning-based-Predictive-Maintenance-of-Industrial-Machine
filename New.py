# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:03:33 2022

@author: Abdul Basit Aftab
"""
import cv2

vid = cv2.VideoCapture()

while True:
    ret,frame = vid.read()
    if cv2.waitkey(1) & 0xff == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
