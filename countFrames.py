# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 01:42:21 2021

@author: lucian
"""
import cv2
cap= cv2.VideoCapture('video_in.mp4')

totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("The total number of frames in this video is ", totalframecount)
