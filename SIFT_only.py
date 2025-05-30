#Imports
from __future__ import print_function
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

#Data loading
os.chdir("C:/Users/4dmer/Desktop/Projects/Fingerprints")
wd = os.getcwd() + '/'
img1 = cv2.imread(wd + 'DB1_B/101/101_1.tif')  # image 1
img2 = cv2.imread(wd + 'DB1_B/101/101_3.tif')  # image 2

#Use sift to read the images and extract keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print("Number of keypoints in image 1: ", len(kp1))
print("Number of keypoints in image 2: ", len(kp2))
kp_image1 = cv2.drawKeypoints(img1, kp1, None, color=( 
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
#cv2.imshow('SIFT', kp_image1) 
#cv2.waitKey() 

kp_image2 = cv2.drawKeypoints(img2, kp2, None, color=( 
    0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
#cv2.imshow('SIFT', kp_image2) 
#cv2.waitKey() 

#Match the descriptors using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)                                               
matches = bf.match(des1,des2)
print("Number of matches: ", len(matches))                   
matches = sorted(matches, key = lambda x:x.distance)                                               
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)                       
#plt.imshow(img3),plt.show()

print("Similarity score: ", len(matches)/((len(kp1)+len(kp2))/2))