#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import cv2
from saxUtils import *

im=retrieveImage('AgBehRingData_plus_some_more/latest_0001130_caz.tiff')
ret,thresh = cv2.threshold(im,2,255,0)
# same as
#thresh=makeBinaryImage(im,2)
cv2.imshow('a',thresh)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
tc=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
cv2.drawContours(tc, contours, -1, (0,255,0), 1)
cv2.imshow('b',tc)


cnt = contours[6]
cv2.contourArea(cnt)
M = cv2.moments(cnt)
# COM
cy = int(M['m01']/M['m00'])
cx = int(M['m10']/M['m00'])
cx
cy

cnt = contours[22]
ellipse = cv2.fitEllipse(cnt)
#                      Center                                   width , height            angle from h axis and 1st side
# ellipse: ((212.01751708984375, 353.942626953125), (55.76272964477539, 57.29179000854492), 47.94578552246094)
 
# struct CvBox2D
# Stores coordinates of a rotated rectangle.

# CvPoint2D32f center
# Center of the box

# CvSize2D32f size
# Box width and height

# float angle

cv2.cv.FitEllipse2(cnt)
cv2.ellipse(tc,ellipse,(0,0,255),1)
cv2.imshow('c',tc)

# Ellipse formula is (x/A)^2+(y/B)^2=1, where A and B are radiuses of ellipse
# Rectangle sides are Rw and Rh
# Let's assume we want ellipse with same proportions as rectangle; then, if we image square in circle (A=B,Rq=Rh) and squeeze it, we well keep ratio of ellipse A/B same as ratio of rectangle sides Rw/Rh;
# This leads us to following system of equations:
# (x/A)^2+(y/B)^2=1
# A/B=Rw/Rh

# Lets solve it: A=B*(Rw/Rh) 
# (Rh/2B)^2+(Rh/2B)^2=1 
# Rh=sqrt(2)*B 

# And final solution:
# A=Rw/sqrt(2)
# B=Rh/sqrt(2)

