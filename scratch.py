#!/usr/bin/env python
# encoding: utf-8

# agbeh//latest_0001130_caz.tiff: (row,col,r) = (354,212,6.59242019653)
# agbeh//latest_0001134_caz.tiff: (row,col,r) = (354,212,6.62872543335)
# agbeh//latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
# agbeh//latest_0001141_caz.tiff: (row,col,r) = (355,212,5.92114868164)


import numpy as np
import cv2

from saxsUtils import *


et1= 5000
et2= 100
apSize=5

im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')

ret,thresh = cv2.threshold(im,40,255,0)
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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Look at lines and symmetry

# agbeh/latest_0001130_caz.tiff: (row,col,r) = (354,212,6.59242019653)
# agbeh/latest_0001134_caz.tiff: (row,col,r) = (354,212,6.62872543335)
# agbeh/latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
# agbeh/latest_0001141_caz.tiff: (row,col,r) = (355,212,5.92114868164)

# smooth(x,window_len=11,window='hanning')
# 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
import numpy as np
import cv2
from smooth import *
from saxsUtils import *

im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')

xCenter=354
yCenter=212
im=retrieveImage('agbeh/latest_0001134_caz.tiff')
imSmall=colorize(cv2.resize(im,(0,0), fx=0.5, fy=0.5))
cv2.imshow('small',imSmall)
row=im[354,:]
col=im[:,212]

wl=9
colS=smooth(col,window_len=wl)
figure(1)
clf()
plot(colS)


im=retrieveImage('agbeh/latest_0001139_caz.tiff')
imSmall=colorize(cv2.resize(im,(0,0), fx=0.5, fy=0.5))
cv2.imshow('small',imSmall)
row=im[354,:]
col=im[:,212]
wl=9
colS=smooth(col,window_len=wl,window='flat')
rowS=smooth(row,window_len=wl,window='flat')
figure(1)
clf()
plot(col)
plot(colS)
figure(2)
clf()
plot(row)
plot(rowS)


# savitzky_golay(y, window_size, order, deriv=0, rate=1):
import numpy as np
import cv2
from smooth import *
from saxsUtils import *
# agbeh/latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
xCenter=353
yCenter=212
im=retrieveImage('agbeh/latest_0001139_caz.tiff')
imSmall=colorize(cv2.resize(im,(0,0), fx=0.5, fy=0.5))
cv2.imshow('small',imSmall)
row=im[354,:]
col=im[:,212]
wl=21
order=5
colS=savitzky_golay(col,wl,order)
rowS=savitzky_golay(row,wl,order)
figure(1)
clf()
plot(col,'0.5')
plot(colS,'r')
figure(2)
clf()
plot(row,'0.5')
plot(rowS,'r')

rows,cols = im.shape
M = cv2.getRotationMatrix2D((yCenter,xCenter),45,1)
rim=cv2.warpAffine(im,M,(cols,rows))
rimCS=colorize(cv2.resize(rim,(0,0), fx=0.5, fy=0.5))
cv2.imshow('small rotated',rimCS)

row=rim[354,:]
col=rim[:,212]
wl=21
order=5
colS=savitzky_golay(col,wl,order)
rowS=savitzky_golay(row,wl,order)

figure(1)
# clf()
plot(col,'0.5')
plot(colS,'b')
figure(2)
# clf()
plot(row,'0.5')
plot(rowS,'b')

rowPrime=savitzky_golay(row,wl,order,deriv=1)
# HMM... have to take the void bands into account...
# Fingind agle of BS: rotate until we get a long line of black that suddenly isn't (and isn't a void band, which are known).

from scipy import signal
from smooth import *
from saxsUtils import *
import cv2
# agbeh/latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')


rowCenter=355
colCenter=212
figure(1)
clf()
imshow(im)

figure(2)
clf()
wl=25
order=3
row=im[rowCenter,:]
col=rim[:,colCenter]
rowS=savitzky_golay(row,wl,order)
# rickerWidths=np.array([10,15,30])
rickerWidths=np.array([8])
peakind = signal.find_peaks_cwt(rowS, rickerWidths,min_snr=1)
# peakind = signal.find_peaks_cwt(rowS, np.arange(1,10),min_snr=1)
# print peakind, row[peakind]
plot(peakind, rowS[peakind],'r*')
plot(rowS,'0.5')

# plot(row,color=(0.0,0.5,0.0))

# let's find how far from center


# looks goot for AgBeh peaks, but gives too wide on the BS - the S.G. speads the function out from the 
# square-well of the BS, but since the AgBeh peaks are more symmetric, it does much better there.

figure(3)
clf()
wl=25
order=3
row=im[rowCenter,:]
col=rim[:,colCenter]
rowS=savitzky_golay(row,wl,order)
# rickerWidths=np.array([10,15,30])
rickerWidths=np.array([8])
peakind = signal.find_peaks_cwt(rowS, rickerWidths,min_snr=1)
# peakind = signal.find_peaks_cwt(rowS, np.arange(1,10),min_snr=1)
# print peakind, row[peakind]


plot(row,'0.5')#color=(0.0,0.3,0.0))
plot(rowS,'b')
plot(peakind, rowS[peakind],'r*')
array(peakind)-colCenter

# look at slopes

# take away: looks good for BS, but not so much for AgBeh peaks

rowPrime=savitzky_golay(row,wl,order,deriv=1)
rickerWidths=np.array([7])
peakind = signal.find_peaks_cwt(rowPrime, rickerWidths,min_snr=1)
min_rowPrime=rowPrime*-1
min_peakind = signal.find_peaks_cwt(min_rowPrime, rickerWidths,min_snr=1)
figure(4)
clf()
plot(peakind, rowPrime[peakind],'r*')
plot(min_peakind, rowPrime[min_peakind],'g*')
plot(rowPrime,'0.5')

figure(5)
clf()
plot(row,'0.5')
plot(peakind, row[peakind],'r*')
plot(min_peakind, row[min_peakind],'g*')

# plot(min_rowPrime,color=(0.5,0.7,0.5))
array(min_peakind)-colCenter
array(min_peakind)-colCenter

 a=hstack((min_peakind,peakind))-colCenter
 a.sort()
 a

 zero_crossings = numpy.where(numpy.diff(numpy.sign(a)))[0]
 zero_crossings







