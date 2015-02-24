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
col=im[:,colCenter]
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


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# make histo of radii of peaks from center
# let's assume we have a good center for now...

from scipy import signal
from smooth import *
from saxsUtils import *
import cv2

# agbeh//latest_0001130_caz.tiff: (row,col,r) = (354,212,6.59242019653)
# agbeh//latest_0001134_caz.tiff: (row,col,r) = (354,212,6.62872543335)
# agbeh//latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
# agbeh//latest_0001141_caz.tiff: (row,col,r) = (355,212,5.92114868164)
# agbeh/im_0005241_caz.tiff: 203 352

rowCenter=352
colCenter=203

# rowCenter=355 #xCenter
# # rowCenter=354 #xCenter
# colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
# /Users/michael/Develop/saxs/AgBeh/agbeh/im_0005241_caz.tiff
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
im=retrieveImage('agbeh/im_0005241_caz.tiff')
figure(1)
clf()
imshow(im)
# rowCenter=355 #xCenter
# colCenter=212 #yCenter
xCenter=rowCenter
yCenter=colCenter
hSize=310 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0
# d=1
rows,cols = im.shape

# rickerWidths=arange(1,30)#np.array([10])
rickerWidths=np.array([8])
wl=25
order=3

row=im[rowCenter,:]
col=im[:,colCenter]

rowS=savitzky_golay(row,wl,order,deriv=d)
peakind = signal.find_peaks_cwt(rowS, rickerWidths,min_snr=1)

figure(2)
clf()
plot(rowS,'b')
plot(peakind, rowS[peakind],'r*')

rH=[array(peakind)-colCenter]
colS=savitzky_golay(col,wl,order,deriv=d)
peakind = signal.find_peaks_cwt(colS, rickerWidths,min_snr=1)
figure(3)
clf()
plot(colS,'b')
plot(peakind, colS[peakind],'r*')

rH.append(array(peakind)-rowCenter)



# figure(1)
# clf()
# plot(row)
# figure(2)
# clf()
# plot(col)

for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((yCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=rim[rowCenter,:]
        col=rim[:,colCenter]

        rowS=savitzky_golay(row,wl,order,deriv=d)
        peakind = signal.find_peaks_cwt(rowS, rickerWidths,min_snr=1)
        # figure(1)
        # clf()
        # plot(rowS,'b')
        # plot(peakind, rowS[peakind],'r*')
        # print array(peakind)-colCenter
        rH.append(array(peakind)-colCenter)
        colS=savitzky_golay(col,wl,order,deriv=d)
        peakind = signal.find_peaks_cwt(colS, rickerWidths,min_snr=1)
        # figure(2)
        # clf()
        # plot(colS,'b')
        # plot(peakind, colS[peakind],'r*')
        # print array(peakind)-rowCenter
        rH.append(array(peakind)-rowCenter)

rHisData=np.hstack(rH)
rHisData=abs(rHisData)
radHist=np.histogram(rHisData,bins=321,range=(0,321))

figure(4)
clf()
wl=15
order=3
rickerWidths=arange(8,30)#np.array([])
radHistS=savitzky_golay(radHist[0],wl,order)
peakind = signal.find_peaks_cwt(radHistS, rickerWidths,min_snr=1)
plot(radHist[1][0:321],radHistS,'0.5')
plot(peakind, radHistS[peakind],'r*')

# the image
figure(1)
clf()
imshow(im)
xPeaks=array(peakind)+colCenter
xPeaks=xPeaks[radHistS[peakind]>(nSteps/8.)]
ringStars=ones_like(xPeaks)+rowCenter
plot(xPeaks,ringStars,'r+')

        # figure(1)
        # plot(row)
        # figure(2)
        # plot(col)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Let'slook at the first deriv, so we can get the BS circle




from scipy import signal
from smooth import *
from saxsUtils import *
import cv2
# agbeh/latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
im0=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')


rowCenter=355
colCenter=212
roi=50
im = takeROISlice(im0,roi,(rowCenter,colCenter))



figure(1)
clf()
imshow(im)


wl=15
order=3
rickerWidths=np.array([5])
rowCenter=roi/2
colCenter=roi/2
xCenter=rowCenter
yCenter=colCenter
hSize=roi/2 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
rows,cols = im.shape

row=im[rowCenter,:]
col=im[:,colCenter]

# look at slopes

# take away: looks good for BS, but not so much for AgBeh peaks

rowPrime=savitzky_golay(row,wl,order,deriv=1)
peakind = signal.find_peaks_cwt(rowPrime, rickerWidths,min_snr=1)
min_rowPrime=rowPrime*-1
min_peakind = signal.find_peaks_cwt(min_rowPrime, rickerWidths,min_snr=1)

rH=[array(peakind)-colCenter]
rH.append(array(min_peakind)-colCenter)

figure(2)
clf()

plot(peakind, rowPrime[peakind],'r*')
plot(min_peakind, rowPrime[min_peakind],'g*')
plot(rowPrime,'0.5')

figure(3)
clf()
plot(row,'0.5')
plot(peakind, row[peakind],'r*')
plot(min_peakind, row[min_peakind],'g*')




colPrime=savitzky_golay(col,wl,order,deriv=1)
peakind = signal.find_peaks_cwt(colPrime, rickerWidths,min_snr=1)
min_colPrime=colPrime*-1
min_peakind = signal.find_peaks_cwt(min_colPrime, rickerWidths,min_snr=1)

rH.append(array(peakind)-rowCenter)
rH.append(array(min_peakind)-rowCenter)




# ======

for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((yCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=rim[rowCenter,:]
        col=rim[:,colCenter]

        rowPrime=savitzky_golay(row,wl,order,deriv=1)
        peakind = signal.find_peaks_cwt(rowPrime, rickerWidths,min_snr=1)
        min_rowPrime=rowPrime*-1
        min_peakind = signal.find_peaks_cwt(min_rowPrime, rickerWidths,min_snr=1)

        rH.append(array(peakind)-colCenter)
        rH.append(array(min_peakind)-colCenter)

        
        colPrime=savitzky_golay(col,wl,order,deriv=1)
        peakind = signal.find_peaks_cwt(colPrime, rickerWidths,min_snr=1)
        min_colPrime=colPrime*-1
        min_peakind = signal.find_peaks_cwt(min_colPrime, rickerWidths,min_snr=1)

        rH.append(array(peakind)-rowCenter)
        rH.append(array(min_peakind)-rowCenter)



rHisData=np.hstack(rH)
rHisData=abs(rHisData)
radHist=np.histogram(rHisData,bins=roi/2+1,range=(0,roi/2+1))

figure(4)
clf()
wl=21
order=9
rickerWidths=np.array([5])
radHistS=savitzky_golay(radHist[0],wl,order)
peakind = signal.find_peaks_cwt(radHistS, rickerWidths,min_snr=1)
bar(radHist[1][0:roi/2+1],radHist[0])
plot(radHist[1][0:roi/2+1],radHistS,'0.5')
plot(peakind, radHistS[peakind],'r*')

# the image
figure(1)
clf()
imshow(im)
xPeaks=array(peakind)+colCenter
# xPeaks=xPeaks[radHistS[peakind]>(nSteps/8.)]
ringStars=ones_like(xPeaks)+rowCenter
plot(xPeaks,ringStars,'r+')


# plot(min_rowPrime,color=(0.5,0.7,0.5))
# array(min_peakind)-colCenter
# array(min_peakind)-colCenter

# a=hstack((min_peakind,peakind))-colCenter
# a.sort()
# a

# # center estimate between the zero_crossings element and the next
# zero_crossings = numpy.where(numpy.diff(numpy.sign(a)))[0]
# zero_crossings



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# root

import ROOT as r
from scipy import signal
from smooth import *
from saxsUtils import *
import cv2
from npRootUtils import *
# agbeh//latest_0001130_caz.tiff: (row,col,r) = (354,212,6.59242019653)
# agbeh//latest_0001134_caz.tiff: (row,col,r) = (354,212,6.62872543335)
# agbeh//latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
# agbeh//latest_0001141_caz.tiff: (row,col,r) = (355,212,5.92114868164)

rowCenter=355 #xCenter
# rowCenter=354 #xCenter
colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
# figure(1)
# clf()
# imshow(im)
# rowCenter=355 #xCenter
# colCenter=212 #yCenter
xCenter=rowCenter
yCenter=colCenter
hSize=310 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0
# d=1
rows,cols = im.shape

# rickerWidths=arange(1,30)#np.array([10])
rickerWidths=np.array([8])
wl=25
order=3

row=im[rowCenter,:]
col=im[:,colCenter]

# setTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):
rowHist=setTH1fFromAr1D(row,name='row0',title='row0')

# ShowPeaks(Double_t sigma = 2, Option_t* option = "", Double_t threshold = 0.050000000000000003)

ShowPeaks(2,"",0.5)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# make histo of radii of peaks from center
# let's assume we have a good center for now...

from scipy import signal
from smooth import *
from saxsUtils import *
import cv2
from npRootUtils import *
# agbeh//latest_0001130_caz.tiff: (row,col,r) = (354,212,6.59242019653)
# agbeh//latest_0001134_caz.tiff: (row,col,r) = (354,212,6.62872543335)
# agbeh//latest_0001139_caz.tiff: (row,col,r) = (355,212,5.94810905457)
# agbeh//latest_0001141_caz.tiff: (row,col,r) = (355,212,5.92114868164)
# agbeh/im_0005241_caz.tiff: 203 352

rowCenter=352
colCenter=203

# rowCenter=355 #xCenter
# # rowCenter=354 #xCenter
# colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
# /Users/michael/Develop/saxs/AgBeh/agbeh/im_0005241_caz.tiff
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
im=retrieveImage('agbeh/im_0005241_caz.tiff')
figure(1)
clf()
imshow(im)
# rowCenter=355 #xCenter
# colCenter=212 #yCenter
xCenter=rowCenter
yCenter=colCenter
hSize=310 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0
# d=1
rows,cols = im.shape

rickerWidths=arange(1,10)
# rickerWidths=np.array([5])
wl=25
order=3

row=im[rowCenter,:]
col=im[:,colCenter]

# rowS=savitzky_golay(row,wl,order,deriv=d)
peakind = signal.find_peaks_cwt(row, rickerWidths,min_snr=1)

figure(2)
clf()
plot(row,'b')
plot(peakind, row[peakind],'r*')

rH=[array(peakind)-colCenter]
# colS=savitzky_golay(col,wl,order,deriv=d)
peakind = signal.find_peaks_cwt(col, rickerWidths,min_snr=1)
figure(3)
clf()
plot(col,'b')
plot(peakind, col[peakind],'r*')

rH.append(array(peakind)-rowCenter)



# figure(1)
# clf()
# plot(row)
# figure(2)
# clf()
# plot(col)

for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((yCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=rim[rowCenter,:]
        col=rim[:,colCenter]

        # rowS=savitzky_golay(row,wl,order,deriv=d)
        peakind = signal.find_peaks_cwt(row, rickerWidths,min_snr=1)
        # figure(1)
        # clf()
        # plot(rowS,'b')
        # plot(peakind, rowS[peakind],'r*')
        # print array(peakind)-colCenter
        rH.append(array(peakind)-colCenter)
        # colS=savitzky_golay(col,wl,order,deriv=d)
        peakind = signal.find_peaks_cwt(col, rickerWidths,min_snr=1)
        # figure(2)
        # clf()
        # plot(colS,'b')
        # plot(peakind, colS[peakind],'r*')
        # print array(peakind)-rowCenter
        rH.append(array(peakind)-rowCenter)

rHisData=np.hstack(rH)
rHisData=abs(rHisData)
radHist=np.histogram(rHisData,bins=321,range=(0,321))

figure(4)
clf()
wl=15
order=3
# rickerWidths=arange(8,30)#np.array([])
radHistS=savitzky_golay(radHist[0],wl,order)
peakind = signal.find_peaks_cwt(radHist[0], rickerWidths,min_snr=1)
plot(radHist[1][0:321],radHist[0],'0.5')
plot(peakind, radHist[0][peakind],'r*')

# the image
figure(1)
clf()
imshow(im)
xPeaks=array(peakind)+colCenter
xPeaks=xPeaks[radHist[0][peakind]>(nSteps/10)]
ringStars=ones_like(xPeaks)+rowCenter
plot(xPeaks,ringStars,'r+')

# look at peaks from ~ 25 pix to ~110 pix, for this image agbeh/im_0005241_caz.tiff
#  the point being that 1st order peak is oftem messed up, and we want 2nd to 8th or so.


# let's write a function:   input: is good center, path to image, radius range to search for peaks.
#                           output: peak spacing (pix). How many peaks, an arr w/ peaks in that range.
#                           
# a=xPeaks[0:7]

# In [89]: a
# Out[89]: array([217, 232, 246, 260, 275, 289, 304])

# In [90]: (304-217)/6.
# Out[90]: 14.5

# In [91]: 32-17
# Out[91]: 15

# In [92]: 
# In [77]: gf=th.Fit('gaus','S','',225,239)
#  FCN=141.449 FROM MIGRAD    STATUS=CONVERGED      82 CALLS          83 TOTAL
#                      EDM=1.97313e-12    STRATEGY= 1      ERROR MATRIX ACCURATE 
#   EXT PARAMETER                                   STEP         FIRST   
#   NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE 
#    1  Constant     4.79398e+02   1.49288e+01   6.35867e-02  -1.56251e-07
#    2  Mean         2.31667e+02   3.74703e-02   2.13111e-04  -2.47214e-05
#    3  Sigma        1.60658e+00   3.41430e-02   2.56546e-05  -3.05188e-04

# In [78]: gf.Value(0)
# Out[78]: 479.3979941558617

# In [79]: gf.Value(1)
# Out[79]: 231.6665489234626

# In [80]: gf.Value(2)
# Out[80]: 1.606575935147517

s=TSpectrum()

nFound=s.Search(th,1,"",0.005) #this updates the th1 with polies on the peaks, can see it in th1.Draw()
xs=s.GetPositionX()
ys=s.GetPositionY()
for idx in range(nFound):
    print xs[idx],ys[idx]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# from scipy import signal
# from smooth import *
from saxsUtils import *
import cv2
from ROOT import TH1F, TSpectrum
from npRootUtils import *
from root_numpy import fill_hist
from numpy import *
# agbeh/im_0005241_caz.tiff: 203 352

firstPeak=25
lastPeak=110
rowCenter=352
colCenter=203

# rowCenter=355 #xCenter
# # rowCenter=354 #xCenter
# colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
# /Users/michael/Develop/saxs/AgBeh/agbeh/im_0005241_caz.tiff
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
im=retrieveImage('agbeh/im_0005241_caz.tiff')
rows,cols = im.shape
xCenter=rowCenter
yCenter=colCenter
hSize=310 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0


# setTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):

peaksHist= TH1F('peaksHist','peaks',hSize*10,0,hSize)
dPeaksHist=TH1F('dPeaksHist','dPeaks',300,0,30)
sRow=TSpectrum()
sCol=TSpectrum()



row=im[rowCenter,:]
col=im[:,colCenter]

rowHist=makeTH1fFromAr1D(row,name='row',title='row')
colHist=makeTH1fFromAr1D(col,name='col',title='col')

#this updates the th1 with polies on the peaks, can see it in th1.Draw(). goff turns that off.
nFoundRow=sRow.Search(rowHist,1,'goff',0.005) 
nFoundCol=sCol.Search(colHist,1,'goff',0.005)

xsRow=sRow.GetPositionX()
xsCol=sCol.GetPositionX()

axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter
# get the peaks in the range we care about and fit gaussians
axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
# can now use fitGausPeaks(th,peaks) to get the peak fits of axP and axM

# fitGausPeaks gives a list of tuples: [(const,mean,sigma),(const,mean,sigma),...]
fitsRowP=fitGausPeaks(rowHist,axRowP)
fitsRowM=fitGausPeaks(rowHist,axRowM)
fitsColP=fitGausPeaks(colHist,axColP)
fitsColM=fitGausPeaks(colHist,axColM)

# Fill the peaks histo with the means of the gaus fits
arFitsRowP=array([x[1] for x in fitsRowP])-colCenter
fill_hist(peaksHist, arFitsRowP)
fill_hist(dPeaksHist,diff(arFitsRowP))

arFitsRowM=array([x[1] for x in fitsRowM])-colCenter
fill_hist(peaksHist, arFitsRowM)
fill_hist(dPeaksHist,diff(arFitsRowM))

arFitsColP=array([x[1] for x in fitsColP])-rowCenter
fill_hist(peaksHist, arFitsColP)
fill_hist(dPeaksHist,diff(arFitsColP))

arFitsColM=array([x[1] for x in fitsColM])-rowCenter
fill_hist(peaksHist, arFitsColM)
fill_hist(dPeaksHist,diff(arFitsColM))


for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((colCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=rim[rowCenter,:]
        col=rim[:,colCenter]

        setBinsToAr1D(rowHist,row)
        setBinsToAr1D(colHist,col)

        nFoundRow=sRow.Search(rowHist,1,'goff',0.005) 
        nFoundCol=sCol.Search(colHist,1,'goff',0.005)

        xsRow=sRow.GetPositionX()
        xsCol=sCol.GetPositionX()

        axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
        axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter
        # get the peaks in the range we care about and fit gaussians
        axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
        axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
        axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
        axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
        # can now use fitGausPeaks(th,peaks) to get the peak fits of axP and axM

        # fitGausPeaks gives a list of tuples: [(const,mean,sigma),(const,mean,sigma),...]
        fitsRowP=fitGausPeaks(rowHist,axRowP)
        fitsRowM=fitGausPeaks(rowHist,axRowM)
        fitsColP=fitGausPeaks(colHist,axColP)
        fitsColM=fitGausPeaks(colHist,axColM)

        # Fill the peaks histo with the means of the gaus fits
        arFitsRowP=array([x[1] for x in fitsRowP])-colCenter
        fill_hist(peaksHist, arFitsRowP)
        fill_hist(dPeaksHist,diff(arFitsRowP))

        arFitsRowM=array([x[1] for x in fitsRowM])-colCenter
        fill_hist(peaksHist, arFitsRowM)
        fill_hist(dPeaksHist,diff(arFitsRowM))

        arFitsColP=array([x[1] for x in fitsColP])-rowCenter
        fill_hist(peaksHist, arFitsColP)
        fill_hist(dPeaksHist,diff(arFitsColP))

        arFitsColM=array([x[1] for x in fitsColM])-rowCenter
        fill_hist(peaksHist, arFitsColM)
        fill_hist(dPeaksHist,diff(arFitsColM))

# these peaks just fill one bin, at x.5, so I'll try fitting the slices to gaus at the peak positions, then fill with mean.
# peaksHist.Draw()
dPeaksHist.Draw()
gf=dPeaksHist.Fit('gaus','QS','goff')
dMean=gf.Value(1)
dMeanEr=gf.Error(1)
dSig=gf.Value(2)
dSigEr=gf.Error(2)
# this gets the peaks array out at the end
nFound = sCol.Search(peaksHist,3.5,'',0.03)
xsPeaks=sCol.GetPositionX()
aPeaks=rwBuf2Array(xsPeaks,nFound)
fitsPeaks=fitGausPeaks(peaksHist,aPeaks)

print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr



