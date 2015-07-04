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
from ROOT import TH1F, TSpectrum, TCanvas
from npRootUtils import *
# from root_numpy import fill_hist
from numpy import *
# agbeh/im_0005241_caz.tiff: 203 352

firstPeak=100
lastPeak=352
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
arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
fill_hist(peaksHist, arFitsRowP)
fill_hist(dPeaksHist,diff(arFitsRowP))

arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
fill_hist(peaksHist, arFitsRowM)
fill_hist(dPeaksHist,diff(arFitsRowM))

arFitsColP=array([x[0] for x in fitsColP])-rowCenter
fill_hist(peaksHist, arFitsColP)
fill_hist(dPeaksHist,diff(arFitsColP))

arFitsColM=array([x[0] for x in fitsColM])-rowCenter
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
        arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
        fill_hist(peaksHist, arFitsRowP)
        fill_hist(dPeaksHist,diff(arFitsRowP))

        arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
        fill_hist(peaksHist, arFitsRowM)
        fill_hist(dPeaksHist,diff(arFitsRowM))

        arFitsColP=array([x[0] for x in fitsColP])-rowCenter
        fill_hist(peaksHist, arFitsColP)
        fill_hist(dPeaksHist,diff(arFitsColP))

        arFitsColM=array([x[0] for x in fitsColM])-rowCenter
        fill_hist(peaksHist, arFitsColM)
        fill_hist(dPeaksHist,diff(arFitsColM))

# these peaks just fill one bin, at x.5, so I'll try fitting the slices to gaus at the peak positions, then fill with mean.
# peaksHist.Draw()
tc=TCanvas()
tc.Divide(1,2)
tc.cd(1)
dPeaksHist.Draw()
gf=dPeaksHist.Fit('gaus','QS','goff')
dMean=gf.Value(1)
dMeanEr=gf.Error(1)
dSig=gf.Value(2)
dSigEr=gf.Error(2)
tc.cd(2)
# this gets the peaks array out at the end
nFound = sCol.Search(peaksHist,3.5,'',0.03)
xsPeaks=sCol.GetPositionX()
aPeaks=rwBuf2Array(xsPeaks,nFound)
aPeaks=aPeaks[aPeaks>=firstPeak]
aPeaks=aPeaks[aPeaks <= lastPeak]
fitsPeaks=fitGausPeaks(peaksHist,aPeaks)
print fitsPeaks
print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr



# ****************************************************************************************************

# from scipy import signal
# from smooth import *
from saxsUtils import *
import cv2
from ROOT import TH1F, TSpectrum, TCanvas
from npRootUtils import *
# from root_numpy import fill_hist
from numpy import *
# agbeh/im_0005241_caz.tiff: 203 352

firstPeak=30
lastPeak=350
rowCenter=350
colCenter=200

# rowCenter=355 #xCenter
# # rowCenter=354 #xCenter
# colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
# /Users/michael/Develop/saxs/AgBeh/agbeh/im_0005241_caz.tiff
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
# im=retrieveImage('SFU/raw/latest_0000146_caz.tiff',doLog=True)
im=retrieveImage('SFU/raw/latest_0000150_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000166_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000137_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000146_caz.tiff')
rows,cols = im.shape[0:2]
xCenter=rowCenter
yCenter=colCenter
hSize=310 # half of the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0
doLogIm=True
smoothingWindow=7
if doLogIm:
    # we will smooth if we do log, so we can (hopefully) safely look at smaller peaks
    peakThresh=0.0025
else:
    peakThresh=0.005

# setTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):

peaksHist= TH1D('peaksHist','peaks',hSize*10,0,hSize)
dPeaksHist=TH1D('dPeaksHist','dPeaks',hSize,0,hSize)
sRow=TSpectrum()
sCol=TSpectrum()

row=array(im[rowCenter,:],dtype=double)
col=array(im[:,colCenter],dtype=double)
sRow.SmoothMarkov(row,len(row),smoothingWindow)
sCol.SmoothMarkov(col,len(col),smoothingWindow)

rowHist=makeTH1DFromAr1D(row,name='row',title='row')
colHist=makeTH1DFromAr1D(col,name='col',title='col')

#this updates the th1 with polies on the peaks, can see it in th1.Draw(). goff turns that off.
# nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh) 
# nFoundCol=sCol.Search(colHist,1,'goff',peakThresh)
nFoundRow=sRow.Search(rowHist,1,'',peakThresh) 
nFoundCol=sCol.Search(colHist,1,'',peakThresh)
xsRow=sRow.GetPositionX()
xsCol=sCol.GetPositionX()

axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter
# get the peaks in the range we care about and fit gaussians
axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
print 'peaks *************\n',axRowP,axRowM,axColP,axColM

# Fill the peaks histo with the means of the gaus fits
# arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
fill_hist(peaksHist, abs(axRowP))
fill_hist(dPeaksHist,diff(axRowP))

# arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
fill_hist(peaksHist, abs(axRowM))
fill_hist(dPeaksHist,diff(axRowM))

# arFitsColP=array([x[0] for x in fitsColP])-rowCenter
fill_hist(peaksHist, abs(axColP))
fill_hist(dPeaksHist,diff(axColP))

# arFitsColM=array([x[0] for x in fitsColM])-rowCenter
fill_hist(peaksHist, abs(axColM))
fill_hist(dPeaksHist,diff(axColM))
# axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
# axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
# axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
# axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
# # can now use fitGausPeaks(th,peaks) to get the peak fits of axP and axM

# # fitGausPeaks gives a list of tuples: [(const,mean,sigma),(const,mean,sigma),...]
# fitsRowP=fitGausPeaks(rowHist,axRowP)
# fitsRowM=fitGausPeaks(rowHist,axRowM)
# fitsColP=fitGausPeaks(colHist,axColP)
# fitsColM=fitGausPeaks(colHist,axColM)

# # Fill the peaks histo with the means of the gaus fits
# arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
# fill_hist(peaksHist, abs(arFitsRowP))
# fill_hist(dPeaksHist,diff(arFitsRowP))

# arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
# fill_hist(peaksHist, abs(arFitsRowM))
# fill_hist(dPeaksHist,diff(arFitsRowM))

# arFitsColP=array([x[0] for x in fitsColP])-rowCenter
# fill_hist(peaksHist, abs(arFitsColP))
# fill_hist(dPeaksHist,diff(arFitsColP))

# arFitsColM=array([x[0] for x in fitsColM])-rowCenter
# fill_hist(peaksHist, abs(arFitsColM))
# fill_hist(dPeaksHist,diff(arFitsColM))


for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((colCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=array(rim[rowCenter,:],dtype=double)
        col=array(rim[:,colCenter],dtype=double)
        sRow.SmoothMarkov(row,len(row),smoothingWindow)
        sCol.SmoothMarkov(col,len(col),smoothingWindow)


        setBinsToAr1D(rowHist,row)
        setBinsToAr1D(colHist,col)

        # nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh) 
        # nFoundCol=sCol.Search(colHist,1,'goff',peakThresh)
        nFoundRow=sRow.Search(rowHist,1,'',peakThresh) 
        nFoundCol=sCol.Search(colHist,1,'',peakThresh)


        xsRow=sRow.GetPositionX()
        xsCol=sCol.GetPositionX()

        axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
        axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter
        print axRow,axCol
        # get the peaks in the range we care about and fit gaussians. These are abs coords, not radial from beam center
        axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])#+colCenter
        axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])#+colCenter
        axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])#+rowCenter
        axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])#+rowCenter
        print 'peaks *************\n',axRowP,axRowM,axColP,axColM

        # Fill the peaks histo with the means of the gaus fits
        # arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
        fill_hist(peaksHist, abs(axRowP))
        fill_hist(dPeaksHist,diff(axRowP))

        # arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
        fill_hist(peaksHist, abs(axRowM))
        fill_hist(dPeaksHist,diff(axRowM))

        # arFitsColP=array([x[0] for x in fitsColP])-rowCenter
        fill_hist(peaksHist, abs(axColP))
        fill_hist(dPeaksHist,diff(axColP))

        # arFitsColM=array([x[0] for x in fitsColM])-rowCenter
        fill_hist(peaksHist, abs(axColM))
        fill_hist(dPeaksHist,diff(axColM))
        # can now use fitGausPeaks(th,peaks) to get the peak fits of axP and axM

        # fitGausPeaks gives a list of tuples: [(const,mean,sigma),(const,mean,sigma),...]
        # fitsRowP=fitGausPeaks(rowHist,axRowP)
        # print 'fits **************\n',fitsRowP
        # fitsRowM=fitGausPeaks(rowHist,axRowM)
        # print fitsRowM
        # fitsColP=fitGausPeaks(colHist,axColP)
        # print fitsColP
        # fitsColM=fitGausPeaks(colHist,axColM)
        # print fitsColM

        # # Fill the peaks histo with the means of the gaus fits
        # arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
        # fill_hist(peaksHist, abs(arFitsRowP))
        # fill_hist(dPeaksHist,diff(arFitsRowP))

        # arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
        # fill_hist(peaksHist, abs(arFitsRowM))
        # fill_hist(dPeaksHist,diff(arFitsRowM))

        # arFitsColP=array([x[0] for x in fitsColP])-rowCenter
        # fill_hist(peaksHist, abs(arFitsColP))
        # fill_hist(dPeaksHist,diff(arFitsColP))

        # arFitsColM=array([x[0] for x in fitsColM])-rowCenter
        # fill_hist(peaksHist, abs(arFitsColM))
        # fill_hist(dPeaksHist,diff(arFitsColM))

# these peaks just fill one bin, at x.5, so I'll try fitting the slices to gaus at the peak positions, then fill with mean.
# peaksHist.Draw()
tc=TCanvas()
tc.Divide(1,2)
tc.cd(1)
dPeaksHist.Draw()
gf=dPeaksHist.Fit('gaus','QS','goff')
dMean=gf.Value(1)
dMeanEr=gf.Error(1)
dSig=gf.Value(2)
dSigEr=gf.Error(2)
tc.cd(2)
# this gets the peaks array out at the end
nFound = sCol.Search(peaksHist,3.5,'',0.03)
xsPeaks=sCol.GetPositionX()
aPeaks=rwBuf2Array(xsPeaks,nFound)
aPeaks=aPeaks[aPeaks>=firstPeak]
aPeaks=aPeaks[aPeaks <= lastPeak]
fitsPeaks=fitGausPeaks(peaksHist,aPeaks)
print fitsPeaks
print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr






******************************************************************************************************
from saxsUtils import *
import cv2
from ROOT import TH1D, TSpectrum, TCanvas
from npRootUtils import *
# from root_numpy import fill_hist
from numpy import *
# agbeh/im_0005241_caz.tiff: 203 352

firstPeak=30
lastPeak=500
rowCenter=350
colCenter=200

# rowCenter=355 #xCenter
# # rowCenter=354 #xCenter
# colCenter=212 #yCenter
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001139_caz.tiff')
# /Users/michael/Develop/saxs/AgBeh/agbeh/im_0005241_caz.tiff
# im=retrieveImage('AgBehRingData_plus_some_more/latest_0001141_caz.tiff')
# im=retrieveImage('SFU/raw/latest_0000146_caz.tiff',doLog=True)
im0=retrieveImage('SFU/raw/latest_0000150_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000166_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000137_caz.tiff',doLog=True)
# im=retrieveImage('SFU/raw/latest_0000146_caz.tiff')
im=zeros((619,619))
im[:,66:553]=im0
rows,cols = im.shape[0:2]
colCenter+=66
xCenter=rowCenter
yCenter=colCenter
hSize=620 # size the tiff on the long dimension (rows)
nSteps=60
stepDeg=90./(nSteps+1)
d=0
doLogIm=True
smoothingWindow=7
if doLogIm:
    # we will smooth if we do log, so we can (hopefully) safely look at smaller peaks
    peakThresh=0.0025
else:
    peakThresh=0.005

# setTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):

peaksHist= TH1D('peaksHist','peaks',hSize*10,0,hSize)
dPeaksHist=TH1D('dPeaksHist','dPeaks',hSize,0,hSize)
sRow=TSpectrum()
sCol=TSpectrum()

row=array(im[rowCenter,:],dtype=double)
col=array(im[:,colCenter],dtype=double)
sRow.SmoothMarkov(row,len(row),smoothingWindow)
sCol.SmoothMarkov(col,len(col),smoothingWindow)

rowHist=makeTH1DFromAr1D(row,name='row',title='row')
colHist=makeTH1DFromAr1D(col,name='col',title='col')

#this updates the th1 with polies on the peaks, can see it in th1.Draw(). goff turns that off.
# nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh) 
# nFoundCol=sCol.Search(colHist,1,'goff',peakThresh)
nFoundRow=sRow.Search(rowHist,1,'',peakThresh) 
nFoundCol=sCol.Search(colHist,1,'',peakThresh)
xsRow=sRow.GetPositionX()
xsCol=sCol.GetPositionX()

# get arrays of peak positions, in coords relative to beam center.
axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter

# get the peaks in the range we care about and fit gaussians

# *****************************************************
# axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
# axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
# axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
# axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
# print 'peaks *************\n',axRowP,axRowM,axColP,axColM

# # Fill the peaks histo with the means of the gaus fits
# # arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
# fill_hist(peaksHist, abs(axRowP))
# fill_hist(dPeaksHist,diff(axRowP))

# # arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
# fill_hist(peaksHist, abs(axRowM))
# fill_hist(dPeaksHist,diff(axRowM))

# # arFitsColP=array([x[0] for x in fitsColP])-rowCenter
# fill_hist(peaksHist, abs(axColP))
# fill_hist(dPeaksHist,diff(axColP))

# # arFitsColM=array([x[0] for x in fitsColM])-rowCenter
# fill_hist(peaksHist, abs(axColM))
# fill_hist(dPeaksHist,diff(axColM))

# *******************************************

# we're going to fit these to gauss in the histo, so we need to go back to pix coords and not
# coords that are relative to beam center, but we need them relative to beam center in order to filter
# on the radius of the rings. So we add the center coords back after filtering.
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
arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
fill_hist(peaksHist, abs(arFitsRowP))
fill_hist(dPeaksHist,diff(arFitsRowP))

arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
fill_hist(peaksHist, abs(arFitsRowM))
fill_hist(dPeaksHist,diff(arFitsRowM))

arFitsColP=array([x[0] for x in fitsColP])-rowCenter
fill_hist(peaksHist, abs(arFitsColP))
fill_hist(dPeaksHist,diff(arFitsColP))

arFitsColM=array([x[0] for x in fitsColM])-rowCenter
fill_hist(peaksHist, abs(arFitsColM))
fill_hist(dPeaksHist,diff(arFitsColM))


for deg in arange(0,90,stepDeg):
    if deg>0.0:
        # print deg
        M = cv2.getRotationMatrix2D((colCenter,rowCenter),deg,1)
        rim=cv2.warpAffine(np.float32(im),M,(cols,rows))
        row=array(rim[rowCenter,:],dtype=double)
        col=array(rim[:,colCenter],dtype=double)
        sRow.SmoothMarkov(row,len(row),smoothingWindow)
        sCol.SmoothMarkov(col,len(col),smoothingWindow)


        setBinsToAr1D(rowHist,row)
        setBinsToAr1D(colHist,col)

        # nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh) 
        # nFoundCol=sCol.Search(colHist,1,'goff',peakThresh)
        nFoundRow=sRow.Search(rowHist,1,'',peakThresh) 
        nFoundCol=sCol.Search(colHist,1,'',peakThresh)


        xsRow=sRow.GetPositionX()
        xsCol=sCol.GetPositionX()

        axRow=rwBuf2Array(xsRow,nFoundRow)-colCenter
        axCol=rwBuf2Array(xsCol,nFoundCol)-rowCenter
        print axRow,axCol

        # ******************************
        # get the peaks in the range we care about and fit gaussians. These are abs coords, not radial from beam center
        axRowP=array([x for x in axRow if x>=firstPeak and x<=lastPeak])+colCenter
        axRowM=array([x for x in axRow if x<=-firstPeak and x>=-lastPeak])+colCenter
        axColP=array([x for x in axCol if x>=firstPeak and x<=lastPeak])+rowCenter
        axColM=array([x for x in axCol if x<=-firstPeak and x>=-lastPeak])+rowCenter
        print 'peaks *************\n',axRowP,axRowM,axColP,axColM

        # # Fill the peaks histo with the means of the gaus fits
        # # arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
        # fill_hist(peaksHist, abs(axRowP))
        # fill_hist(dPeaksHist,diff(axRowP))

        # # arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
        # fill_hist(peaksHist, abs(axRowM))
        # fill_hist(dPeaksHist,diff(axRowM))

        # # arFitsColP=array([x[0] for x in fitsColP])-rowCenter
        # fill_hist(peaksHist, abs(axColP))
        # fill_hist(dPeaksHist,diff(axColP))

        # # arFitsColM=array([x[0] for x in fitsColM])-rowCenter
        # fill_hist(peaksHist, abs(axColM))
        # fill_hist(dPeaksHist,diff(axColM))
        # # ***********************************

        # can now use fitGausPeaks(th,peaks) to get the peak fits of axP and axM

        # fitGausPeaks gives a list of tuples: [(const,mean,sigma),(const,mean,sigma),...]
        fitsRowP=fitGausPeaks(rowHist,axRowP)
        print 'fits **************\n',fitsRowP
        fitsRowM=fitGausPeaks(rowHist,axRowM)
        print fitsRowM
        fitsColP=fitGausPeaks(colHist,axColP)
        print fitsColP
        fitsColM=fitGausPeaks(colHist,axColM)
        print fitsColM

        # Fill the peaks histo with the means of the gaus fits
        arFitsRowP=array([x[0] for x in fitsRowP])-colCenter
        fill_hist(peaksHist, abs(arFitsRowP))
        fill_hist(dPeaksHist,diff(arFitsRowP))

        arFitsRowM=array([x[0] for x in fitsRowM])-colCenter
        fill_hist(peaksHist, abs(arFitsRowM))
        fill_hist(dPeaksHist,diff(arFitsRowM))

        arFitsColP=array([x[0] for x in fitsColP])-rowCenter
        fill_hist(peaksHist, abs(arFitsColP))
        fill_hist(dPeaksHist,diff(arFitsColP))

        arFitsColM=array([x[0] for x in fitsColM])-rowCenter
        fill_hist(peaksHist, abs(arFitsColM))
        fill_hist(dPeaksHist,diff(arFitsColM))

# these peaks just fill one bin, at x.5, so I'll try fitting the slices to gaus at the peak positions, then fill with mean.
# peaksHist.Draw()
tc=TCanvas()
tc.Divide(1,2)
tc.cd(1)
dPeaksHist.Draw()
dPmaxBin=dPeaksHist.GetMaximumBin()
dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
gf=dPeaksHist.Fit('gaus','QS','goff',dPmax-5,dPmax+5)
dMean=gf.Value(1)
dMeanEr=gf.Error(1)
dSig=gf.Value(2)
dSigEr=gf.Error(2)
tc.cd(2)
# this gets the peaks array out at the end
nFound = sCol.Search(peaksHist,3.5,'',0.03)
xsPeaks=sCol.GetPositionX()
aPeaks=rwBuf2Array(xsPeaks,nFound)
aPeaks=aPeaks[aPeaks>=firstPeak]
aPeaks=aPeaks[aPeaks <= lastPeak]
print aPeaks
fitsPeaks=fitGausPeaks(peaksHist,aPeaks)
print fitsPeaks
print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr



# **********************************************************
from saxsUtils import *
import cv2
from ROOT import TH1D, TH2D, TSpectrum, TCanvas,TVector2
from npRootUtils import *
# from root_numpy import fill_hist
from numpy import *

# im0=retrieveImage('SFU/raw/latest_0000150_caz.tiff',makeU8=True,doLog=True)
im0=retrieveImage('SFU/raw/latest_0000154_caz.tiff',doLog=True) # a bad gauss fit to peakshist in the end gives a wrong peak
# im0=retrieveImage('SFU/raw/latest_0000141_caz.tiff',doLog=True) # ~46.2
# im0=retrieveImage('SFU/raw/latest_0000163_caz.tiff',doLog=True) #difficult
onePeak=False#True
# im0=retrieveImage('SFU/raw/latest_0000138_caz.tiff',doLog=True) # ~23.3
# im0=retrieveImage('SFU/raw/latest_0000166_caz.tiff',doLog=True) # ~235.6
# im0=retrieveImage('SFU/raw/latest_0000155_caz.tiff',doLog=True) # ~151.9
rowCenter=350
colCenter=200
pSize=500
peakThresh=0.005
firstPeak=100
lastPeak=pSize
minDiff=10
yM,xM=im0.shape
peaksHist= TH1D('peaksHist','peaks',pSize*10,0,pSize)
dPeaksHist=TH1D('dPeaksHist','dPeaks',pSize,0,pSize)
rowHist=TH1D('rowHist','row',pSize,0,pSize)
    # TH2D(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup, Int_t nbinsy, Double_t ylow, Double_t yup)
# imPolarHist=TH2D('imPolar','im0 Polar',877*4,0,877,720,0,2*pi)
# imPolar=zeros((pSize+1,pSize+1),dtype=uint8)
imPolar=zeros((pSize+1,pSize+1))
yM,xM=im0.shape
vP=TVector2()
# smoothingWindow=5, peakThresh=.01 works ok
smoothingWindow=5
for x in range(xM):
    for y in range(yM):
        vP.SetX(x-colCenter)
        vP.SetY(y-rowCenter)
        p=vP.Phi()*pSize/(2*pi)
        r=vP.Mod()
        # p=1327./(2*pi)*p
        # imPolarHist.Fill(r,p,im0[y,x])
        # print p,r
        imPolar[round(p),round(r)]=im0[y,x]
        

# imPolarHist.Draw()
blur = cv2.GaussianBlur(imPolar,(5,5),0)

sRow=TSpectrum()


for rIdx in range(blur.shape[0]):
    # row=array(imPolar[rIdx,:],dtype=double).copy()
    row=blur[rIdx,:]
    # row=imPolar[0,:]
    # row[0:row.argmax()]=row.max()
    sRow.SmoothMarkov(row,len(row),smoothingWindow)
    setBinsToAr1D(rowHist,row)
    nFoundRow=sRow.Search(rowHist,1,'',peakThresh)
    xsRow=sRow.GetPositionX()
    axRow=rwBuf2Array(xsRow,nFoundRow)
    # axRow[0]=0.0
    axRow=array([x for x in axRow if x>=firstPeak and x<=lastPeak])
    fitsRow=fitGausPeaks(rowHist,axRow)
    
    arFitsRow=array([x[0] for x in fitsRow])
    # print 'arFitsRow ',arFitsRow
    # # arFitsRow[0]=0
    # # if len(arFitsRow)>1:

    # if arFitsRow[0]<firstPeak:
    #     arFitsRow=arFitsRow[1:]
    #     print 'arFitsRow now: ',arFitsRow
    arDiff=diff(arFitsRow)
    # print 'before: ',arDiff
    arDiff=array([x for x in arDiff if x>=minDiff])
    # print 'after: ',arDiff
    fill_hist(peaksHist, arFitsRow)
    fill_hist(dPeaksHist,arDiff)


tc=TCanvas()
tc.Divide(1,2)
tc.cd(1)
dPeaksHist.Draw()
dPmaxBin=dPeaksHist.GetMaximumBin()
dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
gf=dPeaksHist.Fit('gaus','QS','goff',dPmax-5,dPmax+5)
dMean=gf.Value(1)
dMeanEr=gf.Error(1)
dSig=gf.Value(2)
dSigEr=gf.Error(2)
tc.cd(2)
# this gets the peaks array out at the end
peaksHist.Smooth()
nFound = sRow.Search(peaksHist,3.5,'',0.1)
peaksHist.Draw()
xsPeaks=sRow.GetPositionX()
aPeaks=rwBuf2Array(xsPeaks,nFound)
aPeaks=aPeaks[aPeaks>=firstPeak]
aPeaks=aPeaks[aPeaks <= lastPeak]
print aPeaks
fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)

print fitsPeaks
if len(aPeaks)==1:
    # (mean,sigma,errMean,errSig)
    nFound = sRow.Search(peaksHist,3.5,'',0.1)
    xsPeaks=sRow.GetPositionX()
    aPeaks=rwBuf2Array(xsPeaks,nFound)
    fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)
    dMean=fitsPeaks[0][0]
    dMeanEr=fitsPeaks[0][2]
    dSig=fitsPeaks[0][1]
    dSigEr=fitsPeaks[0][3]
print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr








# kernel = ones((5,5),np.float32)/25
# dst = cv2.filter2D(imPolar,-1,kernel)
# blur = cv2.GaussianBlur(imPolar,(5,5),0)

# # gray = cv2.cvtColor(imPolar,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(imPolar,50,150,apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     print x0,y0
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))

#     cv2.line(imPolar,(x1,y1),(x2,y2),(0,0,255),2)

# # plt.subplot(121),plt.imshow(img),plt.title('Original')
# # plt.xticks([]), plt.yticks([])
# # plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

X
out = cv2.linearPolar(im0, center, self.iris_radius, cv2.WARP_FILL_OUTLIERS)
imp=cv2.linearPolar(im0, (350,200), 500, cv2.WARP_FILL_OUTLIERS)
cv.LogPolar(im0,imp,(350,200),100,cv.CV_WARP_FILL_OUTLIERS)

for x in range(xM):
    vP.SetX(x-colCenter)
    for y in range(yM):
        
        vP.SetY(y-rowCenter)
        p=vP.Phi()*pSize/(2*pi) # arctan2(y,x)*pSize/(2*pi)
        r=vP.Mod() # (x**2+y**2)**.5
        # p=1327./(2*pi)*p
        # imPolarHist.Fill(r,p,im0[y,x])
        # print p,r
        imPolar[round(p),round(r)]+=im0[y,x]
                
# yi=arange(486,-1,-1)
# X
# xi=arange(618,-1,-1)      
# Xi,Yi=meshgrid(xi,yi)

X,Y=indices(im0.shape)
Xc=X-350
Yc=Y-200
r=around(((Xc)**2+(Yc)**2)**.5)

at3=arctan2(Yc,Xc)
imshow(at3)
at3
at3[at3<0]+=2*pi
at3
imshow(at3)
at3*=500/(2*pi)
r=r.astype(int)
at3=at3.astype(int)

imp[at3,r]=im0

r2=r.flatten()
at2=at3.flatten()
im0f=im0.flatten()
map(lambda x,y,z:z[x][y], at2,r2,im0f)


impp=zeros((amax(at3)+1,amax(r)+1))
impp[at3,r]=im0
imshow(impp)


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


from saxsUtils import *
# import cv2
# from ROOT import TH1D, TH2D, TSpectrum, TCanvas,TVector2
# from npRootUtils import *
# from root_numpy import fill_hist
# from numpy import *
from polarize import polarize


# im0=retrieveImage('SFU/raw/latest_0000150_caz.tiff',makeU8=True,doLog=True)
im0=retrieveImage('SFU/raw/latest_0000154_caz.tiff',doLog=True) # a bad gauss fit to peakshist in the end gives a wrong peak
# im0=retrieveImage('SFU/raw/latest_0000141_caz.tiff',doLog=True) # ~46.2
# im0=retrieveImage('SFU/raw/latest_0000163_caz.tiff',doLog=True) #difficult
onePeak=False#True
# im0=retrieveImage('SFU/raw/latest_0000138_caz.tiff',doLog=True) # ~23.3
# im0=retrieveImage('SFU/raw/latest_0000166_caz.tiff',doLog=True) # ~235.6
# im0=retrieveImage('SFU/raw/latest_0000155_caz.tiff',doLog=True) # ~151.9
rowCenter=350
colCenter=200
pSize=90
X,Y=indices(im0.shape)
Xc=X-rowCenter
Yc=Y-colCenter
r=around(((Xc)**2+(Yc)**2)**.5)

at3=arctan2(Yc,Xc)
# imshow(at3)
at3
at3[at3<0]+=2*pi
at3
# imshow(at3)
at3*=pSize/(2*pi)
r=r.astype(int)
at3=at3.astype(int)

# imp[at3,r]=im0       

imPolar=zeros((amax(at3)+1,amax(r)+1))
# imPolar[at3,r]=im0
# do this to get the fortran indexing right
# r+=1
# at3+=1
# polarize(im0,at3,r,imPolar)
imPolar = polarize(im0,at3,r,imPolar)
# imp = polarize(im0,at3,r,imp,[n1,n2,rmax,tmax,overwrite_imp])
