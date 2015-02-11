#!/usr/bin/env python
# encoding: utf-8
"""
cvCircle16.py

using edge in the circle finder, adjust dp

Created by michael on 2014-11-29.

"""

import cv2


import numpy as np

import sys
import os

from saxsUtils import *
 
def main(fn):
    # will need globals for this, since the cv2 GUI callbacks don't work right in python (they don't take an args that we have control over)
    # global image
    global fileName
    fileName=fn
    # try:
    #     tif=TIFF.open(fileName)
    #     image = tif.read_image() # this is int32
    #     # print 'image: ', type(image),' of: ',image.dtype, ' of shape: ', image.shape
    # except TypeError: # TIFF.open returns a TypeError when it cant read a file
    #     print 'Could not open',fileName
    #     return
    # print 'Image type: ', image.dtype,' Image name: ',fileName 
    hu = draw()
    return hu
# gui callbacks
def thrs(*arg): 

    draw()
def checkAperture(*arg):
    if arg[0]<=0:
        cv2.setTrackbarPos('CannyAperture', 'ctrl',1)

    # apertureSize must be odd
    if not arg[0]%2:
        apSize=arg[0]+1
        cv2.setTrackbarPos('CannyAperture', 'ctrl',apSize)
    draw()

def checkDP(*arg):
    if arg[0]<=0:
        cv2.setTrackbarPos('erodekern', 'ctrl',1)
    draw()

def checkp1(*arg):

    if arg[0]<=0:
        cv2.setTrackbarPos('p1', 'ctrl',1)
    draw()

def checkp2(*arg):
    if arg[0]<=0:
        cv2.setTrackbarPos('p2', 'ctrl',1)
    draw()
def checkROI(*arg):
    draw()
def checkScale(*arg):
    # if arg[0]<=0:
    #     cv2.setTrackbarPos('Scale', 'ctrl',1)

    sc =cv2.getTrackbarPos('Scale','ctrl')
    psc=cv2.getTrackbarPos('PreScale','ctrl')
    if psc<1:
        cv2.setTrackbarPos('PreScale', 'ctrl',1)
    # if prescale>1, then make sure scale=1
    
    if sc<1 or psc>1:
        cv2.setTrackbarPos('Scale', 'ctrl',1)
    draw()


def doKeyStroke(fileList,fileIdx):
    # deal with inconsistencies across implementations
    if sys.platform=='darwin':
        nextFileKey=3 # -> 
        prevFileKey=2 # <-
    else:
        nextFileKey=83 #110 # n
        prevFileKey=81 #112 # p

    # get a keystroke
    ch = cv2.waitKey(0)
    # turns out waitkey returns all kinds of unpredictable junk in the upper bits, so we strip it out
    ch = ch & 255 if ch + 1 else -1;
    if ch == 27: # pressed esc
        return -1
    
    if ch == nextFileKey:
        if fileIdx<len(fileList)-1:
            fileIdx+=1
    
    if ch == prevFileKey:
        if fileIdx>0:
            fileIdx-=1
    # print 'ch:',ch
    return fileIdx
    

def draw():
    global fileName
    print '\n',fileName
    # retrieve the adjustable values from the GUI
    th = cv2.getTrackbarPos('thrs1', 'ctrl')
    et1= cv2.getTrackbarPos('edgeT1','ctrl')
    et2= cv2.getTrackbarPos('edgeT2','ctrl')
    erodekern= cv2.getTrackbarPos('erodekern','ctrl')
  
  

    apSize=5#cv2.getTrackbarPos('CannyAperture', 'ctrl')

                # pass cols/2 and rows/2, so our rotated image comes back with with even cols and rows)
                
    
    
    im = retrieveImage(fileName)
    
    tName='threshold:'+str(th)
    ret,thresh = cv2.threshold(im,th,255,0)
    print 'lum: ',lum(thresh)
    
    if erodekern>1:
        kernel = np.ones((erodekern,erodekern),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        erodeText='kernel: '+str(erodekern)
    else:
        erodeText='No erosion applied (kern size <2)'
        erosion=thresh.copy()
    edge = cv2.Canny(erosion, et1, et2, apertureSize=apSize)
    erEdge=erosion.copy()
    erEdge[:]=0
    erEdge[edge != 0] = 255
    # ecnt=cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR)
    # ecnt[:,:,0]=erEdge
    
    
    # print erosion.shape,lum(erosion)
    # NOTE: findContours modifies the input image.
    contours, hierarchy = cv2.findContours(erosion.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    

    tc=cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR)
    # print erosion.shape,lum(erosion)
    zc=np.zeros_like(tc)
    cv2.drawContours(zc, contours, -1, (0,255,0), 1)
    cv2.imshow('contours',zc)

    cv2.putText(thresh, basename(fileName), (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255))
    cv2.putText(thresh, tName, (5, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255))
    cv2.imshow('thresh',thresh)
    cv2.putText(erEdge, 'et1,et2: '+str(et1)+','+str(et2), (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255))
    cv2.imshow('edge',erEdge)
    cv2.putText(erosion, erodeText, (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255,255,255))
    cv2.imshow('erosion',erosion)
    
    return #hu
    
if __name__ == '__main__':
    # set up GUI
    

    cv2.namedWindow('thresh')
    
    cv2.namedWindow('edge')
    cv2.namedWindow('contours')
    cv2.namedWindow('erosion')
    cv2.namedWindow('ctrl')
    # cv2.createTrackbar('lTarg', 'ctrl', 3, 10, thrs)
    cv2.createTrackbar('erodekern', 'ctrl', 2, 10, checkDP)
    cv2.createTrackbar('edgeT1', 'ctrl', 5000, 10000, thrs)
    cv2.createTrackbar('edgeT2', 'ctrl', 100, 10000, thrs)

    # cv2.createTrackbar('dp', 'ctrl',14, 20, checkDP) # image resolution/accumulator resolution
    # cv2.createTrackbar('tile', 'ctrl',2, 3, rotate)
    cv2.createTrackbar('thrs1', 'ctrl',30, 255, thrs)
    # cv2.createTrackbar('lTargCirclce', 'ctrl',20, 30, thrs)
    # cv2.createTrackbar('p1', 'ctrl', 1, 10, checkp1)
    # cv2.createTrackbar('p2', 'ctrl', 8, 30, checkp2)
    # cv2.createTrackbar('ROI', 'ctrl', 80, 480, checkROI) # if 0, use whole image, image is 619x487
    # cv2.createTrackbar('Scale', 'ctrl',1, 10, checkScale) # not allowed to be 0
    # cv2.createTrackbar('PreScale', 'ctrl',5, 10, checkScale)
    # cv2.createTrackbar('CannyAperture', 'ctrl',5, 7, checkAperture)
    # cv2.createTrackbar('blotter', 'ctrl',0, 25, checkScale)
    # get the file list
    fl=os.listdir(sys.argv[1])
    tiffDir=sys.argv[1]+'/'
    fileList=[tiffDir+x for x in fl if x.split('.')[-1]=='tiff']

    # set file to first
    fileIdx=0
    fn=fileList[fileIdx]
    # page through files and fit/draw
    # hus=[]
    while 1:
        # hus.append(main(fn))
        main(fn)
        fileIdx=doKeyStroke(fileList,fileIdx)
        if fileIdx<0:
            break

        fn=fileList[fileIdx]
        
    # tschuess
    # for hu in hus:
    #     for idx in range(7):
    #         print '%i: %5.3e' %(idx+1,hu[idx]),
    #     print
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
