#!/usr/bin/env python
# encoding: utf-8
"""
doc string here
"""

import sys
import os
import cv2

# from libtiff import TIFF
import numpy as np
from pilatus_np import JJTiff

def kwa():
    cv2.destroyAllWindows()

def basename(path,sep='/'):
    return path.split(sep)[-1]

def colorize(inPic):
    # return a colorized 
    if len(inPic.shape)==3:
        outPic=inPic.copy()
        inPicGray=cv2.cvtColor(inPic, cv2.COLOR_RGB2GRAY)
    else:
        inPicGray=inPic.copy() 
        outPic=cv2.cvtColor(inPic, cv2.COLOR_GRAY2RGB)
    # print 'inPicGray.shape',inPicGray.shape,'\outPic:',outPic.shape
    
    outPic[:,:,0]=inPicGray
    outPic[:,:,1]=255
    inPicGC=inPicGray.copy()
    inPicGC[inPicGray<1]=0
    inPicGC[inPicGray>1]=255
    outPic[:,:,2]=inPicGC.copy() 
    outPic=cv2.cvtColor(outPic,cv2.COLOR_HSV2RGB)
    return outPic
def rotIm(im):
    # get the rotation in deg from the slider, return im rotated by rotDeg. np array.
    
    # rotDeg = cv2.getTrackbarPos('rotDeg', 'ctrl')*90
    rows,cols = im.shape
    # getRotationMatrix2D(center, angle, scale)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    return cv2.warpAffine(im,M,(cols,rows))

# do a cross correlation of rotdImage and im. the last 4 args are window size.
# TODO: make last 4 args optional, take the size from the shape of inputs arrays.
def crossCorr(rotdImage, im,windowx,windowy,hwx,hwy):
    
    # if !windowx:

    # take rotIm and im, do a cross correlation (a convolutoin, but with x+t instead of x-t), and return the coord of the max.
    #  2d np arrays in, tuple out (xmax, ymax)
    # cv.MatchTemplate(image, templ, result, method) 
    #  If image is W x H and templ is w x h , then result is (W-w+1) x (H-h+1) -> image dims must be bigger than template dims.
    #
    # we have to do our own zero padding for the convolution
    
    rotImagePadded=np.zeros((windowx*2,windowy*2),dtype=rotdImage.dtype)
    try:
        rotImagePadded[windowx/2:3*windowx/2,windowy/2:3*windowy/2]=rotdImage
    except:
        print 'couldn''t fit the slice of ', fileName
        print 'shape of rotImagePadded: ',rotImagePadded.shape
        print 'shape of dst: ', rotdImage.shape
        print 'slice shape: ',windowx/2,3*windowx/2,windowy/2,3*windowy/2
        
        return
    # do the correlation

    cResult=cv2.matchTemplate(np.float32(rotImagePadded), np.float32(im), cv2.cv.CV_TM_CCORR)


    # find the offset
    rMinMax=cv2.minMaxLoc(cResult) #Out[160]: (6227.98291015625, 4536606.0, (50, 0), (25, 25))
    # print rMinMax
    offset=(rMinMax[3][0]-hwx,rMinMax[3][1]-hwy)

    # print offset

    # make ave of offset rotated and imS8
    rotOffsetSlice=rotImagePadded[windowx/2+offset[1]:3*windowx/2+offset[1],windowy/2+offset[0]:3*windowy/2+offset[0]]


    cResult=np.array(cResult/float(np.amax(cResult))*255,dtype='uint8')
    # print cResult.shape,rotOffsetSlice.shape
    return rotOffsetSlice,offset,cResult[0:-1,0:-1]

# retrieve image from file system and convert to uint8
# retrieve image from file system and convert to uint8 if desired
def retrieveImage(filePath,clearVoids=False,makeU8=False,doLog=False):
    # return a np array (2dim) of a saxslab tiff file

    try:
        
        jjIm=JJTiff(filePath,bars = 0.0, nega = 0.0)
        image=np.rot90(np.fliplr(jjIm.arr))
        # print 'image: ', type(image),' of: ',image.dtype, ' of shape: ', image.shape
    except TypeError: # TIFF.open returns a TypeError when it can't read a file
        print 'Could not open',filePath
        return
    # -1, -2, are 255, 254 once we make them uint8, so let's just make the < 0 pix be 0.
    if clearVoids:
        image[image<0]=0
    if doLog:
        image[image<1]=1
        image=np.log(image)
    if makeU8:
        image[image<0]=0 # this is needed for uint8
        imMax=np.amax(image)
        im8=np.array(image/float(imMax)*255,dtype='uint8')
        return im8
    else:
        return image


def u8Image(image):
    image[image<0]=0
    imMax=np.amax(image)
    im8=np.array(image/float(imMax)*255,dtype='uint8')
    return im8

def takeROISlice(im,size,center,maxY=487, maxX=580):
#   im:     original image (wxh)
#   size:   size of (square) ROI
#   center: tuple or list, center of ROI, on im (row from top,col from left)
# returns:  A square slice on input image (im), with dise equal to size. If the slice 
#           would go over the edge of im, then it gets cut off there.
# TODO:     Instead of returning a truncted slice if ROI would go over edge, pad with zeros and
#           return that slice, keeping proper size.
    ROI=size
    ROI=ROI/2*2 # needs to be even, otherwise we get messed up when we take slices 
                # (because when we make rotation matrix, we have to 
                # pass cols/2 and rows/2, so our rotated image comes back with with even cols and rows)
    
    # image is 619x487: x (rows, u/d),y (cols, l/r) - so col major order in memory (FORTRAN origins of numpy?)
    

    # ++ Take the ROI slice

    # USE centroid of image to choose ROI center
    # mx-> row (up down pos), my->col (lr pos)
    mx,my=center
    
    hw=ROI/2
    if mx-hw<0:
        mx=hw

    if my-hw<0:
        my=hw

    if mx+hw>=maxX:
        mx=maxX-1-hw

    if my+hw>=maxY:
        my=maxY-1-hw

    
    xi=[mx-hw,mx+hw,my-hw,my+hw]    
    # actual image slice
    return im[xi[0]:xi[1],xi[2]:xi[3]]
def makeBinaryImage(im,thresh):
# We expect a Uint8 image input.
    outIm=im.copy()
    if thresh>0:
        outIm[im<thresh]=0
        outIm[im>=thresh]=255
    return outIm

def lum (im):

    return float(im.sum())/im.size

def adjustLuminosity(im,lTarg,lThrs=100,eps=5):
#   take input image (im), and return a binary image with the threshold
#   adjusted such that the ave luminosity is as close to lTarg as possible.
#   As it is, we usually get the number above the thresh when it starts above,
  # and we get the number below if it starts low (this is dependent on eps)
    
    imOut=makeBinaryImage(im,lThrs)
    luminosity=float(im.sum())/im.size
    # print 'luminosity:',luminosity,'\tlTarg:',lTarg
    setLq=False
    while luminosity>lTarg+eps and lThrs<255 and lTarg>0: 
        lThrs+=1
        imOut=makeBinaryImage(im,lThrs)
        luminosity=float(imOut.sum())/imOut.size
        setLq=True

    while luminosity<lTarg-eps and lThrs>1 and setLq != True and lTarg>0:

        lThrs-=1
        imOut=makeBinaryImage(im,lThrs)
        luminosity=float(imOut.sum())/imOut.size

    # print 'luminosity:',luminosity,'\tlThrs:',lThrs,'lTarg',lTarg
    return imOut
    