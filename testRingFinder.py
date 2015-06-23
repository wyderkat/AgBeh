#!/usr/bin/env python
# encoding: utf-8

import cv2
import sys
import os
from agbehPeakFinder_SL2 import *
from numpy import *

def main(argv=sys.argv):

    cv2.namedWindow('Pol')
    cv2.namedWindow('Log')

    fl=os.listdir(sys.argv[1])
    tiffDir=sys.argv[1]+'/'
    fileList=[tiffDir+x for x in fl if x.split('.')[-1]=='tiff']
    center=(argv[2],argv[3])
    # set file to first
    fileIdx=0
    fn=fileList[fileIdx]
    y=int(center[0])
    x=int(center[1])
    # page through files and fit/draw
    # hus=[]
    while 1:
        # hus.append(main(fn))
        pX=findPeaks(fn,center,firstPeak=20,verbose=True)
        # pX=findPeaks(fn,center,verbose=True)
        if pX:
            p=pX[0:5]
            im0=colorize(pX[6] )
            imPolar=colorize(pX[5])
            print fn
            print p
            cv2.rectangle(im0,(x,y),(x,y),(0,255,70),-1)
            circs=[c[0] for c in p[4]]
            for c in circs:
                if c>0:
                    cv2.circle(im0,(x,y),int(round(c)),(0,255,70),1)
            cv2.imshow('Pol', imPolar)
            cv2.imshow('Log', im0)
        else:
            print 'Nothing found for ',fn
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
def colorize(inPic):
    # print 'inPic.shape',inPic.shape,'\tlen:',len(inPic.shape)
    if len(inPic.shape)==3:
        outPic=inPic.copy()
        inPicGray=cv2.cvtColor(inPic, cv2.COLOR_RGB2GRAY)
    else:

        inPicGray=inPic.copy() 
        inPicGray[inPicGray<0]=0 # this is needed for uint8
        imMax=amax(inPicGray)
        im8=array(inPicGray/float(imMax)*255,dtype='uint8')
        outPic=cv2.cvtColor(im8, cv2.COLOR_GRAY2RGB)
    # print 'inPicGray.shape',inPicGray.shape,'\outPic:',outPic.shape
    
    outPic[:,:,0]=inPicGray
    outPic[:,:,1]=255
    inPicGC=inPicGray.copy()
    inPicGC[inPicGray<1]=0
    inPicGC[inPicGray>=1]=255
    outPic[:,:,2]=inPicGC.copy() 
    outPic=cv2.cvtColor(outPic,cv2.COLOR_HSV2RGB)
    return outPic

if __name__ == '__main__':
    main()

    

