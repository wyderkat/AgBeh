#!/usr/bin/python
"""
This program is a demonstration of ellipse fitting.

Trackbar controls threshold parameter.

Gray lines are contours.  Colored lines are fit ellipses.

Original C implementation by:  Denis Burenkov.
Python implementation by: Roman Stanchak, James Bowman
"""

import sys
import urllib2
import random
import cv2.cv as cv
import video
import cv2
from math import sqrt, pi


def contour_iterator(contour):
    while contour:
        yield contour
        contour = contour.h_next()
class EllipseStruct:
    def __init__(self,center,size,angle):
        self.center=center
        self.size=size
        self.angle=angle
        self.area=self.size[0]*self.size[1]*pi
    def __str__(self):
        s='EllipseStruct -> (center: '+str(self.center)+', size: '+str(self.size)+', angle: '+str(self.angle)+', area: '+str(self.area)+')'
        # s='abcd'
        # print s
        return s
    # def __repr__(self):
    #     return repr

class FitEllipse:

    def __init__(self, source_image, slider_pos):
        self.source_image = source_image
        self.slider_pos=slider_pos
        cv.CreateTrackbar("Threshold", "Result", self.slider_pos, 255, self.process_image)
        self.stor = cv.CreateMemStorage()
        # self.process_image(self.slider_pos)

    def process_image(self,slider_pos):
        """
        This function finds contours, draws them and their approximation by ellipses.
        """
        self.slider_pos=slider_pos
        # stor = cv.CreateMemStorage()

        # Create the destination images
        image02 = cv.CloneImage(self.source_image)
        compImg=cv.CloneImage(self.source_image);
        cv.Zero(image02)
        image04 = cv.CreateImage(cv.GetSize(self.source_image), cv.IPL_DEPTH_8U, 3)
        cv.Zero(image04)

        # Threshold the source image. This needful for cv.FindContours().
        cv.Threshold(self.source_image, image02, self.slider_pos, 255, cv.CV_THRESH_BINARY)

        # Find all contours.
        cont = cv.FindContours(image02,
            self.stor,
            cv.CV_RETR_LIST,
            cv.CV_CHAIN_APPROX_NONE,
            (0, 0))
        elist=[]
        for c in contour_iterator(cont):
            # Number of points must be more than or equal to 6 for cv.FitEllipse2
            if len(c) >= 6:
                # Copy the contour into an array of (x,y)s
                PointArray2D32f = cv.CreateMat(1, len(c), cv.CV_32FC2)
                for (i, (x, y)) in enumerate(c):
                    PointArray2D32f[0, i] = (x, y)

                # Draw the current contour in gray
                gray = cv.CV_RGB(100, 100, 100)
                cv.DrawContours(image04, c, gray, gray,0,1,8,(0,0))

                # Fits ellipse to current contour.
                (center, size, angle) = cv.FitEllipse2(PointArray2D32f)

                # Convert ellipse data from float to integer representation.
                center = (cv.Round(center[0]), cv.Round(center[1]))
                size = (cv.Round(size[0] * 0.5), cv.Round(size[1] * 0.5))
                elist.append(EllipseStruct(center, size, angle))
                # Draw ellipse in random color
        color = cv.CV_RGB(200,0,200)

        #  though that the second biggest was best for getting the mouth each time, but not reliable enough.        #
                # elist=sorted(elist, key=lambda EllipseStruct: EllipseStruct.size)
                # # for el in elist:
                # #     print el.size
                # print elist[-2]

        # now we will try getting the closest to center of mouth
        elist=sorted(elist, key=lambda EllipseStruct: EllipseStruct.size)
        avCenter=(94,54)
        iEl=0
        d0=200 #start out bigger than anything else
        for idx in range(len(elist)-1):
            # get eucl distance from avCenter and take closest
            d= sqrt((avCenter[0]-elist[idx].center[0])**2 + (avCenter[1]-elist[idx].center[1])**2)
            # print d
            if d<d0 and elist[idx].size[1]>30:  # the size condition is just empirical - I was getting little ellipses on the theeth
                                                # sometimes just using distance to ave center of the ellipses I wanted.
                iEl=idx
                d0=d
        print d0
        print iEl
        print elist[iEl]
        print self.slider_pos
        print
        cv.Ellipse(compImg, elist[iEl].center, elist[iEl].size,
              elist[iEl].angle, 0, 360,
              color, 2, cv.CV_AA, 0)

        # Show image. HighGUI use.
        self.el=elist[iEl]
        cv.ShowImage( "Result", compImg )



if __name__ == '__main__':
    if len(sys.argv) > 1:
        # try:
        fn = sys.argv[1]
        cap = video.create_capture(fn)
        flag = True
        flag, img = cap.read()
        ims=[]
        # imsg=[]
        outFile=open('ellipses_V2N3_pinS12-cro-new.dat','w')
        outFile.write('Frame\ttime\tsize_x\tsize_y\tarea\n')
        while flag:
            imgg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bitmap = cv.CreateImageHeader((imgg.shape[1], imgg.shape[0]), cv.IPL_DEPTH_8U, 1)
            cv.SetData(bitmap, imgg.tostring(), imgg.dtype.itemsize * 1 * imgg.shape[1])
            ims.append(bitmap)
            # imsg.append(imgg)
            flag, img = cap.read()
            print 'len ims: '+str(len(ims))
        # except:
        #       return
        # source = cv2.imread() # source is numpy array
        #         bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
        #         cv.SetData(bitmap, source.tostring(),
        #                    source.dtype.itemsize * 3 * source.shape[1])
        #         bitmap here is cv2.cv.iplimage


        # source_image = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        url = 'http://code.opencv.org/projects/opencv/repository/revisions/master/raw/samples/c/stuff.jpg'
        filedata = urllib2.urlopen(url).read()
        imagefiledata = cv.CreateMatHeader(1, len(filedata), cv.CV_8UC1)
        cv.SetData(imagefiledata, filedata, len(filedata))
        source_image = cv.DecodeImage(imagefiledata, cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Create windows.
    # cv.NamedWindow("Source", 1)
    cv.NamedWindow("Result", 1)
    # cv2.namedWindow('gr')
    # Show the image.
    idx=0
    a=0
    maximg=len(ims)
    print 'maximg: '+str(maximg)
    slider_pos=67
    fe=FitEllipse(ims[idx], slider_pos)
    # for idx in range(len(ims)):
    while a!=27:
        # img=cv2.cvtColor(ims[idx], cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gr', imsg[idx])
        # cv.ShowImage("Source", ims[idx])
        # cv2.imshow("Source", img)

        fe.source_image = ims[idx]
        fe.process_image(slider_pos)
        a=cv.WaitKey(0)
        # turns out waitkey returns all kinds of junk in the upper bits, so we strip it out
        a = a & 255 if a + 1 else -1;
        # a=cv2.waitKey(0)
        print a

        # 'Frame\ttime\tsize_x\tsize_y\tarea\n'
        outstring = '%i\t%f\t%i\t%i\t%f\n' % (idx,idx/25.0,fe.el.size[1],fe.el.size[0],fe.el.area/500.0)
        outFile.write(outstring)
        if sys.platform=='linux2': #linux r and l arrrows
            nxt=93 #46
            prev=91 #44
        else: #osx r and l arrows
            nxt=3
            prev=2
        if a == nxt:
            idx+=1
            idx%=maximg
            # fe.slider_pos-=1
            # fe.slider_pos%=255
            print a
        elif a == prev:
            idx-=1
            idx%=maximg
            # fe.slider_pos+=1
            # fe.slider_pos%=255
            print a

        slider_pos=fe.slider_pos
            # if idx>0:
                # idx-=1
        # right arrow=63235
        # left arrow = 63234
        # print a
    # print "Press any key to exit"
    # cv.WaitKey(0)
    outFile.close()
    # cv.DestroyWindow("Source")
    cv.DestroyWindow("Result")
