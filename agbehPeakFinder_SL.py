#!/usr/bin/env python
# encoding: utf-8
import cv2
from ROOT import TH1D, TSpectrum
from npRootUtils import *
from numpy import *
import sys
from pilatus_np import JJTiff
from polarize import polarize

def findPeaks(image,center,peakThresh=0.05,verbose=False,doLogIm=True,pSize=90,firstPeak=20,lastPeak=None,smoothingWindow=13,minDiff=20,difThresh=70,maxNPeaks=5):
    """

    findPeaks(center,image[,kwargs...])

    Take a saxslab tiff of AgBeh, and return peak spacing, and an array of found peak coordinates (radius from center).

    Peaks in the radial range (firstPeak:lastPeak) are found by first unrolling the image into polar coordinates. We then
    iterate down the image by rows and do a rough peak search on each row. The peak coordinates from this search are then
    fed to a function that separately fits a small range of a row, centered on each peak coordinate, to a gaussian
    curve, and the mean from each fit is added into a histogram. This process is repeated for each row in polar space.

    Also, the spacing between all the peaks in a row is added to a histogram of peak spacings.

    At the end, we have a histogram of peak locations and a histogram of peak spacings. The peak spacing
    histogram is fit to a gaussian, and the mean, sigma, and error are returned.

    Further, the histogram of peak locations is treated similarly to a row in the first step, and it's peaks
    are again fit to gaussians, and the mean, sigma, and error of each are stored in a list of tuples in the output.

     input
     image:         A path to a saxslab tiff, or an np array of a saxslab tiff.
    
     center:        tuple or list -> (rowCenter,colCenter), where rowCenter is the coord in pix of the center row.
                                                   colCenter is defined the same way, but for center col.
     peakThresh:    Parameter used for peak finding. This is the min acceptable ratio of a peak's height to the height
                    of the largest peak, in order to be counted as a peak by the peak finder.

     verbose        Control the level of output from this function. Setting this to false will cause the function to
                    supress any output to the screen. Setting to true will print a small report upon copletion, and also
                    return the histograms of peak locations and spacings.

     doLogIm:       Whether or not to work with the log of the input image.

     pSize:         How many lines in the phi direction will we use in polar space?

     firstPeak:     Min radius of a peak to be considered in the calculations.

     lastPeak:      Max radius of a peak to be considered in the calculations. Set to None go all the way out to edge of
                    image.

     smoothingWindow: How many pixels to use in the smoothing algorithm.

     minDiff:       Min distance of neighboring peaks to consider in the peak spacing calculations.

     difThresh:     Threshold of ave peak distances. If the ave is above this, then we auto-adjust firstPeak=50 in
                    order to avoid false peaks near the beamstop.

     maxNPeaks:     How many peaks out from the center to we use (default is 5).
     
     output
     tuple:        (peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr, peaksList, imPolar, im0)
                    
                   peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr are from fitting a histogram
                   of found peak spacingss to a gaussian. 

                   peaksList is a list of tuples (peakCenter,sig,errPeakCenter,errSig), as radius from center.
                   imPolar is the polar representation of the input image that is used for the peak finding.
                   im0 is the original image passed in.

                   If verbose was set to true, the output will be:

                   (peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr, eaksList, imPolar, im00, peaksHistAr, dPeaksHistAr)

                   With the last two items in the output tuple are the peak location histogram, and the peak spacing
                   histogram as tuples of numpy arrays, where the [0]th element is the data and the [1]st is the axis.

     Returns None on failure.
    """
    try:

        # determine if image is an image or a path:
        if type(image)==str:
            image=retrieveImage(image,doLog=doLogIm)

        im0=image




        rowCenter=float(center[0])# 350 for the sample set of images
        colCenter=float(center[1])# 200
        peakThresh=float(peakThresh)
        pSize=int(pSize)
        
        firstPeak=int(firstPeak)
        smoothingWindow=int(smoothingWindow)
        minDiff=int(minDiff)
        yM,xM=im0.shape

  
        # unroll into polar coordinates
        X,Y=indices(im0.shape)
        Xc=X-rowCenter
        Yc=Y-colCenter
        r=around(((Xc)**2+(Yc)**2)**.5)

        at3=arctan2(Yc,Xc)
        # imshow(at3)
        # convert angles < 0 to positive
        at3[at3<0]+=2*pi
        # imshow(at3)
        at3*=pSize/(2*pi)
        r=r.astype(int)
        at3=at3.astype(int)

        # imp[at3,r]=im0 
        rSize=amax(r)+1
        if not lastPeak:
            lastPeak=int(rSize)
        else:
            lastPeak=int(lastPeak)   

        # Init the histos, now that we know how big to make them.
        peaksHist= TH1D('peaksHist','peaks',rSize*10,0,rSize)
        prePeaksHist= TH1D('prePeaksHist','prePeaksHisteaks',rSize*10,0,rSize)
        dPeaksHist=TH1D('dPeaksHist','dPeaks',rSize,0,rSize)
        rowHist=TH1D('rowHist','row',rSize,0,rSize)

        # allocate the polar image
        imPolar=zeros((amax(at3)+1,rSize))
        # Straight up broadcasting in numpy doesn't do += properly: you just get the last value that mapped to the new coords. So we lose info.
        # imPolar[at3,r]+=im0

        # This one I wrote in Fortran (just because it's really easy to compile fortran modules to work with numpy), it does the proper +=, and it's full speed.
        imPolar = polarize(im0,at3,r,imPolar)
        

        # run a gaus filter over the polar image to blend in the rough spots
        blur = cv2.GaussianBlur(imPolar,(3,3),0)

        sRow=TSpectrum()

        # first pass - roughly find all the peaks and make a histo.
        for rIdx in range(blur.shape[0]):#[1:]:
            
            row=blur[rIdx,:]

            sRow.SmoothMarkov(row,len(row),smoothingWindow)
            setBinsToAr1D(rowHist,row)
            nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh)
            xsRow=sRow.GetPositionX()
            axRow=rwBuf2Array(xsRow,nFoundRow)
            # axRow[0]=0.0
            axRow=array([x for x in axRow if x>=firstPeak and x<=lastPeak])
            fill_hist(prePeaksHist, axRow)

        
        # clean out the noise in our rough estimate of where to look for peaks
        peaksHistAr=setAr1DtoBins(prePeaksHist)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow) # second smoothing kills some outer rings
                                                                              # but the trade off is false positive near 
                                                                              # beam center in the farther-out detector
                                                                              # displacements. 
        setBinsToAr1D(prePeaksHist,peaksHistAr[0])

        # look for peaks and get the gauss fits - we use this instead of the peaks found from sRow.Search
        # bacause sRow.Search can sometimes return multiple peaks that are very close together. If we do guass
        # fits on two close together peaks, we should find the same center for both, and we can then filter them out,
        # keeping only the unique entries.

        # get a list of peaks in our rough peak histo
        nFound = sRow.Search(prePeaksHist,0.33,'goff',0.025)
        if verbose:
          print nFound
        # prePeaksHist.Draw()
        xsPeaks=sRow.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)

        # get the gauss fits and filter for the unique peaks
        fitsPeaks=fitGausPeaks(prePeaksHist,aPeaks)#,showFits=True)
        fitsPeaks=[x[0] for x in fitsPeaks]
        fitsPeaks=unique(fitsPeaks)[0:maxNPeaks]
        

        # now iterate again, and just fit each row to the set of peaks we found above
        for rIdx in range(blur.shape[0]):#[1:]:
            
            row=blur[rIdx,:]
            setBinsToAr1D(rowHist,row)
            fitsRow=fitGausPeaks(rowHist,fitsPeaks)
            
            arFitsRow=array([x[0] for x in fitsRow if x[0]>=firstPeak and x[0]<=lastPeak ])
            arFitsRow.sort()
            arDiff=diff(arFitsRow)
            arDiff=array([x for x in arDiff if x>=minDiff])
            
            # one for peak positions
            fill_hist(peaksHist, arFitsRow)
            # one for peak distances from each other
            fill_hist(dPeaksHist,arDiff)
            


        # the peaks histo seems to need a bit of smoothing
        # peaksHist.Smooth() # don't like the native smooth function contained in TH1
        peaksHistAr=setAr1DtoBins(peaksHist)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        setBinsToAr1D(peaksHist,peaksHistAr[0])

        # now we search the histo we made with our gauss fits for peaks and use them as our final peak locations
        nFound = sRow.Search(peaksHist,0.33,'goff',0.025)
        xsPeaks=sRow.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)
        aPeaks.sort()

        # multiple peaks
        if len(aPeaks)>1:# and std(diff(aPeaks))<1.0 and std(diff(aPeaks))<1.0 !=0:
            
            # dPeaks=diff(aPeaks) # was just curious how close this is to the result we get with the actual dPeaksHist
            fitsPeaks=fitGausPeaks(peaksHist,fitsPeaks,width=10)#,showFits=True)
            # if verbose:
                # print 'mean peaks diff: ',mean(dPeaks),' sig: ',std(dPeaks)
            

            # find the tallest peak in dPeaksHist and fit a gauss to it - this is our working peak distance.
            dPmaxBin=dPeaksHist.GetMaximumBin()
            dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
            gf=dPeaksHist.Fit('gaus','QSNO','goff',dPmax-10,dPmax+10)
            dMean=gf.Value(1)
            dMeanEr=gf.Error(1)
            dSig=gf.Value(2)
            dSigEr=gf.Error(2)
        
        # one peak
        else:
            if verbose:
                print 'One Peak ++++++'
            
            # we take everything for peaksHist, since a diff makes no sense with ooonly one peak.
            # so peak spacing is actually just the location of our single peak.
            peaksHistAr=setAr1DtoBins(peaksHist)
            sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
            sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
            setBinsToAr1D(peaksHist,peaksHistAr[0])
            pMaxBin=peaksHist.GetMaximumBin()
            pMax=peaksHist.GetBinCenter(pMaxBin)
            

            gf=peaksHist.Fit('gaus','QSNO','goff',pMax-5,pMax+5)
            
            dMean=gf.Value(1)
            dMeanEr=gf.Error(1)
            dSig=gf.Value(2)
            dSigEr=gf.Error(2)
 
            # need something to return as peak locations
                
            aPeaks=zeros(1)
            aPeaks[0]=pMax # fitGausPeaks want an np array, not a scalar.

            fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)
            

        if verbose:
            print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr
            peaksHistAr=setAr1DtoBins(peaksHist)
            dPeaksHistAr=setAr1DtoBins(dPeaksHist)
            return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,blur,im0,peaksHistAr,dPeaksHistAr)
        else:       
            return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,blur,im0)
        
    except Exception, e:
        print e
        return None  

def retrieveImage(filePath,clearVoids=False,makeU8=False,doLog=False):
    # return a np array (2dim) of a saxslab tiff file

    try:
        
        jjIm=JJTiff(filePath,bars = 0.0, nega = 0.0)
        image=rot90(fliplr(jjIm.arr)) # these aren't indexed the way I had them before. Easier to fix than to rewrite all my code.
        
    except TypeError: # TIFF.open returns a TypeError when it can't read a file. NOTE: This may be cruft now that I use JJTiff
        print 'Could not open',filePath
        return
    # -1, -2, are 255, 254 once we make them uint8, so let's just make the < 0 pix be 0.
    if clearVoids:
        image[image<0]=0
    if doLog:
        image[image<1]=1
        image=log(image)
    if makeU8:
        image[image<0]=0 # this is needed for uint8
        imMax=amax(image)
        im8=array(image/float(imMax)*255,dtype='uint8')
        return im8
    else:
        return image
def main(argv=sys.argv):
    im=argv[1]
    center=(argv[2],argv[3])
    
    p=findPeaks(im,center,verbose=True)
    print
    p=findPeaks(im,center,verbose=True)
    
    
    # print p

import unittest

class a_testcase(unittest.TestCase):

  def test_4(me):

    files = [ 
      (142, (53.84568342941077, 0.8622330357987071, 0.05311658514998754, 0.05586492186835462) ),
      (143, (61.508508907329706, 0.9031407554284832, 0.06134643997660053, 0.03940823942452154) ),
      (137, (37.12425793776592, 4.454433844722973, 0.8194620422574127, 1.3707865416975364) ),
      (158, (174.63267829976627, 0.20404616811924745, 0.024227341549215325, 0.023505703755918456) ),
      (166, (235.59694110948064, 0.20495192682856153, 0.02373238792327561, 0.021109767427827095) ),
        ]

    for f,r in files:
      p=findPeaks("SFU/raw/latest_%07d_caz.tiff"%f, (350,200),verbose=False)
      # print p[:4]
      me.assertEqual( p[:4], r ) 


if __name__ == '__main__':
    main()
