#!/usr/bin/env python
# encoding: utf-8
"""
doc string here
"""
# from saxsUtils import *
import cv2
from ROOT import TH1F, TSpectrum #, TCanvas
# from root_numpy import fill_hist
from numpy import *
from libtiff import TIFF
from npRootUtils import *
import sys



# let's write a function:   input: is good center, path to image, radius range to search for peaks.
#                           output: peak spacing (pix). How many peaks, an arr w/ peaks in that range.

def findPeaks(image,center,rad=(25,110),nSteps=60,verbose=False):
    """

    findPeaks(center,image,rad)

    Take a saxslab tiff of AgBeh, and return peak spacing, and an array of found peak coordinates.

    Peaks in the range (rad[0]:rad[1]) are found by taking slices through the given center coordinate 
    of the tiff at several rotation agles, and searching for peaks within rad[0]:rad[1]. Then sections of
    the slice containing one peak each from the previous step are fit to gaussian curves, and the mean from
    each fit is added into a histogram.

    Also, the spacing between all the peaks in a slice is added to a histogram of peak spacings. This is all
    repeated for each rotation.

    At the end, we have a histogram of peak locations and a histogram of peak spacings. The peak spacing
    histogram is fit to a gaussian, and the mean, sigma, and error are returned.

    Further, the histogram of peak locations is treated similarly to a slice in the first step, and it's peaks
    are again fit to gaussians, and the mean, sigma, and error of each are stored in a list of tuples in the output.

     input
     center:       tuple or list -> (rowCenter,colCenter), where rowCenter is the coord in pix of the center row.
                                                   colCenter is defined the same way, but for center col.
     image:        A path to a saxslab tiff, or an np array of a saxslab tiff.
     rad:          tuple or list -> (minRadius, maxRadius) in which to search for peaks.
     
     output
     tuple:        (peakSpacing, peakSpacingSigma, peakSpacingErr, peaksList)
                   peakSpacing, peakSpacingSigma, peakSpacingErr are from fitting a histogram of found peak coordinates
                   to a gaussian. 
                   peaksList is a list of tuples (peak,sig,errPeak,errSig), as radius from center. The peaks are
                   located again by doing gaussian fits to the peaks in the histogram built from the ocation of every 
                   peak found during the slicing and fitting process.

     Returns None on failure.
    """

    try:
        # determine if image is an image or a path:
        if type(image)==str:
            image=retrieveImage(image)

        im=image
        firstPeak=int(rad[0])
        lastPeak=int(rad[1])
        rowCenter=int(center[0])
        colCenter=int(center[1])

        rows,cols = im.shape
        # print im.shape
        hSize=max(im.shape)/2+1 # half of the tiff on the long dimension (rows)
        stepDeg=90./(nSteps+1)

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
                rim=cv2.warpAffine(float32(im),M,(cols,rows))
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
        # tc=TCanvas()
        # tc.Divide(1,2)
        # tc.cd(1)
        # dPeaksHist.Draw()

        gf=dPeaksHist.Fit('gaus','QSNO','goff')
        dMean=gf.Value(1)
        dMeanEr=gf.Error(1)
        dSig=gf.Value(2)
        dSigEr=gf.Error(2)
        # tc.cd(2)
        # this gets the peaks array out at the end

        nFound = sCol.Search(peaksHist,3.5,'goff',0.03)
        xsPeaks=sCol.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)
        aPeaks=aPeaks[aPeaks>=firstPeak]
        aPeaks=aPeaks[aPeaks <= lastPeak]
        fitsPeaks=fitGausPeaks(peaksHist,aPeaks)

        # print fitsPeaks
        # print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr
        # tuple:        (peakSpacing, peakSpacingSigma, peakSpacingErr, peaksList)
               # peakSpacing, peakSpacingSigma, peakSpacingErr are from fitting a histogram of found peak coordinates
               # to a gaussian. 
               # peaksList is a list of tuples (peak,sig,errPeak,errSig), as radius from center. The peaks are
               # located again by doing gaussian fits to the peaks in the histogram built from the ocation of every 
               # peak found during the slicing and fitting process.
        if verbose:
            print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr
            
        return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks)

    except Exception, e:
        print e
        return None    



# retrieve image from file system and convert to uint8 if desired
def retrieveImage(filePath,clearVoids=False,makeU8=False):
    # return a np array (2dim) of a saxslab tiff file

    try:
        tif=TIFF.open(filePath)
        image = tif.read_image() # this is int32
        # print 'image: ', type(image),' of: ',image.dtype, ' of shape: ', image.shape
    except TypeError: # TIFF.open returns a TypeError when it can't read a file
        print 'Could not open',filePath
        return
    # -1, -2, are 255, 254 once we make them uint8, so let's just make the < 0 pix be 0.
    if clearVoids:
        image[image<0]=0
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

    p=findPeaks(im,center)
    print p

if __name__ == '__main__':
    main()