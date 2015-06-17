#!/usr/bin/env python
# encoding: utf-8
import cv2
from ROOT import TH1D, TH2D, TSpectrum, TCanvas,TVector2
from npRootUtils import *
# from root_numpy import fill_hist
from numpy import *
import sys
from pilatus_np import JJTiff
# im0=retrieveImage('SFU/raw/latest_0000150_caz.tiff',makeU8=True,doLog=True)
# im0=retrieveImage('SFU/raw/latest_0000150_caz.tiff',doLog=True)
# im0=retrieveImage('SFU/raw/latest_0000141_caz.tiff',doLog=True) # ~46.2
# im0=retrieveImage('SFU/raw/latest_0000157_caz.tiff',doLog=True)

# im0=retrieveImage('SFU/raw/latest_0000138_caz.tiff',doLog=True) # ~23.3
# im0=retrieveImage('SFU/raw/latest_0000166_caz.tiff',doLog=True) # ~235.6
# im0=retrieveImage('SFU/raw/latest_0000155_caz.tiff',doLog=True) # ~151.9
# rowCenter=350
# colCenter=200
# pSize=500
# peakThresh=0.005
# firstPeak=100
# lastPeak=pSize
# minDiff=10
def findPeaks(image,center,peakThresh=0.01,verbose=False,doLogIm=True,pSize=360,firstPeak=50,lastPeak=None,smoothingWindow=5,minDiff=10):
    """

    findPeaks(center,image,rad)

    Take a saxslab tiff of AgBeh, and return peak spacing, and an array of found peak coordinates (radius from center).

    Peaks in the range (rad[0]:rad[1]) are found by taking slices through the given center coordinate 
    of the tiff at several rotation agles, and searching for peaks within rad[0]:rad[1]. Then each peak in a slice from
    the pervious step is separately fit to a gaussian curve, and the mean from each fit is added into a histogram.

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
                   peakSpacing, peakSpacingSigma, peakSpacingErr are from fitting a histogram of found peak spacings
                   to a gaussian. 
                   peaksList is a list of tuples (peak,sig,errPeak,errSig), as radius from center. The peaks are
                   located again by doing gaussian fits to the peaks in the histogram built from the location of every 
                   peak found during the slicing and fitting process.

     Returns None on failure.
    """
    try:

        # determine if image is an image or a path:
        if type(image)==str:
            image=retrieveImage(image,doLog=doLogIm)

        im0=image
        rSize=500
        if not lastPeak:
            lastPeak=int(rSize)
        else:
            lastPeak=int(lastPeak)
        rowCenter=int(center[0])# 350
        colCenter=int(center[1])# 200
        peakThresh=float(peakThresh)
        pSize=int(pSize)
        
        firstPeak=int(firstPeak)
        smoothingWindow=int(smoothingWindow)
        minDiff=int(minDiff)
        yM,xM=im0.shape
        peaksHist= TH1D('peaksHist','peaks',rSize*10,0,rSize)
        dPeaksHist=TH1D('dPeaksHist','dPeaks',rSize,0,rSize)
        rowHist=TH1D('rowHist','row',rSize,0,rSize)
            # TH2D(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup, Int_t nbinsy, Double_t ylow, Double_t yup)
        # imPolarHist=TH2D('imPolar','im0 Polar',877*4,0,877,720,0,2*pi)
        # imPolar=zeros((pSize+1,pSize+1),dtype=uint8)
        imPolar=zeros((pSize+1,rSize+1))
        yM,xM=im0.shape
        vP=TVector2()
        
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
            nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh)
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


        # tc=TCanvas()
        # tc.Divide(1,2)
        # tc.cd(1)
        # dPeaksHist.Draw()
        dPmaxBin=dPeaksHist.GetMaximumBin()
        dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
        gf=dPeaksHist.Fit('gaus','QSNO','goff',dPmax-5,dPmax+5)
        dMean=gf.Value(1)
        dMeanEr=gf.Error(1)
        dSig=gf.Value(2)
        dSigEr=gf.Error(2)
        # tc.cd(2)
        # this gets the peaks array out at the end
        peaksHist.Smooth()
        nFound = sRow.Search(peaksHist,3.5,'goff',0.1)
        # peaksHist.Draw()
        xsPeaks=sRow.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)
        aPeaks=aPeaks[aPeaks>=firstPeak]
        aPeaks=aPeaks[aPeaks <= lastPeak]
        # print aPeaks
        fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)

        # print fitsPeaks
        if len(aPeaks)==1:
            print 'One Peak ++++++'
            # (mean,sigma,errMean,errSig)
            nFound = sRow.Search(peaksHist,3.5,'goff',0.1)
            xsPeaks=sRow.GetPositionX()
            aPeaks=rwBuf2Array(xsPeaks,nFound)
            aPeaks=aPeaks[aPeaks>=firstPeak]
            aPeaks=aPeaks[aPeaks <= lastPeak]
            fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)
            # print fitsPeaks
            # fitsPeaks=fitsPeaks[fitsPeaks>=firstPeak]
            # fitsPeaks=fitsPeaks[fitsPeaks<=lastPeak]
            # print fitsPeaks
            idx=len(fitsPeaks)-1
            dMean=fitsPeaks[idx][0]
            dMeanEr=fitsPeaks[idx][2]
            dSig=fitsPeaks[idx][1]
            dSigEr=fitsPeaks[idx][3]
        if verbose:
            print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr
                
        return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,blur,im0)
    except Exception, e:
        print e
        return None  

def retrieveImage(filePath,clearVoids=False,makeU8=False,doLog=False):
    # return a np array (2dim) of a saxslab tiff file

    try:
        
        jjIm=JJTiff(filePath,bars = 0.0, nega = 0.0)
        image=rot90(fliplr(jjIm.arr))
        # print 'image: ', type(image),' of: ',image.dtype, ' of shape: ', image.shape
    except TypeError: # TIFF.open returns a TypeError when it can't read a file
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
    # doLog=int(argv[4])
    p=findPeaks(im,center,verbose=True)
    
    # p=findPeaks(im,center,verbose=True,rad=(0,110))
    print
    print (p,im0,imPolar)

if __name__ == '__main__':
    main()
