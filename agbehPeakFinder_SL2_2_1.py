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
def findPeaks(image,center,peakThresh=0.05,verbose=False,doLogIm=True,pSize=90,firstPeak=20,lastPeak=None,smoothingWindow=13,minDiff=20,difThresh=70,maxNPeaks=5):
    """

    findPeaks(center,image,rad)

    Take a saxslab tiff of AgBeh, and return peak spacing, and an array of found peak coordinates (radius from center).

    Peaks in the radial range (firstPeak:lastPeak) are found by first unrolling the image into polar coordinates. We then
    iterate down the image by rows and do a rough peak search on each row. The peak coordinates from this search are then
    fed to a function that separately fits a small range of the row, centered on each peak coordinate, to a gaussian
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
                    spress any output to the screen.

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
        rowCenter=float(center[0])# 350
        colCenter=float(center[1])# 200
        peakThresh=float(peakThresh)
        pSize=int(pSize)
        
        firstPeak=int(firstPeak)
        smoothingWindow=int(smoothingWindow)
        minDiff=int(minDiff)
        yM,xM=im0.shape
        peaksHist= TH1D('peaksHist','peaks',rSize*10,0,rSize)
        prePeaksHist= TH1D('prePeaksHist','prePeaksHisteaks',rSize*10,0,rSize)
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
                imPolar[round(p),round(r)]+=im0[y,x]
                

        # run a gaus filter over the polar image to blend in the rough spots
        blur = cv2.GaussianBlur(imPolar,(3,3),0)

        sRow=TSpectrum()

        # check the first row to see if we are far enough away from source that
        # the beamstop halo would give us a false peak near the center, and adjust firstPeak
        # to compensate if needed. Might want to parameterise the default setting to adjust to
        # instead of hard coded to 50....
        # row=blur[0,:]
        # sRow.SmoothMarkov(row,len(row),smoothingWindow)
        # setBinsToAr1D(rowHist,row)
        # nFoundRow=sRow.Search(rowHist,1,'goff',peakThresh)
        # xsRow=sRow.GetPositionX()
        # axRow=rwBuf2Array(xsRow,nFoundRow)
        # axRow=array([x for x in axRow if x>=firstPeak and x<=lastPeak])
        # sort(axRow)
        # # if len(axRow)>maxNPeaks:
        # #     axRow=axRow[0:maxNPeaks]
        # fitsRow=fitGausPeaks(rowHist,axRow)
        
        # arFitsRow=array([x[0] for x in fitsRow])
        # # print arFitsRow
        # arDiff=diff(arFitsRow)
        # print 'arDiff,mean,std: ',arDiff,mean(arDiff),std(arDiff)
        # if mean(arDiff)>difThresh:
        #     firstPeak=50
        #     print '+++++++++++++++++== set to 50\n'
        #     axRow=rwBuf2Array(xsRow,nFoundRow)
        #     axRow=array([x for x in axRow if x>=firstPeak and x<=lastPeak])
        #     sort(axRow)
        #     # if len(axRow)>maxNPeaks:
        #     #     axRow=axRow[0:maxNPeaks]
        #     fitsRow=fitGausPeaks(rowHist,axRow)
        #     arFitsRow=array([x[0] for x in fitsRow])
        #     arDiff=diff(arFitsRow)
        # fill_hist(peaksHist, arFitsRow)
        # fill_hist(dPeaksHist,arDiff)
        # dPmaxBin=dPeaksHist.GetMaximumBin()
        # dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
        # if dPmax<1: #one
        #         lastPeak=int(rSize)
        # else:
        #     # print dPmax,(dPmax+1)*maxNPeaks,':'
        #     lastPeak=(dPmax+1)*maxNPeaks
        # lastPeak=(dPmax+1)*maxNPeaks
        # now that we have decided whether to change firstPeak or not, do the rest of the rows.
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

            # print lastPeak
        # now go through and do gauss fits on the rows, using the first maxNPeaks for our center param
        # but we need to make sure that we don't have false+ in the center < 50 
        
        # this seems to clean out the noise in the center...
        peaksHistAr=setAr1DtoBins(prePeaksHist)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        setBinsToAr1D(prePeaksHist,peaksHistAr[0])

        # look for peaks and get the gauss fits - we use this instead of the peaks found from sRow.Search
        # bacause sRow.Search can sometimes return multiple peaks that are very close to gether. If we do guass
        # fits on two close together peaks, we should find the same center for both, and we can then filter them out.

        nFound = sRow.Search(prePeaksHist,0.33,'goff',0.025)
        # prePeaksHist.Draw()
        xsPeaks=sRow.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)

        # get the gauss fits and filter out the unique peaks
        fitsPeaks=fitGausPeaks(prePeaksHist,aPeaks)#,showFits=True)
        fitsPeaks=[x[0] for x in fitsPeaks]
        fitsPeaks=unique(fitsPeaks)

        # let's figure out if there is some false peak we are seeing near the beam center.
            # -- don't seem to need it naymore...
        

        # now iterate again, and just fit each row to the set of peaks we found above
        for rIdx in range(blur.shape[0]):#[1:]:
            
            row=blur[rIdx,:]
            setBinsToAr1D(rowHist,row)
            fitsRow=fitGausPeaks(rowHist,fitsPeaks[0:maxNPeaks])
            
            arFitsRow=array([x[0] for x in fitsRow if x[0]>=firstPeak and x[0]<=lastPeak ])
            arFitsRow.sort()
            arDiff=diff(arFitsRow)
            arDiff=array([x for x in arDiff if x>=minDiff])
            # print 'axRow: ',len(axRow), 'maxNPeaks: ',maxNPeaks
            fill_hist(peaksHist, arFitsRow)
            fill_hist(dPeaksHist,arDiff)
            
        print 'out of second iter'

        # tc=TCanvas()
        # tc.Divide(1,2)
        # tc.cd(1)
        # gPad.Divide(1,2)
        # gPad.cd(1)
        # Might want to smooth this - otherwise good.
        # dPeaksHist.Draw()
        # dPmaxBin=dPeaksHist.GetMaximumBin()
        # dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
        # # gf=dPeaksHist.Fit('gaus','QSNO','goff',dPmax-10,dPmax+10)
        # gf=dPeaksHist.Fit('gaus','QS','',dPmax-10,dPmax+10)
        # dMean=gf.Value(1)
        # dMeanEr=gf.Error(1)
        # dSig=gf.Value(2)
        # dSigEr=gf.Error(2)
        # print 'fit the dPeaksHist'
        # dPeaksHist.Draw()
        # tc.cd(2)
        # gPad.cd(2)
        # this gets the peaks array out at the end

        peaksHistAr=setAr1DtoBins(peaksHist)
        sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        # sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
        setBinsToAr1D(peaksHist,peaksHistAr[0])


        # peaksHist.Smooth()
        # nFound = sRow.Search(peaksHist,3.5,'goff',0.05)
        nFound = sRow.Search(peaksHist,0.33,'goff',0.025)
        
        # peaksHist.Draw()
        xsPeaks=sRow.GetPositionX()
        aPeaks=rwBuf2Array(xsPeaks,nFound)
        # # fitsRow=fitGausPeaks(rowHist,axRow)
        # # print aPeaks
        # aPeaks=aPeaks[aPeaks>=firstPeak]   
        # # print aPeaks 
        # aPeaks=aPeaks[aPeaks <= lastPeak]
        # # print aPeaks
        # # print aPeaks
        aPeaks.sort()
        # print 'sdt(diff(aPeaks)', std(diff(aPeaks)), 'aPeaks',aPeaks,'diff(aPeaks)',diff(aPeaks)
        # if len(aPeaks)>maxNPeaks:
        #     aPeaks=aPeaks[0:maxNPeaks]
        print 'aPeaks',aPeaks
        if len(aPeaks)>1:# and std(diff(aPeaks))<1.0 and std(diff(aPeaks))<1.0 !=0:
            dPeaks=diff(aPeaks)
            print 'mean peaks diff: ',mean(dPeaks),' sig: ',std(dPeaks)
            fitsPeaks=fitGausPeaks(peaksHist,fitsPeaks,width=10)
            dPmaxBin=dPeaksHist.GetMaximumBin()
            dPmax=dPeaksHist.GetBinCenter(dPmaxBin)
            # gf=dPeaksHist.Fit('gaus','QSNO','goff',dPmax-10,dPmax+10)
            gf=dPeaksHist.Fit('gaus','QS','',dPmax-10,dPmax+10)
            dMean=gf.Value(1)
            dMeanEr=gf.Error(1)
            dSig=gf.Value(2)
            dSigEr=gf.Error(2)
            print 'fit the dPeaksHist'
            dPeaksHist.Draw()
        # fitsPeaks=fitGausPeaks(rowHist,fitsPeaks[0:maxNPeaks])
        # arFitsPeaks=array([x[0] for x in fitsPeaks])
        # arDiff=diff(arFitsPeaks)
        # dMean=mean(arDiff)
        # dSig=std(arDiff)
        # dSigEr=0.0
        # dMeanEr=0.0
        # print fitsPeaks
        # if len(aPeaks)==1:
        else:
            print 'One Peak ++++++'
            # (mean,sigma,errMean,errSig)
            # print peaksHistAr
            peaksHistAr=setAr1DtoBins(peaksHist)
            sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
            sRow.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
            setBinsToAr1D(peaksHist,peaksHistAr[0])
            pMaxBin=peaksHist.GetMaximumBin()
            pMax=peaksHist.GetBinCenter(pMaxBin)
            # nFound = sRow.Search(peaksHist,3.5,'',0.5)
            # xsPeaks=sRow.GetPositionX()
            # aPeaks=rwBuf2Array(xsPeaks,nFound)

            gf=peaksHist.Fit('gaus','QS','',pMax-5,pMax+5)
            print 'fit the peaksHist'
        # gf=dPeaksHist.Fit('gaus','QS','',dPmax-10,dPmax+10)
            dMean=gf.Value(1)
            dMeanEr=gf.Error(1)
            dSig=gf.Value(2)
            dSigEr=gf.Error(2)
                # print aPeaks
            # aPeaks=aPeaks[aPeaks>=firstPeak]
            # # aPeaks=aPeaks[aPeaks <= lastPeak]
            # print aPeaks
            aPeaks=zeros(1)
            aPeaks[0]=pMax
            fitsPeaks=fitGausPeaks(peaksHist,aPeaks,width=10)
            # print fitsPeaks,
            # Fit('gaus','QSNO','goff',dPmax-10,dPmax+10)
            # print fitsPeaks
            # fitsPeaks=fitsPeaks[fitsPeaks>=firstPeak]
            # fitsPeaks=fitsPeaks[fitsPeaks<=lastPeak]
            # print fitsPeaks
            # idx=len(fitsPeaks)-1
        # dMean=0
        # dMeanEr=0
        # dSig=0
        # dSigEr=0
        # peaksHist.Draw()
        if verbose:
            print '\nMean Spacing : ',dMean,' +/- ',dMeanEr,'\nSigma        : ',dSig, '+/- ',dSigEr
                
        return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,blur,im0)
        # return (0, 0, 0, 0,fitsPeaks,blur,im0)
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
