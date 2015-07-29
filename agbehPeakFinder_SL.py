#!/usr/bin/env python
# encoding: utf-8
import cv2
from ROOT import TH1D, TSpectrum
from npRootUtils import *
import numpy as np
# TODO
from numpy import *
import sys
from pilatus_np import JJTiff
from polarize import polarize

from time import sleep
import matplotlib.pyplot as plt

def findPeaks(
              imageOrFilename,
              center,
              peakThresh=0.05,
              verbose=False,
              doLogIm=True,
              polarSize=90,
              firstPeak=20,
              lastPeak=None,
              smoothingWindow=13,
              minDiff=20,
              difThresh=70,
              maxNPeaks=5,
  ):
  """

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
     imageOrFilename:         A path to a saxslab tiff, or an np array of a saxslab tiff.
    
     center:        tuple or list -> (rowCenter,colCenter), where rowCenter is the coord in pix of the center row.
                                                   colCenter is defined the same way, but for center col.
     peakThresh:    Parameter used for peak finding. This is the min acceptable ratio of a peak's height to the height
                    of the largest peak, in order to be counted as a peak by the peak finder.

     verbose        Control the level of output from this function. Setting this to false will cause the function to
                    supress any output to the screen. Setting to true will print a small report upon copletion, and also
                    return the histograms of peak locations and spacings.

     doLogIm:       Whether or not to work with the log of the input image.

     polarSize:         How many lines in the phi direction will we use in polar space?

     firstPeak:     Min radius of a peak to be considered in the calculations.

     lastPeak:      Max radius of a peak to be considered in the calculations. Set to None go all the way out to edge of
                    image.

     smoothingWindow: How many pixels to use in the smoothing algorithm.

     minDiff:       Min distance of neighboring peaks to consider in the peak spacing calculations.

     difThresh:     Threshold of ave peak distances. If the ave is above this, then we auto-adjust firstPeak=50 in
                    order to avoid false peaks near the beamstop.

     maxNPeaks:     How many peaks out from the center to we use (default is 5).
     
     output
     tuple:        (peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr, peaksList, polarImage, im0)
                    
                   peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr are from fitting a histogram
                   of found peak spacingss to a gaussian. 

                   peaksList is a list of tuples (peakCenter,sig,errPeakCenter,errSig), as radius from center.
                   polarImage is the polar representation of the input image that is used for the peak finding.
                   im0 is the original image passed in.

                   If verbose was set to true, the output will be:

                   (peakSpacing, peakSpacingSigma, peakSpacingErr, peakSpacingSigmaErr, eaksList, polarImage, im00, peaksHistAr, dPeaksHistAr)

                   With the last two items in the output tuple are the peak location histogram, and the peak spacing
                   histogram as tuples of numpy arrays, where the [0]th element is the data and the [1]st is the axis.

     Returns None on failure.
    """

  if type(imageOrFilename)==str:
    image = retrieveImage(imageOrFilename,doLog=doLogIm)
  else:
    image = imageOrFilename

  peakThresh=float(peakThresh)
  polarSize=int(polarSize)
  
  firstPeak=int(firstPeak)
  smoothingWindow=int(smoothingWindow)
  minDiff=int(minDiff)
  yM,xM= image.shape
  # print yM,xM # 619 486

  polarImage,radiusSize = imageToPolar( image, center, polarSize ) 

  if not lastPeak:
      lastPeak=int(radiusSize)
  else:
      lastPeak=int(lastPeak)   
  # print lastPeak
  # 453

  # Init the histos, now that we know how big to make them.
  peaksHist= TH1D('peaksHist','peaks',radiusSize*10,0,radiusSize)
  prePeaksHist= TH1D('prePeaksHist','prePeaksHisteaks',radiusSize*10,0,radiusSize)
  prePeaksHistON= TH1D('prePeaksHistON','prePeaksHisteaksON',radiusSize*10,0,radiusSize)
  dPeaksHist=TH1D('dPeaksHist','dPeaks',radiusSize,0,radiusSize)
  rowHist=TH1D('rowHist','row',radiusSize,0,radiusSize)

  S=TSpectrum()

  
# 2) first loop -> prePeaksHist

  # run a gaus filter over the polar image to blend in the rough spots
  # wyderkat - not needed - but different type...
  polarImage = cv2.GaussianBlur(polarImage,(3,3),0)
  # show_array( polarImage )


  #################################################################################
  polarImageON = np.apply_along_axis( smoothMarkov, 1, polarImage, smoothingWindow )
  # show_array( polarimageON )

  rowPeaksON = np.array( [] )
  for row in polarImageON:
    peaks = peakMarkov( row, 1.0, peakThresh, radiusSize,0,radiusSize)
    # TODO optimize. maybe normal array
    rowPeaksON = np.append( rowPeaksON, peaks )
  rowPeaksON = np.array([x for x in rowPeaksON if x>=firstPeak and x<=lastPeak])
  # print rowPeaksON
  prePeaksH,e = np.histogram( rowPeaksON , bins=radiusSize*10, range=(0,radiusSize) )
  prePeaksH = prePeaksH.astype( np.float )
  # print len(prePeaksH)
  # show_vector( prePeaksH )
  #################################################################################

  # first pass - roughly find all the peaks and make a histo.
  for rIdx in range(polarImage.shape[0]):#[1:]:
    # print polarImage.shape[0] 
    # 90
    
    row=polarImage[rIdx,:]
    if rIdx == 0:
      # show_vector(row)
      # print row
      pass

    S.SmoothMarkov(row,len(row),smoothingWindow)
    if rIdx == 0:
      # show_vector(row)
      # print row
      pass

    # just for using it in Search()
    setBinsToAr1D(rowHist,row)
    # how many peaks
    nFoundRow=S.Search(rowHist,1,'goff',peakThresh)
    # peaks positions in ROOT format...
    xsRow=S.GetPositionX()
    # peaks position in arrary
    axRow=rwBuf2Array(xsRow,nFoundRow)
    if rIdx == 0:
      # print axRow
      pass
    axRow=array([x for x in axRow if x>=firstPeak and x<=lastPeak])
    fill_hist(prePeaksHist, axRow)

  
  # prePeaksHist.Draw(); raw_input("continue?")


  # TEST
  hArr,eArr=setAr1DtoBins(prePeaksHist)
  # print len(hArr)
  for i in xrange( len(hArr) ):
    if prePeaksH[i] != hArr[i]:
      print "Mismatch at %s" % i


# 3) proper Gauss fit -> fitsPeaks

  #################################################################################
  # show_vector( prePeaksH )
  prePeaksH = smoothMarkov( prePeaksH, smoothingWindow )
  # show_vector( prePeaksH )
  prePeaksH = smoothMarkov( prePeaksH, smoothingWindow )
  # show_vector( prePeaksH )
  #################################################################################

  # clean out the noise in our rough estimate of where to look for peaks
  peaksHistAr=setAr1DtoBins(prePeaksHist)
  # print len( peaksHistAr[0] )
  # 10 times bigger because of peaksHistAr bins
  # show_vector( peaksHistAr[0] )
  
  S.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
  # show_vector( peaksHistAr[0] )
  S.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow) 
  # show_vector( peaksHistAr[0] )
  # second smoothing kills some outer rings
  # but the trade off is false positive near 
  # beam center in the farther-out detector
  # displacements. 

  setBinsToAr1D(prePeaksHist,peaksHistAr[0])
  # prePeaksHist.Draw(); raw_input("continue?")

  # look for peaks and get the gauss fits - we use this instead of the peaks found from S.Search bacause S.Search can
  # sometimes return multiple peaks that are very close together. If we do guass fits on two close together peaks, we
  # should find the same center for both, and we can then filter them out, keeping only the unique entries.

  #################################################################################
  aPeaksON = peakMarkov( prePeaksH, 0.33, 0.025, radiusSize*10,0,radiusSize )
  # print aPeaksON
  #################################################################################
  # get a list of peaks in our rough peak histo
  nFound = S.Search(prePeaksHist,0.33,'goff',0.025)
  # if verbose:
    # print nFound
  # prePeaksHist.Draw()
  xsPeaks=S.GetPositionX()
  aPeaks=rwBuf2Array(xsPeaks,nFound)
  # print aPeaks
  
  # TEST
  hArr,eArr=setAr1DtoBins(prePeaksHist)
  # print len(hArr)
  for i in xrange( len(hArr) ):
   if prePeaksH[i] != hArr[i]:
     print "Mismatch2 at %s" % i


  # # Proof the Fit is not deterministic!
  # one=fitGausPeaks(prePeaksHist,aPeaks)
  # two=fitGausPeaks(prePeaksHist,aPeaks)
  # three=fitGausPeaks(prePeaksHist,aPeaks)
  # print one
  # print two
  # print three
  # print np.array_equal( one, two)
  # print np.array_equal( two, three)
  # print np.array_equal( one, three)
  #################################################################################
  # TODO better syntax
  fitsPeaksON = peakGaus( prePeaksH, aPeaksON, 30, radiusSize*10,0,radiusSize )
  # print fitsPeaksON
  fitsPeaksON=[x[0] for x in fitsPeaksON]
  fitsPeaksON=np.unique(fitsPeaksON)[0:maxNPeaks]
  # print fitsPeaksON
  #################################################################################
  # get the gauss fits and filter for the unique peaks
  fitsPeaks=fitGausPeaks(prePeaksHist,aPeaks)#,showFits=True)
  # print fitsPeaks
  fitsPeaks=[x[0] for x in fitsPeaks]
  fitsPeaks=np.unique(fitsPeaks)[0:maxNPeaks]
  # print fitsPeaks
  # print np.array_equal( aPeaksON, aPeaks)
  # print np.array_equal( fitsPeaksON, fitsPeaks )

  ########################################
  #### TODO Gaus fix
  if np.all( fitsPeaksON - fitsPeaks < 10e-7 ):
    # print "Fixing gaus fitting"
    fitsPeaksON = fitsPeaks
  
  
# 4) second loop with Gauss fit -> peaksHist, dPeaksHist

  # now iterate again, and just fit each row to the set of peaks we found above
  peakscorr = []
  for rIdx in range(polarImage.shape[0]):#[1:]:
      
    row=polarImage[rIdx,:]
    setBinsToAr1D(rowHist,row)
    fitsRow=fitGausPeaks(rowHist,fitsPeaks)

    arFitsRow=array([x[0] for x in fitsRow if x[0]>=firstPeak and x[0]<=lastPeak ])
    arFitsRow.sort()

    peakscorr.append(arFitsRow)

    arDiff=diff(arFitsRow)
    arDiff=array([x for x in arDiff if x>=minDiff])
    # print arDiff
    
    # one for peak positions
    fill_hist(peaksHist, arFitsRow)
    # one for peak distances from each other
    fill_hist(dPeaksHist,arDiff)
    if rIdx == 0:
      # print fitsRow
      # print arFitsRow
      # print arDiff
      # peaksHist.Draw(); raw_input("continue?\n")
      # dPeaksHist.Draw(); raw_input("continue?\n")
      pass

  # peaksHist.Draw(); raw_input("continue?\n")
  # dPeaksHist.Draw(); raw_input("continue?\n")
    
  #################################################################################
  rowPeaks2ndON = np.array( [] )
  rowDiff2ndON = np.array( [] )
  i = 0
  for row in polarImageON:
    peaks = peakGaus( row, fitsPeaksON, 30, radiusSize,0,radiusSize )
    peaks = np.array( [ x[0] for x in peaks if x[0]>=firstPeak and x[0]<=lastPeak ] )
    peaks.sort()
    ########################################
    #### TODO Gaus fix
    peaks = peakscorr[ i ]
    i+=1
    #### TODO Gaus fix
    ########################################

    rowPeaks2ndON = np.append(rowPeaks2ndON, peaks)
    rowDiff2ndON = np.append(rowDiff2ndON, np.diff( peaks ) )
  peaksHistON,e = np.histogram( rowPeaks2ndON , bins=radiusSize*10, range=(0,radiusSize) )
  peaksHistON = peaksHistON.astype( np.float )
  # show_vector( peaksHistON )

  rowDiff2ndON = rowDiff2ndON[ rowDiff2ndON>=minDiff ]
  diffHistON,e = np.histogram( rowDiff2ndON , bins=radiusSize, range=(0,radiusSize) )
  diffHistON = diffHistON.astype( np.float )
  # show_vector( diffHistON )
  #################################################################################


  hArr,eArr=setAr1DtoBins(peaksHist)
  # print len(hArr)
  for i in xrange( len(hArr) ):
    if peaksHistON[i] != hArr[i]:
      print "Mismatch 3 at %s (%s %s)" % (i, peaksHistON[i], hArr[i])

  hArr,eArr=setAr1DtoBins(dPeaksHist)
  # print len(hArr)
  for i in xrange( len(hArr) ):
    if diffHistON[i] != hArr[i]:
      print "Mismatch 4 at %s (%s %s)" % (i, diffHistON[i], hArr[i])


# 5) final search (no Gauss) -> aPeaks

  # the peaks histo seems to need a bit of smoothing
  # peaksHist.Smooth() # don't like the native smooth function contained in TH1
  peaksHistAr=setAr1DtoBins(peaksHist)
  S.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
  setBinsToAr1D(peaksHist,peaksHistAr[0])

  # peaksHist.Draw(); raw_input("continue?\n")

  # now we search the histo we made with our gauss fits for peaks and use them as our final peak locations
  nFound = S.Search(peaksHist,0.33,'goff',0.025)
  xsPeaks=S.GetPositionX()
  aPeaks=rwBuf2Array(xsPeaks,nFound)
  aPeaks.sort()
  # print aPeaks

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
    S.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
    S.SmoothMarkov(peaksHistAr[0],len(peaksHistAr[0]),smoothingWindow)
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
      return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,polarImage, image,peaksHistAr,dPeaksHistAr)
  else:       
      return (dMean, dSig, dMeanEr, dSigEr,fitsPeaks,polarImage, image)
        

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

def imageToPolar( image, center, polarSize ):

  rowCenter=float(center[0])
  colCenter=float(center[1])

# 1) polar system

  # unroll into polar coordinates
  X,Y = indices( image.shape )
  # print X
  # 0 0 0 0 ...
  # 1 1 1 1 ...
  # ....
  # 618 618 618

  Xc=X-rowCenter
  Yc=Y-colCenter
  r = np.around( np.sqrt(Xc**2+Yc**2) )
  # print r
  #[403.  403.  402. ...,  451.  451.  452.]
  #[ 402.  402.  401. ...,  450.  451.  451.]
  #[ 401.  401.  400. ...,  449.  450.  450.]
  #..., 
  #[ 333.  332.  332. ...,  389.  390.  391.]
  #[ 334.  333.  332. ...,  390.  391.  391.]
  #[ 334.  334.  333. ...,  390.  391.  392.]]
  

  at3 = np.arctan2(Yc,Xc)
  # print at3
  #[-2.62244654 -2.62460304 -2.62676483 ...,  2.45992182  2.45820141
  #  2.45648581]
  #[-2.62121311 -2.62337275 -2.62553772 ...,  2.45852146  2.45680006
  #  2.45508348]
  #[-2.61997436 -2.62213713 -2.62430527 ...,  2.45711627  2.45539388
  #  2.45367633]
  #..., 
  #[-0.64470303 -0.64229702 -0.63988231 ...,  0.81811376  0.81986726
  #  0.82161421]
  #[-0.64290159 -0.64049811 -0.638086   ...,  0.81624137  0.81799531
  #  0.8197427 ]
  #[-0.64110877 -0.63870786 -0.63629835 ...,  0.81437556  0.8161299
  #  0.81787771]]
  # imshow(at3)

  # convert angles < 0 to positive
  at3[ at3<0.0 ] += 2*np.pi
  # print at3
  #[[ 3.66073877  3.65858227  3.65642047 ...,  2.45992182  2.45820141
  #   2.45648581]
  # [ 3.66197219  3.65981256  3.65764759 ...,  2.45852146  2.45680006
  #   2.45508348]
  # [ 3.66321095  3.66104817  3.65888003 ...,  2.45711627  2.45539388
  #   2.45367633]
  # ..., 
  # [ 5.63848228  5.64088829  5.643303   ...,  0.81811376  0.81986726
  #   0.82161421]
  # [ 5.64028372  5.64268719  5.64509931 ...,  0.81624137  0.81799531
  #   0.8197427 ]
  # [ 5.64207654  5.64447745  5.64688695 ...,  0.81437556  0.8161299
  #   0.81787771]]
  
  # imshow(at3)

  at3 *= polarSize/(2*pi)
  # print at3
  #[[ 52.43622032  52.40533078  52.3743653  ...,  35.23578449  35.21114147
  #   35.1865673 ]
  # [ 52.45388784  52.42295332  52.39194243 ...,  35.21572592  35.19106859
  #   35.16648039]
  # [ 52.47163174  52.44065224  52.40959592 ...,  35.19559808  35.17092658
  #   35.14632449]
  # ..., 
  # [ 80.76530932  80.79977296  80.83436111 ...,  11.71861639  11.7437335
  #   11.76875659]
  # [ 80.79111308  80.82554031  80.86009132 ...,  11.69179645  11.71691971
  #   11.74194925]
  # [ 80.81679333  80.85118388  80.88569748 ...,  11.66507059  11.69019964
  #   11.71523527]]

  r = r.astype(int)
  # print r
  #[[403 403 402 ..., 451 451 452]
  # [402 402 401 ..., 450 451 451]
  # [401 401 400 ..., 449 450 450]
  # ..., 
  # [333 332 332 ..., 389 390 391]
  # [334 333 332 ..., 390 391 391]
  # [334 334 333 ..., 390 391 392]]

  at3 = at3.astype(int)
  # print at3
  #[[52 52 52 ..., 35 35 35]
  #[52 52 52 ..., 35 35 35]
  #[52 52 52 ..., 35 35 35]
  #..., 
  #[80 80 80 ..., 11 11 11]
  #[80 80 80 ..., 11 11 11]
  #[80 80 80 ..., 11 11 11]]

  # imp[at3,r]= image 
  radiusSize = np.amax(r)+1
  # print radiusSize 
  # 453

  # allocate the polar image
  polarImage=zeros((amax(at3)+1,radiusSize))
  # print amax(at3)+1
  # 90


  # Straight up broadcasting in numpy doesn't do += properly: you just get the last value that mapped to the new coords. So we lose info.
  # polarImage[at3,r]+= image

  # This one I wrote in Fortran (just because it's really easy to compile fortran modules to work with numpy), it does the proper +=, and it's full speed.
  polarImage = polarize( image,at3,r,polarImage)
  # show_array( polarImage )

  return (polarImage, radiusSize )

# TODO !!!! Now is slower than before

# do it inplace

def smoothMarkov( row, window ):
  copy = np.copy( row )
  S=TSpectrum()
  S.SmoothMarkov( copy, copy.shape[0], window )
  return copy

def peakMarkov( row, sigma, threshold, hbins, hmin, hmax):
  S=TSpectrum()
  hist = TH1D('','',hbins, hmin, hmax)
  setBinsToAr1D(hist,row)
  # how many peaks
  npeaks = S.Search(hist,sigma,'goff', threshold)
  # peaks positions in ROOT format...
  posROOT = S.GetPositionX()
  pos = rwBuf2Array( posROOT, npeaks)
  return pos

def peakGaus( row,peaks,width,hbins, hmin, hmax, write=True):
  # returns a list of tuples (mean,sigma,errMean,errSig), one entry for each peak in peaks

  peaks.sort()
  fits=[]

  if write==True:
    hist = TH1D('','',hbins, hmin, hmax)
    setBinsToAr1D(hist,row)
  else:
    hist = row
  # hist.Draw(); raw_input("continue?\n")

  nBins= hist.GetNbinsX() #thists have nBins+2 bins - 0 is underflow and nBins+1 is overflow.
  minBin= hist.GetBinCenter(1)
  maxBin= hist.GetBinCenter(nBins)

  for idx in range(len(peaks)):
    if peaks[idx]<minBin and peaks[idx] > maxBin:
        print 'fitGausPeaks: ERROR ***************** peak outside of histogram range.'
        print 'Histo name: ', hist.GetName(),'\nHisto Title: ', hist.GetTitle()
        continue
    gf= hist.Fit('gaus','QSNO','goff',peaks[idx]-width/2.,peaks[idx]+width/2.)
    # print peaks[idx]

    #53.2320715094
    #107.518033347
    #160.601812267
    #214.335510293
    #267.682490065

    # print peaks[idx]-width/2.,peaks[idx]+width/2.
    #38.25 68.25
    #92.55 122.55
    #145.55 175.55
    #199.35 229.35
    #252.65 282.65

    #38.2320715094 68.2320715094
    #92.5180333473 122.518033347
    #145.601812267 175.601812267
    #199.335510293 229.335510293
    #252.682490065 282.682490065
    fits.append((gf.Value(1),gf.Value(2),gf.Error(1),gf.Error(2)))
    # print "Vn %s n=%s m=%s x=%s" % (gf.Value(1), nBins, minBin, maxBin) 
  return fits

def main(argv=sys.argv):
    im=argv[1]
    center=(argv[2],argv[3])
    
    p = findPeaks( im, center, verbose=True )
    # print p
    
def show_array( a ):
  plt.imshow( a )
  plt.colorbar(orientation='horizontal')
  plt.show()
    
def show_vector( v ):
  x = arange(v.shape[0])
  plt.plot( x, v )
  plt.show()

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
