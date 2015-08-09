#!/usr/bin/env python
# encoding: utf-8
import cv2
from ROOT import TH1D, TSpectrum
from npRootUtils import *
import numpy as np
import sys
from pilatus_np import JJTiff
from pdb import set_trace as t
from scipy.optimize import leastsq

import markov

class a_histogram(object):
  def __init__( me, data, lower, upper, resolution ):
    me.lower = lower
    me.upper = upper
    me.resolution = resolution
    me.bins = [] 
    me.edges = [] # len(me.edges) == len(me.bins) + 1
    me.centers = [] # len(me.centers) == len(me.bins)

    me.bins, me.edges = \
      np.histogram( data , bins=me.resolution, range=(me.lower,me.upper) )
    me.bins = me.bins.astype( np.float )

    me.centers = np.zeros_like( me.bins, dtype=np.float )
    for center, left, right in np.nditer(
        [me.centers, me.edges[:-1], me.edges[1:]],
        op_flags=['readwrite']):
      center[...] = (left+right)/2.0;

  def maxbin_center( me ):
    idx = me.bins.argmax()
    return me.centers[ idx ]




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


  if type(imageOrFilename) == str:
    image = retrieveImage( imageOrFilename, doLog=doLogIm)
  else:
    image = imageOrFilename

  polarImage, radiusSize = imageToPolar( image, center, polarSize ) 

  if lastPeak is None:
      lastPeak=radiusSize
  # print lastPeak # 453

# 1st STAGE

  # run a gaus filter over the polar image to blend in the rough spots
  polarImage = cv2.GaussianBlur(polarImage,(3,3),0)
  # show_array( polarImage )


  allpeaks1st = []
  for row in polarImage:
    markov.smooth( row, smoothingWindow )
    allpeaks1st.extend( peakMarkov( row, 1.0, peakThresh, radiusSize,0,radiusSize) )
    # allpeaks1st.extend( markov.search( row, 1.0, peakThresh ) )
  # print allpeaks1st
  # show_array( polarImage )
  allpeaks1st = np.array( [x for x in allpeaks1st if x>=firstPeak and x<=lastPeak] )
  # print allpeaks1st
  hist1st,hist1stEdges = np.histogram( allpeaks1st , bins=radiusSize*10, range=(0,radiusSize) )
  hist1st = hist1st.astype( np.float )
  # show_vector( hist1st )


  # be careful, dtype has to be dynamic (float, but no float32 or no float64)
  markov.smooth( hist1st, smoothingWindow )
  # show_vector( hist1st )
  markov.smooth( hist1st, smoothingWindow )
  # show_vector( hist1st )

  #peaks1st = peakMarkov( hist1st, 0.33, 0.025, radiusSize*10,0,radiusSize )
  # peaks1st = peakMarkovVerbose( hist1st, 0.33, 0.025, radiusSize*10,0,radiusSize )
  peaks1st = peakMarkovPorted( hist1st, 0.33, 0.025, hist1stEdges )
  print "peaks1st", peaks1st
  peaks1st.sort()
  # print "peaks1st", peaks1st
  
  peaks1st = [ fitGaus( hist1stEdges, hist1st, p, 30, 0, radiusSize, radiusSize*10, draw=False )[0] \
               for p in peaks1st ]
  # peaks1st = peakGaus( hist1st, peaks1st, 30, radiusSize*10,0,radiusSize )
  # for o,n in zip(peaks1st, peaks1stTEST):
    # print o[0], "--new->", n
  peaks1st = np.unique(peaks1st)[0:maxNPeaks]
  # print peaks1st


# 2nd STAGE

  allpeaks2nd = []
  diffs2nd = []
  for row in polarImage:
    xdata = np.arange( len(row)+1, dtype=np.float )
    peaks = [ fitGaus( xdata, row, p, 30, 0, radiusSize, radiusSize )[0] \
              for p in peaks1st ]
    # peaks = peakGaus( row, peaks1st, 30, radiusSize,0,radiusSize )
    # peaks = [ x[0] for x in peaks if x[0]>=firstPeak and x[0]<=lastPeak ]
    peaks = [ x for x in peaks if x>=firstPeak and x<=lastPeak ]
    peaks.sort()
    allpeaks2nd.extend( peaks )
    diffs2nd.extend( np.diff( peaks ) ) # still not a numpy array

  allpeaks2nd = np.array( allpeaks2nd )
  # print allpeaks2nd
  diffs2nd = np.array( diffs2nd )

  hist2nd, hist2ndEdges = \
      np.histogram( allpeaks2nd , bins=radiusSize*10, range=(0,radiusSize) )
  hist2nd = hist2nd.astype( np.float )
  # show_vector( hist2nd )

  diffs2nd = diffs2nd[ diffs2nd>=minDiff ]
  diffhist2nd, diffhist2ndEdges = \
      np.histogram( diffs2nd , bins=radiusSize, range=(0,radiusSize) )
  diffhist2nd = diffhist2nd.astype( np.float )
  # show_vector( diffhist2nd )


  # TODO Is this logic fine ?
  # TODO should be taken normal aPeaks

  # show_vector( hist1st )
  markov.smooth( hist2nd, smoothingWindow )
  # show_vector( hist1st )
  peaks2nd = peakMarkov( hist2nd, 0.33, 0.025, radiusSize*10,0,radiusSize )
  peaks2nd.sort()
  # print peaks2nd
  if len(peaks2nd) > 1:
    targethist = diffhist2nd
    targethistEdges = diffhist2ndEdges
    width=10
    nbins=radiusSize
  else:
    markov.smooth( hist2nd, smoothingWindow )
    markov.smooth( hist2nd, smoothingWindow )
    targethist = hist2nd
    targethistEdges = hist2ndEdges
    width=5
    nbins=radiusSize*10 # TODO rm when no ROOT

  maxbinIndex = targethist.argmax()
  maxbinCenter = (targethistEdges[maxbinIndex] + targethistEdges[maxbinIndex+1])/2.0
  # dMeanON, dSigON, dMeanErON, dSigErON = \
      # peak1Gaus( targethist, maxbinCenter, width, nbins,0,radiusSize )
  peak, sigma = fitGaus( targethistEdges, targethist, maxbinCenter, 30, 0, radiusSize, radiusSize, draw=False )
  # print dMeanON, dSigON, dMeanErON, dSigErON 
  # print dMeanON, maxbinCenter


  return (peak, sigma)
  # return (dMeanON, dSigON, dMeanErON, dSigErON,polarImage, image)
        




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
  X,Y = indices( image.shape )
  Xc=X-rowCenter
  Yc=Y-colCenter
  r = np.around( np.sqrt(Xc**2+Yc**2) )
  at3 = np.arctan2(Yc,Xc)
  # convert angles < 0 to positive
  at3[ at3<0.0 ] += 2*np.pi
  at3 *= polarSize/(2*pi)
  r = r.astype(int)
  at3 = at3.astype(int)
  radiusSize = np.amax(r)+1
  polarImage=zeros((amax(at3)+1,radiusSize))
  it = np.nditer(image, flags=['multi_index'])
  # polarize
  while not it.finished:
    polarImage[ at3[it.multi_index],r[it.multi_index] ] += it[0]
    it.iternext()

  return (polarImage, radiusSize )

def peakMarkovPorted( row, sigma, threshold, histedges = None ):
  # print "markov.search( ssize = %s" % len(row)
  peaks = markov.search( row, sigma, threshold )
  #peaks[i] =  (int)(peaks[i] + 0.5) + 0.5;
  result = []

  if histedges is not None:
    first = histedges[0]
    i = 0
    for p in peaks:
      j = first + int(p + 0.5);
      center = (histedges[j] + histedges[j+1])/2.0
      result.append( center )
      i+=1

  return result

  #    for (i = 0; i < npeaks; i++) {
  #       bin = first + Int_t(fPositionX[i] + 0.5);
  #       fPositionX[i] = hin->GetBinCenter(bin);
  #       fPositionY[i] = hin->GetBinContent(bin);
  #    }

def peakMarkovInternal( row, sigma, threshold, histedges ):
  S=TSpectrum()
  # hist = TH1D('','',hbins, hmin, hmax)
  # setBinsToAr1D(hist,row)
  size = len(row)
  # how many peaks
  if (sigma < 1):
     sigma = size/100
     if sigma < 1: sigma = 1
     if sigma > 8: sigma = 8

  dest = np.zeros(100,dtype=np.float)
  # npeaks = S.Search(hist,sigma,'goff', threshold)
  npeaks = S.SearchHighRes(row, dest, size, sigma, 100*threshold,
                             True, 3, True, 3)
  result = []

  first = histedges[0]
  i = 0
  for p in S.fPositionX:
    j = first + int(p + 0.5);
    center = (histedges[j] + histedges[j+1])/2.0
    result.append( center )
    i+=1

  return result

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
def peakMarkovVerbose( row, sigma, threshold, hbins, hmin, hmax):
  S=TSpectrum()
  hist = TH1D('','',hbins, hmin, hmax)
  setBinsToAr1D(hist,row)
  # how many peaks
  npeaks = S.Search(hist,sigma,'goff', threshold, True)
  # peaks positions in ROOT format...
  posROOT = S.GetPositionX()
  pos = rwBuf2Array( posROOT, npeaks)
  return pos

# has to be sorted??? TODO
def fitGaus( xdata, ydata, peak, width, histmin, histmax, histres,maxfev=0, draw=False):

  fitfunc = lambda p, x: p[0]*exp(-0.5*((x-p[1])/p[2])**2)
  errfunc  = lambda p, x, y: (y - fitfunc(p, x))

  init  = [ 1.0, peak, 1.0]

  left = peak - float(width)/2
  if left < histmin:
    print "!!! Below boundaries"
    return init
  left -= histmin
  left *=  histres/(histmax-histmin)
  left = int(left)
  right = peak + float(width)/2
  if right > histmax:
    print "!!! Above boundaries"
    return init
  right -= histmin
  right *=  histres/(histmax-histmin)
  right = int(right)

  # TODO better and in the class
  # centers of bins
  xdata1 = np.array( [ ((xdata[i]+xdata[i+1])/2.0) for i in xrange(left,right+1) ] )
  # xdata1 = xdata1 - peak

  # show_vector( xdata1 )
  ydata1 = ydata[left:right+1]
  # show_vector( ydata1 )


  out   = leastsq( errfunc, init, args=(xdata1, ydata1), maxfev=maxfev)
  c = out[0]
  # print c
  xdelta = c[1]
  sigma  = c[2]

  if draw:
    import pylab
    pylab.plot(xdata1, ydata1)
    pylab.plot(xdata1, fitfunc(c, xdata1))
    pylab.title(r'$A = %.6f\  \mu = %.6f\  \sigma = %.6f$' %(c[0],c[1],c[2]));
    pylab.show()

  return (xdelta,sigma)



def peak1Gaus( row,peak,width,hbins, hmin, hmax, write=True):
  # returns a list of tuples (mean,sigma,errMean,errSig), one entry for each peak in peaks

  if write==True:
    hist = TH1D('','',hbins, hmin, hmax)
    setBinsToAr1D(hist,row)
  else:
    hist = row
  # hist.Draw(); raw_input("continue?\n")

  gf= hist.Fit('gaus','QSNO','goff',peak-width,peak+width)

  return (gf.Value(1),gf.Value(2),gf.Error(1),gf.Error(2))


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

    fits.append((gf.Value(1),gf.Value(2),gf.Error(1),gf.Error(2)))
    # print "Vn %s n=%s m=%s x=%s" % (gf.Value(1), nBins, minBin, maxBin) 
  return fits

def main(argv=sys.argv):
    im=argv[1]
    center=(argv[2],argv[3])
    
    p = findPeaks( im, center, verbose=True )
    print p[:4]
    
import matplotlib.pyplot as plt

def show_array( a ):
  plt.imshow( a )
  plt.colorbar(orientation='horizontal')
  plt.show()
    
def show_vector( v ):
  x = arange(v.shape[0])
  plt.plot( x, v )
  plt.show()

import unittest



def smooth_np(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def test():

    files = [ 
      (143, (61.508508907329706, 0.9031407554284832, 0.06134643997660053, 0.03940823942452154) ),
      (142, (53.84568342941077, 0.8622330357987071, 0.05311658514998754, 0.05586492186835462) ),
      (137, (37.12425793776592, 4.454433844722973, 0.8194620422574127, 1.3707865416975364) ),
      (158, (174.63267829976627, 0.20404616811924745, 0.024227341549215325, 0.023505703755918456) ),
      (166, (235.59694110948064, 0.20495192682856153, 0.02373238792327561, 0.021109767427827095) ),
        ]

    for f,r in files:
      p=findPeaks("SFU/raw/latest_%07d_caz.tiff"%f, (350,200),verbose=False)
      # print p[:4]
      if p[:4] != r:
        print
        print "Diff for %s: " % f,
        for i in range(2):
          print "%.4f" % abs((p[i]-r[i])/r[i]), "(%.4f,%.4f)" % (r[i],p[i]),
        print
        print
      else:
        print "Exact results."

if __name__ == '__main__':
    # main()
    test()
    #h = a_histogram( [1,1,2,3,6,6], 0,10,11 )
    #print "bins", h.bins
    #print "edges", h.edges
    #print "centers", h.centers
