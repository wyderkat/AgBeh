#!/usr/bin/env python
# encoding: utf-8

import sys
import cv2
import numpy as np
import scipy.optimize
import markov

from pilatus_np import JJTiff

class a_histogram(object):
  def __init__( me, data, lower, upper, resolution ):
    me.lower = lower
    me.upper = upper
    me.resolution = resolution

    me.bins, me.edges = \
      np.histogram( data , bins=me.resolution, range=(me.lower,me.upper) )
    me.bins = me.bins.astype( np.float )

    me.centers = np.zeros_like( me.bins, dtype=np.float )
    for center, left, right in np.nditer(
        [me.centers, me.edges[:-1], me.edges[1:]],
        op_flags=['readwrite']):
      center[...] = (left+right)/2.0;

    # len(me.edges) == len(me.bins) + 1
    # len(me.centers) == len(me.bins)

  def bin_to_idx( me, value ):
    value -= me.lower
    value *=  me.resolution/(me.upper-me.lower)
    value = int(value)
    return value

  def maxbin_center( me ):
    idx = me.bins.argmax()
    return me.centers[ idx ]




def findDistance(
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
    lastPeak = radiusSize
  # print lastPeak # 453

# 1st STAGE

  # run a gaus filter over the polar image to blend in the rough spots
  polarImage = cv2.GaussianBlur(polarImage,(3,3),0)
  # show_array( polarImage )


  allpeaks1st = []
  for row in polarImage:
    markov.smooth( row, smoothingWindow )
    allpeaks1st.extend( peakMarkov( row, 1.0, peakThresh ) )
  # print allpeaks1st
  # show_array( polarImage )
  allpeaks1st = np.array( [x for x in allpeaks1st if x>=firstPeak and x<=lastPeak] )
  # print allpeaks1st
  hist1st = a_histogram( allpeaks1st, 0, radiusSize, radiusSize*10 )
  # show_vector( hist1st.bins )


  # be careful, dtype has to be dynamic (float, but no float32 or no float64)
  markov.smooth( hist1st.bins, smoothingWindow )
  # show_vector( hist1st.bins )
  markov.smooth( hist1st.bins, smoothingWindow )
  # show_vector( hist1st.bins )

  peaks1st = peakMarkov( hist1st, 0.33, 0.025 )
  if not peaks1st: 
    return (0,0)
  # print "\n%s\n" % peaks1st
  peaks1st.sort()
  # print "peaks1st", peaks1st
  
  peaks1st = [ fitGaus( hist1st, p, 30 )[0] for p in peaks1st ]
  peaks1st = np.unique(peaks1st)[0:maxNPeaks]
  # print peaks1st


# 2nd STAGE

  allpeaks2nd = []
  diffs2nd = []
  for row in polarImage:
    xdata = np.arange( len(row)+1, dtype=np.float )
    peaks = [ fitGaus( row, p, 30 )[0]  for p in peaks1st ]

    peaks = [ x for x in peaks if x>=firstPeak and x<=lastPeak ]
    peaks.sort()
    allpeaks2nd.extend( peaks )
    diffs2nd.extend( np.diff( peaks ) ) # still not a numpy array

  allpeaks2nd = np.array( allpeaks2nd )
  # print allpeaks2nd
  diffs2nd = np.array( diffs2nd )

  hist2nd = a_histogram( allpeaks2nd, 0, radiusSize, radiusSize*10 )
  # show_vector( hist2nd.bins )

  diffs2nd = diffs2nd[ diffs2nd>=minDiff ]
  diffhist2nd = a_histogram( diffs2nd, 0, radiusSize, radiusSize )
  # show_vector( diffhist2nd.bins )


  markov.smooth( hist2nd.bins, smoothingWindow )
  peaks2nd = peakMarkov( hist2nd, 0.33, 0.025 )
  if not peaks2nd: 
    return (0,0)
  peaks2nd.sort()
  # print peaks2nd
  if len(peaks2nd) > 1:
    targethist = diffhist2nd
    width=10
  else:
    markov.smooth( hist2nd.bins, smoothingWindow )
    markov.smooth( hist2nd.bins, smoothingWindow )
    targethist = hist2nd
    width=5

  p = targethist.maxbin_center()
  peak, sigma = fitGaus( targethist, p, 30 )

  return (peak, sigma)
        




def retrieveImage(filePath,clearVoids=False,makeU8=False,doLog=False):
    # return a np array (2dim) of a saxslab tiff file

    try:
        
        jjIm=JJTiff(filePath,bars = 0.0, nega = 0.0)
        image=np.rot90(np.fliplr(jjIm.arr)) # these aren't indexed the way I had them before. Easier to fix than to rewrite all my code.
        
    except TypeError: # TIFF.open returns a TypeError when it can't read a file. NOTE: This may be cruft now that I use JJTiff
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
        im8=array(image/float(imMax)*255,dtype='uint8')
        return im8
    else:
        return image

def imageToPolar( image, center, polarSize ):

  rowCenter=float(center[0])
  colCenter=float(center[1])
  X,Y = np.indices( image.shape )
  Xc=X-rowCenter
  Yc=Y-colCenter
  r = np.around( np.sqrt(Xc**2+Yc**2) )
  at3 = np.arctan2(Yc,Xc)
  # convert angles < 0 to positive
  at3[ at3<0.0 ] += 2*np.pi
  at3 *= polarSize/(2*np.pi)
  r = r.astype(int)
  at3 = np.rint( at3 )
  at3 = at3.astype(int)
  radiusSize = np.amax(r)+1
  polarImage= np.zeros((np.amax(at3)+1,radiusSize))
  it = np.nditer(image, flags=['multi_index'])
  # polarize
  while not it.finished:
    polarImage[ at3[it.multi_index],r[it.multi_index] ] += it[0]
    it.iternext()

  return (polarImage, radiusSize )


def peakMarkov( container, sigma, threshold ):
  if isinstance( container, a_histogram ):
    data = container.bins
  else:
    data = container

  peaks = markov.search( data, sigma, threshold )

  result = []
  for p in peaks:
    if isinstance( container, a_histogram ):
      # Michael, do you think this is OK? From TSpectrum.cxx
      first = container.edges[0]
      # Michael, do you think this is OK? From TSpectrum.cxx
      j = first + int(p + 0.5);
      center = container.centers[j]
    else:
      # Michael, do you think this is OK? From TSpectrum.cxx
      center = int(p + 0.5) + 0.5
    result.append( center )

  return result

def fitGaus( container, peak, width, maxfev=0, draw=False):

  fitfunc = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
  errfunc  = lambda p, x, y: (y - fitfunc(p, x))

  init  = [ 1.0, peak, 1.0]

  if isinstance( container, a_histogram ):
    lower = container.lower
    upper = container.upper
  else:
    lower = 0
    upper = len(container)

  left = peak - float(width)/2
  right = peak + float(width)/2

  if left < lower:
    print "!!! Below lower boundry: %s < %s" % (left, lower)
    return init
  if right > upper:
    print "!!! Above upper boundry: %s > %s" % (right, upper)
    return init

  if isinstance( container, a_histogram ):
    left = container.bin_to_idx( left )
    right = container.bin_to_idx( right )
    
    # Michael, in TSpectrum.cxx is actually right+1. Any thoughts?
    xdata = container.centers[ left : right ]
    ydata = container.bins   [ left : right ]
  else:
    left = int(left)
    right = int(right)

    # Michael, in TSpectrum.cxx is actually right+1. Any thoughts?
    xdata = np.arange( left, right )
    ydata = container[ left : right ]


  out   = scipy.optimize.leastsq( errfunc, init, args=(xdata, ydata), maxfev=maxfev)
  C = out[0]
  # print C
  xdelta = C[1]
  sigma  = C[2]

  if draw:
    import pylab
    pylab.plot(xdata, ydata)
    pylab.plot(xdata, fitfunc(C, xdata))
    pylab.title(r'$A = %.6f\  \mu = %.6f\  \sigma = %.6f$' %(C[0],C[1],C[2]));
    pylab.show()

  return (xdelta,sigma)




def main(argv=sys.argv):
    im=argv[1]
    center=(argv[2],argv[3])
    
    p = findDistance( im, center, verbose=True )
    print p[:4]
    

def test():

    files = [ 
      (143, (61.508508907329706, 0.9031407554284832, 0.06134643997660053, 0.03940823942452154) ),
      (142, (53.84568342941077, 0.8622330357987071, 0.05311658514998754, 0.05586492186835462) ),
      (137, (37.12425793776592, 4.454433844722973, 0.8194620422574127, 1.3707865416975364) ),
      (158, (174.63267829976627, 0.20404616811924745, 0.024227341549215325, 0.023505703755918456) ),
      (166, (235.59694110948064, 0.20495192682856153, 0.02373238792327561, 0.021109767427827095) ),
        ]

    for f,r in files:
      p=findDistance("SFU/raw/latest_%07d_caz.tiff"%f, (350,200),verbose=False)
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

def test1():

    files = [ 
      (143, (61.508508907329706, 0.9031407554284832, 0.06134643997660053, 0.03940823942452154) ),
        ]

    for f,r in files:
      p=findDistance("SFU/raw/latest_%07d_caz.tiff"%f, (350,200),verbose=False)
      # print p[:4]


import matplotlib.pyplot as plt

def show_array( a ):
  plt.imshow( a )
  plt.colorbar(orientation='horizontal')
  plt.show()
    
def show_vector( v ):
  x = arange(v.shape[0])
  plt.plot( x, v )
  plt.show()



if __name__ == '__main__':
    # main()
    test()
