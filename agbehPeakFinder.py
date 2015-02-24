#!/usr/bin/env python
# encoding: utf-8
"""
doc string here
"""
from saxsUtils import *
import cv2
from ROOT import TH1F, TSpectrum
from root_numpy import fill_hist
from numpy import *
from libtiff import TIFF


# let's write a function:   input: is good center, path to image, radius range to search for peaks.
#                           output: peak spacing (pix). How many peaks, an arr w/ peaks in that range.

def findPeaks(center,image,rad):
"""

findPeaks(center,image,rad)

Take a saxslab tiff of AgBeh, and return peak spacing, and an array of found peak coordinates.
Peaks in the range (rad[0]:rad[1]) are found by taking slices of the tiff at several rotation agles,
and searching for peaks within rad[0]:rad[1]. Then sections of the slice containing one peak each from
the previous step are fit to gaussian curves, and the mean from each fit, is added into a histogram.
Also, the spacing between all the fits in a slice is added to a histogram of peak spacings. This is all
repeated for each rotation.

At the end, we have a histogram of peak locations and a histogram of peak spacings. The peak spacing
histogram is fit to a gaussian, and the mean, sigma, and error are returned. Further, the      
 input
 center:       tuple or list -> (rowCenter,colCenter), where rowCenter is the coord in pix of the center row.
                                               colCenter is defined the same way, but for center col.
 image:        A path to a saxslab tiff, or an np array of a saxslab tiff.
 rad:          tuple or list -> (minRadius, maxRadius) in which to search for peaks.
 
 output
 tuple:        (peakSpacing, peakSpacingSigma, peakSpacingErr, peaksAr)
               peakSpacing, peakSpacingSigma, peakSpacingErr are from fitting a histogram of found peak coordinates to a gaussian. 
               peaksArr is an np.array of the the peak locations found (radius from center).
"""


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



def setBinsToAr1D(hist,ar):#,xlow,xup):
    for i in range(len(ar)):
        hist.SetBinContent(i+1,ar[i])
def makeTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):
    # TH1(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup)
    nbinsx=len(ar)
    if not xup:
        xup=nbinsx

    tHist=TH1F(name,title,nbinsx,xlow,xup)
    setBinsToAr1D(tHist,ar)
    return tHist

def rwBuf2Array(buf,bufLen):
    al=[buf[idx] for idx in range(bufLen)]
    return array(al)


def fitGausPeaks(th,peaks):
    # th:    a thist which we've done some peak fitting to, and we want to get gaussian fits to those peaks
    # peaks: an np array of the x coords of the peaks we want to fit.
    # returns a list of tuples (const,mean,sigma), one entry for each peak in peaks

    peaks.sort()
    
    dxP=diff(peaks)
    # peaks: [ 231.5,  245.5,  260.5,  274.5,  288.5]
    # dxP:         [ 14.,    15.,    14.,    14.]
    fits=[]
    for idx in range(len(peaks)):
        if idx==0:
            dm=dxP[idx]/2.
        else:
            dm=dxP[idx-1]/2.
        if idx==len(peaks)-1:
            dp=dxP[idx-1]/2.
        else:
            dp=dxP[idx]/2.

        gf=th.Fit('gaus','QSNO','goff',peaks[idx]-dm,peaks[idx]+dp)
        fits.append((gf.Value(0),gf.Value(1),gf.Value(2)))
    return fits
def main(argv=sys.argv):
    pass


if __name__ == '__main__':
    main()