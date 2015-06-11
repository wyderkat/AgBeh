from ROOT import TH1D
from numpy import *
def setBinsToAr1D(hist,ar):#,xlow,xup):
    for i in range(len(ar)):
        hist.SetBinContent(i+1,ar[i])

def setAr1DtoBins(hist):
    nBins=hist.GetNbinsX()
    ar=zeros((nBins))
    ax=zeros((nBins))
    for idx in range(nBins):
        ar[idx]=hist.GetBinContent(idx+1)
        ax[idx]=hist.GetBinCenter(idx+1)
    return (ar,ax)
# Default x-axis is 0-len(ar). Pass different xlow and xup to change x-axis
def makeTH1DFromAr1D(ar,name='array',title='title',xlow=0,xup=None):
    # TH1(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup)
    nbinsx=len(ar)
    if not xup:
        xup=nbinsx

    tHist=TH1D(name,title,nbinsx,xlow,xup)
    setBinsToAr1D(tHist,ar)
    return tHist

def rwBuf2Array(buf,bufLen):
    al=[buf[idx] for idx in range(bufLen)]
    return array(al)

def fill_hist(th1,ar1):
    # fill a th1 from a 1d np array
    # print 'filling',ar1
    for idx in range(len(ar1)):
        th1.Fill(ar1[idx])


def fitGausPeaks(th,peaks,width=30):
    # th:    a tHist which we've done some peak fitting to, and we want to get gaussian fits to those peaks
    # peaks: an np array of the approx x coords of the peaks we want to fit.
    # returns a list of tuples (mean,sigma,errMean,errSig), one entry for each peak in peaks

    peaks.sort()
    # print peaks
    # dxP=diff(peaks)
    # peaks: [ 231.5,  245.5,  260.5,  274.5,  288.5]
    # dxP:         [ 14.,    15.,    14.,    14.]
    fits=[]

    # y,x=setAr1DtoBins(th)
    nBins=th.GetNbinsX() #thists have nBins+2 bins - 0 is underflow and nBins+1 is overflow.
    minBin=th.GetBinCenter(1)
    maxBin=th.GetBinCenter(nBins)

    for idx in range(len(peaks)):
        
        # root doesn't seem to mind if we put fit limits outside of lowBin,highBin. So 
        # we don't bother to check if we would overflow the range of the histo with our
        # fit limits.
        if peaks[idx]<minBin and peaks[idx] > maxBin:
            print 'fitGausPeaks: ERROR ***************** peak outside of histogram range.'
            print 'Histo name: ',th.GetName(),'\nHisto Title: ',th.GetTitle()
            continue
        # print 'fit: low,high ',peaks[idx]-width/2.,peaks[idx]+width/2.
        gf=th.Fit('gaus','QSNO','goff',peaks[idx]-width/2.,peaks[idx]+width/2.)
        # print 'did we fit?\t',gf
        # if gf:
            # print 'gf: ',gf.Value(1),gf.Value(2),gf.Error(1),gf.Error(2)
        fits.append((gf.Value(1),gf.Value(2),gf.Error(1),gf.Error(2)))

        

        
    return fits


 # EXT PARAMETER                                   STEP         FIRST   
 #  NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE 
 #   1  Constant     8.59870e+01   6.10376e+00   1.45273e-02  -4.15609e-05
 #   2  Mean         2.59977e+02   1.20712e-01   4.91596e-04   1.20419e-04
 #   3  Sigma        2.81622e+00   1.74553e-01   4.66133e-05  -4.52088e-03


        # In [78]: gf.Value(0)
        # Out[78]: 479.3979941558617

        # In [79]: gf.Value(1)
        # Out[79]: 231.6665489234626

        # In [80]: gf.Value(2)
        # Out[80]: 1.606575935147517
