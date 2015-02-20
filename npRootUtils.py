import ROOT as r
from numpy import *
def setBinsToAr1D(hist,ar):#,xlow,xup):
    for i in range(len(ar)):
        hist.SetBinContent(i+1,ar[i])
def makeTH1fFromAr1D(ar,name='array',title='title',xlow=0,xup=None):
    # TH1(const char* name, const char* title, Int_t nbinsx, Double_t xlow, Double_t xup)
    nbinsx=len(ar)
    if not xup:
        xup=nbinsx

    tHist=r.TH1F(name,title,nbinsx,xlow,xup)
    setBinsToAr1D(tHist,ar)
    return tHist

def rwBuf2Array(buf,bufLen):
    al=[buf[idx] for idx in range(bufLen)]
    return array(al)



