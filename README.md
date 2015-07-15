You'll need:

OpenCV -> apt-get install python-opencv should do it.

Cern ROOT -> https://root.cern.ch/drupal/content/downloading-root

Get the latest ROOT version (6.04/00 at the time of this writing, earlier versions will do a core-dump if fitting fails, so stay away from that). I just used the tarball for my linux distribution, and put it at /usr/local/root/

Then to set up all the relevant PATH variables, put this in your ~/.bashrc

. /usr/local/root/bin/thisroot.sh


++++++++++++

Not sure if you need gfortran to run the little python module written in fortran. If polarize chokes, that's probably why.

To test the agbehPeakFinder_SL module on real saxs images, run

./testRingFinder.py SFU/raw 350 200

350 200 is the beam center in all of the sample images. Long dimension is up/down (so it's Y, zero at the top).


testRingFinder.py shows how to use the module.


