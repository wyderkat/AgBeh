#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdbool.h>

static PyObject *smooth(PyObject *self, PyObject *args); 
static PyObject *search(PyObject *self, PyObject *args); 

static PyMethodDef methods[] = {
  { "smooth", smooth, METH_VARARGS, "Smooth using Markov chain"},
  { "search", search, METH_VARARGS, "Peak finder with Markov chain"},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

PyMODINIT_FUNC initmarkov(void) {
  (void) Py_InitModule("markov", methods);
  import_array();
}


static PyObject *smooth(PyObject *self, PyObject *args) {
  PyArrayObject *py_source;
  // NO npy_float, npy_float64 because we need for 32 and 64
  double *source=NULL;
  int ssize;
  int averWindow;

  if (!PyArg_ParseTuple(args, "O!i",
                        &PyArray_Type, &py_source, &averWindow)) {
    return NULL;
  }
  
  source = (double*)PyArray_DATA(py_source);
  ssize = (int)PyArray_DIM(py_source,0);

  // From ROOT.TSpectrum
  //
  int xmin, xmax, i, l;
  double a, b, maxch;
  double nom, nip, nim, sp, sm, area = 0;
  double *working_space = calloc( ssize, sizeof(double) );
  xmin = 0;
  xmax = ssize - 1;
  for(i = 0, maxch = 0; i < ssize; i++){
     if(maxch < source[i])
        maxch = source[i];

     area += source[i];
  }
  if(maxch == 0) {
     free( working_space );
     Py_RETURN_NONE;
  }

  nom = 1;
  working_space[xmin] = 1;
  for(i = xmin; i < xmax; i++){
     nip = source[i] / maxch;
     nim = source[i + 1] / maxch;
     sp = 0,sm = 0;
     for(l = 1; l <= averWindow; l++){
        if((i + l) > xmax)
           a = source[xmax] / maxch;

        else
           a = source[i + l] / maxch;
        b = a - nip;
        if(a + nip <= 0)
           a = 1;

        else
           a = sqrt(a + nip);
        b = b / a;
        b = exp(b);
        sp = sp + b;
        if((i - l + 1) < xmin)
           a = source[xmin] / maxch;

        else
           a = source[i - l + 1] / maxch;
        b = a - nim;
        if(a + nim <= 0)
           a = 1;
        else
           a = sqrt(a + nim);
        b = b / a;
        b = exp(b);
        sm = sm + b;
     }
     a = sp / sm;
     a = working_space[i + 1] = working_space[i] * a;
     nom = nom + a;
  }
  for(i = xmin; i <= xmax; i++){
     working_space[i] = working_space[i] / nom;
  }
  for(i = 0; i < ssize; i++)
     source[i] = working_space[i] * area;
  free(working_space);

  Py_RETURN_NONE;
}

#define PEAK_WINDOW 1024
#define MAX_PEAKS 100

static int search_ROOT(
  double *source,
  /*double *destVector,*/
  int ssize,
  double sigma, 
  double threshold,
  bool backgroundRemove,
  int deconIterations,
  bool markov, 
  int averWindow, 
  double* resultPeaks)
{
    printf("TomMarkov: ssize=%d, sigma=%f, th=%f, bg=%d, iter=%d, markov=%d, win=%d\n",
      ssize, sigma, threshold, backgroundRemove, deconIterations, markov, averWindow);
  // ROOT's code. Don't touch that ... puzzle!
   int i, j, numberIterations = (int)(7 * sigma + 0.5);
   double a, b, c;
   int k, lindex, posit, imin, imax, jmin, jmax, lh_gold, priz;
   double lda, ldb, ldc, area, maximum, maximum_decon;
   int xmin, xmax, l, peak_index = 0, size_ext = ssize + 2 * numberIterations, shift = numberIterations, bw = 2, w;
   double maxch;
   double nom, nip, nim, sp, sm, plocha = 0;
   double m0low=0,m1low=0,m2low=0,l0low=0,l1low=0,detlow,av,men;
   if (sigma < 1) {
      return 0;
   }
   if(threshold<=0 || threshold>=100){
      return 0;
   }

   j = (int) (5.0 * sigma + 0.5);
   if (j >= PEAK_WINDOW / 2) {
      //Error("SearchHighRes", "Too large sigma");
      return 0;
   }

   if (markov == true) {
      if (averWindow <= 0) {
         //Error("SearchHighRes", "Averanging window must be positive");
         return 0;
      }
   }

   if(backgroundRemove == true){
      if(ssize < 2 * numberIterations + 1){
         //Error("SearchHighRes", "Too large clipping window");
         return 0;
      }
   }

   k = (int)(2 * sigma+0.5);
   if(k >= 2){
      for(i = 0;i < k;i++){
         a = i,b = source[i];
         m0low += 1,m1low += a,m2low += a * a,l0low += b,l1low += a * b;
      }
      detlow = m0low * m2low - m1low * m1low;
      if(detlow != 0)
         l1low = (-l0low * m1low + l1low * m0low) / detlow;

      else
         l1low = 0;
      if(l1low > 0)
         l1low=0;
   }

   else{
      l1low = 0;
   }

   i = (int)(7 * sigma + 0.5);
   i = 2 * i;
   double *working_space = calloc( 7*(ssize+i), sizeof(double) );

   for(i = 0; i < size_ext; i++){
      if(i < shift){
         a = i - shift;
         working_space[i + size_ext] = source[0] + l1low * a;
         if(working_space[i + size_ext] < 0)
            working_space[i + size_ext]=0;
      }

      else if(i >= ssize + shift){
         a = i - (ssize - 1 + shift);
         working_space[i + size_ext] = source[ssize - 1];
         if(working_space[i + size_ext] < 0)
            working_space[i + size_ext]=0;
      }

      else
         working_space[i + size_ext] = source[i - shift];
   }

   if(backgroundRemove == true){
      for(i = 1; i <= numberIterations; i++){
         for(j = i; j < size_ext - i; j++){
            if(markov == false){
               a = working_space[size_ext + j];
               b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
               if(b < a)
                  a = b;

               working_space[j]=a;
            }

            else{
               a = working_space[size_ext + j];
               av = 0;
               men = 0;
               for (w = j - bw; w <= j + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     av += working_space[size_ext + w];
                     men +=1;
                  }
               }
               av = av / men;
               b = 0;
               men = 0;
               for (w = j - i - bw; w <= j - i + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     b += working_space[size_ext + w];
                     men +=1;
                  }
               }
               b = b / men;
               c = 0;
               men = 0;
               for (w = j + i - bw; w <= j + i + bw; w++){
                  if ( w >= 0 && w < size_ext){
                     c += working_space[size_ext + w];
                     men +=1;
                  }
               }
               c = c / men;
               b = (b + c) / 2;
               if (b < a)
                  av = b;
               working_space[j]=av;
            }
         }
         for(j = i; j < size_ext - i; j++)
            working_space[size_ext + j] = working_space[j];
      }
      for(j = 0;j < size_ext; j++){
         if(j < shift){
                  a = j - shift;
                  b = source[0] + l1low * a;
                  if (b < 0) b = 0;
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else if(j >= ssize + shift){
                  a = j - (ssize - 1 + shift);
                  b = source[ssize - 1];
                  if (b < 0) b = 0;
            working_space[size_ext + j] = b - working_space[size_ext + j];
         }

         else{
            working_space[size_ext + j] = source[j - shift] - working_space[size_ext + j];
         }
      }
      for(j = 0;j < size_ext; j++){
         if(working_space[size_ext + j] < 0) working_space[size_ext + j] = 0;
      }
   }

   for(i = 0; i < size_ext; i++){
      working_space[i + 6*size_ext] = working_space[i + size_ext];
   }

   if(markov == true){
      for(j = 0; j < size_ext; j++)
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      xmin = 0,xmax = size_ext - 1;
      for(i = 0, maxch = 0; i < size_ext; i++){
         working_space[i] = 0;
         if(maxch < working_space[2 * size_ext + i])
            maxch = working_space[2 * size_ext + i];
         plocha += working_space[2 * size_ext + i];
      }
      if(maxch == 0) {
         free(working_space);
         return 0;
      }

      nom = 1;
      working_space[xmin] = 1;
      for(i = xmin; i < xmax; i++){
         nip = working_space[2 * size_ext + i] / maxch;
         nim = working_space[2 * size_ext + i + 1] / maxch;
         sp = 0,sm = 0;
         for(l = 1; l <= averWindow; l++){
            if((i + l) > xmax)
               a = working_space[2 * size_ext + xmax] / maxch;

            else
               a = working_space[2 * size_ext + i + l] / maxch;

            b = a - nip;
            if(a + nip <= 0)
               a=1;

            else
               a = sqrt(a + nip);

            b = b / a;
            b = exp(b);
            sp = sp + b;
            if((i - l + 1) < xmin)
               a = working_space[2 * size_ext + xmin] / maxch;

            else
               a = working_space[2 * size_ext + i - l + 1] / maxch;

            b = a - nim;
            if(a + nim <= 0)
               a = 1;

            else
               a = sqrt(a + nim);

            b = b / a;
            b = exp(b);
            sm = sm + b;
         }
         a = sp / sm;
         a = working_space[i + 1] = working_space[i] * a;
         nom = nom + a;
      }
      for(i = xmin; i <= xmax; i++){
         working_space[i] = working_space[i] / nom;
      }
      for(j = 0; j < size_ext; j++)
         working_space[size_ext + j] = working_space[j] * plocha;
      for(j = 0; j < size_ext; j++){
         working_space[2 * size_ext + j] = working_space[size_ext + j];
      }
      if(backgroundRemove == true){
         for(i = 1; i <= numberIterations; i++){
            for(j = i; j < size_ext - i; j++){
               a = working_space[size_ext + j];
               b = (working_space[size_ext + j - i] + working_space[size_ext + j + i]) / 2.0;
               if(b < a)
                  a = b;
               working_space[j] = a;
            }
            for(j = i; j < size_ext - i; j++)
               working_space[size_ext + j] = working_space[j];
         }
         for(j = 0; j < size_ext; j++){
            working_space[size_ext + j] = working_space[2 * size_ext + j] - working_space[size_ext + j];
         }
      }
   }
//deconvolution starts
   area = 0;
   lh_gold = -1;
   posit = 0;
   maximum = 0;
//generate response vector
   for(i = 0; i < size_ext; i++){
      lda = (double)i - 3 * sigma;
      lda = lda * lda / (2 * sigma * sigma);
      j = (int)(1000 * exp(-lda));
      lda = j;
      if(lda != 0)
         lh_gold = i + 1;

      working_space[i] = lda;
      area = area + lda;
      if(lda > maximum){
         maximum = lda;
         posit = i;
      }
   }
//read source vector
   for(i = 0; i < size_ext; i++)
      working_space[2 * size_ext + i] = abs(working_space[size_ext + i]);
//create matrix at*a(vector b)
   i = lh_gold - 1;
   if(i > size_ext)
      i = size_ext;

   imin = -i,imax = i;
   for(i = imin; i <= imax; i++){
      lda = 0;
      jmin = 0;
      if(i < 0)
         jmin = -i;
      jmax = lh_gold - 1 - i;
      if(jmax > (lh_gold - 1))
         jmax = lh_gold - 1;

      for(j = jmin;j <= jmax; j++){
         ldb = working_space[j];
         ldc = working_space[i + j];
         lda = lda + ldb * ldc;
      }
      working_space[size_ext + i - imin] = lda;
   }
//create vector p
   i = lh_gold - 1;
   imin = -i,imax = size_ext + i - 1;
   for(i = imin; i <= imax; i++){
      lda = 0;
      for(j = 0; j <= (lh_gold - 1); j++){
         ldb = working_space[j];
         k = i + j;
         if(k >= 0 && k < size_ext){
            ldc = working_space[2 * size_ext + k];
            lda = lda + ldb * ldc;
         }

      }
      working_space[4 * size_ext + i - imin] = lda;
   }
//move vector p
   for(i = imin; i <= imax; i++)
      working_space[2 * size_ext + i - imin] = working_space[4 * size_ext + i - imin];
//initialization of resulting vector
   for(i = 0; i < size_ext; i++)
      working_space[i] = 1;
//START OF ITERATIONS
   for(lindex = 0; lindex < deconIterations; lindex++){
      for(i = 0; i < size_ext; i++){
         if(abs(working_space[2 * size_ext + i]) > 0.00001 && abs(working_space[i]) > 0.00001){
            lda=0;
            jmin = lh_gold - 1;
            if(jmin > i)
               jmin = i;

            jmin = -jmin;
            jmax = lh_gold - 1;
            if(jmax > (size_ext - 1 - i))
               jmax=size_ext-1-i;

            for(j = jmin; j <= jmax; j++){
               ldb = working_space[j + lh_gold - 1 + size_ext];
               ldc = working_space[i + j];
               lda = lda + ldb * ldc;
            }
            ldb = working_space[2 * size_ext + i];
            if(lda != 0)
               lda = ldb / lda;

            else
               lda = 0;

            ldb = working_space[i];
            lda = lda * ldb;
            working_space[3 * size_ext + i] = lda;
         }
      }
      for(i = 0; i < size_ext; i++){
         working_space[i] = working_space[3 * size_ext + i];
      }
   }
//shift resulting spectrum
   for(i=0;i<size_ext;i++){
      lda = working_space[i];
      j = i + posit;
      j = j % size_ext;
      working_space[size_ext + j] = lda;
   }
//write back resulting spectrum
   maximum = 0, maximum_decon = 0;
   j = lh_gold - 1;
   for(i = 0; i < size_ext - j; i++){
      if(i >= shift && i < ssize + shift){
         working_space[i] = area * working_space[size_ext + i + j];
         if(maximum_decon < working_space[i])
            maximum_decon = working_space[i];
         if(maximum < working_space[6 * size_ext + i])
            maximum = working_space[6 * size_ext + i];
      }

      else
         working_space[i] = 0;
   }
   lda=1;
   if(lda>threshold)
      lda=threshold;
   lda=lda/100;

//searching for peaks in deconvolved spectrum
   for(i = 1; i < size_ext - 1; i++){
      if(working_space[i] > working_space[i - 1] && working_space[i] > working_space[i + 1]){
         if(i >= shift && i < ssize + shift){
            if(working_space[i] > lda*maximum_decon && working_space[6 * size_ext + i] > threshold * maximum / 100.0){
               for(j = i - 1, a = 0, b = 0; j <= i + 1; j++){
                  a += (double)(j - shift) * working_space[j];
                  b += working_space[j];
               }
               a = a / b;
               if(a < 0)
                  a = 0;

               if(a >= ssize)
                  a = ssize - 1;
               if(peak_index == 0){
                  resultPeaks[0] = a;
                  peak_index = 1;
               }

               else{
                  for(j = 0, priz = 0; j < peak_index && priz == 0; j++){
                     if(working_space[6 * size_ext + shift + (int)a] > working_space[6 * size_ext + shift + (int)resultPeaks[j]])
                        priz = 1;
                  }
                  if(priz == 0){
                     if(j < MAX_PEAKS){
                        resultPeaks[j] = a;
                     }
                  }

                  else{
                     for(k = peak_index; k >= j; k--){
                        if(k < MAX_PEAKS){
                           resultPeaks[k] = resultPeaks[k - 1];
                        }
                     }
                     resultPeaks[j - 1] = a;
                  }
                  if(peak_index < MAX_PEAKS)
                     peak_index += 1;
               }
            }
         }
      }
   }

   free(working_space);
   return peak_index;
}

static PyObject *search(PyObject *self, PyObject *args) {
  PyArrayObject *py_source;
  // NO npy_float, npy_float64 because we need for 32 and 64
  double *source=NULL;
  int ssize;
  double sigma;
  double threshold;
  double *peaks;
  int peakscount;
  int i;
  PyObject *resultpeaks,*item;

  if (!PyArg_ParseTuple(args, "O!dd",
                        &PyArray_Type, &py_source, &sigma, &threshold)) {
    return NULL;
  }
  
  source = (double*)PyArray_DATA(py_source);
  ssize = (int)PyArray_DIM(py_source,0);

  // From TSpectrum.Search
  if (sigma < 1) {
    //sigma = (float)ssize/MAX_PEAKS;
    sigma = ssize/MAX_PEAKS;
    if (sigma < 1) sigma = 1;
    if (sigma > 8) sigma = 8;
  }

  peaks = calloc(MAX_PEAKS, sizeof(double));
  
  peakscount = search_ROOT( source, ssize, sigma, threshold*100, true, 3, true, 3, peaks);
  
  resultpeaks = PyList_New(peakscount);
  for(i=0; i<peakscount; ++i) {
    item = PyFloat_FromDouble( peaks[i] );
    PyList_SetItem( resultpeaks, i, item );
  }

  free(peaks);
  return resultpeaks;
}

