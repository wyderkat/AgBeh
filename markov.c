#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *smooth(PyObject *self, PyObject *args); 

static PyMethodDef methods[] = {
  { "smooth", smooth, METH_VARARGS, "Smooth using Markov chain"},
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

