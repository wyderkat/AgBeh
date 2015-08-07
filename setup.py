#!/usr/bin/env python

from distutils.core import setup, Extension

setup (name = 'Markov',
       version = '1.0',
       description = 'Markov based smoothing and peak finding',
       ext_modules = [ Extension('markov', sources = ['markov.c'],
              extra_compile_args=["-Ofast", "-march=native"] ) ] )

  # -Ofast -O3 is needed for accesing numpy arrays




