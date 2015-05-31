###
# Copyright 2014 Tomasz Wyderka <wyderkat@cofoh.com>
#  www.cofoh.com
##

import jjlibtiff as jjt
import conf_system 
import re
import numpy as np
import time
#import logging as ___

bar1_start = 194
bar1_end   = 213 # because of Python bounderies, +1 !
bar2_start = 406
bar2_end   = 425 # because of Python bounderies, +1 !


class JJTiff( object ):
  """
  float32 version of JJLibTiff

  Size probably (487, 619),
  means 487 first index, and 619 second
  (dshort, dlong) [s, l]

  This class should be subclass of np.array,
  but with numpy it is not so easy.
  So now, array is under "arr" attribute

  Saving this object means using all specific tiff data 
  from contructor. Use saveas for another filename.
  """
  exposure_re = re.compile( 
    "Exposure_time (.*?) s.*Exposure_period (.*?) s", 
                           re.DOTALL)
  def __init__(me, 
               file_or_name , # filename or file object
               #TODO
               bars = np.nan , 
               nega = np.nan ,
               save_nega = False  # save negatives position to restore when saving
              ):

    if type(file_or_name) == type(""):
      me.filename = file_or_name
    else:
      me.filename = ""
    
    me.tiff = jjt.JJLibTiff( file_or_name, print_warnings=False )

    me.negatives = None

    # MAIN object
    me.arr = me.tiff.arr_ro.astype( np.float32 )

    # add bars
    if bars != None:
      me.arr[ : , bar1_start : bar1_end ] = bars
      me.arr[ : , bar2_start : bar2_end ] = bars

    # remeber negatives
    if save_nega:
      me.negatives = np.nonzero( me.arr < 0.0 ) # return indexes

    # remove negatives
    if nega != None:
      me.arr[ me.arr < 0.0 ] = nega

  def dims( me ):
    return me.arr.shape
    # probably (487, 619)

  def save(me,
            bars = -1.0, # has to be float, convertion to int inside
            nan = -2.0,
            round = True
           ):
    me.save_as( me.filename, bars, nan, round )

  def save_as(me,
              new_filename, 
              bars = -1, # has to be int
              nan = -2,
              round = True
             ):

    if round:
      # round to the nearest integer
      arr1 = np.rint( me.arr )
    else:
      arr1 = me.arr

    arr2 = arr1.astype(np.int32) # deep copy is needed too

    if bars != None:
      arr2[ : , bar1_start : bar1_end ] = bars
      arr2[ : , bar2_start : bar2_end ] = bars
    if nan != None:
      arr2[ np.isnan( arr2 ) ] = nan
    if me.negatives != None:
      arr2[ me.negatives ] = -2

    # this is the fastes way in Image
    me.tiff.set_arr( arr2 )

    me.tiff.save_as( new_filename )

  def read_old_tag( me ):
    return me.read_tag( conf_system.OLD_TAG )

  def read_new_tag( me ):
    return me.read_tag( conf_system.NEW_TAG )

  def read_tag( me, tag_no ):
    try:
      return me.tiff.get_ascii_tag( tag_no )
    except KeyError:
      return None

  def write_old_tag(me, old_tag ):
    me.tiff.set_ascii_tag( conf_system.OLD_TAG, old_tag )

  def write_new_tag(me, new_tag ):
    me.tiff.set_ascii_tag( conf_system.NEW_TAG, new_tag )
 
  def read_exposure( me ):
    exposures = ( None, None )
    tag = me.read_old_tag()
    if tag:
      matched = JJTiff.exposure_re.search( tag )
      if matched:
        exposures = ( float( matched.group(1) ), 
                      float( matched.group(2) ) 
                    )
    return exposures

  def write_exposure( me, exposures ):
    tag =  me.read_old_tag()
    if tag:
      tag1 = JJTiff.exposure_re.sub( 
        "Exposure_time %f s\r\n# Exposure_period %f s"% exposures,
        tag)
    else:
      tag1 = "# Exposure_time %f s\r\n# Exposure_period %f s\r\n"% exposures

    me.write_old_tag( tag1 ) 

if __name__ == "__main__":
  # test
  t = JJTiff("i.tiff")
  print t.read_old_tag()
  print t.read_new_tag()
  print t.read_exposure()

  
  t.write_exposure( ( 17.0, 27.0 ) )
  t.write_new_tag("New tag content")


  t.arr[0,0] = -13

  t.save_as("i_o1.tiff")

  t_o1 = JJTiff("i_o1.tiff")
  print t_o1.dims()


