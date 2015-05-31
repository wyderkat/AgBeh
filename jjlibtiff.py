#!/usr/bin/env python
###
# Copyright 2013-2014 Tomasz Wyderka <wyderkat@cofoh.com>
#  www.cofoh.com
##

# Word about images classification in TIFF
#
#     type  BiLevel     Grayscale     Colormap    RGB
# tag
# PI(262)   0 or 1       0 or 1        3          2
# BPS(258)               4 or 8        4 or 8     8,8,8
# CM(320)                              cm table

import struct as st
import numpy as np

class MissingFeatureError(Exception):
  pass

class EmptyFileError(Exception):
  pass

# print and stop
def error( str ):
  print str
  raise MissingFeatureError(str)

# print 
global_print_warnings = True
def warning( str ):
  if global_print_warnings:
    print str

# print in debug only
def debug( str ):
  #print str
  pass

types = { # type: bytes, # NAME
          1: 1, # BYTE
          2: 1, # ASCII
          3: 2, # SHORT
          4: 4, # LONG
          5: 8, # RATIONAL
          # TIFF v6.0
          6: 1, # SBYTE
          7: 1, # UNDEFINED
          8: 2, # SSHORT
          9: 4, # SLONG
         10: 8, # SRATIONAL
         11: 4, # FLOAT
         12: 8, # DOUBLE
        }

class JJLibTiff():
  """ Representation of tiff file,
  with all tags parsed and structured,
  and with read only numpy array (arr_ro).

  To get writable array use:
    get_arr_rw()
  Result is a reference which will be saved
  later when save() invoke.
  Alternatively using 
    set_arr()
  you can make faster writing array if you don't 
  care about previous values.
  """
  fmt_hdr = "<2sHL"
  fmt_hdr_size = st.calcsize( fmt_hdr )
  fmt_ifd_beg = "<H"
  fmt_ifd_beg_size = st.calcsize( fmt_ifd_beg )
  fmt_ifd_end = "<L"
  fmt_ifd_end_size = st.calcsize( fmt_ifd_end )

  def __init__( me,  file_or_name, print_warnings = True, floats_not_integers=False ):
    if type(file_or_name) == type(""):
      me.filename = file_or_name
      with open( me.filename , mode='rb') as file:
        me.content = file.read()
    else:
      me.filename = None
      me.content = file_or_name.read()
    if len( me.content ) == 0:
      raise EmptyFileError() 

    global global_print_warnings 
    global_print_warnings = print_warnings

    me.unpack( 0 )
    me.arr_rw = None 
    me._initialize( floats_not_integers )
  
  def get_dims( me ):
    return me.arr_ro.shape

  def unpack( me, off ):
    (me.endian, me.version, me.ifd) =  st.unpack_from( JJLibTiff.fmt_hdr, me.content, off) 
    if me.endian != "II": 
      error("Not Little Endian Layout")
    debug("Header: endian=%s, version=%u, ifd=%u" % 
          (me.endian, me.version, me.ifd))
    off += JJLibTiff.fmt_hdr_size
    debug("Offset: %u" % off)

    me._unpack_ifd ( me.ifd )

  # only one IFD support
  def _unpack_ifd( me, off ):
    me.read_no_of_fields, = st.unpack_from( JJLibTiff.fmt_ifd_beg, me.content, off )
    debug("IFD at %u: no_of_fields: %u" % (off, me.read_no_of_fields))
    off += JJLibTiff.fmt_ifd_beg_size
    me.fields = {}
    for i in xrange(me.read_no_of_fields):
      f = JJLibTiffField()
      f.unpack( me.content, off ) 
      # dictionary for quick tag access
      me.fields[ f.tag ] = f
      off += JJLibTiffField.fmt_size
    next_ifd, = st.unpack_from( JJLibTiff.fmt_ifd_end, me.content, off )
    if next_ifd != 0:
      warning("More than one IFD! Stop parsing.")
      debug("IFD next_ifd: %u" % next_ifd)
    off += JJLibTiff.fmt_ifd_end_size
    debug("Offset: %d" % off)

  def get_tag( me, tag_no ):
    # KeyError in case of missing value
    return me.fields[tag_no].value

  def get_tags_names( me ):
    return sorted(me.fields.keys())

  def get_ascii_tag( me, tag_no ):
    # like normal get_tag, but remove zero values at the end
    f = me.fields[tag_no]
    # KeyError in case of missing value
    if f.type == 2: # ascii
      return f.value.rstrip("\0")
    else:
      error("Not ascii tag!")

  def set_ascii_tag( me, tag_no, str ):
    f = JJLibTiffField()
    f.new_ascii( tag_no, str )
    me.fields[ tag_no ] = f

  def _initialize( me, floats_not_integers ):
    """ mostly support checks """
    me.im_width = me.get_tag( 256 )
    me.im_lenght = me.get_tag( 257 )
    me.rows_per_strip = me.get_tag( 278 )
    me.strip_off = me.get_tag( 273 )
    me.strip_byte_counts = me.get_tag( 279 )
    me.bits_per_sample = me.get_tag( 258 )
    debug("Image width=%u, length=%u, rps=%u, off=%u, sbc=%u, bps=%u" % \
         (me.im_width, 
          me.im_lenght, 
          me.rows_per_strip, 
          me.strip_off, 
          me.strip_byte_counts, \
          me.bits_per_sample) )
    if me.bits_per_sample != 32:
      error("Not supported bit sample ratio")
    if me.im_lenght != me.rows_per_strip:
      error("Not supported more than one strip!")
    if me.im_lenght*me.im_width*(me.bits_per_sample/8) != me.strip_byte_counts:
      error("Not supported compression on image data!")

    if floats_not_integers:
      dt = np.dtype('<f4') # floats
    else:
      dt = np.dtype('<i4') #signed integer

    me.arr_ro = np.frombuffer( 
                  me.content, 
                  dtype=dt, 
                  count=me.im_lenght*me.im_width, 
                  offset=me.strip_off
                )
    #arr = arr.reshape((im_lenght,im_width)).transpose()
    me.arr_ro = me.arr_ro.reshape((me.im_width, me.im_lenght), order="F")

  def get_arr_rw( me ): #, asfloat = False ):
    #if not asfloat:
    me.arr_rw = me.arr_ro.copy()
    #else:
    #  me.arr_rw = me.arr_ro.astype( np.float32 )
    return me.arr_rw

  def set_arr( me, a ):
    me.arr_rw = a

  def save ( me ):
    if not me.filename:
      error("MIssing filename for tiff file")
    else:
      me.save_as( me.filename )

  def save_as ( me, filename ):
    with open( filename , mode='wb') as file:
      off = me._pack_header( file )
      im_off = me._estimate_extra_space( off )
      off = me._pack_ifd( file, off, im_off)
      off = me._pack_ifd_data( file, off )
      me._pack_image( file, off, im_off )

  def _pack_header( me, file ):
    ifd_off = JJLibTiff.fmt_hdr_size
    file.write( st.pack( JJLibTiff.fmt_hdr, me.endian, me.version, ifd_off ) )
    return ifd_off

  def _estimate_extra_space( me, off ):
    no_of_fields = len( me.fields )
    fs = off +        JJLibTiff.fmt_ifd_beg_size + \
         no_of_fields*JJLibTiffField.fmt_size + \
                      JJLibTiff.fmt_ifd_end_size
    
    for t in sorted( me.fields ):
      field = me.fields[ t ]
      fs += field.allocate_extra_space( fs ) 
    return fs

  def _pack_ifd( me, file, off, im_off ):
    no_of_fields = len( me.fields )
    fs = off +        JJLibTiff.fmt_ifd_beg_size + \
         no_of_fields*JJLibTiffField.fmt_size + \
                      JJLibTiff.fmt_ifd_end_size

    file.write( st.pack( JJLibTiff.fmt_ifd_beg, no_of_fields ) )
    off += JJLibTiff.fmt_ifd_beg_size

    # it has to be the same loop like in upper function
    for t in sorted( me.fields ):
      field = me.fields[ t ]
      if t == 273: # special case! image data
        tag273 = im_off
      else:
        tag273 = None

      field.pack( file, tag273 )
      off += JJLibTiffField.fmt_size

    file.write( st.pack( JJLibTiff.fmt_ifd_end, 0 ) ) # end of IDF
    off += JJLibTiff.fmt_ifd_end_size
    if off != fs:
      error( "Internal error: ifd corrupted (%u, %u)" % (off, fs) )

    return off

  def _pack_ifd_data( me, file, off ):
    for t in sorted( me.fields ):
      field = me.fields[ t ]
      if field.write_size != 0:
        if off != field.write_place:
          error("Internal error - corrupted data in IFD (%u,%u)" % \
               (off, field.write_place))
        off += field.pack_data( file, off ) 

    return off

  def _pack_image( me, file, off, im_off ):
    debug("Writing image at %u, from %u size %u" %\
          (off, me.strip_off, me.strip_byte_counts))
    if off != im_off:
        error("Internal error - corrupted image location (%u,%u)" % \
             (off, im_off))
    if me.arr_rw == None: # not modified image
      file.write( me.content[
                    me.strip_off 
                    : 
                    me.strip_off + me.strip_byte_counts
                  ])
    else:
      me.arr_rw.transpose().tofile(file)


class JJLibTiffField():
  fmt = "<HHLL"
  fmt_size = st.calcsize( fmt )

  def __init__( me ):
    me.tag = -1

  def unpack( me, content,  off ):
    me.content = content
    (me.tag, me.type, me.count, me.valueoff) = \
        st.unpack_from( JJLibTiffField.fmt, me.content, off )
    me.val_size = me.count * types[me.type]
    debug("At %d: tag=%u, type=%u, count=%u, valueoff=%u, val_size: %u" % \
          (off, me.tag, me.type, me.count, me.valueoff, me.val_size))

    if me.val_size <= 4: 
      me._unpack_value()
    else:
      me._unpack_offset()

  def new_ascii( me, tag, ascii_val ):
    me.tag = tag
    me.type = 2
    me.value = ascii_val + "\0" # terminal NUL
    me.count = len(ascii_val) + 1
    me.val_size = me.count

    if me.val_size <= 4: 
      me.valueoff = st.unpack( "<L", me.value + "\0"*(4-me.val_size))[0]
    else:
      me.valueoff = None

  def _unpack_value( me ):
    debug("Value int: %d, hex=%x, str=%s" % \
          (me.valueoff, me.valueoff, \
           st.pack("<L",me.valueoff)))
    test = me.valueoff >> ((me.val_size)*8)
    if test:
      warning("THIS SHOULD BE ZERO, NOT 0x%x" % test)
    me.value = me.valueoff

  def _unpack_offset( me ):
    debug("Value STR: %s" % me.content[me.valueoff: me.valueoff+me.val_size%600])
    me.value = me.content[ me.valueoff : me.valueoff+me.val_size]

  def allocate_extra_space( me, fs ):
    if me.val_size <= 4: 
      me.write_size = 0
      me.write_place = None
    else:
      # even numbers
      me.write_size = ( me.val_size+1 if me.val_size%2 else me.val_size)
      me.write_place = fs
    return me.write_size
    
  def pack( me, file, overwrite_valueoff=None ):
    # overwrite_valueoff is usefull for tag 273 - image pointer
    if me.val_size <= 4: 
      if overwrite_valueoff != None:
        v = overwrite_valueoff
      else:
        v = me.valueoff
      file.write( st.pack( JJLibTiffField.fmt, me.tag, me.type, me.count, v))
    else:
      file.write( st.pack( JJLibTiffField.fmt, me.tag, me.type, me.count, me.write_place))

  def pack_data( me, file, off ):
    if me.val_size <= 4: 
      error("Internal error - val_size not more than 4")
    else:
      file.write( me.value )
      if me.val_size % 2:
        file.write( st.pack( "x" ) ) # pad byte
        return me.val_size+1
      else:
        return me.val_size



if __name__ == "__main__":
  #tests
  import sys
  filename = sys.argv[1]
  t = JJLibTiff( filename )
  a = t.get_arr_rw()
  a[0,0] = -1
  a[0,1] = -1000.5
  try:
    t.save_as( sys.argv[2] )
  except IndexError:
    pass
  #print "TAG: "+str( t.get_tag( int(sys.argv[2]) ) )


