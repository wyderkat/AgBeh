###
# Copyright 2014 Tomasz Wyderka <wyderkat@cofoh.com>
#  www.cofoh.com
##

NEW_TAG=315
OLD_TAG=270
NEW_TAG_NOT_USED_ANYMORE=1334

TAGS_TO_UPDATE=[NEW_TAG,OLD_TAG,272]

ROOT = "/disk2/"
INPUT_TIFF_DIR = ROOT+"datatemp/"
IMAGES_TIFF_DIR = ROOT+"data/images/"
LATEST_TIFF_DIR = ROOT+"data/latest/"
ZIP_TIFF_DIR = ROOT+"data/frames/"
TMP_TIF_DIR = ROOT+"data/.tmp/" # use for corrections
TMP_INPUT_TIFF_DIR = TMP_TIF_DIR+"datatemp/"
TMP_INPUT_ORYG_TIFF_DIR = TMP_TIF_DIR+"datatemp_oryg/"
CORRECTED_DIR = "corrected/"
ROUND_1ST_DIR = "round1st/"

LOST_PLUS_FOUND_ZIP = ZIP_TIFF_DIR+"lost+found.zip"
