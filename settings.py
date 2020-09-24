# -*- coding: utf-8 -*-

# bluit-in import 
import ntpath

# General Definition 
MODE_DEBUG = False # default = False 
MODE_FORCE_UPDATE = False # default = False 
PREDICT = True # default = True 
MODEL_TFLITE = True # default = True 
PATH_TO_SAVE_DATA = ntpath.dirname(__file__)
NB_PERIOD_PLOT = 9
# AWS
BUCKET_NAME = 'app-covid-visu-bucket'