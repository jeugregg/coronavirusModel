# -*- coding: utf-8 -*-
''' 
Module for utility helper functions 
'''
# IMPORT

# import bluit-in
import datetime
import re
import shutil
# import third-party
import numpy as np
import pandas as pd


def display_msg(my_message):
    print("{} : {}".format(\
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), my_message))

# save file before update
def clean_file(path_file_name, flag_copy=False, suffix=None):
    '''
    Clean file already traited : rename file with date
    '''
    
    try:
        if (suffix is None):
            d = datetime.datetime.now()
            suffix = '_' + d.strftime("%Y%m%d_%H_%M_%S")
        else:
            suffix = '_' + suffix

        res_re = re.search(r'\.\w+$', path_file_name)
        
        path_file_name_saved = \
            path_file_name[0:res_re.start()] + suffix + res_re.group(0)
        
        if flag_copy:
            shutil.copy(path_file_name, path_file_name_saved)
        else:
            shutil.move(path_file_name, path_file_name_saved)

        print('File {} moved!'.format(path_file_name_saved))
    except:
        print('File {} does not exist!'.format(path_file_name))

def sum_between(ser_val, str_date_start, str_date_end):
    '''
    sum up values in series between 2 dates (index = date)
    '''
    b_range = (ser_val.index >= str_date_start) & \
        (ser_val.index <= str_date_end) 
    
    return ser_val[b_range].sum()

def sum_mobile(ser_val, ser_start, ser_end):
    '''
    mobile sums between dates start & end for ser_val (index = date)
    '''
    ser_sum = ser_val.copy()*np.NaN
    # for each date range
    for date_end, date_start in zip(ser_end, ser_start):
        # calculate sum 
        sum_curr = sum_between(ser_val, date_start, date_end)
        # store at date
        ser_sum.loc[date_end] = sum_curr

    return ser_sum