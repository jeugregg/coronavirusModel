# -*- coding: utf-8 -*-
import datetime
import os
import pandas as pd
import numpy as np

# FOR DATES
def add_days(str_date_0, nb_days_CV):
    '''
    add days to string dates
    '''
    date_format = "%Y-%m-%d"
    date_last = datetime.datetime.strptime(str_date_0, date_format)
    date_start = date_last + datetime.timedelta(days=nb_days_CV)
    str_date_start = date_start.strftime(date_format)
    return str_date_start

def generate_list_dates(str_date_0, str_date_1, date_format=None):
    '''
    Generate a list of dates between 2 dates
    '''
    if date_format is None:
        date_format = "%Y-%m-%d"
    date_0 = datetime.datetime.strptime(str_date_0, date_format)
    date_1 = datetime.datetime.strptime(str_date_1, date_format)
    delta = date_1 - date_0
    if delta.days > 0:
        return [(date_0 + \
                 datetime.timedelta(days=I)).strftime(date_format) \
               for I in range(1, delta.days+1)]
    else:
        return None

def days_between(str_date_0, str_date_1):
    '''
    calculate days between 2 string dates
    '''
    date_format = "%Y-%m-%d"
    
    date_0 = datetime.datetime.strptime(str_date_0, date_format)
    date_1 = datetime.datetime.strptime(str_date_1, date_format)
    delta = date_1 - date_0
    '''date_start = date_last + datetime.timedelta(days=nb_days_CV)
    str_date_start = date_start.strftime(date_format)
    return str_date_start'''
    return delta

def check_cont_dates(list_dates, date_format="%Y-%m-%d"):
    '''
    Check dates continuity in a list

    return list missing dates
    '''
    # create range from min & max dates
    str_date_min = min(list_dates)
    str_date_min = add_days(str_date_min, -1) # add first day
    str_date_max = max(list_dates)
    list_range = generate_list_dates(str_date_min, str_date_max)
    # search missing days
    list_missing = []
    for day_curr in list_range:
        if day_curr not in list_dates:
            list_missing.append(day_curr)
    return list_missing

def get_file_date(path_to_file):
    '''
    get file modification date 
    '''
    #return datetime.datetime.utcfromtimestamp(os.path.getmtime(path_to_file))
    return datetime.datetime.fromtimestamp(os.path.getmtime(path_to_file))

def conv_dt_2_str(dt_in):
    '''
    Convert datetime to string date
    '''
    return  dt_in.strftime("%Y-%m-%d %H:%M:%S")

def find_close_date(ser_dates, str_date_0, str_date_min):
    '''
    find closest date but not developped
    '''
    return str_date_0

def create_date_ranges(ser_dates, nb_days_CV):
    '''
    Find first and last dates in "ser_dates" for last "nb_days_CV" days 
    '''
    ser_start = []
    ser_end = []
    # find first date : 
    str_date_min = ser_dates.min()
    str_date_max = ser_dates.max()
    #print("str_date_min: ", str_date_min)
    #print("str_date_max: ", str_date_max)
    ser_end.append(str_date_max)
    str_date_start = add_days(str_date_max, -(nb_days_CV-1))
    next_date = find_close_date(ser_dates, str_date_start, str_date_min)
    ser_start.append(next_date)
    while ser_start[-1] > str_date_min:
        ser_end.append(add_days(ser_end[-1], -1))
        ser_start.append(add_days(ser_end[-1], -(nb_days_CV-1)))
    return ser_start, ser_end

def create_date_range_lim(date_req_start, date_req_end, n_days=32):
    '''
    Create lists of date range limited to n_days
    '''
    date_req_start_lim = add_days(date_req_start, -1)
    list_dates = generate_list_dates(date_req_start_lim, date_req_end)
    list_dates_start = []
    list_dates_end = []

    if len(list_dates) > n_days:
        for I in range(0, len(list_dates), n_days):
            list_dates_start.append(list_dates[I])
            list_dates_end.append(min(add_days(list_dates[I], n_days-1),
                                     date_req_end))
    else:
        list_dates_start = [date_req_start]
        list_dates_end = [date_req_end]
    return list_dates_start, list_dates_end