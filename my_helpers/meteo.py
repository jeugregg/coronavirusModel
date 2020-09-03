# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
import urllib.request

# helpers project modules
import settings
from my_helpers.dates import days_between

# DEFINITIONS
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
PATH_JSON_METEO_FR = PATH_TO_SAVE_DATA + '/' + 'data_meteo_fr.json'


# For METEO
def create_url_meteo_date(str_date):
    # str_date = 2020-05-10
    num_records_max = 10000
    return 'https://public.opendatasoft.com/api/records/1.0/search/' + \
        f'?dataset=donnees-synop-essentielles-omm&q=&rows={num_records_max}' + \
        f'&sort=date&refine.date={str_date}'

def get_data_meteo_by_date(str_date):
    '''
    get data meteo for 1 day on date str_date 
    
    example : get_data_meteo_by_date("2020-01-24")
    '''
    # download raw json object
    url = create_url_meteo_date(str_date)
    data = urllib.request.urlopen(url).read().decode()
    # parse json object
    return json.loads(data)

def get_data_meteo_by_list(list_date):
    '''
    Retrieve data meteo for a list of dates
    '''
    for I, date_curr in enumerate(list_date):
        
        data_curr = get_data_meteo_by_date(date_curr)
        
        if I:
            data_out["records"] = data_out["records"] + data_curr["records"]
        else:
            data_out = data_curr.copy()
            
    return data_out
    

def create_url_meteo(num_records, num_start=0):
    # https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm  
    return 'https://public.opendatasoft.com/api/records/1.0/search/' + \
        f'?dataset=donnees-synop-essentielles-omm&q=&rows={num_records}' + \
        f'&sort=date&start={num_start}'

def get_data_meteo(num_records, num_start=0):
    # num_records max = 10000
    # download raw json object
    url = create_url_meteo(num_records, num_start)
    data = urllib.request.urlopen(url).read().decode()
    # parse json object
    return json.loads(data)

# function about data meteo from json 
def get_data_meteo_date_list(data): 
    return np.unique([data["records"][I]["fields"]["date"][0:10] \
                for I in range(len(data["records"]))]).tolist()

def get_data_meteo_date_min(data):
    list_date = get_data_meteo_date_list(data)
    return min(list_date)[0:10]

def get_data_meteo_date_max(data):
    list_date = get_data_meteo_date_list(data)
    return max(list_date)[0:10]

def get_rec_by_date(data_meteo, date_str):
    '''
    get one date data 
    '''
    
    # date_str = '2020-05-14'
    list_rec = []
    for rec_curr in data_meteo["records"]:
        if rec_curr['fields']["date"][0:10] == date_str:
            list_rec.append(rec_curr)
    data_out = data_meteo.copy()
    data_out["records"] = list_rec
    return data_out

def select_rec_by_station(data_meteo):
    '''
    Select list of list of data by station
    '''
    list_station = [data_meteo["records"][I]['fields']['numer_sta'] \
                    for I in range(len(data_meteo["records"]))]
    list_station = np.unique(list_station).tolist()
    
    list_rec = []
    for sta_curr in list_station:
        list_rec_curr = []
        for rec_curr in data_meteo["records"]:
            if rec_curr['fields']['numer_sta'] == sta_curr:
                list_rec_curr.append(rec_curr)
        list_rec.append(list_rec_curr)
            
    return list_rec


def get_field_in_list(list_rec, field_name):
    '''
    get a field from a list of list of data 
    '''
    list_field = []
    for I in range(len(list_rec)):
        list_field_curr = []
        for J in range(len(list_rec[I])):
            try:
                list_field_curr.append(list_rec[I][J]["fields"][field_name])
            except:
                continue
        if list_field_curr != []:
            list_field.append(list_field_curr)
    return list_field

def calculate_mean_field(list_field, fun):
    """
    calculate mean with fun on list of data
    """
    list_by_sta = []
    for list_curr in list_field:
        list_by_sta.append(fun(list_curr))

    return np.mean(list_by_sta)

def calc_list_mean_field(data_meteo, fieldname, fun):
    list_date = get_data_meteo_date_list(data_meteo)
    list_mean = []
    for date_curr in list_date:
        data_out = get_rec_by_date(data_meteo, date_curr)
        list_by_sta = select_rec_by_station(data_out)
        list_field = get_field_in_list(list_by_sta, fieldname)
        list_mean.append(calculate_mean_field(list_field, fun))
    return list_mean

def update_data_meteo(df_pos_fr):
    '''Update missing data from meteo france'''
    # meteo
    if os.path.isfile(PATH_JSON_METEO_FR):
        f_reload_from_start = False
        f_load_missing = False
        # load
        with open(PATH_JSON_METEO_FR) as f:
            data_meteo = json.load(f)
        # check start date
        date_meteo_start = get_data_meteo_date_min(data_meteo)
        delta_days = days_between(df_pos_fr.index.min(), date_meteo_start)
        if delta_days.days > 0:
            print(f"Must reload from start, {delta_days.days} days missing")
            f_reload_from_start = True
        # check last date
        date_meteo_end = get_data_meteo_date_max(data_meteo)
        delta_days = days_between(date_meteo_end, df_pos_fr.index.max())
        if delta_days.days > 0:
            print(f"Must load more last days, {delta_days.days} days missing")
            f_load_missing = True
        
        # determine list of days to download
        list_dates = None
        if f_reload_from_start:
            # all dates between [FORCED]
            list_dates = df_pos_fr.index.tolist()
        elif f_load_missing:
            # from date
            list_dates = df_pos_fr.index.tolist()
            # remove days already downloaded:
            list_remove = get_data_meteo_date_list(data_meteo)
            #get_data_meteo_date_list(data_meteo)
            for item_curr in  list_remove:
                try:
                    list_dates.remove(item_curr)
                except:
                    print(f'{item_curr} not found in dates list')
        else:
            # download NOT needed
            list_dates = None
    else:
        # all dates between [FORCED]
        f_reload_from_start = True
        f_load_missing = True
        list_dates = df_pos_fr.index.tolist()
    # if download needed
    if list_dates is not None:
        data_meteo_new = get_data_meteo_by_list(list_dates)
        print(f'{len(data_meteo_new["records"])} records downloaded')

        if f_reload_from_start:
            # reload all
            data_meteo = data_meteo_new
        else:
            # add data
            data_meteo["records"] = data_meteo["records"] + \
                data_meteo_new["records"]      
        # save
        with open(PATH_JSON_METEO_FR, 'w') as outfile:
            json.dump(data_meteo, outfile)
    else:
        print("Data meteo OK")

    return data_meteo

def precompute_data_meteo(data_meteo):
    '''pre-compute data meteo'''

    list_t_min = calc_list_mean_field(data_meteo, "t", min)
    list_t_max = calc_list_mean_field(data_meteo, "t", max)
    list_u_min = calc_list_mean_field(data_meteo, "u", min)
    list_u_max = calc_list_mean_field(data_meteo, "u", max)

    dict_meteo = dict()
    dict_meteo["date"] = get_data_meteo_date_list(data_meteo)
    dict_meteo["t_min"] = list_t_min
    dict_meteo["t_max"] = list_t_max
    dict_meteo["u_min"] = list_u_min
    dict_meteo["u_max"] = list_u_max

    df_feat_fr = pd.DataFrame(data=dict_meteo)
    df_feat_fr.columns = ["date", "T_min", "T_max", "H_min", "H_max"]
    df_feat_fr.sort_values(by="date", inplace=True)
    df_feat_fr.index = df_feat_fr["date"]

    return df_feat_fr