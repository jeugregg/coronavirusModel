# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import tensorflow as tf

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime
import math
import re
import os

# read json from http
import json
import urllib.request

# read csv from http
import io
import requests

# DEFINITIONS
PATH_TO_SAVE_DATA = "."
PATH_DF_POS_FR = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr.csv'
PATH_DF_TEST_FR = PATH_TO_SAVE_DATA + '/' + 'df_test_fr.csv'
PATH_JSON_METEO_FR = PATH_TO_SAVE_DATA + '/' + 'data_meteo_fr.json'
PATH_DF_FEAT_FR = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr.csv' 
PATH_DF_GOUV_FR_RAW = PATH_TO_SAVE_DATA + '/' + 'df_gouv_fr_raw.csv'
PATH_GEO_DEP_FR = PATH_TO_SAVE_DATA + '/sources/geofrance/' + 'departments.csv'
PATH_MDL_SINGLE_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_single_step_pos_fr"
PATH_MDL_MULTI_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_multi_step_pos_fr"
URL_CSV_GOUV_FR = \
    'https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675'
URL_GEOJSON_DEP_FR = 'sources/departements-avec-outre-mer_simple.json'
NB_POS_DATE_MIN_DF_FEAT = 140227 # on 12/05/2020
date_format = "%Y-%m-%d"
# model parameters
train_split = 58
past_history = 10 # days used to predict next values in future
future_target = 3 # predict 3 days later
STEP = 1
NB_DAY_PLOT = 60

# Rt model
nb_days_CV = 14

###################
# HELPER FUNCTIONS

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

def get_file_date(path_to_file):
    '''
    get file modification date 
    '''
    return datetime.datetime.utcfromtimestamp(os.path.getmtime(path_to_file))

def conv_dt_2_str(dt_in):
    '''
    Convert datatime to string date
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
    date_format = "%Y-%m-%d"
    # find first date : 
    str_date_min = ser_dates.min()
    str_date_max = ser_dates.max()
    print("str_date_min: ", str_date_min)
    print("str_date_max: ", str_date_max)
    ser_end.append(str_date_max)
    str_date_start = add_days(str_date_max, -(nb_days_CV-1))
    next_date = find_close_date(ser_dates, str_date_start, str_date_min)
    ser_start.append(next_date)
    while ser_start[-1] > str_date_min:
        ser_end.append(add_days(ser_end[-1], -1))
        ser_start.append(add_days(ser_end[-1], -(nb_days_CV-1)))
    return ser_start, ser_end

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

# For Calculation
def mdl_R0_estim(nb_cases, nb_cases_init=1, nb_day_contag=14, delta_days=14):
    '''
    R0 Model with exact formulation : 
    Nb_cases(D) - Nb_cases(D-1) = Nb_cases(D-1) * R_0_CV / NB_DAY_CONTAG_CV
    
    Nb_cases(D) = Nb_cases(D0) * exp(R_0_CV / NB_DAY_CONTAG_CV * (D - DO))
    
    => 
    R_0_CV = NB_DAY_CONTAG_CV / (D - DO) * ln( Nb_cases(D) / Nb_cases(D0))
    
    return R0
    
    '''
    if type(nb_cases) == np.float64:
        if nb_cases == 0:
            return 0
        return nb_day_contag / delta_days * math.log(nb_cases / \
                                                     max(1, nb_cases_init))
    else:
        list_out = []
        for I in range(len(nb_cases)):
            try:
                if nb_cases[I] == 0:
                    list_out.append(0)
                else:
                    list_out.append(nb_day_contag / delta_days * \
                        math.log(nb_cases[I] / max(1, nb_cases_init[I])))
            except:
                print("I = ",I)
                print("nb_day_contag = ", nb_day_contag)
                print("nb_cases[I] = ", nb_cases[I])
                print("nb_cases_init[I] = ", nb_cases_init[I])
                raise
        return list_out
# For DATA
def get_geo_fr():
    ###########
    # GEOJSON : dep france : source : https://france-geojson.gregoiredavid.fr/
    #

    #URL_GEOJSON_DEP_FR = 'sources/geojson-departements.json'
    # source : https://github.com/gregoiredavid/france-geojson

    with open(URL_GEOJSON_DEP_FR) as f:
        dep_fr = json.load(f)

    # example : 
    # dep_fr['features'][0]['geometry']['type']
    # dep_fr['features'][0]['geometry']["coordinates"]
    # dep_fr['features'][0]["properties"]["code"]
    # dep_fr['features'][0]["properties"]["nom"]

    # get list dep / code
    list_code = \
        [feat_curr["properties"]["code"] for feat_curr in dep_fr['features']]
    list_name = \
        [feat_curr["properties"]["nom"] for feat_curr in dep_fr['features']]
    df_code_dep = pd.DataFrame(data=list_code, columns=["code"])
    df_code_dep["name"] = list_name

    return dep_fr, df_code_dep

def get_data_rt(df_gouv_fr_raw):
    ############################
    # Create data last 14 days : FRANCE Tested and Positive
    # output : pt_fr_test_last DataFrame

    pt_fr_test = pd.pivot_table(df_gouv_fr_raw, values=['t', 'p'], 
                            index=["jour"],
                    columns=["dep"], aggfunc=np.sum) 
    pt_fr_test["date"] = pt_fr_test.index

    df_dep_pos = pt_fr_test["p"].copy()
    df_dep_pos.index = pt_fr_test["date"].index

    df_dep_test = pt_fr_test["t"].copy()
    df_dep_test.index = pt_fr_test["date"].index

    # find last date 
    date_format = "%Y-%m-%d"
    str_date_last = df_gouv_fr_raw["jour"].max() 

    # find start cumulative sum of confirmed cases / test
    date_last = datetime.datetime.strptime(str_date_last, date_format)
    date_start = date_last - datetime.timedelta(days=14-1)
    str_date_start = date_start.strftime(date_format)

    # create table of nb_cases of last date : sum of all last 14 days
    # sum all from date_start :
    bol_date_last14d = df_gouv_fr_raw["jour"] >= str_date_start

    pt_fr_test_last = pd.pivot_table(df_gouv_fr_raw[bol_date_last14d], 
                                    values=['t', 'p'], 
                                index=["dep"], aggfunc=np.sum) 

    pt_fr_test_last.index.name = ''
    pt_fr_test_last["dep"] = pt_fr_test_last.index

    ser_start , ser_end = create_date_ranges(df_gouv_fr_raw["jour"], nb_days_CV)
    print("ser_start : ", ser_start)
    print("ser_end : ", ser_end)

    df_dep_sum = pd.DataFrame(index=df_dep_pos.index, columns=["date"],
                            data=df_dep_pos.index.tolist())
    for dep_curr in df_dep_pos.columns:
        df_dep_sum[dep_curr] = sum_mobile(df_dep_pos[dep_curr], ser_start, 
            ser_end)

    df_dep_r0 = pd.DataFrame(index=df_dep_pos.index, columns=["date"],
                            data=df_dep_pos.index.tolist())

    for dep_curr in df_dep_sum.columns[1:]:
        ser_val = df_dep_sum[dep_curr].copy()
        date_min = add_days(ser_val.index.min(), nb_days_CV) 
        ser_r0 = ser_val.copy()*np.nan
        for date_curr in ser_val[ser_val.index >= date_min].index:
            date_0 = add_days(date_curr, -nb_days_CV)
            if not(np.isnan(ser_val.loc[date_0])):
                sum_0 = ser_val.loc[date_0]
                sum_1 = sum_0 + ser_val.loc[date_curr]
                ser_r0.loc[date_curr] = mdl_R0_estim(nb_cases=sum_1, 
                                                    nb_cases_init=sum_0,
                                                    nb_day_contag=nb_days_CV, 
                                                    delta_days=nb_days_CV)
        df_dep_r0[dep_curr] = ser_r0
        
    df_dep_r0.dropna(inplace=True)

    #################
    # last R0 for MAP
    #
    dep_fr, df_code_dep = get_geo_fr()
    # add departement name
    pt_fr_test_last = pt_fr_test_last.merge(df_code_dep, left_on='dep', 
                                            right_on='code')
    # find last date 
    date_format = "%Y-%m-%d"
    date_p0 = date_start - datetime.timedelta(days=14)
    str_date_p0 = date_p0.strftime(date_format)

    # Nb_cases 14 days before: p_0
    # sum cases 14 days period before current 14 days period 
    # => period : 28 days -> 14 days before last date:
    bol_date_p0 = (df_gouv_fr_raw["jour"] < str_date_start) & \
        (df_gouv_fr_raw["jour"] >= str_date_p0)
    pt_fr_test_p0 = pd.pivot_table(df_gouv_fr_raw[bol_date_p0], 
                                    values=['p'], 
                                index=["dep"], aggfunc=np.sum) 
    pt_fr_test_p0.index.name = ''
    pt_fr_test_p0["dep"] = pt_fr_test_p0.index
    pt_fr_test_p0.columns = ["p_0", "dep"]
    pt_fr_test_last = pt_fr_test_last.merge(pt_fr_test_p0, left_on='dep', 
                                            right_on='dep')

    # R0 Estimation :
    # Nb_cases(T0) sum of confirmed cases with T0=T-14days = between T0-14days -> T0 
    #   <=> (28 days before T -> 14 days before T)
    #
    # Nb cases(T):  sum of confirmed cases between T-28days -> T

    pt_fr_test_last["R0"] = mdl_R0_estim(nb_cases=pt_fr_test_last["p_0"] + \
                                        pt_fr_test_last["p"] , 
                                        nb_cases_init=pt_fr_test_last["p_0"], 
                                        nb_day_contag=14, 
                                        delta_days=14)

    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep

def get_data_gouv_fr():
    '''
    Get from Gouv  SFP page data cases in France 
    Clean & Save
    '''
    # patch 29/07/2020 : SSL error patch
    req = requests.get(URL_CSV_GOUV_FR).content
    df_gouv_fr_raw = pd.read_csv(io.StringIO(req.decode('utf-8')), sep=";")

    # past treat data upper cases -> lower cases
    if "t" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["t"] =  df_gouv_fr_raw["T"]
    if "p" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["p"] =  df_gouv_fr_raw["P"]
    # patch : clear data in double !!!
    df_gouv_fr_raw = df_gouv_fr_raw[df_gouv_fr_raw["cl_age90"] != 0]

    df_gouv_fr_raw.to_csv(PATH_DF_GOUV_FR_RAW, index=False)

    return df_gouv_fr_raw

def get_data_pos():
    '''
    1) Retrieve data from Sante Publique France direct CSV URL 
        (updated every days but with 4 to 5 days delay...)
    2) Proceed this data by departements (tested - positive)
    3) Retrieve data from Méteo France
    4) Proceed this data to have mean feature all over France every days
    5) Proceed features data for model by combining all these data

    Every database is saved in CSV format.
    '''

    df_gouv_fr_raw = get_data_gouv_fr()
    # creation of table data : 't':tested 'p':positive
    # data =  f(line : date, dep / col: t) => f(line : date / col: dep = f(t)) 
    pt_fr_test = pd.pivot_table(df_gouv_fr_raw, values=['t', 'p'], 
                                index=["jour"],
                        columns=["dep"], aggfunc=np.sum) 
    pt_fr_test["date"] = pt_fr_test.index

    # save data
    df_pos_fr = pt_fr_test["p"].copy()
    df_pos_fr.index = pt_fr_test["date"].index
    #df_pos_fr["date"] = df_pos_fr.index
    df_pos_fr.to_csv(PATH_DF_POS_FR, index=False)

    df_test_fr = pt_fr_test["t"].copy()
    df_test_fr.index = pt_fr_test["date"].index
    #df_test_fr["date"] = df_test_fr.index
    df_test_fr.to_csv(PATH_DF_TEST_FR, index=False)

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
    
    # create features for model

    # add meteo data

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

    # add positive cases
    df_pos_fr["pos"] = 0
    list_dep = []
    for col_curr in df_pos_fr.columns:
        if re.search("^\d", col_curr):
            list_dep.append(col_curr)

    for dep_curr in list_dep:
        df_pos_fr["pos"] += df_pos_fr[dep_curr]

    df_feat_fr["pos"] = df_pos_fr["pos"].copy()

    # add tested cases
    df_test_fr["test"] = 0
    list_dep = []
    for col_curr in df_test_fr.columns:
        if re.search("^\d", col_curr):
            list_dep.append(col_curr)

    for dep_curr in list_dep:
        df_test_fr["test"] += df_test_fr[dep_curr]
        
    df_feat_fr["test"] = df_test_fr["test"].copy()

    # add num days
    df_feat_fr['day_num'] = \
        df_feat_fr["date"].astype(np.datetime64).dt.strftime("%w")

    # add nb_cases
    arr_nb_cases = df_feat_fr["pos"].cumsum().values
    df_feat_fr["nb_cases"] = NB_POS_DATE_MIN_DF_FEAT + arr_nb_cases
    
    # save for future uses
    df_feat_fr.to_csv(PATH_DF_FEAT_FR, index=False)

    


# FOR data to plot
def load_data_pos():
    '''
    Load data positive cases France
    '''
    df_feat_fr = pd.read_csv(PATH_DF_FEAT_FR)
    df_feat_fr.index = df_feat_fr["date"]
    return df_feat_fr

def load_data_gouv():
    '''
    Load data gouv France
    '''
    try:
        df_gouv_fr_raw = pd.read_csv(PATH_DF_GOUV_FR_RAW)
    except:
        # try to get from URL
        df_gouv_fr_raw = get_data_gouv_fr()

    return df_gouv_fr_raw

def update_pos(df_feat_fr):
    '''
    Update plot data positive cases France
    '''
    # pos last 60 days : date, pos, total (sum)
    str_date_0 = add_days(df_feat_fr.date.max(), -60)
    df_plot = df_feat_fr[df_feat_fr["date"] >= str_date_0].copy()
    return df_plot

def update_pred_pos(df_feat_fr):
    '''
    Update prediction data positive cases France
    '''
    # load model
    multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)

    # prepare features
    features = df_feat_fr.copy().filter(items=['T_min', 'T_max', 'H_min',
                                            'H_max', 'pos', 'test', 'day_num'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    # prepare data : very last days
    x_multi = np.array([dataset[-past_history:,:]]) 
    # predict next days
    y_multi_pred = multi_step_model.predict(x_multi)
    # convert in positive cases
    y_pos_pred = y_multi_pred * data_std[4] + data_mean[4]
    # pos pred next 3 days from last day : date, pos, total (sum)
    str_date_pred_0 = df_feat_fr.date.max()
    str_date_pred_1 = add_days(str_date_pred_0, 3)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)
    # figure 
    df_plot_pred = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                        data=list_dates_pred)

    df_plot_pred["pos"] = y_pos_pred[0].astype(int)
    arr_nb_pred = df_plot_pred["pos"].cumsum().values
    df_plot_pred["nb_cases"] = df_feat_fr["nb_cases"].max() + arr_nb_pred

    return df_plot_pred

def update_pred_pos_all(df_feat_fr):
    '''
    Update prediction data positive cases France for all days
    '''
    # load model
    multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)

    # prepare features
    features = df_feat_fr.copy().filter(items=['T_min', 'T_max', 'H_min',
                                            'H_max', 'pos', 'test', 'day_num'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    list_x = []
    K_days = 0
    # prepare data : very last days
    nb_max = int((NB_DAY_PLOT)/future_target)
    for I in range(nb_max, 0, -1):
        I_start = I * future_target - past_history
        if I_start < 0:
            break
        I_end = I * future_target
        list_x.append(np.array([dataset[I_start:I_end, :]]))
        K_days += future_target

    str_date_pred_1 = df_feat_fr.date.max()
    str_date_pred_0 = add_days(str_date_pred_1, -1*K_days)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)

    # model prediction
    for I, x_multi in enumerate(list_x):
        if I:
            y_multi_pred = np.concatenate([y_multi_pred, 
                                        multi_step_model.predict(x_multi)],
                                axis=1)
        else:
            y_multi_pred = multi_step_model.predict(x_multi)
        
    # convert in positive cases
    y_pos_pred = y_multi_pred * data_std[4] + data_mean[4]

    # create df output
    df_plot_pred_all = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                       data=list_dates_pred)
    df_plot_pred_all["pos"] = y_pos_pred[0].astype(int)
    arr_nb_pred = df_plot_pred_all["pos"].cumsum().values
    df_plot_pred_all["nb_cases"] = df_feat_fr[df_feat_fr["date"] == \
        df_plot_pred_all["date"].min()]["nb_cases"][0] + arr_nb_pred

    return df_plot_pred_all


def jsonifed_pred(df_plot_pred):
     return df_plot_pred.to_json(date_format='iso', orient='split')

# FOR FIGURE
def create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, str_date_mdl):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Create and style traces
    # total
    fig.add_trace(go.Scatter(x=df_plot["date"].astype(np.datetime64), 
                            y=df_plot["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        connectgaps=True, name="Total"),
                secondary_y=False)

    fig.add_trace(go.Scatter(x=df_plot_pred["date"].astype(np.datetime64), 
                            y=df_plot_pred["nb_cases"],
                        mode='lines+markers',
                        line_shape='linear',
                        connectgaps=True, name="Future pred."),
                secondary_y=False)
    # new cases
    fig.add_trace(go.Bar(x=df_plot["date"].astype(np.datetime64), 
                        y=df_plot["pos"], 
                        name="Daily"), 
                secondary_y=True)

    fig.add_trace(go.Bar(x=df_plot_pred["date"].astype(np.datetime64), 
                y=df_plot_pred["pos"], 
                name="Daily pred."), 
                secondary_y=True)

    fig.add_trace(go.Scatter(x=df_plot_pred_all["date"].astype(np.datetime64), 
                            y=df_plot_pred_all["nb_cases"],
                        mode='lines',
                        line_shape='linear',
                        connectgaps=True, name="Past pred."),
                secondary_y=False)
    # Edit the layout
    title_fig = '<b>COVID-19 Confirmed cases in France with prediction</b>' + \
        '<br>LMST Deep Learning Model : ' + \
        'prediction of <b>3 days</b> from <b>last 10 days</b>' + \
        '<br>Trained until ' + str_date_mdl
    fig.update_layout(title=title_fig, yaxis_title='nb <b>Total</b> cases')
    fig.update_layout(legend_orientation="h", legend=dict(x=0, y=1))

    fig.update_yaxes(title_text="nb <b>Daily</b> cases", 
                    range=[0, 5000], secondary_y=True)

    return fig

def create_fig_rt(df_dep_r0, df_code_dep, pt_fr_test_last):
    list_num_dep = df_dep_r0.columns[1:].tolist()
    # path 975 & 977 & 978 doesn't exist in dep name data
    list_num_dep.remove('975')
    list_num_dep.remove('977')
    list_num_dep.remove('978')

    list_name_dep = [f'{dep_num_curr} - ' + \
                df_code_dep.loc[df_code_dep["code"] == dep_num_curr,
                                    "name"].values[0] + \
                "<br>Rt=<b>{:.2f}</b>".format(df_dep_r0[dep_num_curr][-1]) + \
                " cases={}".format(pt_fr_test_last.loc[ \
                pt_fr_test_last.dep == dep_num_curr, "p"].values[0]) \
                for dep_num_curr in list_num_dep]

    nb_dep = len(list_num_dep)
    nb_col = 4
    nb_row = math.ceil(nb_dep/nb_col)
    fig = make_subplots(rows=nb_row, cols=nb_col, shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=list_name_dep)
    I_dep = 0
    #list_color = []
    for row in range(nb_row):
        for col in range(nb_col):   
            dep_num_curr = list_num_dep[I_dep]
            dep_curr = df_code_dep.loc[df_code_dep["code"] == dep_num_curr, 
                                    "name"].values[0]
        
            if (df_dep_r0[dep_num_curr][-1] > 1) & \
                (pt_fr_test_last.loc[ \
                    pt_fr_test_last.dep == dep_num_curr, "p"].values[0] > 400):
                color_curr = "red"
            elif (df_dep_r0[dep_num_curr][-1] > 1):
                color_curr = "orange"
            else:
                color_curr = "blue"
        
            
            fig.add_trace(go.Scatter(x=df_dep_r0["date"], 
                        y=df_dep_r0[dep_num_curr],
                        mode='lines', name=dep_curr, 
                        line=dict(color=color_curr),
                        fill='tozeroy'), 
                        row=row+1, col=col+1)
            
            fig.add_trace(go.Scatter(x=[df_dep_r0["date"][0], 
                                        df_dep_r0["date"][-1]], 
                                    y=[1,1],
                                    mode='lines', 
                                    line=dict(color="red", dash='dash'),
                                    hoverinfo="skip"), 
                        row=row+1, col=col+1)
            I_dep +=1
            
            if I_dep >= nb_dep: #nb_dep:
                break
        if I_dep >= nb_dep: #nb_dep:
                break 

    #fig.update_traces(patch=dict(font=dict(size=6)))
    for I, subplot_title_curr in enumerate(fig['layout']['annotations']):
        subplot_title_curr['font'] = dict(size=10)
        subplot_title_curr['xshift'] = 0
        subplot_title_curr['yshift'] = -10

    fig.update_layout(
        height=1800,
        title="Rt: Estimated Reproduction Nb. in France ( until {} )".format( \
            df_dep_r0['date'].max()),
        showlegend=False,
        font=dict(
            size=12,
        )
    )

    return fig

def check_update():
    '''
    Just check if new data possibly available
    (if file date older than 24 hours)
    '''
    time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    dtime_now  = datetime.datetime.now() - time_file_df_feat_date
    # update only if more than 24 hours without update
    if dtime_now.days > 0:
        flag_old = True
        # update data from external 
        print("Maybe new data available...")
        #get_data_pos()
    else:
        flag_old = False
        print("No new data available.")
    return flag_old

# APP DASH
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def startup_layout():
    '''
    startup web page
    '''
    #time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    #dtime_now  = datetime.datetime.now() - time_file_df_feat_date

    # update 
    if check_update():
        get_data_pos()
    
    df_feat_fr = load_data_pos()

    df_plot = update_pos(df_feat_fr)
    # predict 3 future days
    df_plot_pred = update_pred_pos(df_feat_fr)
    # predict all past days
    df_plot_pred_all = update_pred_pos_all(df_feat_fr)
    # last date of training
    str_date_mdl =  df_feat_fr.iloc[train_split]["date"]

    # rt plots
    df_gouv_fr_raw = load_data_gouv()
    df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
        get_data_rt(df_gouv_fr_raw)

    return html.Div(children=[
        html.H1(children='COVID-19 Dashboard in France : Model & Dataviz'),
        html.Div(children=html.Button('Update Data', id='update-data', 
        n_clicks=0), style={'display': 'inline-block', 'margin-right': 10}),
        html.Div(children=dcc.Loading(
            id="loading-fig-pos",
            type="default",
            children=html.Div(id="loading-output-1")), 
            style={'display': 'inline-block', 'margin-right': 10}),
        html.Div(children=html.A(children="By G.LANG, Data Scientist Freelance",
            href="http://greg.coolplace.fr/data-scientist-freelance", 
            target="_blank"), style={'display': 'inline-block'}),
        dcc.Graph(id='covid-pos-graph',
            figure=create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all, 
                str_date_mdl)
        ),
        dcc.Graph(id='covid-rt-graph',
            figure=create_fig_rt(df_dep_r0, df_code_dep, pt_fr_test_last)
        ),
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='predicted-value', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred)),
        html.Div(id='predicted-value-all', style={'display': 'none'},
            children=jsonifed_pred(df_plot_pred_all))
        ])

app.layout = startup_layout


# button update data
'''
@app.callback(
    dash.dependencies.Output('covid-pos-graph', 'figure'),
    [dash.dependencies.Input('update-data', 'n_clicks')])
def update_fig_pos(n_clicks):
    df_feat_fr = load_data_pos()
    df_plot = update_pos(df_feat_fr)
    df_plot_pred = update_pred_pos(df_feat_fr)
    return create_fig_pos(df_plot, df_plot_pred)
'''

@app.callback(
    [dash.dependencies.Output('loading-output-1', 'children') , 
    dash.dependencies.Output('covid-pos-graph', 'figure')],
    [dash.dependencies.Input('update-data', 'n_clicks')],
    [dash.dependencies.State('predicted-value', 'children'),
    dash.dependencies.State('predicted-value-all', 'children')])
def load_figure(n_clicks, jsonified_pred, jsonified_pred_all):
    flag_update = check_update()
    if flag_update:
        get_data_pos()
    df_feat_fr = load_data_pos()
    df_plot = update_pos(df_feat_fr)
    
    if flag_update:
        # model predicting
        df_plot_pred = update_pred_pos(df_feat_fr)
        df_plot_pred_all = update_pred_pos_all(df_feat_fr)
    else:
        # load from hidden div (no model predicting again)
        print("loading prediction from hidden div...")
        df_plot_pred = pd.read_json(jsonified_pred, orient='split')
        df_plot_pred_all = pd.read_json(jsonified_pred_all, orient='split')
    # last date of training
    str_date_mdl =  df_feat_fr.iloc[train_split]["date"]
    return conv_dt_2_str(get_file_date(PATH_DF_FEAT_FR)), \
                        create_fig_pos(df_plot, df_plot_pred, df_plot_pred_all,
                            str_date_mdl)

if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(host='0.0.0.0', debug=True, port=80)
    app.config.suppress_callback_exceptions = True

