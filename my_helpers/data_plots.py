# -*- coding: utf-8 -*-
''' Prepare for plot positive cases data and model features
'''

# import built-in
import re
import datetime
import io
import os

# import third party
import pandas as pd
import numpy as np
import requests

# import project libraries
import settings
from my_helpers.dates import days_between, add_days, get_file_date
from my_helpers.meteo import update_data_meteo_light
from my_helpers.meteo import precompute_data_meteo_light
from my_helpers.meteo import PATH_DF_METEO_FR
from my_helpers.model import FUTURE_TARGET, TRAIN_SPLIT
from my_helpers.model import update_pred_pos, update_pred_pos_all

# DEFINITIONS
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
URL_CSV_GOUV_FR = 'https://www.data.gouv.fr/' + \
    'fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675'
PATH_DF_GOUV_FR_RAW = PATH_TO_SAVE_DATA + '/' + 'df_gouv_fr_raw.csv'
NB_POS_DATE_MIN_DF_FEAT = 140227 # on 12/05/2020
PATH_DF_POS_FR = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr.csv' 
PATH_DF_TEST_FR = PATH_TO_SAVE_DATA + '/' + 'df_test_fr.csv'
PATH_DF_FEAT_FR = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr.csv' 
NB_PERIOD_PLOT = settings.NB_PERIOD_PLOT
NB_DAY_PLOT = FUTURE_TARGET * NB_PERIOD_PLOT
from my_helpers.meteo import PATH_DF_METEO_FR

# DATA from SPF
def get_data_gouv_fr():
    '''
    Get from Gouv  SFP page data cases in France 
    Clean & Save
    '''
    # patch 29/07/2020 : SSL error patch
    req = requests.get(URL_CSV_GOUV_FR).content
    df_gouv_fr_raw = pd.read_csv(io.StringIO(req.decode('utf-8')), sep=";", 
        low_memory=False) # patch dtype 2020-09-08

    # past treat data upper cases -> lower cases
    if "t" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["t"] =  df_gouv_fr_raw["T"]
    if "p" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["p"] =  df_gouv_fr_raw["P"]
    # patch : clear data in double !!!
    df_gouv_fr_raw = df_gouv_fr_raw[df_gouv_fr_raw["cl_age90"] != 0]

    df_gouv_fr_raw.to_csv(PATH_DF_GOUV_FR_RAW, index=False)

    return df_gouv_fr_raw

def compute_sum_dep(df_in):
    df_in["daily"] = 0
    list_dep = []
    for col_curr in df_in.columns:
        if re.search("^\d", col_curr):
            list_dep.append(col_curr)
    for dep_curr in list_dep:
        df_in["daily"]  += df_in[dep_curr]
    return df_in

def precompute_data_pos(df_gouv_fr_raw):
    '''Pre-compute data from Sante Publique France'''
    # creation of table data : 't':tested 'p':positive
    # data =  f(line : date, dep / col: t) => f(line : date / col: dep = f(t)) 
    pt_fr_test = pd.pivot_table(df_gouv_fr_raw, values=['t', 'p'], 
                                index=["jour"],
                                columns=["dep"], aggfunc=np.sum) 
    pt_fr_test["date"] = pt_fr_test.index

    # age (new feature)
    # date / dep age pos test
    # =>  date / pos mean(age) 
    df_gouv_fr_raw_0 = df_gouv_fr_raw.copy()
    df_gouv_fr_raw_0["prod_p_age"] = \
        df_gouv_fr_raw_0["p"] * df_gouv_fr_raw_0["cl_age90"]
    df_gouv_fr_raw_0["prod_t_age"] = \
        df_gouv_fr_raw_0["t"] * df_gouv_fr_raw_0["cl_age90"]
    ser_p_age = df_gouv_fr_raw_0.groupby("jour") \
        ["prod_p_age"].sum() / df_gouv_fr_raw_0.groupby("jour")["p"].sum()
    df_age = pd.DataFrame(index=ser_p_age.index, columns=["pos_mean_age"], 
                        data=ser_p_age.values)
    ser_t_age = df_gouv_fr_raw_0.groupby("jour") \
        ["prod_t_age"].sum() / df_gouv_fr_raw_0.groupby("jour")["t"].sum()
    df_age["test_mean_age"] = ser_t_age

    # prepare data positive
    df_pos_fr = pt_fr_test["p"].copy()
    df_pos_fr.index = pt_fr_test["date"].index
    # add date
    df_pos_fr["date"] = df_pos_fr.index
    # add age
    df_pos_fr["age"] = df_age["pos_mean_age"].copy()
    # add positive cases sum   
    df_pos_fr = compute_sum_dep(df_pos_fr)
    # add nb_cases confirmed cummulative sum
    arr_nb_cases = df_pos_fr["daily"].cumsum().values
    df_pos_fr["nb_cases"] = NB_POS_DATE_MIN_DF_FEAT + arr_nb_cases
    # save data pos
    df_pos_fr.to_csv(PATH_DF_POS_FR, index=False)

    # prepare data tested
    df_test_fr = pt_fr_test["t"].copy()
    df_test_fr.index = pt_fr_test["date"].index
    # add date
    df_test_fr["date"] = df_test_fr.index
    # add age
    df_test_fr["age"] = df_age["test_mean_age"].copy()
    # add cases sum
    df_test_fr = compute_sum_dep(df_test_fr)
    # save data tested
    df_test_fr.to_csv(PATH_DF_TEST_FR, index=False)

    return df_pos_fr, df_test_fr

def precompute_data_pos_disk():
    ''' Pre-compute data from Sante Publique France from disk
    '''
    df_gouv_fr_raw = load_data_gouv()
    precompute_data_pos(df_gouv_fr_raw)

def prepare_features(df_feat_fr, df_pos_fr, df_test_fr):
    '''Finalize preparation of model features df_feat_fr table 
    to input model.
    Result is saved only. no output.
    '''
    # add daily positive cases for all departements
    df_feat_fr["pos"] = df_pos_fr["daily"].copy()
    # add age positive cases
    df_feat_fr["age_pos"] = df_pos_fr["age"].copy()

    # add daily tested cases for all departements    
    df_feat_fr["test"] = df_test_fr["daily"].copy()
    # add age tested cases
    df_feat_fr["age_test"] = df_test_fr["age"].copy()

    # add num days
    df_feat_fr['day_num'] = \
        df_feat_fr["date"].astype(np.datetime64).dt.strftime("%w")

    # add nb_cases confirmed cummulative sum
    df_feat_fr["nb_cases"] = df_pos_fr["nb_cases"].copy()

    # save for future uses
    df_feat_fr.to_csv(PATH_DF_FEAT_FR, index=False)

def prepare_features_disk():
    '''
    Prepare features from disk
    '''
    df_feat_fr = load_data_pos()
    df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
    df_test_fr = pd.read_csv(PATH_DF_TEST_FR)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr)

def prepare_features_disk_emr():
    '''
    Prepare features from disk after EMR
    Take meteo dataFrame as input for features 
    '''
    df_feat_fr = pd.read_csv(PATH_DF_METEO_FR)
    df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
    df_test_fr = pd.read_csv(PATH_DF_TEST_FR)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr)

def update_data_meteo_disk():
    ''' Update meteo light from disk
    '''
    df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
    update_data_meteo_light(df_pos_fr["date"].tolist())

def get_data_pos():
    '''
    1) Retrieve data from Sante Publique France direct CSV URL 
        (updated every days but with 4 to 5 days delay...)
    2) Proceed this data by departements (tested - positive)
    3) Retrieve data from Méteo France
    4) Proceed this data to have mean feature all over France every days
    5) Proceed features data for model by combining all these data

    Every databases are saved in CSV format.
    '''
    df_gouv_fr_raw = get_data_gouv_fr()
    # creation of data tables : tested & positive
    df_pos_fr, df_test_fr = precompute_data_pos(df_gouv_fr_raw)
    # meteo
    data_meteo = update_data_meteo_light(df_pos_fr.index.tolist())
    # create features for model
    # pre-compute data meteo & add
    df_feat_fr = precompute_data_meteo_light(data_meteo)
    # finalize features and save (df_feat_fr on disk)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr)

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
        df_gouv_fr_raw = pd.read_csv(PATH_DF_GOUV_FR_RAW, low_memory=False)
    except:
        # try to get from URL
        df_gouv_fr_raw = get_data_gouv_fr()

    return df_gouv_fr_raw

def update_pos(df_feat_fr):
    '''
    Update plot data positive cases France
    '''
    # pos last NB_DAY_PLOT days : date, pos, total (sum)
    str_date_0 = add_days(df_feat_fr.date.max(), -NB_DAY_PLOT)
    df_plot = df_feat_fr[df_feat_fr["date"] >= str_date_0].copy()
    return df_plot

def prepare_data_input(flag_update):
    '''Prepare data input'''
    if flag_update:
        get_data_pos()
    # load from disk
    df_feat_fr = load_data_pos()
    # last date of training
    str_date_mdl =  df_feat_fr.iloc[TRAIN_SPLIT]["date"]
    # date of last data
    str_data_date = "last data available: " + df_feat_fr["date"].max()

    return df_feat_fr, str_date_mdl, str_data_date

def prepare_plot_data_pos(df_feat_fr, flag_update):
    '''Prepare data for plot positive cases'''
    # plot data for positive cases
    df_plot = update_pos(df_feat_fr)
    str_date_last = df_plot.date.max()
    # predict 3 future days
    flag_pred_disk = not(flag_update)
    df_plot_pred = update_pred_pos(df_feat_fr, flag_pred_disk)
    # predict all past days
    df_plot_pred_all = update_pred_pos_all(df_feat_fr, flag_pred_disk)

    return df_plot, df_plot_pred, df_plot_pred_all, str_date_last

def check_update():
    '''
    Just check if new data possibly available
    (if file date older than 24 hours or 
    different dates of postive cases and meteo)
    '''
    time_file_df_feat_date = get_file_date(PATH_DF_FEAT_FR)
    dtime_now  = datetime.datetime.now() - time_file_df_feat_date

    # meteo check
    # if date df_pos != date df_meteo
    if (os.path.isfile(PATH_DF_POS_FR) & os.path.isfile(PATH_DF_METEO_FR)):
        df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR)
        if df_pos_fr["date"].max() !=  df_meteo_fr["date"].max():
            flag_meteo = True
        else:
            flag_meteo = False
    else:
        flag_meteo = True

    # update only if more than 24 hours without update
    if ((dtime_now.days > 0) | flag_meteo):
        flag_old = True
        # update data from external 
        print("Maybe new data available...")
        #get_data_pos()
    else:
        flag_old = False
        print("No new data available.")
    return flag_old