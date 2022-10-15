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
from my_helpers.meteo import extrapolate_df_meteo
from my_helpers.meteo import PATH_DF_METEO_FR
from my_helpers.meteo import PATH_DF_METEO_FR_OLD
from my_helpers.meteo import PATH_JSON_METEO_TEMP_FR_OLD
from my_helpers.model import FUTURE_TARGET, TRAIN_SPLIT
from my_helpers.model import calc_rt, update_pred_pos, update_pred_pos_all
from my_helpers.model import NB_DAYS_CV, calc_rt_from_sum, calc_sum_mobile
#from my_helpers.data_maps import calc_rt
#from my_helpers.data_maps import sum_mobile

# DEFINITIONS
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
URL_CSV_GOUV_FR = 'https://www.data.gouv.fr/' + \
    'fr/datasets/r/674bddab-6d61-4e59-b0bd-0be535490db0'
PATH_DF_GOUV_FR_RAW = PATH_TO_SAVE_DATA + '/' + 'df_gouv_fr_raw.csv'
NB_POS_DATE_MIN_DF_FEAT = 140227 # on 12/05/2020
NB_POS_DATE_MIN_DF_FEAT_OLD = NB_POS_DATE_MIN_DF_FEAT - 38892
PATH_DF_GOUV_FR_RAW_OLD = os.path.join(PATH_TO_SAVE_DATA,'sources/csv_fr' ,
    'donnees-tests-covid19-labo-quotidien-2020-05-29-19h00.csv')

PATH_DF_GOUV_FR_OLD = os.path.join(PATH_TO_SAVE_DATA, 'df_gouv_fr_raw_old.csv')
PATH_DF_POS_FR = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr.csv' 
PATH_DF_POS_FR_OLD = PATH_TO_SAVE_DATA + '/' + 'df_pos_fr_old.csv' 
PATH_DF_TEST_FR = PATH_TO_SAVE_DATA + '/' + 'df_test_fr.csv'
PATH_DF_TEST_FR_OLD = PATH_TO_SAVE_DATA + '/' + 'df_test_fr_old.csv'
PATH_DF_FEAT_FR = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr.csv'
PATH_DF_FEAT_FR_OLD = PATH_TO_SAVE_DATA + '/' + 'df_feat_fr_old.csv'
DATE_START_NEW = "2020-05-13"

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
    df_gouv_fr_raw = pd.read_csv(
        io.StringIO(req.decode('utf-8')),
        sep=";",
        low_memory=False
    ) # patch dtype 2020-09-08

    # past treat data upper cases -> lower cases
    if "t" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["t"] =  df_gouv_fr_raw["T"]
    if "p" not in df_gouv_fr_raw.columns:
        df_gouv_fr_raw["p"] =  df_gouv_fr_raw["P"]
    # patch : clear data in double !!!
    df_gouv_fr_raw = df_gouv_fr_raw[df_gouv_fr_raw["cl_age90"] != 0]

    # patch str -> double (after update data link on 18/05/2022)
    # "00,0" -> 0.0
    df_gouv_fr_raw["t"] = df_gouv_fr_raw["t"].str.replace(",",".").astype(np.float64)
    df_gouv_fr_raw["p"] = df_gouv_fr_raw["p"].str.replace(",",".").astype(np.float64)
    # save to csv
    df_gouv_fr_raw.to_csv(PATH_DF_GOUV_FR_RAW, index=False)

    return df_gouv_fr_raw

def apply_conv_age(str_in):
    ''' Convert age classes to mean age
    '''
    #TODO : add precision with age of total pop"
    dict_age = {"A": 14/2, 
                "B": (15+44)/2,
                "C": (45+64)/2,
                "D": (65+74)/2,
                "E": (75+100)/2}
    return dict_age[str_in]
    
def get_old_data_gouv_fr():
    '''
    Get from Gouv SFP page data cases in France before May 2020
    Clean & Save
    '''
    df_gouv_fr_raw_old = pd.read_csv(PATH_DF_GOUV_FR_RAW_OLD, sep=";")
    # reformat to new format
    df_gouv_fr_raw_old.rename(columns={"nb_test": "t",
        "nb_pos": "p" },
        inplace=True)
    # patch : clear data in double !!!
    df_gouv_fr_raw_old = \
        df_gouv_fr_raw_old[df_gouv_fr_raw_old["clage_covid"] != 0]
    df_gouv_fr_raw_old = \
        df_gouv_fr_raw_old[df_gouv_fr_raw_old["clage_covid"] != "0"]
    # "clage_covid": "cl_age90"
    df_gouv_fr_raw_old["cl_age90"] = \
        df_gouv_fr_raw_old["clage_covid"].apply(apply_conv_age)
    # select until first date of new data DATE_START_NEW
    df_gouv_fr_raw_old = \
        df_gouv_fr_raw_old[df_gouv_fr_raw_old["jour"] < DATE_START_NEW]
    # save
    df_gouv_fr_raw_old.to_csv(PATH_DF_GOUV_FR_OLD, index=False)

    return df_gouv_fr_raw_old

def compute_sum_dep(df_in):
    df_in["daily"] = 0
    list_dep = []
    for col_curr in df_in.columns:
        if re.search(r"^\d", col_curr):
            list_dep.append(col_curr)
    for dep_curr in list_dep:
        df_in["daily"]  += df_in[dep_curr]
    return df_in

def precompute_data_pos(df_gouv_fr_raw, nb_pos_start=NB_POS_DATE_MIN_DF_FEAT,
        path_df_pos_fr=PATH_DF_POS_FR, path_df_test_fr=PATH_DF_TEST_FR):
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
    df_pos_fr["nb_cases"] = nb_pos_start + arr_nb_cases
    # save data pos
    df_pos_fr.to_csv(path_df_pos_fr, index=False)

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
    df_test_fr.to_csv(path_df_test_fr, index=False)

    return df_pos_fr, df_test_fr

def precompute_data_pos_disk():
    ''' Pre-compute data from Sante Publique France from disk
    '''
    df_gouv_fr_raw = load_data_gouv()
    precompute_data_pos(df_gouv_fr_raw)

def precompute_old_data_pos_disk():
    ''' 
    Pre-compute old data from Sante Publique France from disk
    '''
    df_gouv_fr_raw_old = load_old_data_gouv()
    precompute_data_pos(df_gouv_fr_raw_old, nb_pos_start=0,
        path_df_pos_fr=PATH_DF_POS_FR_OLD, path_df_test_fr=PATH_DF_TEST_FR_OLD)

def prepare_features(df_feat_fr, df_pos_fr, df_test_fr,
        path_df_feat_fr=PATH_DF_FEAT_FR):
    '''
    Finalize preparation of model features df_feat_fr table
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

    # calculate sum-cases
    ser_sum = calc_sum_mobile(df_feat_fr["date"], df_feat_fr["pos"], NB_DAYS_CV)
    ser_sum.name = "sum_cases"
    df_feat_fr.drop(columns=["sum_cases"], inplace=True, errors="ignore")
    df_feat_fr = df_feat_fr.join(ser_sum)

    # calculate Rt country : Rt
    ser_rt = calc_rt_from_sum(df_feat_fr["sum_cases"], NB_DAYS_CV)
    ser_rt.name = "Rt"
    df_feat_fr.drop(columns=["Rt"], inplace=True, errors="ignore")
    df_feat_fr = df_feat_fr.join(ser_rt)
    
    # positive rate
    df_feat_fr["rate_pos"] = 100*df_feat_fr["pos"] / df_feat_fr["test"]
    # save for future uses
    df_feat_fr.to_csv(path_df_feat_fr, index=False)

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
    list_dates = df_pos_fr["date"].tolist()
    df_feat_fr = extrapolate_df_meteo(df_feat_fr, list_dates)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr)

def update_data_meteo_disk():
    ''' Update meteo light from disk for Airflow DAG
    '''
    df_pos_fr = pd.read_csv(PATH_DF_POS_FR)
    data_meteo_new = update_data_meteo_light(df_pos_fr["date"].tolist())
    assert data_meteo_new != None

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
    # list dates 
    list_dates = df_pos_fr["date"].tolist()
    # meteo
    data_meteo = update_data_meteo_light(list_dates)
    # create features for model
    # pre-compute data meteo & add
    df_meteo_fr = precompute_data_meteo_light(data_meteo)
    # extrapolation meteo
    df_feat_fr = extrapolate_df_meteo(df_meteo_fr, list_dates)
    # finalize features and save (df_feat_fr on disk)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr)

def get_old_data_pos():
    ''' Get Old data (before 13/05/2020)
    1) Retrieve data from Sante Publique France direct CSV URL 
        (updated every days but with 4 to 5 days delay...)
    2) Proceed this data by departements (tested - positive)
    3) Retrieve data from Méteo France
    4) Proceed this data to have mean feature all over France every days
    5) Proceed features data for model by combining all these data

    Every databases are saved in CSV format.
    '''
    df_gouv_fr_raw_old = get_old_data_gouv_fr()
    # creation of data tables : tested & positive
    df_pos_fr, df_test_fr = precompute_data_pos(df_gouv_fr_raw_old, 
        nb_pos_start=NB_POS_DATE_MIN_DF_FEAT_OLD, 
        path_df_pos_fr=PATH_DF_POS_FR_OLD, path_df_test_fr=PATH_DF_TEST_FR_OLD)
    # meteo
    data_meteo = update_data_meteo_light(df_pos_fr.index.tolist(), 
        path_df_meteo_fr=PATH_DF_METEO_FR_OLD,
        path_json_meteo_temp_fr=PATH_JSON_METEO_TEMP_FR_OLD)
    # create features for model
    # pre-compute data meteo & add
    df_feat_fr = precompute_data_meteo_light(data_meteo, 
        path_df_meteo_fr=PATH_DF_METEO_FR_OLD)
    # finalize features and save (df_feat_fr on disk)
    prepare_features(df_feat_fr, df_pos_fr, df_test_fr, 
        path_df_feat_fr=PATH_DF_FEAT_FR_OLD)

# FOR data to plot
def load_data_pos(path_df_feat_fr=PATH_DF_FEAT_FR):
    '''
    Load data positive cases France
    '''
    df_feat_fr = pd.read_csv(path_df_feat_fr)
    df_feat_fr.index = df_feat_fr["date"]
    return df_feat_fr

def load_old_data_pos():
    '''
    Load Old data positive cases France
    '''
    df_feat_fr_old = pd.read_csv(PATH_DF_FEAT_FR_OLD)
    df_feat_fr_old.index = df_feat_fr_old["date"]
    return df_feat_fr_old


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


def load_old_data_gouv():
    '''
    Load old data gouv France
    '''
    df_gouv_fr_raw_old = pd.read_csv(PATH_DF_GOUV_FR_OLD, low_memory=False)
    return df_gouv_fr_raw_old

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
    str_data_date = "(up to: " + df_feat_fr["date"].max() + ")"

    return df_feat_fr, str_date_mdl, str_data_date

def prepare_old_data_input(flag_update):
    '''Prepare old data input'''
    if flag_update:
        get_old_data_pos()
    # load from disk
    df_feat_fr_old = load_old_data_pos()
    return df_feat_fr_old

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

    # update only if more than 6 hours without update
    if ((dtime_now.total_seconds() > 6*3600) | flag_meteo):
        flag_old = True
        # update data from external 
        print("Maybe new data available...")
        #get_data_pos()
    else:
        flag_old = False
        print("No new data available.")
    return flag_old