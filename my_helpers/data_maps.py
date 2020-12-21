# -*- coding: utf-8 -*-

# IMPORT

# import bluit-in
import math
import json
import datetime
# import thirs-party
import pandas as pd
import numpy as np
# import project modules
import settings
from my_helpers.dates import create_date_ranges, add_days
from my_helpers.data_plots import load_data_gouv
from my_helpers.utils import sum_mobile
from my_helpers.model import mdl_R0_estim
from my_helpers.model import NB_DAYS_CV, calc_rt_from_sum
# DEFINITIONS

# path local
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
PATH_DF_DEP_SUM = PATH_TO_SAVE_DATA + '/' + 'df_dep_sum.csv'
PATH_DF_DEP_R0 = PATH_TO_SAVE_DATA + '/' + 'df_dep_r0.csv'
PATH_PT_FR_TEST_LAST = PATH_TO_SAVE_DATA + '/' + 'pt_fr_test_last.csv'
PATH_DEP_FR = PATH_TO_SAVE_DATA + '/' + 'dep_fr.csv'
PATH_DF_CODE_DEP = PATH_TO_SAVE_DATA + '/' + 'df_code_dep.csv'
PATH_GEO_DEP_FR = PATH_TO_SAVE_DATA + '/sources/geofrance/' + 'departments.csv'
URL_GEOJSON_DEP_FR = PATH_TO_SAVE_DATA + \
    '/sources/departements-avec-outre-mer_simple.json'


# HELPERS FUNCTIONS

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

    ser_start, ser_end = create_date_ranges(df_gouv_fr_raw["jour"], NB_DAYS_CV)
    #print("ser_start : ", ser_start)
    #print("ser_end : ", ser_end)

    df_dep_sum = pd.DataFrame(index=df_dep_pos.index, columns=["date"],
                            data=df_dep_pos.index.tolist())
    for dep_curr in df_dep_pos.columns:
        df_dep_sum[dep_curr] = sum_mobile(df_dep_pos[dep_curr], ser_start, 
            ser_end)

    df_dep_sum.to_csv(PATH_DF_DEP_SUM, index=False)

    df_dep_r0 = pd.DataFrame(index=df_dep_sum.index, columns=["date"],
                            data=df_dep_sum.index.tolist())
    for dep_curr in df_dep_sum.columns:
        if dep_curr != "date":
            ser_rt = calc_rt_from_sum(df_dep_sum[dep_curr], NB_DAYS_CV)
            ser_rt.name = dep_curr
            df_dep_r0 = df_dep_r0.join(ser_rt)

    df_dep_r0.to_csv(PATH_DF_DEP_R0, index=False)

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

    pt_fr_test_last["R0"] = pt_fr_test_last["p"] / pt_fr_test_last["p_0"]

    '''pt_fr_test_last["R0"] = mdl_R0_estim(nb_cases=pt_fr_test_last["p_0"] + \
                                        pt_fr_test_last["p"] , 
                                        nb_cases_init=pt_fr_test_last["p_0"], 
                                        nb_day_contag=14, 
                                        delta_days=14)'''

    

    pt_fr_test_last.to_csv(PATH_PT_FR_TEST_LAST, index=False)

    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum

def load_data_rt():
    '''
    Load from disk pre-computed data for RT map
    output : df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep DataFrame
    '''
    dep_fr, df_code_dep = get_geo_fr()
    df_dep_r0 = load_df_dep_r0()
    df_dep_sum = load_df_dep_sum()
    pt_fr_test_last = load_pt_fr_test_last()
    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum

def load_df_dep_r0():
    df_dep_r0 = pd.read_csv(PATH_DF_DEP_R0)
    df_dep_r0.index = df_dep_r0["date"]
    return df_dep_r0

def load_df_dep_sum():
    df_dep_sum = pd.read_csv(PATH_DF_DEP_SUM)
    df_dep_sum.index = df_dep_sum["date"]
    return df_dep_sum

def load_pt_fr_test_last():
    return pd.read_csv(PATH_PT_FR_TEST_LAST)

def prepare_plot_data_map(flag_update=False):
    '''Prepare plot data for RT MAP'''
    # plot data for MAPS
    # rt plots
    if flag_update:
        df_gouv_fr_raw = load_data_gouv()
        df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum = \
            get_data_rt(df_gouv_fr_raw)
        
    else:
        df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum = \
            load_data_rt()
    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep, df_dep_sum