# -*- coding: utf-8 -*-

# IMPORT

# settings
import settings
# classical
import pandas as pd
import numpy as np
import math
import json
import datetime
# helpers project modules
from my_helpers.dates import create_date_ranges, add_days
from my_helpers.data_plots import load_data_gouv 

# DEFINITIONS

# path local
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
PATH_DF_DEP_R0 = PATH_TO_SAVE_DATA + '/' + 'df_dep_r0.csv'
PATH_PT_FR_TEST_LAST = PATH_TO_SAVE_DATA + '/' + 'pt_fr_test_last.csv'
PATH_DEP_FR = PATH_TO_SAVE_DATA + '/' + 'dep_fr.csv'
PATH_DF_CODE_DEP = PATH_TO_SAVE_DATA + '/' + 'df_code_dep.csv'
PATH_GEO_DEP_FR = PATH_TO_SAVE_DATA + '/sources/geofrance/' + 'departments.csv'
URL_GEOJSON_DEP_FR = PATH_TO_SAVE_DATA + \
    '/sources/departements-avec-outre-mer_simple.json'

NB_DAYS_CV = 14

# HELPERS FUNCTIONS

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

    ser_start , ser_end = create_date_ranges(df_gouv_fr_raw["jour"], NB_DAYS_CV)
    #print("ser_start : ", ser_start)
    #print("ser_end : ", ser_end)

    df_dep_sum = pd.DataFrame(index=df_dep_pos.index, columns=["date"],
                            data=df_dep_pos.index.tolist())
    for dep_curr in df_dep_pos.columns:
        df_dep_sum[dep_curr] = sum_mobile(df_dep_pos[dep_curr], ser_start, 
            ser_end)

    df_dep_r0 = pd.DataFrame(index=df_dep_pos.index, columns=["date"],
                            data=df_dep_pos.index.tolist())

    for dep_curr in df_dep_sum.columns[1:]:
        ser_val = df_dep_sum[dep_curr].copy()
        date_min = add_days(ser_val.index.min(), NB_DAYS_CV) 
        ser_r0 = ser_val.copy()*np.nan
        for date_curr in ser_val[ser_val.index >= date_min].index:
            date_0 = add_days(date_curr, -NB_DAYS_CV)
            if not(np.isnan(ser_val.loc[date_0])):
                sum_0 = ser_val.loc[date_0]
                sum_1 = sum_0 + ser_val.loc[date_curr]
                ser_r0.loc[date_curr] = mdl_R0_estim(nb_cases=sum_1, 
                                                    nb_cases_init=sum_0,
                                                    nb_day_contag=NB_DAYS_CV, 
                                                    delta_days=NB_DAYS_CV)
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

    df_dep_r0.to_csv(PATH_DF_DEP_R0, index=False)

    pt_fr_test_last.to_csv(PATH_PT_FR_TEST_LAST, index=False)

    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep

def load_data_rt():
    '''
    Load from disk pre-computed data for RT map
    output : df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep DataFrame
    '''
    dep_fr, df_code_dep = get_geo_fr()
    df_dep_r0 = load_df_dep_r0()
    pt_fr_test_last = load_pt_fr_test_last()
    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep

def load_df_dep_r0():
    df_dep_r0 = pd.read_csv(PATH_DF_DEP_R0)
    df_dep_r0.index = df_dep_r0["date"]
    return df_dep_r0

def load_pt_fr_test_last():
    return pd.read_csv(PATH_PT_FR_TEST_LAST)

def prepare_plot_data_map(flag_update=False):
    '''Prepare plot data for RT MAP'''
    # plot data for MAPS
    # rt plots
    if flag_update:
        df_gouv_fr_raw = load_data_gouv()
        df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
            get_data_rt(df_gouv_fr_raw)
    else:
        df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep = \
            load_data_rt()
    return df_dep_r0, pt_fr_test_last, dep_fr, df_code_dep