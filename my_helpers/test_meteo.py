# -*- coding: utf-8 -*-

# import 

# built-in libs
import os
import shutil
import math
import datetime
import json
# third party libs
import numpy as np
import pandas as pd
# projects libs
from my_helpers.meteo import update_data_meteo_light
from my_helpers.meteo import precompute_data_meteo_light
from my_helpers.meteo import get_data_meteo_date_list
from my_helpers.meteo import extrapolate_df_meteo
from my_helpers.data_plots import load_data_gouv
from my_helpers.data_plots import precompute_data_pos
from my_helpers.data_plots import prepare_features
from my_helpers.data_plots import load_data_pos


# definitions
from settings import PATH_TO_SAVE_DATA
from my_helpers.data_plots import NB_POS_DATE_MIN_DF_FEAT
PATH_DF_POS_FR_TEST = os.path.join(PATH_TO_SAVE_DATA,
    'df_pos_fr_test.csv' )
PATH_DF_TEST_FR_TEST = os.path.join(PATH_TO_SAVE_DATA,
    'df_test_fr_test.csv')
PATH_JSON_METEO_TEMP_FR_TEST = os.path.join(PATH_TO_SAVE_DATA,
    'data_meteo_temp_fr_test.json')
PATH_DF_METEO_FR_TEST_DEF = os.path.join(PATH_TO_SAVE_DATA,
    'df_meteo_fr_for_test.csv')
PATH_DF_METEO_FR_TEST = os.path.join(PATH_TO_SAVE_DATA,
    'df_meteo_fr_test.csv')
PATH_DF_FEAT_FR_TEST = os.path.join(PATH_TO_SAVE_DATA,
    'df_feat_fr_test.csv')

# prepare data
df_gouv_fr_raw = load_data_gouv()
df_pos_fr, df_test_fr = precompute_data_pos(df_gouv_fr_raw,
    nb_pos_start=NB_POS_DATE_MIN_DF_FEAT,
    path_df_pos_fr=PATH_DF_POS_FR_TEST,
    path_df_test_fr=PATH_DF_TEST_FR_TEST)

list_dates = df_pos_fr["date"].tolist() # list dates to be treated
print("liste dates from URL gouv.fr to be treated: ", list_dates)

# TESTS
class TestMeteo:

    @classmethod
    def setup_class(cls):
        '''
        Prepare initial meteo data
        '''
        shutil.copyfile(PATH_DF_METEO_FR_TEST_DEF, PATH_DF_METEO_FR_TEST)

    def test_update_data_meteo_light(self):
        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR_TEST)
        list_dates_meteo = df_meteo_fr["date"].tolist()
        flag_update_meteo = False
        list_dates_to_proceed = list()
        for date_curr in list_dates:
            if date_curr not in list_dates_meteo:
                flag_update_meteo = True
                list_dates_to_proceed.append(date_curr)
                if len(list_dates_to_proceed) > 10:
                    break
                
        if flag_update_meteo:
            data_meteo = update_data_meteo_light(list_dates_to_proceed,
                path_df_meteo_fr=PATH_DF_METEO_FR_TEST,
                path_json_meteo_temp_fr=PATH_JSON_METEO_TEMP_FR_TEST)
            assert data_meteo is not None
    
    def test_precompute_data_meteo_light(self):
        with open(PATH_JSON_METEO_TEMP_FR_TEST) as f:
            data_meteo = json.load(f)
        df_feat_fr = precompute_data_meteo_light(data_meteo,
            path_df_meteo_fr=PATH_DF_METEO_FR_TEST)
        list_dates_final = df_feat_fr["date"].tolist()
        list_dates_meteo = get_data_meteo_date_list(data_meteo)
        for date_curr in list_dates_meteo:
            assert  date_curr in list_dates_final,  "{} not in final dates". \
                format(date_curr)

    def test_extrapolate_df_meteo(self):
        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR_TEST)
        df_meteo_fr.index = df_meteo_fr["date"]
        df_meteo_fr = extrapolate_df_meteo(df_meteo_fr, list_dates,
            path_df_meteo_fr=PATH_DF_METEO_FR_TEST)

        list_dates_final = df_meteo_fr["date"].tolist()

        for date_curr in list_dates:
            assert  date_curr in list_dates_final, "{} not in final dates". \
                format(date_curr)

    def test_prepare_features(self):

        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR_TEST)
        df_meteo_fr.index = df_meteo_fr["date"]
        list_dates_meteo = df_meteo_fr["date"].to_list()
        prepare_features(df_meteo_fr, df_pos_fr, df_test_fr,
            path_df_feat_fr=PATH_DF_FEAT_FR_TEST)

        df_feat_fr = load_data_pos(path_df_feat_fr=PATH_DF_FEAT_FR_TEST)
        list_dates_final = df_feat_fr["date"].tolist()

        assert sorted(list_dates_final) == sorted(list_dates_meteo), "some dates are missing or different!"


