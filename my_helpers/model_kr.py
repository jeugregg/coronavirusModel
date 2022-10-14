# -*- coding: utf-8 -*-

# IMPORT

# settings (special)
import settings

# import built-in 
import os
import math
import numpy as np
import pandas as pd
import json
import requests
if settings.PREDICT:
    if not settings.MODEL_TFLITE:
        import tensorflow as tf

# import project modules
from my_helpers.dates import add_days
from my_helpers.dates import generate_list_dates
from my_helpers.dates import create_date_ranges
from my_helpers.utils import sum_mobile
from my_helpers.model import prepare_to_lambda_future
from my_helpers.model import retrieve_from_lambda
from my_helpers.model import prepare_to_lambda

# DEFINITIONS
NB_DAYS_CV = 14 # state duration in nb days for contagious confirmed people

# plot
NB_PERIOD_PLOT = settings.NB_PERIOD_PLOT
# model parameters
TRAIN_SPLIT = 439 #347
PAST_HISTORY= NB_DAYS_CV # days used to predict next values in future
FUTURE_TARGET = 7 # predict 3 days later
STEP = 1

# model deep learning TLITE AWS LAMBDA
URL_PREDICT_KR = \
    "https://hauojq3o6f.execute-api.us-east-2.amazonaws.com/dev/infer"

# path local
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
PATH_DF_PLOT_PRED_KR = PATH_TO_SAVE_DATA + '/' + 'df_plot_pred_kr.csv'
PATH_DF_PLOT_PRED_ALL_KR = PATH_TO_SAVE_DATA + '/' + 'df_plot_pred_all_kr.csv'
PATH_MDL_MULTI_STEP_KR = PATH_TO_SAVE_DATA + '/' + "mdl_multi_step_pos_kr"

PATH_MDL_MULTI_TFLITE_KR = PATH_TO_SAVE_DATA + '/' + \
    'serverless/tensorflow_lite_on_aws_lambda_kr'
PATH_MDL_MULTI_TFLITE_FILE_KR = PATH_MDL_MULTI_TFLITE_KR + '/' + \
    "converted_model.tflite"
PATH_SERVERLESS_KR = PATH_MDL_MULTI_TFLITE_KR + '/' + 'serverless.yml'

def prepare_data_features_kr(df_feat_kr):
    '''
    Prepare DafaFrame to be used with model
    Reduce timeframe to usable dates without NaN values for features
    '''
    df_out = df_feat_kr.copy()
    
    # correct age
    df_age_pos = df_out["age_pos"].copy()
    df_age_pos.fillna(method="pad", inplace=True)
    df_out["age_pos"] = df_age_pos
    
    # check by drop if nan
    df_out.dropna(inplace=True, subset=["date", 'nb_cases', 
                                        'T_min', 'T_max', 'H_mean',
                                        'W_speed', 'pos', 'test', 'day_num',
                                        'age_pos'])
    date_old = add_days(df_out.index[0].strftime("%Y-%m-%d"), -1)
    dates_index =  df_out.index.strftime("%Y-%m-%d")
    for date_curr in dates_index:
        if date_curr != add_days(date_old, 1):
            print("ERROR : ", date_curr)
            #break
        assert date_curr == add_days(date_old, 1)
        date_old = date_curr
    return df_out

def prepare_dataset_kr(df_feat_kr_clean):
    '''
    Prepare final model inputs features
    Outputs numpy arrays of normalized dataset, std, and mean 
    '''
    # prepare features
    df_out = df_feat_kr_clean.copy()
    features = df_out.filter(items=['T_min', 'T_max', 'H_mean',
                                           'W_speed', 'pos', 'test', 'day_num',
                                           'age_pos'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    return dataset, data_std, data_mean

# Prediction

def update_pred_pos_kr(df_feat_kr, from_disk=False):
    '''
    Update prediction data positive cases France
    '''

    # check if last prediction is after last known date
    if os.path.isfile(PATH_DF_PLOT_PRED_KR):
        df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED_KR)
        df_plot_pred.index = df_plot_pred["date"]
        if df_plot_pred["date"].min() <= df_feat_kr["date"].max():
            from_disk = False
    else:
        from_disk = False
        
    # if no prediction or if from disk 
    if (not settings.PREDICT) | from_disk:
        df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED_KR)
        df_plot_pred.index = df_plot_pred["date"]
        return df_plot_pred

    # prepare features
    dataset, data_std, data_mean = prepare_dataset_kr(df_feat_kr)
    # predict next days
    if settings.MODEL_TFLITE:
        json_list_list_x = prepare_to_lambda_future(dataset)
        resp = requests.post(URL_PREDICT_KR, json=json_list_list_x)
        print("status code : ", resp.status_code) 
        if resp.status_code == 200:
            y_multi_pred = retrieve_from_lambda(resp)
        else:
            print("AWS Lamdba future pred ERROR!")
            df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED_KR)
            df_plot_pred.index = df_plot_pred["date"]
            return df_plot_pred            
    else:
        # prepare data : very last days
        x_multi = np.array([dataset[-PAST_HISTORY:,:]]) 
        # load model
        multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP_KR)
        y_multi_pred = multi_step_model.predict(x_multi)

    # convert in positive cases
    y_pos_pred = y_multi_pred * data_std[4] + data_mean[4]
    # pos pred next 3 days from last day : date, pos, total (sum)
    str_date_pred_0 = df_feat_kr.date.max()
    str_date_pred_1 = add_days(str_date_pred_0, FUTURE_TARGET)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)
    # figure 
    df_plot_pred = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                        data=list_dates_pred)

    df_plot_pred["pos"] = y_pos_pred[0].astype(int)


    arr_nb_pred = df_plot_pred["pos"].cumsum().values
    df_plot_pred["nb_cases"] = df_feat_kr["nb_cases"].max() + arr_nb_pred

    # save for future pred
    df_plot_pred.to_csv(PATH_DF_PLOT_PRED_KR, index=False)

    return df_plot_pred

def update_pred_pos_all_kr(df_feat_kr, from_disk=False):
    '''
    Update prediction data positive cases France for all days
    '''
    if os.path.isfile(PATH_DF_PLOT_PRED_ALL_KR):
        df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED_ALL_KR)
        df_plot_pred.index = df_plot_pred["date"]
        if df_plot_pred["date"].max() < df_feat_kr["date"].max():
            from_disk = False
    else:
        from_disk = False

    if (not settings.PREDICT) | from_disk:
        df_plot_pred_all = pd.read_csv(PATH_DF_PLOT_PRED_ALL_KR)
        df_plot_pred_all.index = df_plot_pred_all["date"]
        return df_plot_pred_all

    # prepare features
    dataset, data_std, data_mean = prepare_dataset_kr(df_feat_kr)

    # predict
    if settings.MODEL_TFLITE:
        json_list_list_x = prepare_to_lambda(dataset)
        resp = requests.post(URL_PREDICT_KR, json=json_list_list_x)
        print("status code : ", resp.status_code) 
        if resp.status_code == 200:
            y_multi_pred = retrieve_from_lambda(resp)
        else:
            print("AWS Lamdba future pred ERROR!")
            df_plot_pred_all = pd.read_csv(PATH_DF_PLOT_PRED_ALL_KR)
            df_plot_pred_all.index = df_plot_pred_all["date"]
            return df_plot_pred_all       
    else:

        # load model
        multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP_KR)

        list_x = []

        # prepare data : very last days
        nb_max = NB_PERIOD_PLOT
        for I in range(nb_max, 0, -1):
            I_start = I * FUTURE_TARGET - PAST_HISTORY
            if I_start < 0:
                break
            I_end = I * FUTURE_TARGET
            list_x.append(np.array([dataset[I_start:I_end, :]]))

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

    # list of dates
    K_days = y_pos_pred.shape[1]
    print("K_days = ", K_days)
    print("y_pos_pred.shape = ", y_pos_pred.shape)
    str_date_pred_1 = df_feat_kr.date.max()
    str_date_pred_0 = add_days(str_date_pred_1, -1*K_days)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)

    # create df output
    df_plot_pred_all = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                       data=list_dates_pred)
    # daily
    df_plot_pred_all["pos"] = y_pos_pred[0].astype(int)

    # Total : cumulate sum 
    list_nb_cases =[]
    str_date_nb_0 = str_date_pred_0
    for I in range(0, df_plot_pred_all["pos"].shape[0], FUTURE_TARGET):
        str_date_nb_0 = add_days(str_date_pred_0, I)
        nb_0 = df_feat_kr[df_feat_kr["date"] == str_date_nb_0]["nb_cases"][0]
        arr_nb = nb_0 + \
            df_plot_pred_all.iloc[I:I+FUTURE_TARGET]["pos"].cumsum().values
        list_nb_cases = list_nb_cases + arr_nb.tolist()
    df_plot_pred_all["nb_cases"] = list_nb_cases

    # save for future pred
    df_plot_pred_all.to_csv(PATH_DF_PLOT_PRED_ALL_KR, index=False)

    return df_plot_pred_all
