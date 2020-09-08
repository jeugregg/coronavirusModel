# -*- coding: utf-8 -*-

# IMPORT

# settings
import settings
# classical
import math
import numpy as np
import pandas as pd
import json
import requests
if settings.PREDICT:
    if not settings.MODEL_TFLITE:
        import tensorflow as tf
# helpers project modules
from my_helpers.dates import add_days, generate_list_dates

# DEFINITIONS 

# plot
NB_PERIOD_PLOT = settings.NB_PERIOD_PLOT
# model parameters
TRAIN_SPLIT = 93
PAST_HISTORY= 14 # days used to predict next values in future
FUTURE_TARGET = 7 # predict 3 days later
STEP = 1

# model deep learning TLITE AWS LAMBDA
URL_PREDICT = 'https://yl0910jrga.execute-api.us-east-2.amazonaws.com/dev/infer'

# path local
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA
PATH_DF_PLOT_PRED = PATH_TO_SAVE_DATA + '/' + 'df_plot_pred.csv'
PATH_DF_PLOT_PRED_ALL = PATH_TO_SAVE_DATA + '/' + 'df_plot_pred_all.csv'
PATH_MDL_SINGLE_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_single_step_pos_fr"
PATH_MDL_MULTI_STEP = PATH_TO_SAVE_DATA + '/' + "mdl_multi_step_pos_fr"

# FOR AWS Lambda predict
def prepare_to_lambda(dataset):
    '''
    Prepare data input model to be used by lambda: 
    
    for prediction all past days
    '''
    list_list_x = []
    nb_max = math.ceil(NB_PERIOD_PLOT)
    I_start_pred = TRAIN_SPLIT - nb_max*FUTURE_TARGET
    for I in range(nb_max):
        I_start = I_start_pred + I * FUTURE_TARGET - PAST_HISTORY
        I_end =   I_start_pred + I * FUTURE_TARGET
        print(f"[{I_start} - {I_end}]")
        list_list_x.append(np.array([dataset[I_start:I_end, :]]).tolist())
        
    json_list_list_x = json.dumps(list_list_x)
    return json_list_list_x

def retrieve_from_lambda(response):
    '''
    To retrieve prediction from AWS Lambda
    '''

    if type(response)  == requests.models.Response:
        list_list_out = response.json()
    else: # for local test
        json_list_list_out = response.get("body")
        list_list_out = json.loads(json_list_list_out)
    
    y_multi_pred_out = []
    for I, list_x_multi in enumerate(list_list_out):
        if I:
            y_multi_pred_out = np.concatenate([y_multi_pred_out, 
                                           np.array(list_x_multi)],
                                  axis=1)
        else: # for first entry
            y_multi_pred_out = np.array(list_x_multi)
    return y_multi_pred_out   

def prepare_to_lambda_future(dataset):
    '''
    Prepare data input model to be used by lambda: 
    
    for prediction of very last days
    '''
    return json.dumps([[dataset[-PAST_HISTORY:,:].tolist()]])

# Prediction

def update_pred_pos(df_feat_fr, from_disk=False):
    '''
    Update prediction data positive cases France
    '''
    if (not settings.PREDICT) | from_disk:
        df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED)
        df_plot_pred.index = df_plot_pred["date"]
        return df_plot_pred

    # prepare features
    features = df_feat_fr.copy().filter(items=['T_min', 'T_max', 'H_min',
                                           'H_max', 'pos', 'test', 'day_num',
                                           'age_pos', 'age_test'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    # predict next days
    if settings.MODEL_TFLITE:
        json_list_list_x = prepare_to_lambda_future(dataset)
        resp = requests.post(URL_PREDICT, json=json_list_list_x)
        print("status code : ", resp.status_code) 
        if resp.status_code == 200:
            y_multi_pred = retrieve_from_lambda(resp)
        else:
            print("AWS Lamdba future pred ERROR!")
            df_plot_pred = pd.read_csv(PATH_DF_PLOT_PRED)
            df_plot_pred.index = df_plot_pred["date"]
            return df_plot_pred            
    else:
        # prepare data : very last days
        x_multi = np.array([dataset[-PAST_HISTORY:,:]]) 
        # load model
        multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)
        y_multi_pred = multi_step_model.predict(x_multi)

    # convert in positive cases
    y_pos_pred = y_multi_pred * data_std[4] + data_mean[4]
    # pos pred next 3 days from last day : date, pos, total (sum)
    str_date_pred_0 = df_feat_fr.date.max()
    str_date_pred_1 = add_days(str_date_pred_0, FUTURE_TARGET)
    list_dates_pred = generate_list_dates(str_date_pred_0, str_date_pred_1)
    # figure 
    df_plot_pred = pd.DataFrame(index=list_dates_pred, columns=["date"], 
                        data=list_dates_pred)

    df_plot_pred["pos"] = y_pos_pred[0].astype(int)


    arr_nb_pred = df_plot_pred["pos"].cumsum().values
    df_plot_pred["nb_cases"] = df_feat_fr["nb_cases"].max() + arr_nb_pred

    # save for future pred
    df_plot_pred.to_csv(PATH_DF_PLOT_PRED, index=False)

    return df_plot_pred

def update_pred_pos_all(df_feat_fr, from_disk=False):
    '''
    Update prediction data positive cases France for all days
    '''
    if (not settings.PREDICT) | from_disk:
        df_plot_pred_all = pd.read_csv(PATH_DF_PLOT_PRED_ALL)
        df_plot_pred_all.index = df_plot_pred_all["date"]
        return df_plot_pred_all

    # prepare features
    features = df_feat_fr.copy().filter(items=['T_min', 'T_max', 'H_min',
                                           'H_max', 'pos', 'test', 'day_num',
                                           'age_pos', 'age_test'])
    # prepare dataset 
    dataset = features.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std

    # predict
    if settings.MODEL_TFLITE:
        json_list_list_x = prepare_to_lambda(dataset)
        resp = requests.post(URL_PREDICT, json=json_list_list_x)
        print("status code : ", resp.status_code) 
        if resp.status_code == 200:
            y_multi_pred = retrieve_from_lambda(resp)
        else:
            print("AWS Lamdba future pred ERROR!")
            df_plot_pred_all = pd.read_csv(PATH_DF_PLOT_PRED_ALL)
            df_plot_pred_all.index = df_plot_pred_all["date"]
            return df_plot_pred_all       
    else:

        # load model
        multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP)

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

    # list od dates
    K_days = y_pos_pred.shape[1]
    print("K_days = ", K_days)
    print("y_pos_pred.shape = ", y_pos_pred.shape)
    str_date_pred_1 = df_feat_fr.date.max()
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
        nb_0 = df_feat_fr[df_feat_fr["date"] == str_date_nb_0]["nb_cases"][0]
        arr_nb = nb_0 + \
            df_plot_pred_all.iloc[I:I+FUTURE_TARGET]["pos"].cumsum().values
        list_nb_cases = list_nb_cases + arr_nb.tolist()
    df_plot_pred_all["nb_cases"] = list_nb_cases

    # save for future pred
    df_plot_pred_all.to_csv(PATH_DF_PLOT_PRED_ALL, index=False)

    return df_plot_pred_all