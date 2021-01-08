# -*- coding: utf-8 -*-
'''
Module Test model KR
'''
# import 

# classical
import math
import numpy as np
import pandas as pd
import json
import requests

# import third party
import pytest
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# import project libs
import settings

from my_helpers.model import multivariate_data
from my_helpers.model import create_list_past_hist
from my_helpers.model import prepare_to_lambda
from my_helpers.model import prepare_to_lambda_future
from my_helpers.model import retrieve_from_lambda
from my_helpers.model import predict_list

import my_helpers.model_kr as model
from my_helpers.model_kr import prepare_data_features_kr
from my_helpers.model_kr import prepare_dataset_kr

from my_helpers.data_plots_kr import load_df_feat_kr

from serverless.tensorflow_lite_on_aws_lambda_kr.handler import predict

# definitions
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA 
from my_helpers.model_kr import PATH_MDL_MULTI_TFLITE_KR
from my_helpers.model_kr import PATH_MDL_MULTI_TFLITE_FILE_KR
PATH_MDL_MULTI_STEP_KR = model.PATH_MDL_MULTI_STEP_KR
PAST_HISTORY = model.PAST_HISTORY
URL_PREDICT = model.URL_PREDICT_KR
TRAIN_SPLIT = model.TRAIN_SPLIT
FUTURE_TARGET = model.FUTURE_TARGET
STEP = model.STEP

ERROR_REL_MAX = 20 # in %

# prepare test

# load data
# load
df_feat_kr = load_df_feat_kr()
# clean
df_feat_kr = prepare_data_features_kr(df_feat_kr)

dataset, data_std, data_mean = prepare_dataset_kr(df_feat_kr)

# load model Tensorflow
multi_step_model = tf.keras.models.load_model(PATH_MDL_MULTI_STEP_KR)

# predict with TensorFlow model
list_x = create_list_past_hist(dataset)
y_multi_pred = predict_list(list_x, multi_step_model)
x_for_future = np.array([dataset[-PAST_HISTORY:,:]]) 
y_future_pred = multi_step_model.predict(x_for_future)

# load converted model Tensorflow LITE
interpreter = tf.lite.Interpreter(model_path=PATH_MDL_MULTI_TFLITE_FILE_KR)

# TESTS
class TestModel:

    def test_relative_error(self):
        '''
        Test model Tensorflow relative error :  less than ERROR_REL_MAX %  ?
        '''
        x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 4],
                                TRAIN_SPLIT-PAST_HISTORY, None, PAST_HISTORY,
                                FUTURE_TARGET, STEP, single_step=False)

        y_val_multi_pred = multi_step_model.predict(x_val_multi)
        mean_error_val = data_std[4] * mean_absolute_error(y_val_multi, 
                                                   y_val_multi_pred)
        print("Mean error  [nb of cases]: ", mean_error_val)
        mean_val = np.mean(data_std[4] * y_val_multi + data_mean[4])
        print("Mean nb cases  [nb of cases]: ", mean_val)
        rel_error_pc = 100*mean_error_val/mean_val
        print("mean relative error [%]: ", rel_error_pc)

        assert rel_error_pc < ERROR_REL_MAX

class TestConvertedModel:

    def test_predict(self):
        '''
        Test if TFLITE model has same prediction than TF model  
        '''
        list_x = create_list_past_hist(dataset)
        # Run the model with TensorFlow Lite
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check if same results 
        for x_multi in list_x:
            expected = multi_step_model.predict(x_multi)
            interpreter.set_tensor(input_details[0]["index"], 
                           x_multi.astype(np.float32))
            interpreter.invoke()
            result = interpreter.get_tensor(output_details[0]["index"])
            # Assert if result of TFLite model is consistent with the TF one.
            np.testing.assert_almost_equal(expected, result, decimal=3)
            print("Done. The results of TensorFlow / TensorFlow Lite matches.")
            # Please note: TfLite fused Lstm kernel is stateful, 
            # so we need to reset the states.
            # Clean up internal states.
            interpreter.reset_all_variables()
    
    def test_lambda_interface(self):
        '''
        Test interface with lambda AWS (IN/OUT)
        '''
        # test past days

        json_list_list_x = prepare_to_lambda(dataset)
        # simulate input to lambda (double dumps ? why ? i don't know yet)
        json_list_list_x = json.dumps(json_list_list_x)
        # simulate lambda
        event = {"body": json_list_list_x}
        # lambda
        context = None
        response = predict(event, context)
        # Retrieve from lambda in App code
        y_multi_pred_out = retrieve_from_lambda(response)      
        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(y_multi_pred, y_multi_pred_out, 
            decimal=3)
        print("Done. PAST DAYS : TensorFlow and TensorFlow Lite matches.")

        # test future days

        json_list_list_x = prepare_to_lambda_future(dataset)
        json_list_list_x = json.dumps(json_list_list_x) # dumps again : I dont know why
        event = {"body": json_list_list_x}
        context = None
        response = predict(event, context)
        y_future_pred_out = retrieve_from_lambda(response)
        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(y_future_pred, y_future_pred_out, 
            decimal=3)
        print("Done. FUTURE DAYS : TensorFlow and TensorFlow Lite matches.")


    def test_lambda(self):
        '''
        Test Lambda function on AWS
        '''
        # prepare input
        json_list_list_x = prepare_to_lambda(dataset)
        # REQUEST
        resp = requests.post(URL_PREDICT, json=json_list_list_x)
        #print("status code : ", resp.status_code) 
        #print(resp.json())
        y_multi_pred_out = retrieve_from_lambda(resp)     
        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(y_multi_pred, y_multi_pred_out, 
            decimal=3)
        print("Done. PAST DAYS : TensorFlow and TensorFlow Lite matches.")

        # prepare input
        json_list_list_x = prepare_to_lambda_future(dataset)
        # REQUEST 
        resp = requests.post(URL_PREDICT, json=json_list_list_x)
        y_future_pred_out = retrieve_from_lambda(resp)   
        # Assert if the result of TFLite model is consistent with the TF model.
        np.testing.assert_almost_equal(y_future_pred, y_future_pred_out, 
            decimal=3)
        print("Done. FUTURE DAYS : TensorFlow and TensorFlow Lite matches.")
