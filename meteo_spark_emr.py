#!/usr/bin/python
# -*-coding:utf-8 -*

''' Data meteo france treatment : script on AWS EMR

Push this script on S3

- input files from S3: 
        - PATH_DF_METEO_FR : df_meteo_fr (csv)
        - PATH_JSON_METEO_TEMP_FR : data_meteo_temp_fr (json)
- output files to S3: 
        - PATH_DF_METEO_FR_OUT : df_meteo_fr updated (csv)
        - PATH_DF_METEO_FR_TMP : df_meteo_fr_tmp (new processed data) (csv)

'''

# import bluit-in
import os
from io import StringIO  # python3 (or BytesIO for python2)
import sys
import ntpath

# import third-party
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import explode, substring
import pyspark.sql.functions as fun_sql
import boto3
import pandas as pd

# definitions 
BUCKET_NAME = 'app-covid-visu-bucket'
MODE_S3 = True
MODE_TEST = True

if MODE_S3:
        PATH_TO_SAVE_DATA = f's3://{BUCKET_NAME}' 
else:
        PATH_TO_SAVE_DATA = ntpath.dirname(__file__)

if MODE_TEST:
        # inputs
        PATH_DF_METEO_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_for_emr_test.csv')
        PATH_JSON_METEO_TEMP_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'data_meteo_temp_fr_for_emr_test.json')
        # outputs
        PATH_DF_METEO_FR_TMP = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_tmp_for_emr_test.csv')
        PATH_DF_METEO_FR_OUT = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_out_for_emr_test.csv')
else:
        # inputs
        PATH_DF_METEO_FR = os.path.join(PATH_TO_SAVE_DATA, 'df_meteo_fr.csv')
        PATH_JSON_METEO_TEMP_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'data_meteo_temp_fr.json')
        # outputs
        PATH_DF_METEO_FR_TMP = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_tmp.csv')
        PATH_DF_METEO_FR_OUT = PATH_DF_METEO_FR

s3 = boto3.resource('s3')

def download_file_S3(path_file, bucket_name):
        filename = ntpath.basename(path_file)
        s3.Bucket(bucket_name).download_file(filename, path_file)

def upload_file_to_S3(filename, bucket_name):
        key = ntpath.basename(filename)
        s3.Bucket(bucket_name).upload_file(filename, key)

def upload_df_to_S3(df, path_file, bucket_name=BUCKET_NAME):
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        filename = ntpath.basename(path_file)
        s3.Object(bucket_name, filename).put(Body=csv_buffer.getvalue())

# get data input from S3
if MODE_S3:
        #download_file_S3(PATH_JSON_METEO_TEMP_FR, BUCKET_NAME)
        #download_file_S3(PATH_DF_METEO_FR, BUCKET_NAME)
        # Configure Spark
        sparkcontext = SparkContext()
else:
        conf = SparkConf().set('spark.driver.host','127.0.0.1')
        sparkcontext = SparkContext(conf=conf)

# Create an sql context so that we can query data files in sql like syntax
sqlContext = SQLContext(sparkcontext)

# read the json data file and select only the field labeled as "text"
# this returns a spark data frame
#jsondata = sqlContext.read.json(os.path.join(abspath, datafile_json))
jsondata = sqlContext.read.json(PATH_JSON_METEO_TEMP_FR)

df = jsondata.select("records")
try:
        dfMeteo = df.select(explode("records").alias("rec")). \
        selectExpr("rec.fields as records"). \
        selectExpr("records.numer_sta as numer_sta", 
                "records.date as date", "records.t as t", "records.u as u"). \
        alias("df").withColumn("date", substring("df.date", 1, 10))
        flag_ok = True
except:
        print('meteo spark : NO new data ?')
        flag_ok = False

if flag_ok: 
        dfMeteo.persist()

        # T_min group by station and date
        df_tmin = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.min('t').alias('T_min'))
        df_tmin = df_tmin.groupby('date').agg(fun_sql.mean('T_min'). \
                alias('T_min')).orderBy('date')
        #df_tmin.show()
        #df_tmin.persist()
        # T_max group by station and date
        df_tmax = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.max('t').alias('T_max'))
        df_tmax = df_tmax.groupby('date').agg(fun_sql.mean('T_max'). \
                alias('T_max')).orderBy('date')

        # join
        df_meteo =  df_tmin.join(df_tmax, df_tmin.date == df_tmax.date). \
                drop(df_tmax.date)

        # H_min group by station and date
        df_hmin = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.min('u').alias('H_min'))
        df_hmin = df_hmin.groupby('date').agg(fun_sql.mean('H_min'). \
                alias('H_min')).orderBy('date')

        # join
        df_meteo =  df_meteo.join(df_hmin, df_meteo.date == df_hmin.date). \
                drop(df_hmin.date)

        # H_min group by station and date
        df_hmax = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.max('u').alias('H_max'))
        df_hmax = df_hmax.groupby('date').agg(fun_sql.mean('H_max'). \
                alias('H_max')).orderBy('date')

        # join
        df_meteo =  df_meteo.join(df_hmax, df_meteo.date == df_hmax.date).\
                drop(df_hmax.date)

        # prepare to save
        df_meteo_fr_new = df_meteo.toPandas()
        df_meteo_fr_new.index = df_meteo_fr_new["date"]

        if MODE_S3:
                upload_df_to_S3(df_meteo_fr_new, PATH_DF_METEO_FR_TMP)
                #df_meteo_fr_new.to_csv(PATH_DF_METEO_FR_TMP, index=False) 
                #upload_file_to_S3(PATH_DF_METEO_FR_TMP, BUCKET_NAME)
        # load old data 
        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR)
        df_meteo_fr.index = df_meteo_fr["date"]

        # append new data 
        try:
                df_meteo_fr = df_meteo_fr.append(df_meteo_fr_new, 
                        verify_integrity=True)
                # save df_meteo
                upload_df_to_S3(df_meteo_fr, PATH_DF_METEO_FR_OUT)
                #df_meteo_fr.to_csv(PATH_DF_METEO_FR_OUT, index=False) 

                # export on S3
                #if MODE_S3:
                #        upload_file_to_S3(PATH_DF_METEO_FR_OUT, BUCKET_NAME)
        except: 
                print('meteo spark : data duplicates ?')

        df_meteo.show()

#input("press ctrl+c to exit")
