#!/usr/bin/python
# -*-coding:utf-8 -*

''' Data meteo france treatment in local
/usr/local/opt/apache-spark/bin/spark-submit ./meteo_spark.py
'''

# import bluit-in
import os
import sys
import ntpath

# import third-party
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import explode, substring
import pyspark.sql.functions as fun_sql
import boto3
import pandas as pd

# import from project
import settings


# definition 
MODE_S3 = not(settings.MODE_DEBUG)
MODE_TEST = False
BUCKET_NAME = settings.BUCKET_NAME
PATH_TO_SAVE_DATA = settings.PATH_TO_SAVE_DATA

if MODE_TEST:
        PATH_DF_METEO_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_for_test.csv')
        PATH_JSON_METEO_TEMP_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'data_meteo_temp_fr_for_test.json')
        PATH_DF_METEO_FR_OUT = os.path.join(PATH_TO_SAVE_DATA, 
                'df_meteo_fr_for_test_out.csv')
else:
        PATH_DF_METEO_FR = os.path.join(PATH_TO_SAVE_DATA, 'df_meteo_fr.csv')
        PATH_DF_METEO_FR_OUT = PATH_DF_METEO_FR
        PATH_JSON_METEO_TEMP_FR = os.path.join(PATH_TO_SAVE_DATA, 
                'data_meteo_temp_fr.json')

s3 = boto3.resource('s3')

def download_file_S3(path_file, bucket_name):
        filename = ntpath.basename(path_file)
        s3.Bucket(bucket_name).download_file(filename, path_file)

def upload_file_to_S3(filename, bucket_name):
        key = ntpath.basename(filename)
        s3.Bucket(bucket_name).upload_file(filename, key)

# get data input from S3
if MODE_S3:
        download_file_S3(PATH_JSON_METEO_TEMP_FR, BUCKET_NAME)
        download_file_S3(PATH_DF_METEO_FR, BUCKET_NAME)
# Configure Spark
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
        df_tmin = df_tmin.groupby('date').agg(fun_sql.mean('T_min').alias('T_min')).\
                orderBy('date')
        #df_tmin.show()
        #df_tmin.persist()
        # T_max group by station and date
        df_tmax = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.max('t').alias('T_max'))
        df_tmax = df_tmax.groupby('date').agg(fun_sql.mean('T_max').alias('T_max')).\
                orderBy('date')

        # join
        df_meteo =  df_tmin.join(df_tmax, df_tmin.date == df_tmax.date).\
                drop(df_tmax.date)

        # H_min group by station and date
        df_hmin = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.min('u').alias('H_min'))
        df_hmin = df_hmin.groupby('date').agg(fun_sql.mean('H_min').alias('H_min')).\
                orderBy('date')

        # join
        df_meteo =  df_meteo.join(df_hmin, df_meteo.date == df_hmin.date).\
                drop(df_hmin.date)

        # H_min group by station and date
        df_hmax = dfMeteo.groupby('date', 
                'numer_sta').agg(fun_sql.max('u').alias('H_max'))
        df_hmax = df_hmax.groupby('date').agg(fun_sql.mean('H_max').alias('H_max')).\
                orderBy('date')

        # join
        df_meteo =  df_meteo.join(df_hmax, df_meteo.date == df_hmax.date).\
                drop(df_hmax.date)

        # prepare to save
        df_meteo_fr_new = df_meteo.toPandas()
        df_meteo_fr_new.index = df_meteo_fr_new["date"]
        # load old data 
        df_meteo_fr = pd.read_csv(PATH_DF_METEO_FR)
        df_meteo_fr.index = df_meteo_fr["date"]

        # append new data 
        try:
                df_meteo_fr = df_meteo_fr.append(df_meteo_fr_new, verify_integrity=True)
                # save df_meteo
                df_meteo_fr.to_csv(PATH_DF_METEO_FR_OUT, index=False) 
                # export on S3
                if MODE_S3:
                        upload_file_to_S3(PATH_DF_METEO_FR_OUT, BUCKET_NAME)
        except: 
                print('meteo spark : data duplicates ?')

        df_meteo.show()

#input("press ctrl+c to exit")
