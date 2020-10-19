# -*- coding: utf-8 -*-
''' 
Helpers S3 AWS Storage for Airflow 
'''

import datetime
import re
import shutil

import airflow.hooks.S3_hook
import ntpath
import boto3

# prepare helpers
s3 = boto3.resource('s3')
hook = airflow.hooks.S3_hook.S3Hook('my_S3_conn')

# save file before update
def clean_file(path_file_name):
    '''
    Clean file already traited : rename file with date
    '''
    try:
        d = datetime.datetime.now()
        str_date = '_' + d.strftime("%Y%m%d_%H_%M_%S")
       
        res_re = re.search('\.\w+$', path_file_name)
        
        path_file_name_saved = \
            path_file_name[0:res_re.start()] + str_date + res_re.group(0)
         
        shutil.move(path_file_name, path_file_name_saved) 
        print('File {} moved!'.format(path_file_name_saved))
    except:
        print('File {} does not exist!'.format(path_file_name))

def upload_files_to_S3_with_hook(filenames, bucket_name):
    ''' 
    Upload list of files to S3 (replace if already exist)
    '''
    for filename in filenames:
        key = ntpath.basename(filename)
        hook.load_file(filename, key, bucket_name, replace=True)

def download_files_from_S3(filenames, bucket_name):
    ''' 
    Download list of files from S3 (replace if already exist)
    but saved with date
    '''
    for filename in filenames:
        clean_file(filename)
        key = ntpath.basename(filename)
        s3.Bucket(bucket_name).download_file(key, filename)