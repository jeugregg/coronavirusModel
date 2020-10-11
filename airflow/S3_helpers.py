# -*- coding: utf-8 -*-
''' Helpers S3 AWS Storage for Airflow 
'''
import airflow.hooks.S3_hook
import ntpath

def upload_files_to_S3_with_hook(filenames, bucket_name):
    ''' Upload list of files (replace if already exist)
    '''
    hook = airflow.hooks.S3_hook.S3Hook('my_S3_conn')

    for filename in filenames:
        key = ntpath.basename(filename)
        hook.load_file(filename, key, bucket_name, replace=True)