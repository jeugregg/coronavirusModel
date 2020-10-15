# -*- coding: utf-8 -*-

''' AirFlow DAG : update DATA with AWS EMR Spark to precompute meteo
On AWS S3 from SPF and Meteo France
- check if update is available
- download raw data on S3
- treat data with AWS EMR Spark ephemere 
- save results in tables (df_gouv_fr_raw / df_meteo_fr or df_feat_fr )
- 
'''

# built-in import
import os, sys
import datetime

# third party import
from airflow import DAG
from airflow.operators.python_operator import PythonOperator 
from airflow.operators.dummy_operator import DummyOperator 
from airflow.operators.bash_operator import BashOperator 

# project import 
PATH_PROJECT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PATH_PROJECT)
import settings
from my_helpers.data_plots import get_data_gouv_fr
from my_helpers.data_plots import precompute_data_pos_disk
from my_helpers.data_plots import update_data_meteo_disk
from my_helpers.data_plots import precompute_data_meteo_light
from my_helpers.data_plots import prepare_features_disk
from my_helpers.data_maps import prepare_plot_data_map
from S3_helpers import upload_files_to_S3_with_hook

# definitions
from my_helpers.data_plots import PATH_DF_GOUV_FR_RAW, PATH_DF_POS_FR
from my_helpers.data_plots import PATH_DF_TEST_FR, PATH_DF_FEAT_FR
from my_helpers.data_maps import PATH_DF_DEP_R0, PATH_PT_FR_TEST_LAST
from my_helpers.meteo import PATH_DF_METEO_FR
from my_helpers.meteo import PATH_JSON_METEO_TEMP_FR

PATH_METEO_SPARK = os.path.join(PATH_PROJECT, 'meteo_spark.py')

default_args = {
    'owner': 'gregory',
    'start_date': datetime.datetime(2020, 9, 25),
    'retry_delay': datetime.timedelta(minutes=5),
}
# Using the context manager alllows you not to duplicate the dag parameter in each operator
with DAG('app_visu_covid_update', default_args=default_args, 
    schedule_interval='@daily') as my_dag:

    start_task = DummyOperator(
            task_id='dummy_start'
    )

    get_data_gouv_fr_task = PythonOperator(
        task_id='get_data_gouv_fr',
        python_callable=get_data_gouv_fr,
        dag=my_dag)

    precompute_data_pos_task = PythonOperator(
        task_id='precompute_data_pos',
        python_callable=precompute_data_pos_disk,
        dag=my_dag)

    update_data_meteo_task = PythonOperator(
        task_id='update_data_meteo',
        python_callable=update_data_meteo_disk,
        dag=my_dag)  

    precompute_data_meteo_spark_task = BashOperator(
        task_id='precompute_data_meteo_spark',
        bash_command='/usr/local/opt/apache-spark/bin/spark-submit ' + \
            PATH_METEO_SPARK,
        dag=my_dag)

    prepare_features_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features_disk,
        dag=my_dag) 

    prepare_plot_data_map_task = PythonOperator(
        task_id='prepare_plot_data_map',
        python_callable=prepare_plot_data_map,
        op_kwargs={'flag_update': True},
        dag=my_dag)

    upload_to_S3_task = PythonOperator(
        task_id='upload_to_S3',
        python_callable=upload_files_to_S3_with_hook,
        op_kwargs={
            'filenames': [PATH_DF_GOUV_FR_RAW, PATH_DF_POS_FR, PATH_DF_TEST_FR,
                PATH_DF_FEAT_FR, PATH_DF_DEP_R0, PATH_PT_FR_TEST_LAST, 
                PATH_DF_METEO_FR, PATH_JSON_METEO_TEMP_FR],
            'bucket_name': 'app-covid-visu-bucket',
        },
        dag=my_dag)

    # Use arrows to set dependencies between tasks
    start_task >> get_data_gouv_fr_task >> precompute_data_pos_task >> \
        update_data_meteo_task >> precompute_data_meteo_spark_task >> \
        prepare_features_task >> prepare_plot_data_map_task >> upload_to_S3_task