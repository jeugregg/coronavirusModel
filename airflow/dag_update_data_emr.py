# -*- coding: utf-8 -*-

''' AirFlow DAG : update DATA with AWS EMR Spark to precompute meteo
On AWS S3 from SPF and Meteo France
- check if update is available 
- download COVID data
- download raw METEO data
- upload on S3 raw METEO data
- treat data with AWS EMR Spark ephemere 
- download METEO processed data from S3
- prepare features mixing METEO and COVID data
- prepare map data from COVID data
'''

# import 

# built-in import
import os
import sys
import datetime
import ntpath

# third party import
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator 
from airflow.operators.dummy_operator import DummyOperator 
from airflow.operators.bash_operator import BashOperator 
from airflow.contrib.operators.emr_create_job_flow_operator \
    import EmrCreateJobFlowOperator
from airflow.contrib.sensors.emr_job_flow_sensor import EmrJobFlowSensor

# project import 
PATH_PROJECT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PATH_PROJECT)
import settings
from my_helpers.data_plots import get_data_gouv_fr
from my_helpers.data_plots import precompute_data_pos_disk
from my_helpers.data_plots import update_data_meteo_disk
from my_helpers.data_plots import precompute_data_meteo_light
from my_helpers.data_plots import prepare_features_disk_emr
from my_helpers.data_maps import prepare_plot_data_map
from S3_helpers import upload_files_to_S3_with_hook
from S3_helpers import download_files_from_S3

# definitions

from my_helpers.data_plots import PATH_DF_GOUV_FR_RAW, PATH_DF_POS_FR
from my_helpers.data_plots import PATH_DF_TEST_FR, PATH_DF_FEAT_FR
from my_helpers.data_maps import PATH_DF_DEP_R0, PATH_PT_FR_TEST_LAST
from my_helpers.meteo import PATH_DF_METEO_FR
from my_helpers.meteo import PATH_JSON_METEO_TEMP_FR
from my_helpers.meteo import PATH_METEO_SPARK
BUCKET_NAME = settings.BUCKET_NAME

# for DAG
default_args = {
    'owner': 'gregory',
    'start_date': days_ago(2),
    'retries': 3,
    'retry_delay': datetime.timedelta(minutes=60),
}
# For EMR
FILE_SPARK = ntpath.basename(PATH_METEO_SPARK)
SPARK_STEPS = [
    {
        'Name': 'Application Spark',
        'ActionOnFailure': 'TERMINATE_CLUSTER',
        'HadoopJarStep': {
            'Jar': 'command-runner.jar',
            'Args': ["spark-submit","--deploy-mode","cluster",
                f"s3://{BUCKET_NAME}/{FILE_SPARK}"],
        },
    }
]

EBS_CONFIG = {
    'EbsBlockDeviceConfigs': [
        {
            'VolumeSpecification': {
                'VolumeType': 'gp2',
                'SizeInGB': 32
            },
            'VolumesPerInstance': 1
        },
    ]
}
# applications Name=Ganglia Name=Spark Name=Zeppelin 
JOB_FLOW_OVERRIDES = {
    'Name': 'Treat_Data_Meteo',
    'ReleaseLabel': 'emr-5.30.1',
    "Applications": [ 
      { 
         "Name": "Ganglia",
      },
      { 
         "Name": "Spark",
      },
      { 
         "Name": "Zeppelin",
      }
    ],
    'Instances': {
        'InstanceGroups': [
            {
                'Name': 'Master node',
                'Market': 'ON_DEMAND',
                'InstanceRole': 'MASTER',
                'InstanceType': 'm4.large',
                'InstanceCount': 1,
                'EbsConfiguration': EBS_CONFIG,
            }, 
            {
                'Name': 'Principal node',
                'Market': 'ON_DEMAND',
                'InstanceRole': 'CORE',
                'InstanceType': 'm4.large',
                'InstanceCount': 2,
                'EbsConfiguration': EBS_CONFIG,
            }
        ],
        'KeepJobFlowAliveWhenNoSteps': False,
        'TerminationProtected': False,
        'Ec2KeyName': 'EC2_AWS_01',
        'Ec2SubnetId': 'subnet-6583b71f',
        'EmrManagedMasterSecurityGroup': 'sg-0c777b0f367bb7681',
        'EmrManagedSlaveSecurityGroup': 'sg-09fe5f546208999ab',
    },
    'Steps': SPARK_STEPS,
    'JobFlowRole': 'EMR_EC2_DefaultRole',
    'ServiceRole': 'EMR_DefaultRole',
    'AutoScalingRole': 'EMR_AutoScaling_DefaultRole',
    'ScaleDownBehavior': 'TERMINATE_AT_TASK_COMPLETION',
    'EbsRootVolumeSize': 10,
    'LogUri': 's3n://aws-logs-324466407431-us-east-2/elasticmapreduce/',
    "BootstrapActions": [ 
      { 
         "Name": "Personal action",
         "ScriptBootstrapAction": { 
            "Path": "s3://app-covid-visu-bucket/bootstrap-emr-meteo.sh"
         }
      }
    ],
}


# Execute at 5am UTC (7am Paris) to be sure that meteo data is ok
with DAG('app_visu_covid_upd_emr', default_args=default_args, 
    schedule_interval='0 5 * * *', catchup=False) as my_dag:

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

    # upload on S3 meteo data to be processed in EMR 
    upload_to_S3_meteo_task = PythonOperator(
        task_id='upload_to_S3_meteo',
        python_callable=upload_files_to_S3_with_hook,
        op_kwargs={
            'filenames': [PATH_DF_METEO_FR, PATH_JSON_METEO_TEMP_FR, 
            PATH_METEO_SPARK],
            'bucket_name': BUCKET_NAME,
        },
        dag=my_dag)

    # launch ephemere EMR task to process data uploaded
    precompute_data_meteo_emr_task = EmrCreateJobFlowOperator(
        task_id='precompute_data_meteo_emr',
        job_flow_overrides=JOB_FLOW_OVERRIDES,
        aws_conn_id='aws_default',
        emr_conn_id='emr_default',
    )
    # wait for EMR end
    emr_job_sensor = EmrJobFlowSensor(
        task_id='check_job_flow',
        job_flow_id="{{ task_instance.xcom_pull(" +  \
            "task_ids='precompute_data_meteo_emr', key='return_value') }}",
        aws_conn_id='aws_default',
    )

    # retrieve data processed
    download_files_from_S3_task = PythonOperator(
        task_id='download_from_S3_meteo',
        python_callable=download_files_from_S3,
        op_kwargs={
            'filenames': [PATH_DF_METEO_FR],
            'bucket_name': BUCKET_NAME,
        },
        dag=my_dag)

    prepare_features_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features_disk_emr,
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
                PATH_DF_METEO_FR],
            'bucket_name': BUCKET_NAME,
        },
        dag=my_dag)

    # Use arrows to set dependencies between tasks
    start_task >> get_data_gouv_fr_task >> precompute_data_pos_task >> \
        update_data_meteo_task >> upload_to_S3_meteo_task >> \
        precompute_data_meteo_emr_task >> emr_job_sensor >> \
        download_files_from_S3_task >> prepare_features_task >> \
        prepare_plot_data_map_task >> upload_to_S3_task