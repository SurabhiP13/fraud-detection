###THIS DOES NOT HAVE THE NEWER CHANGES FROM THE KUBERNETES VERSION. IT IS ONLY FOR LOCAL DEVELOPMENT WITH DOCKER COMPOSE.###

"""
Airflow DAG for Fraud Detection Pipeline (Docker Compose Version).

This version uses DockerOperator instead of KubernetesPodOperator
for local development with Docker Compose.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

# Default arguments for the DAG
default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Docker configuration
IMAGE = 'fraud-detection-pipeline:latest'
NETWORK = 'fraud-detection_fraud-detection'  # Docker Compose network

# Volume mounts for shared data. Names are pinned in docker-compose.yml
# (volumes.*.name) so they do not depend on the compose project name.
mounts = [
    Mount(source='fraud-detection_data', target='/opt/airflow/data', type='volume'),
]
# The MLflow server uses a non-proxied artifact root, so the training container
# must see the same /mlflow path the server and Streamlit use.
mlflow_mounts = mounts + [
    Mount(source='fraud-detection_mlflow', target='/mlflow', type='volume'),
]

# Define the DAG
dag = DAG(
    'fraud_detection_pipeline_docker',
    default_args=default_args,
    description='End-to-end fraud detection pipeline (Docker Compose)',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['fraud-detection', 'machine-learning', 'docker'],
)

# Task 1: Setup
setup_task = DockerOperator(
    task_id='setup_directories',
    image=IMAGE,
    command='python3 /opt/airflow/scripts/run_setup.py',
    mounts=mounts,
    network_mode=NETWORK,
    auto_remove=False,
    docker_url='unix://var/run/docker.sock',
    retries=0,
    dag=dag,
)

# Task 2: Data Ingestion
data_ingestion_task = DockerOperator(
    task_id='data_ingestion',
    image=IMAGE,
    command='python3 /opt/airflow/scripts/run_data_ingestion.py',
    mounts=mounts,
    network_mode=NETWORK,
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    dag=dag,
)

# Task 3: Data Cleaning
data_cleaning_task = DockerOperator(
    task_id='data_cleaning',
    image=IMAGE,
    command='python3 /opt/airflow/scripts/run_data_cleaning.py',
    mounts=mounts,
    network_mode=NETWORK,
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    dag=dag,
)

# Task 4: Feature Engineering
feature_engineering_task = DockerOperator(
    task_id='feature_engineering',
    image=IMAGE,
    command='python3 /opt/airflow/scripts/run_feature_engineering.py',
    mounts=mounts,
    network_mode=NETWORK,
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    dag=dag,
)

# Task 5: Model Training
model_training_task = DockerOperator(
    task_id='model_training',
    image=IMAGE,
    command='python3 /opt/airflow/scripts/run_model_training.py',
    mounts=mlflow_mounts,
    network_mode=NETWORK,
    environment={'MLFLOW_TRACKING_URI': 'http://mlflow:5000'},
    auto_remove=True,
    docker_url='unix://var/run/docker.sock',
    dag=dag,
)

# Define task dependencies
setup_task >> data_ingestion_task >> data_cleaning_task >> feature_engineering_task >> model_training_task
