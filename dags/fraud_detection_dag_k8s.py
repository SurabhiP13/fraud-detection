"""
Airflow DAG for Fraud Detection Pipeline (Kubernetes Version).

This version uses KubernetesPodOperator for running tasks in Kubernetes pods.
"""
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.dates import days_ago
from kubernetes.client import models as k8s

# Default arguments for the DAG
default_args = {
    'owner': 'data-science',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Kubernetes configuration.
# Image names default to the local tags built into minikube's docker daemon
# (see scripts/start_cluster.sh). The CD workflow overrides them through the
# Helm chart's `env:` list so the same DAG can run registry-tagged images.
IMAGE = os.getenv('FRAUD_AIRFLOW_IMAGE', 'fraud-detection-airflow:v1')
MODEL_TRAINING_IMAGE = os.getenv('FRAUD_TRAINING_IMAGE', 'fraud-detection-model-training:v1')  # Separate image with mlflow
NAMESPACE = os.getenv('FRAUD_NAMESPACE', 'airflow')

# Volume configuration for shared data
volume = k8s.V1Volume(
    name='fraud-data',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='fraud-data-pvc'),
)

volume_mount = k8s.V1VolumeMount(
    name='fraud-data',
    mount_path='/opt/airflow/data',
    sub_path=None,
    read_only=False
)

# Security context to fix permission issues
security_context = k8s.V1PodSecurityContext(
    fs_group=50000,  # Airflow user group
    run_as_user=50000,  # Airflow user
)

# Init container to fix volume permissions
init_container = k8s.V1Container(
    name="fix-permissions",
    image="busybox:latest",
    command=["sh", "-c", "chmod -R 777 /opt/airflow/data && chown -R 50000:50000 /opt/airflow/data"],
    volume_mounts=[volume_mount],
    security_context=k8s.V1SecurityContext(run_as_user=0)  # Run as root
)

# MLflow artifacts volume (shared with MLflow server)
mlflow_volume = k8s.V1Volume(
    name='mlflow-artifacts',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='mlflow-pvc'),
)

mlflow_volume_mount = k8s.V1VolumeMount(
    name='mlflow-artifacts',
    mount_path='/mlflow',
    sub_path=None,
    read_only=False
)

# The MLflow server uses a non-proxied artifact root, so the training pod writes
# model artifacts straight into /mlflow/artifacts on the shared PVC. Make sure
# uid 50000 can write there too (k8s/mlflow-deployment.yaml runs the server as
# the same uid; this covers PVCs created before that change).
training_init_container = k8s.V1Container(
    name="fix-permissions",
    image="busybox:latest",
    command=["sh", "-c",
             "chmod -R 777 /opt/airflow/data && chown -R 50000:50000 /opt/airflow/data && "
             "mkdir -p /mlflow/artifacts && chown -R 50000:50000 /mlflow && chmod -R 775 /mlflow"],
    volume_mounts=[volume_mount, mlflow_volume_mount],
    security_context=k8s.V1SecurityContext(run_as_user=0)
)

# Define the DAG
# dag = DAG(
#     'fraud_detection_pipeline_k8s',
#     default_args=default_args,
#     description='End-to-end fraud detection pipeline (Kubernetes)',
#     schedule_interval=None,
#     start_date=days_ago(1), #moving start date to yesterday to avoid scheduling issues
#     catchup=False,
#     tags=['fraud-detection', 'machine-learning', 'kubernetes'],
# )
#one with scheduler interval
dag = DAG(
    'fraud_detection_pipeline_k8s',
    default_args=default_args,
    description='Scheduled retraining: runs only when new raw data has landed',
    schedule='@weekly',                 # or a cron string, e.g. '0 2 * * 1'
    start_date=datetime(2026, 1, 1),    # fixed date, not days_ago()
    catchup=False, #means only the latest scheduled run is executed, not all missed runs
    max_active_runs=1,                  # never overlap two trainings on the same PVC
    tags=['fraud-detection', 'machine-learning', 'kubernetes'],
)


# Task 1: Setup
setup_task = KubernetesPodOperator(
    task_id='setup_directories',
    name='setup-directories',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_setup.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    security_context=security_context,
    init_containers=[init_container],  # Fix permissions before running
    get_logs=True,
    is_delete_operator_pod=False,  # Keep pods for debugging
    dag=dag,
)

# Task 2: Data Ingestion
data_ingestion_task = KubernetesPodOperator(
    task_id='data_ingestion',
    name='data-ingestion',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_data_ingestion.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    security_context=security_context,
    init_containers=[init_container],
    get_logs=True,
    is_delete_operator_pod=False,  # Keep pods for debugging
    dag=dag,
)

# Task 3: Data Cleaning
data_cleaning_task = KubernetesPodOperator(
    task_id='data_cleaning',
    name='data-cleaning',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_data_cleaning.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    security_context=security_context,
    init_containers=[init_container],
    get_logs=True,
    is_delete_operator_pod=False,
    dag=dag,
)

# Task 4: Feature Engineering
feature_engineering_task = KubernetesPodOperator(
    task_id='feature_engineering',
    name='feature-engineering',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_feature_engineering.py'],
    volumes=[volume],
    volume_mounts=[volume_mount],
    security_context=security_context,
    init_containers=[init_container],
    get_logs=True,
    is_delete_operator_pod=False,
    dag=dag,
)

# Task 5: Model Training (uses separate image with mlflow)
model_training_task = KubernetesPodOperator(
    task_id='model_training',
    name='model-training',
    namespace=NAMESPACE,
    image=MODEL_TRAINING_IMAGE,  # Use model-training specific image
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/run_model_training.py'],
    volumes=[volume, mlflow_volume],  # Mount both data and mlflow PVCs
    volume_mounts=[volume_mount, mlflow_volume_mount],  # Mount both volumes
    security_context=security_context,
    init_containers=[training_init_container],  # Also fixes /mlflow ownership
    env_vars=[
        k8s.V1EnvVar(name='MLFLOW_TRACKING_URI', value='http://mlflow-service.airflow.svc.cluster.local:5000'),
    ],
    get_logs=True,
    is_delete_operator_pod=False,  # Keep pod for inspection
    dag=dag,
)

check_new_data_task = KubernetesPodOperator(
    task_id='check_new_data',
    name='check-new-data',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/check_new_data.py'],
    skip_on_exit_code=99,               # exit 99 => task "skipped", downstream skipped too
    volumes=[volume], volume_mounts=[volume_mount],
    security_context=security_context, init_containers=[init_container],
    get_logs=True, is_delete_operator_pod=False, dag=dag,
)

mark_processed_task = KubernetesPodOperator(
    task_id='mark_data_processed',
    name='mark-data-processed',
    namespace=NAMESPACE,
    image=IMAGE,
    image_pull_policy='IfNotPresent',
    cmds=['python3'],
    arguments=['/opt/airflow/scripts/check_new_data.py', '--commit'],
    volumes=[volume], volume_mounts=[volume_mount],
    security_context=security_context, init_containers=[init_container],
    get_logs=True, is_delete_operator_pod=False, dag=dag,
)


# # Define task dependencies
# setup_task >> data_ingestion_task >> data_cleaning_task >> feature_engineering_task >> model_training_task

#new for conditional execution: check for new data first, then run the rest of the pipeline only if new data is detected
check_new_data_task >> setup_task >> data_ingestion_task >> data_cleaning_task \
    >> feature_engineering_task >> model_training_task >> mark_processed_task
