"""
Airflow DAG for Fraud Detection ETL Pipeline
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import random

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'fraud-detection',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def extract_transactions(**context):
    """Extract transactions from source"""
    logger.info("Extracting transactions from source...")
    
    # Simulate extraction of transaction data
    transactions = []
    merchants = ['Amazon', 'Walmart', 'Target', 'Best Buy', 'Starbucks', 'Unknown Store', 'Test Merchant']
    categories = ['shopping', 'groceries', 'entertainment', 'food', 'travel']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    # Generate sample transactions
    for i in range(10):
        transaction = {
            'amount': random.uniform(10, 15000),
            'merchant': random.choice(merchants),
            'category': random.choice(categories),
            'location': random.choice(locations),
            'user_id': f'user_{random.randint(1, 100)}'
        }
        transactions.append(transaction)
    
    logger.info(f"Extracted {len(transactions)} transactions")
    context['ti'].xcom_push(key='transactions', value=transactions)


def transform_transactions(**context):
    """Transform and validate transactions"""
    logger.info("Transforming transactions...")
    
    transactions = context['ti'].xcom_pull(key='transactions', task_ids='extract_transactions')
    
    transformed = []
    for txn in transactions:
        # Add validation and transformation logic
        if txn['amount'] > 0:
            txn['amount'] = round(txn['amount'], 2)
            transformed.append(txn)
    
    logger.info(f"Transformed {len(transformed)} valid transactions")
    context['ti'].xcom_push(key='transformed_transactions', value=transformed)


def detect_fraud(**context):
    """Apply fraud detection model"""
    logger.info("Running fraud detection...")
    
    transactions = context['ti'].xcom_pull(key='transformed_transactions', task_ids='transform_transactions')
    
    # Simple fraud detection logic
    for txn in transactions:
        # Flag high-value transactions or suspicious merchants
        if txn['amount'] > 10000 or 'unknown' in txn['merchant'].lower():
            txn['is_fraud'] = True
            txn['fraud_score'] = 0.9
        else:
            txn['is_fraud'] = False
            txn['fraud_score'] = 0.1
    
    fraud_count = sum(1 for txn in transactions if txn['is_fraud'])
    logger.info(f"Detected {fraud_count} fraudulent transactions out of {len(transactions)}")
    
    context['ti'].xcom_push(key='processed_transactions', value=transactions)


def load_transactions(**context):
    """Load transactions into database"""
    logger.info("Loading transactions into database...")
    
    transactions = context['ti'].xcom_pull(key='processed_transactions', task_ids='detect_fraud')
    
    # In a real implementation, this would insert into the database
    # For now, we'll just log the results
    for txn in transactions:
        logger.info(f"Transaction: {txn['merchant']} - ${txn['amount']:.2f} - Fraud: {txn['is_fraud']}")
    
    logger.info(f"Successfully loaded {len(transactions)} transactions")


def send_alerts(**context):
    """Send alerts for fraudulent transactions"""
    logger.info("Sending fraud alerts...")
    
    transactions = context['ti'].xcom_pull(key='processed_transactions', task_ids='detect_fraud')
    
    fraud_transactions = [txn for txn in transactions if txn['is_fraud']]
    
    if fraud_transactions:
        logger.warning(f"ALERT: {len(fraud_transactions)} fraudulent transactions detected!")
        for txn in fraud_transactions:
            logger.warning(f"Fraud Alert: {txn['merchant']} - ${txn['amount']:.2f}")
    else:
        logger.info("No fraudulent transactions detected")


# Define the DAG
dag = DAG(
    'fraud_detection_etl',
    default_args=default_args,
    description='ETL pipeline for fraud detection',
    schedule_interval=timedelta(minutes=15),  # Run every 15 minutes
    catchup=False,
    tags=['fraud-detection', 'etl'],
)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_transactions',
    python_callable=extract_transactions,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_transactions',
    python_callable=transform_transactions,
    dag=dag,
)

detect_fraud_task = PythonOperator(
    task_id='detect_fraud',
    python_callable=detect_fraud,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_transactions',
    python_callable=load_transactions,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='send_alerts',
    python_callable=send_alerts,
    dag=dag,
)

# Set task dependencies
extract_task >> transform_task >> detect_fraud_task >> [load_task, alert_task]
